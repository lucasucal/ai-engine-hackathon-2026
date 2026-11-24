# /// script
# requires-python = ">=3.12"
# # rclpy comes from the sourced ROS install through PYTHONPATH, but uv's environment hides the
# # system packages it imports (its package.xml's python3-* dependencies). Its C extensions are
# # built for the system Python only.
# dependencies = ["numpy", "pyyaml", "typing-extensions"]
#
# [tool.uv]
# python-preference = "only-system"
# ///
"""Record demonstrations as episodes of API.md's /act observations, one JSON file each.

On start it puts the arm in freedrive: it deactivates whichever controllers command the arm's joints,
activates freedrive_mode_controller and keeps freedrive enabled. When it stops, by Ctrl-C or
otherwise, it ends freedrive and reactivates those controllers, which hold the arm where it is. If
freedrive_mode_controller is already active, it cannot tell what came before and reactivates
joint_trajectory_controller. With --no-freedrive it leaves the controllers alone.

Start it with the button released, as whatever the digital input reads then counts as released,
unless --pressed-state says otherwise. An episode runs while the button is held: it starts when the
button is pressed and ends when it is released.

Each 50 ms step in the file's `steps` has the `step`, `images` and `proprio` of an /act request,
with the palm pose computed as the policy runner computes it, plus `t`, the seconds since the
episode started. `proprio.gripper` is null while /gripper/joint_states is not publishing, and
`images.top` is null, as nothing records the camera yet.

    uv run record.py --task-id bench_lift_can_v0

Episodes go to recording/<task_id>/<episode_id>.json beside this script.
"""

import argparse
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from math import pi
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.client import Client
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from ur_msgs.msg import IOStates

SCHEMA_VERSION = 2
STEP_S = 0.05
STALE_AFTER_S = 0.1
STARTUP_TIMEOUT_S = 10.0
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
FINGER_JOINT = "finger_joint"
FINGER_CLOSED = 0.695
FREEDRIVE_CONTROLLER = "freedrive_mode_controller"
DEFAULT_ARM_CONTROLLER = "joint_trajectory_controller"
# freedrive_mode_controller ends freedrive after its inactive_timeout, 1 s, without a message.
FREEDRIVE_KEEPALIVE_S = 0.2

# (d, a, alpha) per joint, and the palm's offset and axes, as crates/policy-runner/src/kinematics.rs
# has them, so that recorded ee_pos and ee_quat match what the policy is sent.
UR12E_DH = (
    (0.1807, 0.0, pi / 2),
    (0.0, -0.6127, 0.0),
    (0.0, -0.57155, 0.0),
    (0.17415, 0.0, pi / 2),
    (0.11985, 0.0, -pi / 2),
    (0.11655, 0.0, 0.0),
)
PALM_OFFSET_IN_TOOL0 = np.array([0.0, 0.107, 0.0])
PALM_AXES_IN_TOOL0 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


def quaternion_wxyz(m: np.ndarray) -> np.ndarray:
    trace = np.trace(m)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    return np.array(q) / np.linalg.norm(q)


def palm_pose(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The palm's position and wxyz orientation in base_link, the DH base turned 180° about z."""
    pose = transform(rot_z(pi), np.zeros(3))
    for (d, a, alpha), theta in zip(UR12E_DH, joints, strict=True):
        translation = np.array([a * np.cos(theta), a * np.sin(theta), d])
        pose = pose @ transform(rot_z(theta) @ rot_x(alpha), translation)
    pose = pose @ transform(PALM_AXES_IN_TOOL0, PALM_OFFSET_IN_TOOL0)
    orientation = quaternion_wxyz(pose[:3, :3])
    # The contract's sign: the largest-magnitude component positive, the first of equals deciding.
    if orientation[np.argmax(np.abs(orientation))] < 0:
        orientation = -orientation
    return pose[:3, 3].copy(), orientation


class Debouncer:
    """Reports a new state only once the input has held it for `hold_s`."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.stable: int | None = None
        self.candidate: int | None = None
        self.candidate_since = 0.0

    def update(self, raw: int, now: float) -> int | None:
        if self.stable is None:
            self.stable = raw
            return None
        if raw == self.stable:
            self.candidate = None
            return None
        if raw != self.candidate:
            self.candidate, self.candidate_since = raw, now
        if now - self.candidate_since < self.hold_s:
            return None
        self.stable, self.candidate = raw, None
        return raw


@dataclass
class ArmState:
    at: float
    position: np.ndarray
    velocity: np.ndarray


class Robot:
    def __init__(self, node: Node, pin: int, pressed_state: int | None, debounce_s: float) -> None:
        self.pin = pin
        self.pressed_state = pressed_state
        self.debouncer = Debouncer(debounce_s)
        self.button_pressed: queue.Queue[bool] = queue.Queue()
        self.arm: ArmState | None = None
        self.finger: tuple[float, float] | None = None
        self.io_first_at: float | None = None
        self.io_last_at = 0.0
        node.create_subscription(JointState, "/joint_states", self.on_arm, 10)
        node.create_subscription(JointState, "/gripper/joint_states", self.on_finger, 10)
        node.create_subscription(IOStates, "/io_and_status_controller/io_states", self.on_io, 100)

    def on_arm(self, msg: JointState) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(j in index for j in ARM_JOINTS) or len(msg.velocity) != len(msg.name):
            return
        order = [index[j] for j in ARM_JOINTS]
        self.arm = ArmState(
            time.monotonic(),
            np.array([msg.position[i] for i in order]),
            np.array([msg.velocity[i] for i in order]),
        )

    def on_finger(self, msg: JointState) -> None:
        if FINGER_JOINT in msg.name:
            self.finger = (time.monotonic(), msg.position[msg.name.index(FINGER_JOINT)])

    def on_io(self, msg: IOStates) -> None:
        now = time.monotonic()
        state = next((int(d.state) for d in msg.digital_in_states if d.pin == self.pin), None)
        if state is None:
            return
        if self.io_first_at is None:
            self.io_first_at = now
            if self.pressed_state is None:
                self.pressed_state = 1 - state
                print(
                    f"Digital input {self.pin} reads {state} at rest, so a press reads "
                    f"{self.pressed_state}. Hold the button to record an episode."
                )
            elif state == self.pressed_state:
                print(f"Digital input {self.pin} reads {state}, pressed: release it first.")
            else:
                print(f"Digital input {self.pin} reads {state}. Hold the button to record.")
        self.io_last_at = now
        changed_to = self.debouncer.update(state, now)
        if changed_to is not None:
            self.button_pressed.put(changed_to == self.pressed_state)

    def fresh_arm(self, now: float) -> ArmState | None:
        arm = self.arm
        return arm if arm is not None and now - arm.at <= STALE_AFTER_S else None

    def fresh_gripper_opening(self, now: float) -> float | None:
        finger = self.finger
        if finger is None or now - finger[0] > STALE_AFTER_S:
            return None
        return float(np.clip(1.0 - finger[1] / FINGER_CLOSED, 0.0, 1.0))


@dataclass
class Episode:
    task_id: str
    start: float
    episode_id: str = field(default_factory=lambda: f"ep_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3])
    started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    steps: list[dict[str, Any]] = field(default_factory=list)
    stale_steps: int = 0

    def add_step(self, now: float, robot: Robot) -> None:
        proprio: dict[str, Any] = {
            "joint_pos": None,
            "joint_vel": None,
            "gripper": robot.fresh_gripper_opening(now),
            "ee_pos": None,
            "ee_quat": None,
        }
        arm = robot.fresh_arm(now)
        if arm is None:
            self.stale_steps += 1
        else:
            position, orientation = palm_pose(arm.position)
            proprio |= {
                "joint_pos": arm.position.tolist(),
                "joint_vel": arm.velocity.tolist(),
                "ee_pos": position.tolist(),
                "ee_quat": orientation.tolist(),
            }
        self.steps.append(
            {
                "step": len(self.steps),
                "t": round(now - self.start, 4),
                "images": {"top": None},
                "proprio": proprio,
            }
        )

    def save(self, directory: Path, ended_by: str) -> Path:
        path = directory / self.task_id / f"{self.episode_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "schema_version": SCHEMA_VERSION,
            "step_s": STEP_S,
            "started_at": self.started_at,
            "ended_by": ended_by,
            "stale_steps": self.stale_steps,
            "joint_names": list(ARM_JOINTS),
        }
        # One step per line, so the file stays readable however long the episode.
        steps = ",\n".join(f"    {json.dumps(step, allow_nan=False)}" for step in self.steps)
        head = json.dumps(header, indent=2, allow_nan=False).removesuffix("\n}")
        partial = path.with_name(path.name + ".part")
        partial.write_text(f'{head},\n  "steps": [\n{steps}\n  ]\n}}\n')
        partial.rename(path)
        return path


def spin(executor: SingleThreadedExecutor) -> None:
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


class ControllerError(Exception):
    pass


def call[Request, Response](
    client: Client[Request, Response], request: Request, timeout_s: float = 5.0
) -> Response:
    future = client.call_async(request)
    deadline = time.monotonic() + timeout_s
    while not future.done():
        if time.monotonic() > deadline:
            raise ControllerError(f"{client.srv_name} did not answer within {timeout_s:g} s")
        time.sleep(0.01)
    response = future.result()
    if response is None:
        raise ControllerError(f"{client.srv_name} gave no answer")
    return response


class Freedrive:
    """Swaps the controllers commanding the arm's joints for freedrive_mode_controller, and back."""

    def __init__(self, node: Node, controller_manager: str) -> None:
        self.controller_manager = controller_manager
        self.lister = node.create_client(ListControllers, f"{controller_manager}/list_controllers")
        self.switcher = node.create_client(
            SwitchController, f"{controller_manager}/switch_controller"
        )
        self.enable = node.create_publisher(
            Bool, f"/{FREEDRIVE_CONTROLLER}/enable_freedrive_mode", 10
        )
        self.restore: list[str] = []
        self.engaged = False
        node.create_timer(FREEDRIVE_KEEPALIVE_S, self.keep_alive)

    def keep_alive(self) -> None:
        if self.engaged:
            self.enable.publish(Bool(data=True))

    def start(self) -> None:
        for client in (self.lister, self.switcher):
            if not client.wait_for_service(timeout_sec=5.0):
                raise ControllerError(f"cannot reach {self.controller_manager}")
        active = [
            c
            for c in call(self.lister, ListControllers.Request()).controller
            if c.state == "active"
        ]
        if any(c.name == FREEDRIVE_CONTROLLER for c in active):
            self.restore = [DEFAULT_ARM_CONTROLLER]
            print(
                f"{FREEDRIVE_CONTROLLER} was already active, so {DEFAULT_ARM_CONTROLLER} "
                "takes over on exit."
            )
        else:
            self.restore = [
                c.name
                for c in active
                if any(i.split("/")[0] in ARM_JOINTS for i in c.claimed_interfaces)
            ]
            self.switch(activate=[FREEDRIVE_CONTROLLER], deactivate=self.restore)
        self.engaged = True
        self.keep_alive()
        print(f"Freedrive on. On exit it ends and {self.restored()} takes over again.")

    def stop(self) -> None:
        if not self.engaged:
            return
        self.engaged = False
        self.enable.publish(Bool(data=False))
        self.switch(activate=self.restore, deactivate=[FREEDRIVE_CONTROLLER])
        print(f"Freedrive off; {self.restored()} active again.")

    def switch(self, activate: list[str], deactivate: list[str]) -> None:
        request = SwitchController.Request(
            activate_controllers=activate,
            deactivate_controllers=deactivate,
            strictness=SwitchController.Request.STRICT,
        )
        request.timeout.sec = 3
        response = call(self.switcher, request)
        if not response.ok:
            detail = f": {response.message}" if response.message else ""
            raise ControllerError(f"activating {activate} for {deactivate} failed{detail}")

    def restored(self) -> str:
        return ", ".join(self.restore) or "no controller"

    def manual_stop_command(self) -> str:
        activate = f" --activate {' '.join(self.restore)}" if self.restore else ""
        return f"ros2 control switch_controllers --deactivate {FREEDRIVE_CONTROLLER}{activate}"


def interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def finish(episode: Episode, out_dir: Path, ended_by: str) -> Path:
    path = episode.save(out_dir, ended_by)
    steps = len(episode.steps)
    stale = f", {episode.stale_steps} without a fresh arm state" if episode.stale_steps else ""
    print(f"\nSaved {steps} steps ({steps * STEP_S:.2f} s{stale}) to {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task-id", required=True, help='e.g. bench_lift_can_v0, or "hidden"')
    parser.add_argument("--pin", type=int, default=0, help="the button's digital input (default 0)")
    parser.add_argument(
        "--pressed-state",
        type=int,
        choices=(0, 1),
        help="what the input reads while the button is pressed; "
        "default the opposite of what it reads when the recorder starts",
    )
    parser.add_argument(
        "--debounce-ms",
        type=float,
        default=20.0,
        help="a press or release counts once the input has held it this long (default 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "recording",
        help="default recording/ beside this script",
    )
    parser.add_argument(
        "--controller-manager",
        default="/controller_manager",
        help="the arm's controller manager (default /controller_manager)",
    )
    parser.add_argument(
        "--no-freedrive",
        action="store_true",
        help="leave the controllers alone, for freedrive started and ended by hand",
    )
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    # rclpy's own Ctrl-C handler would shut ROS down before the controllers can be switched back.
    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    node = rclpy.create_node("demo_recorder")
    robot = Robot(node, args.pin, args.pressed_state, args.debounce_ms / 1000)
    freedrive = None if args.no_freedrive else Freedrive(node, args.controller_manager)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=spin, args=(executor,), daemon=True)
    spinner.start()

    saved: list[Path] = []
    episode: Episode | None = None
    next_step_at = 0.0
    stop_reason = "Ctrl-C"
    try:
        if freedrive is not None:
            freedrive.start()
        print(f"Waiting for the button on digital input {args.pin}. Ctrl-C to quit.")
        started_waiting = time.monotonic()
        while rclpy.ok():
            now = time.monotonic()
            if robot.io_first_at is None:
                if now - started_waiting > STARTUP_TIMEOUT_S:
                    stop_reason = (
                        f"no io_states within {STARTUP_TIMEOUT_S:.0f} s; is the driver running?"
                    )
                    break
            elif now - robot.io_last_at > 2.0:
                stop_reason = "io_states stopped for 2 s"
                break

            while not robot.button_pressed.empty():
                pressed = robot.button_pressed.get_nowait()
                if not pressed and episode is not None:
                    saved.append(finish(episode, args.out_dir, "release"))
                    episode = None
                elif not pressed or episode is not None:
                    continue
                elif robot.fresh_arm(now) is None:
                    print("Not starting: no fresh /joint_states.")
                else:
                    episode = Episode(args.task_id, now)
                    next_step_at = now
                    no_gripper = robot.fresh_gripper_opening(now) is None
                    gripper = (
                        "; no fresh /gripper/joint_states, gripper is null" if no_gripper else ""
                    )
                    print(f"Recording {episode.episode_id}. Release to stop{gripper}.")

            if episode is not None and now >= next_step_at:
                episode.add_step(now, robot)
                next_step_at += STEP_S
                if now - next_step_at > STEP_S:
                    next_step_at = now + STEP_S
                steps = len(episode.steps)
                if steps % 20 == 0:
                    print(f"\r  step {steps} ({steps * STEP_S:.0f} s)", end="", flush=True)
            time.sleep(0.002 if episode is not None else 0.01)
    except KeyboardInterrupt:
        pass
    except ControllerError as error:
        stop_reason = f"could not start freedrive, {error}"
    finally:
        # A second Ctrl-C, or a closing terminal's signal arriving twice, must not cut short saving
        # the episode and leaving freedrive.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        if episode is not None:
            saved.append(finish(episode, args.out_dir, stop_reason))
        if freedrive is not None:
            try:
                freedrive.stop()
            except ControllerError as error:
                print(
                    f"Could not leave freedrive ({error}). Run:\n  {freedrive.manual_stop_command()}"
                )
        rclpy.try_shutdown()
        spinner.join(timeout=2.0)

    print(f"\nStopped: {stop_reason}. {len(saved)} episode(s) saved this session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
