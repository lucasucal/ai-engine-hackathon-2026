# /// script
# requires-python = ">=3.12"
# # rclpy comes from the sourced ROS install through PYTHONPATH, but uv's environment hides the
# # system packages it imports (its package.xml's python3-* dependencies). Its C extensions are
# # built for the system Python only.
# dependencies = ["numpy", "pillow", "pyyaml", "typing-extensions"]
#
# [tool.uv]
# python-preference = "only-system"
# ///
"""Record demonstrations as episodes of each task, one JSON file each."""

import argparse
import base64
import io
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
from PIL import Image, UnidentifiedImageError
from rclpy.client import Client
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64MultiArray
from ur_msgs.msg import Analog, IOStates

SCHEMA_VERSION = 2
STEP_S = 0.05
STALE_AFTER_S = 0.1
STARTUP_TIMEOUT_S = 10.0
CAMERA_TOPIC = "/camera/color/image_raw/compressed"
IMAGE_SIZE = (640, 480)
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
FINGER_COMMAND_TOPIC = "/gripper/finger_position_controller/commands"
# The gripper's driver passes a new target on only every 15-20 ms, so faster gains nothing.
GRIPPER_COMMAND_S = 0.02
# throttle_cal measured 0.84 V released and 4.35 V pressed fully; these ends sit just inside, so
# noise and a resting thumb still read as released, and every full press as fully pressed.
THROTTLE_RELEASED_V = 0.90
THROTTLE_PRESSED_V = 4.32
THROTTLE_VALID_V = (0.75, 4.50)
# The gripper's driver spreads 0 to FINGER_CLOSED over 227 steps of its position register.
GRIPPER_STEP_RAD = FINGER_CLOSED / 227
# Held still mid-travel, the throttle's raw reading spanned up to 1.5 steps, peak to peak.
GRIPPER_HYSTERESIS_STEPS = 2.0
# How far the fully pressed throttle opens the gripper, in proprio.gripper's units: 1 fully open,
# 0 closed.
GRIPPER_MAX_OPENING = 0.7
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


@dataclass
class AnalogReading:
    at: float
    domain: int
    value: float


@dataclass
class CameraFrame:
    at: float
    jpeg: bytes
    problem: str | None


def frame_problem(jpeg: bytes) -> str | None:
    """Why the policy runner would reject this frame, as its check_frame does, or None."""
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG":
                return f"not a JPEG but {image.format}"
            if image.size != IMAGE_SIZE:
                return f"{image.size[0]}x{image.size[1]}, not {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    except (UnidentifiedImageError, OSError):
        return "not an image"
    return None


class Robot:
    def __init__(
        self,
        node: Node,
        pin: int,
        pressed_state: int | None,
        debounce_s: float,
        throttle_pin: int,
        camera_topic: str,
    ) -> None:
        self.pin = pin
        self.pressed_state = pressed_state
        self.debouncer = Debouncer(debounce_s)
        self.button_pressed: queue.Queue[bool] = queue.Queue()
        self.throttle_pin = throttle_pin
        self.throttle: AnalogReading | None = None
        self.camera_topic = camera_topic
        self.camera: CameraFrame | None = None
        self.arm: ArmState | None = None
        self.finger: tuple[float, float] | None = None
        self.io_first_at: float | None = None
        self.io_last_at = 0.0
        node.create_subscription(JointState, "/joint_states", self.on_arm, 10)
        node.create_subscription(JointState, "/gripper/joint_states", self.on_finger, 10)
        node.create_subscription(IOStates, "/io_and_status_controller/io_states", self.on_io, 100)
        node.create_subscription(
            CompressedImage, camera_topic, self.on_camera, qos_profile_sensor_data
        )

    def on_camera(self, msg: CompressedImage) -> None:
        jpeg = bytes(msg.data)
        problem = frame_problem(jpeg)
        previous = self.camera
        if problem is not None and (previous is None or previous.problem != problem):
            print(f"Camera frames on {self.camera_topic} are {problem}: images.top stays null.")
        self.camera = CameraFrame(time.monotonic(), jpeg, problem)

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
        throttle = next((a for a in msg.analog_in_states if a.pin == self.throttle_pin), None)
        if throttle is not None:
            self.throttle = AnalogReading(now, throttle.domain, throttle.state)
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

    def fresh_image(self, now: float) -> str | None:
        frame = self.camera
        if frame is None or frame.problem is not None or now - frame.at > STALE_AFTER_S:
            return None
        return base64.b64encode(frame.jpeg).decode("ascii")


def opening_to_angle(opening: float) -> float:
    return (1.0 - opening) * FINGER_CLOSED


def finger_angle(volts: float, released_v: float, pressed_v: float, open_angle: float) -> float:
    pressed = (volts - released_v) / (pressed_v - released_v)
    if pressed <= 0.0:
        return FINGER_CLOSED
    if pressed >= 1.0:
        return open_angle
    return FINGER_CLOSED - pressed * (FINGER_CLOSED - open_angle)


class Hysteresis:
    """Holds a value until its input moves `band` from it, or reaches either end of its range."""

    def __init__(self, band: float, low: float, high: float) -> None:
        self.band, self.low, self.high = band, low, high
        self.value: float | None = None

    def update(self, target: float) -> float:
        held = self.value
        at_end = target in (self.low, self.high)
        if held is None or abs(target - held) >= self.band or (at_end and target != held):
            self.value = held = target
        return held


class Throttle:
    """Commands the gripper from the throttle's voltage while enabled: released closes it, pressed
    opens it."""

    def __init__(
        self,
        node: Node,
        robot: Robot,
        released_v: float,
        pressed_v: float,
        max_opening: float,
        hysteresis_rad: float,
    ) -> None:
        self.robot = robot
        self.released_v, self.pressed_v = released_v, pressed_v
        self.open_angle = opening_to_angle(max_opening)
        self.command = Hysteresis(hysteresis_rad, self.open_angle, FINGER_CLOSED)
        self.publisher = node.create_publisher(Float64MultiArray, FINGER_COMMAND_TOPIC, 10)
        self.angle: float | None = None
        self.fault: str | None = "no fresh reading"
        self.enabled = False
        node.create_timer(GRIPPER_COMMAND_S, self.update)

    def update(self) -> None:
        reading = self.robot.throttle
        fault = self.check(reading, time.monotonic())
        if fault != self.fault:
            pin = self.robot.throttle_pin
            print(
                f"Throttle on analog input {pin}: {fault}, so the gripper holds."
                if fault
                else f"Throttle on analog input {pin} reads normally."
            )
            self.fault = fault
        if self.enabled and fault is None and reading is not None:
            target = finger_angle(reading.value, self.released_v, self.pressed_v, self.open_angle)
            self.angle = self.command.update(target)
            self.publisher.publish(Float64MultiArray(data=[self.angle]))

    def check(self, reading: AnalogReading | None, now: float) -> str | None:
        if reading is None or now - reading.at > STALE_AFTER_S:
            return "no fresh reading"
        if reading.domain != Analog.VOLTAGE:
            return "not in voltage mode"
        low, high = THROTTLE_VALID_V
        if not low <= reading.value <= high:
            return f"reads {reading.value:.2f} V, outside {low}-{high} V"
        return None

    def commanded_opening(self) -> float | None:
        return None if self.angle is None else 1.0 - self.angle / FINGER_CLOSED


@dataclass
class Episode:
    task_id: str
    start: float
    episode_id: str = field(default_factory=lambda: f"ep_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3])
    started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    steps: list[dict[str, Any]] = field(default_factory=list)
    stale_steps: int = 0
    steps_without_image: int = 0

    def add_step(self, now: float, robot: Robot, gripper_command: float | None) -> None:
        image = robot.fresh_image(now)
        if image is None:
            self.steps_without_image += 1
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
                "images": {"top": image},
                "proprio": proprio,
                "gripper_command": gripper_command,
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
            "steps_without_image": self.steps_without_image,
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

    def connect(self) -> None:
        for client in (self.lister, self.switcher):
            if not client.wait_for_service(timeout_sec=5.0):
                raise ControllerError(f"cannot reach {self.controller_manager}")

    def start(self) -> None:
        active = [
            c
            for c in call(self.lister, ListControllers.Request()).controller
            if c.state == "active"
        ]
        if any(c.name == FREEDRIVE_CONTROLLER for c in active):
            self.restore = [DEFAULT_ARM_CONTROLLER]
            print(
                f"{FREEDRIVE_CONTROLLER} was already active, so {DEFAULT_ARM_CONTROLLER} "
                "takes over when it ends."
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
        print("Freedrive on.")

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


def enter_freedrive(freedrive: Freedrive) -> bool:
    try:
        freedrive.start()
    except ControllerError as error:
        print(f"Not starting: could not enter freedrive ({error}).")
        return False
    return True


def leave_freedrive(freedrive: Freedrive) -> None:
    try:
        freedrive.stop()
    except ControllerError as error:
        print(f"Could not leave freedrive ({error}). Run:\n  {freedrive.manual_stop_command()}")


def interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def finish(episode: Episode, out_dir: Path, ended_by: str) -> Path:
    path = episode.save(out_dir, ended_by)
    steps = len(episode.steps)
    gaps = []
    if episode.stale_steps:
        gaps.append(f"{episode.stale_steps} without a fresh arm state")
    if episode.steps_without_image:
        gaps.append(f"{episode.steps_without_image} without a camera frame")
    detail = "".join(f", {gap}" for gap in gaps)
    print(f"Saved {steps} steps ({steps * STEP_S:.2f} s{detail}) to {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True, help='e.g. bench_lift_can_v0, or "hidden"')
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
    parser.add_argument(
        "--throttle-pin", type=int, default=1, help="the throttle's analog input (default 1)"
    )
    parser.add_argument(
        "--camera",
        default=CAMERA_TOPIC,
        help=f"the camera's JPEG topic for images.top (default {CAMERA_TOPIC})",
    )
    parser.add_argument(
        "--throttle-released-v",
        type=float,
        default=THROTTLE_RELEASED_V,
        help=f"at or below this the gripper is closed (default {THROTTLE_RELEASED_V} V)",
    )
    parser.add_argument(
        "--throttle-pressed-v",
        type=float,
        default=THROTTLE_PRESSED_V,
        help="at or above this the gripper is open as far as --gripper-max-opening "
        f"(default {THROTTLE_PRESSED_V} V)",
    )
    parser.add_argument(
        "--gripper-hysteresis-steps",
        type=float,
        default=GRIPPER_HYSTERESIS_STEPS,
        help="the gripper command changes only when the throttle asks for this many of the "
        "gripper's 227 position steps away; 0 turns it off "
        f"(default {GRIPPER_HYSTERESIS_STEPS:g}, about 0.6 mm at the fingertips each)",
    )
    parser.add_argument(
        "--gripper-max-opening",
        type=float,
        default=GRIPPER_MAX_OPENING,
        help="how far the fully pressed throttle opens the gripper: "
        f"1 fully open, 0 closed (default {GRIPPER_MAX_OPENING:g})",
    )
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if args.throttle_released_v >= args.throttle_pressed_v:
        parser.error("--throttle-released-v must be below --throttle-pressed-v")
    if not 0.0 < args.gripper_max_opening <= 1.0:
        parser.error("--gripper-max-opening must be above 0 and at most 1")

    # rclpy's own Ctrl-C handler would shut ROS down before the controllers can be switched back.
    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    node = rclpy.create_node("demo_recorder")
    robot = Robot(
        node,
        args.pin,
        args.pressed_state,
        args.debounce_ms / 1000,
        args.throttle_pin,
        args.camera,
    )
    throttle = Throttle(
        node,
        robot,
        args.throttle_released_v,
        args.throttle_pressed_v,
        args.gripper_max_opening,
        args.gripper_hysteresis_steps * GRIPPER_STEP_RAD,
    )
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
            freedrive.connect()
        print(
            f"Waiting for the button on digital input {args.pin}; the throttle drives the gripper "
            "only while it is held. Ctrl-C to quit."
        )
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
                if not pressed:
                    throttle.enabled = False
                    if episode is not None:
                        saved.append(finish(episode, args.out_dir, "release"))
                        episode = None
                    if freedrive is not None:
                        leave_freedrive(freedrive)
                elif episode is not None:
                    continue
                elif robot.fresh_arm(now) is None:
                    print("Not starting: no fresh /joint_states.")
                elif freedrive is not None and not enter_freedrive(freedrive):
                    continue
                else:
                    throttle.enabled = True
                    # Switching into freedrive takes a moment; the episode starts once it is on.
                    now = time.monotonic()
                    episode = Episode(args.task, now)
                    next_step_at = now
                    missing = []
                    if robot.fresh_gripper_opening(now) is None:
                        missing.append("no fresh /gripper/joint_states, gripper is null")
                    if robot.fresh_image(now) is None:
                        missing.append(f"no fresh frame on {args.camera}, images.top is null")
                    notes = "".join(f"; {note}" for note in missing)
                    print(f"Recording {episode.episode_id}. Release to stop{notes}.")

            if episode is not None and now >= next_step_at:
                episode.add_step(now, robot, throttle.commanded_opening())
                next_step_at += STEP_S
                if now - next_step_at > STEP_S:
                    next_step_at = now + STEP_S
                steps = len(episode.steps)
                if steps % 20 == 0:
                    print(f"  step {steps} ({steps * STEP_S:.0f} s)", flush=True)
            time.sleep(0.002 if episode is not None else 0.01)
    except KeyboardInterrupt:
        pass
    except ControllerError as error:
        stop_reason = f"freedrive unavailable, {error}"
    finally:
        # A second Ctrl-C, or a closing terminal's signal arriving twice, must not cut short saving
        # the episode and leaving freedrive.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        throttle.enabled = False
        if episode is not None:
            saved.append(finish(episode, args.out_dir, stop_reason))
        if freedrive is not None:
            leave_freedrive(freedrive)
        rclpy.try_shutdown()
        spinner.join(timeout=2.0)

    print(f"\nStopped: {stop_reason}. {len(saved)} episode(s) saved this session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
