use crate::{
    app::{Command, Phase, Shared},
    episode::Episode,
    policy::Policy,
    ros::{Received, Ros},
};
use color_eyre::{Result, eyre::bail};
use policy_contract::{Joints, arm_joints, gripper};
use std::{
    sync::{atomic::Ordering, mpsc::Receiver},
    thread,
    time::{Duration, Instant},
};

/// Reality Check's `HOME_Q`, where its episodes start.
#[expect(clippy::approx_constant, reason = "Reality Check's HOME_Q, as written")]
const HOME_Q: Joints = [-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708];
pub const START_POSE: Joints = arm_joints(HOME_Q);
/// Reality Check's gripper reads 0.9968 open at the start.
const GRIPPER_OPEN: f64 = 0.95;
const GRIPPER_WAIT: Duration = Duration::from_secs(3);
/// Half a turn takes 10 s, as every move to the start pose used to.
const START_SPEED: f64 = std::f64::consts::PI / 10.0;
const MIN_START_MOVE: Duration = Duration::from_secs(1);

/// Carries out the operator's commands until the sender is dropped. Returns an error, ending
/// the runner, when the arm may not be holding.
pub fn serve(
    ros: &Ros,
    policy: &Policy,
    shared: &Shared,
    commands: &Receiver<Command>,
) -> Result<()> {
    for command in commands {
        let phase = match command {
            Command::MoveToStart => {
                if move_to_start(ros, shared)? {
                    Phase::AtStart
                } else {
                    Phase::Idle
                }
            }
            Command::RunEpisode(task) => {
                let report = Episode::run(ros, policy, &task, &shared.cancel, &shared.progress);
                let line = report.to_string();
                shared.log.push(line.clone());
                shared.reports.lock().push(line);
                if report.hold_failed.is_some() {
                    // Commands to the arm are likely failing; someone should look before it
                    // moves again.
                    bail!("stopping, as the arm may not be holding");
                }
                Phase::Idle
            }
        };
        *shared.phase.lock() = phase;
    }
    Ok(())
}

/// The largest joint error from `position` to the start pose, in radians.
pub fn distance_to_start(position: &Joints) -> f64 {
    position
        .iter()
        .zip(START_POSE)
        .map(|(q, s)| (q - s).abs())
        .fold(0.0, f64::max)
}

fn start_move(from: &Joints) -> Duration {
    Duration::from_secs_f64(distance_to_start(from) / START_SPEED).max(MIN_START_MOVE)
}

/// Moves the arm slowly to the start pose and opens the gripper. Returns false if the gripper did
/// not open, or if the operator aborted, after making the arm hold.
fn move_to_start(ros: &Ros, shared: &Shared) -> Result<bool> {
    let Some(arm) = ros.arm() else {
        bail!("no joint state from the arm")
    };
    let duration = start_move(&arm.value.position);
    shared.log.push(format!(
        "Moving to the start pose over {:.1} s.",
        duration.as_secs_f64()
    ));
    ros.command_finger(gripper::finger(1.0))?;
    ros.send_trajectory(&[START_POSE], duration)?;

    let gripper_open = || {
        ros.finger()
            .filter(Received::is_fresh)
            .is_some_and(|finger| gripper::opening(finger.value) >= GRIPPER_OPEN)
    };
    let arrival = Instant::now() + duration + Duration::from_millis(500);
    while Instant::now() < arrival || (!gripper_open() && Instant::now() < arrival + GRIPPER_WAIT) {
        if shared.cancel.load(Ordering::Relaxed) {
            if let Some(arm) = ros.arm() {
                ros.hold(&arm.value.position)?;
            }
            shared.log.push("Move to the start pose aborted; holding.");
            return Ok(false);
        }
        thread::sleep(Duration::from_millis(10));
    }

    let Some(arm) = ros.arm() else {
        bail!("no joint state from the arm")
    };
    let error = distance_to_start(&arm.value.position);
    if !gripper_open() {
        shared.log.push(format!(
            "At the start pose, within {error:.4} rad, but the gripper has not opened."
        ));
        return Ok(false);
    }
    shared
        .log
        .push(format!("At the start pose, within {error:.4} rad."));
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use policy_contract::policy_joints;
    use std::f64::consts::PI;
    use test_case::test_case;

    fn offset(joint: usize, by: f64) -> Joints {
        let mut position = START_POSE;
        position[joint] += by;
        position
    }

    #[test_case(START_POSE => MIN_START_MOVE; "already there")]
    #[test_case(offset(5, -2.0 * PI) => Duration::from_secs(20); "full turn backwards")]
    fn start_move_duration(from: Joints) -> Duration {
        start_move(&from)
    }

    #[test]
    fn start_pose_is_reported_as_home() {
        let reported = policy_joints(START_POSE);
        assert!(
            reported
                .iter()
                .zip(HOME_Q)
                .all(|(r, h)| (r - h).abs() < 1e-12),
            "{reported:?}"
        );
        assert!((START_POSE[5] - -1.1781).abs() < 1e-4, "{START_POSE:?}");
    }
}
