use crate::{
    app::{Command, Phase, Shared},
    episode::Episode,
    kinematics::Joints,
    policy::Policy,
    ros::Ros,
};
use color_eyre::{Result, eyre::bail};
use std::{
    sync::{atomic::Ordering, mpsc::Receiver},
    thread,
    time::{Duration, Instant},
};

/// Captured from the real arm on 2026-11-20; not the contract's example pose.
pub const START_POSE: Joints = [-1.6262, -1.6024, -2.7163, -2.0559, -0.1211, -1.6580];
const START_MOVE: Duration = Duration::from_secs(10);

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
            Command::RunEpisode { task_id } => {
                let report = Episode::run(ros, policy, &task_id, &shared.cancel, &shared.progress);
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

/// Moves the arm slowly to the start pose. Returns false if the operator aborted, after making
/// the arm hold.
fn move_to_start(ros: &Ros, shared: &Shared) -> Result<bool> {
    shared.log.push(format!(
        "Moving to the start pose over {} s.",
        START_MOVE.as_secs()
    ));
    ros.send_trajectory(&[START_POSE], START_MOVE)?;

    let arrival = Instant::now() + START_MOVE + Duration::from_millis(500);
    while Instant::now() < arrival {
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
    let error = arm
        .value
        .position
        .iter()
        .zip(START_POSE)
        .map(|(q, s)| (q - s).abs())
        .fold(0.0, f64::max);
    shared
        .log
        .push(format!("At the start pose, within {error:.4} rad."));
    Ok(true)
}
