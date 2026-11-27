//! The `/act` contract `v0`, schema version 2, as `API.md` describes it: what the policy server and
//! the tools that observe or command the robot for it must agree on.

pub mod gripper;
pub mod image;
pub mod kinematics;
pub mod wire;

use std::{f64::consts::FRAC_PI_8, time::Duration};

pub const SCHEMA_VERSION: u32 = 2;
pub const STEP: Duration = Duration::from_millis(50);

/// Joint positions in the contract's order: shoulder pan, shoulder lift, elbow, wrist 1, 2, 3.
pub type Joints = [f64; 6];

/// How far Reality Check's wrist 3 is ahead of this arm's for the same tool pose.
///
/// The wrist bracket here, and the camera and gripper on it, is turned 7π/8 about tool0 so that
/// its dowel meets the flange's; the simulation mounts them unturned.
pub const WRIST_3_OFFSET: f64 = 7.0 * FRAC_PI_8;

/// The joints the policy is given, and the palm pose is computed from, for the arm's.
#[must_use]
pub const fn policy_joints(mut arm: Joints) -> Joints {
    arm[5] += WRIST_3_OFFSET;
    arm
}

/// The arm's joints for those the policy is given.
#[must_use]
pub const fn arm_joints(mut policy: Joints) -> Joints {
    policy[5] -= WRIST_3_OFFSET;
    policy
}

/// The arm's joint names, in the contract's order.
pub const ARM_JOINTS: [&str; 6] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
];
