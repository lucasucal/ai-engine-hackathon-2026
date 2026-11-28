use crate::Joints;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use std::f64::consts::{FRAC_PI_2, PI};

struct DhLink {
    d: f64,
    a: f64,
    alpha: f64,
}

/// The `UR12e`'s published, not calibrated, DH parameters, on the assumption that the policy's
/// training simulation used them too. They must agree with `ur_description`'s
/// `config/ur12e/default_kinematics.yaml`, which gives them as URDF joint origins; they are not
/// read from there so that passing the robot's calibration to its bringup cannot change them.
const UR12E: [DhLink; 6] = [
    DhLink {
        d: 0.1807,
        a: 0.0,
        alpha: FRAC_PI_2,
    },
    DhLink {
        d: 0.0,
        a: -0.6127,
        alpha: 0.0,
    },
    DhLink {
        d: 0.0,
        a: -0.57155,
        alpha: 0.0,
    },
    DhLink {
        d: 0.17415,
        a: 0.0,
        alpha: FRAC_PI_2,
    },
    DhLink {
        d: 0.11985,
        a: 0.0,
        alpha: -FRAC_PI_2,
    },
    DhLink {
        d: 0.11655,
        a: 0.0,
        alpha: 0.0,
    },
];

/// Reality Check's palm, its `gripperframe` site, along tool0's z: 24.4 mm of bracket, a 7 mm
/// spacer and 100 mm of gripper.
const PALM_DISTANCE: f64 = 0.1314;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PalmPose {
    pub position: [f64; 3],
    /// `w, x, y, z`, signed as `mju_mat2Quat` signs it.
    pub orientation: [f64; 4],
}

/// The palm's pose in `base_link`, which is the DH base (`base`) turned 180° about z, for the
/// joints the policy is given.
#[must_use]
pub fn palm_pose(joints: &Joints) -> PalmPose {
    PalmPose::new(&(tool0_pose(joints) * tool0_to_palm()))
}

impl PalmPose {
    fn new(palm: &Isometry3<f64>) -> Self {
        let position = palm.translation.vector;
        let rotation = palm.rotation;
        let orientation = [rotation.w, rotation.i, rotation.j, rotation.k];
        // `w` while the rotation matrix's trace, 4w² − 1, is positive, else the component of its
        // largest diagonal entry, the first of equals losing.
        let [w, x, y, z] = orientation.map(|c| c * c);
        let positive = if w > 0.25 {
            rotation.w
        } else if x > y && x > z {
            rotation.i
        } else if y > z {
            rotation.j
        } else {
            rotation.k
        };
        Self {
            position: [position.x, position.y, position.z],
            orientation: orientation.map(|c| if positive < 0.0 { -c } else { c }),
        }
    }
}

fn tool0_pose(joints: &Joints) -> Isometry3<f64> {
    let base_link_to_base = Isometry3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI),
    );
    UR12E
        .iter()
        .zip(joints)
        .fold(base_link_to_base, |pose, (link, &theta)| {
            pose * dh_transform(link, theta)
        })
}

fn dh_transform(link: &DhLink, theta: f64) -> Isometry3<f64> {
    let translation = Translation3::new(link.a * theta.cos(), link.a * theta.sin(), link.d);
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), theta)
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), link.alpha);
    Isometry3::from_parts(translation, rotation)
}

/// The palm's z is the approach and its y the fingers' opening axis, tool0's x.
fn tool0_to_palm() -> Isometry3<f64> {
    Isometry3::new(Vector3::z() * PALM_DISTANCE, Vector3::z() * -FRAC_PI_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ARM_JOINTS, WRIST_3_OFFSET};
    use approx::assert_abs_diff_eq;
    use nalgebra::{Unit, Vector4};
    use roxmltree::{Document, Node};
    use std::{process::Command, sync::LazyLock};
    use test_case::test_case;

    static ARM_URDF: LazyLock<Document<'static>> = LazyLock::new(|| {
        Document::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../robot/urdf/ur12e.urdf"
        )))
        .unwrap()
    });

    /// What launch.xml gives `robot_state_publisher`, which needs `robotiq_description` sourced.
    static GRIPPER_URDF: LazyLock<Document<'static>> = LazyLock::new(|| {
        let xacro = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../robot/gripper.urdf.xacro"
        );
        let output = Command::new("xacro").arg(xacro).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Document::parse(String::from_utf8(output.stdout).unwrap().leak()).unwrap()
    });

    fn gripper_pose(link: &str) -> Isometry3<f64> {
        urdf_pose(&GRIPPER_URDF, "tool0", link, &[0.0; 6]).unwrap()
    }

    /// `link`'s pose in `root`, by composing the URDF's joints down to it.
    fn urdf_pose(
        urdf: &Document<'_>,
        root: &str,
        link: &str,
        joints: &Joints,
    ) -> Option<Isometry3<f64>> {
        if link == root {
            return Some(Isometry3::identity());
        }
        let joint = urdf.root_element().children().find(|n| {
            n.has_tag_name("joint")
                && child_element(*n, "child").and_then(|c| c.attribute("link")) == Some(link)
        })?;
        let parent = child_element(joint, "parent")?.attribute("link")?;
        let origin = child_element(joint, "origin")?;
        let rpy = vector(origin, "rpy")?;
        let joint_to_parent = Isometry3::from_parts(
            vector(origin, "xyz")?.into(),
            UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z),
        );
        let motion = match joint.attribute("type")? {
            "fixed" => UnitQuaternion::identity(),
            "revolute" => {
                let name = joint.attribute("name")?;
                let index = ARM_JOINTS.iter().position(|&j| j == name)?;
                let axis = Unit::new_normalize(vector(child_element(joint, "axis")?, "xyz")?);
                UnitQuaternion::from_axis_angle(&axis, joints[index])
            }
            other => panic!("joint type {other}"),
        };
        Some(urdf_pose(urdf, root, parent, joints)? * joint_to_parent * motion)
    }

    fn child_element<'a, 'input>(node: Node<'a, 'input>, tag: &str) -> Option<Node<'a, 'input>> {
        node.children().find(|n| n.has_tag_name(tag))
    }

    fn vector(node: Node<'_, '_>, attribute: &str) -> Option<Vector3<f64>> {
        let values: Vec<f64> = node
            .attribute(attribute)?
            .split_whitespace()
            .map(|v| v.parse().ok())
            .collect::<Option<_>>()?;
        Some(Vector3::from_column_slice(&values))
    }

    #[test_case([0.0; 6])]
    #[test_case([-0.2, -0.6, 1.15, -3.6, 1.6, 0.0])]
    #[test_case([1.0, -1.2, 2.0, -0.5, -1.0, 2.5])]
    #[test_case([-2.8, 0.4, -2.6, 3.1, 0.3, -4.0])]
    fn dh_matches_urdf(joints: Joints) {
        let dh = tool0_pose(&joints);
        let urdf = urdf_pose(&ARM_URDF, "base_link", "tool0", &joints).unwrap();

        assert_abs_diff_eq!(dh.to_homogeneous(), urdf.to_homogeneous(), epsilon = 1e-6);
    }

    /// On this arm the palm is on the gripper's axis with the gripper's axes, 107 mm out from its
    /// base: this URDF has no 7 mm spacer, but the policy was trained with one.
    #[test]
    fn palm_matches_gripper() {
        let gripper = gripper_pose("robotiq_140_base_link");
        let wrist_3_offset = Isometry3::rotation(Vector3::z() * WRIST_3_OFFSET);

        let palm_in_gripper = gripper.inverse() * wrist_3_offset * tool0_to_palm();

        assert_abs_diff_eq!(
            palm_in_gripper.to_homogeneous(),
            Isometry3::translation(0.0, 0.0, 0.107).to_homogeneous(),
            epsilon = 1e-6
        );
    }

    /// The example in Reality Check's `docs/api.md`, at its `HOME_Q`.
    #[test]
    #[expect(clippy::approx_constant, reason = "Reality Check's HOME_Q, as written")]
    fn contract_example() {
        let joints = [-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708];

        let pose = palm_pose(&joints);

        assert_abs_diff_eq!(
            Vector3::from(pose.position),
            Vector3::new(0.8906, -0.0078, 0.2016),
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            Vector4::from(pose.orientation),
            Vector4::new(-0.0289, -0.6302, 0.7751, 0.0355),
            epsilon = 1e-4
        );
    }

    #[test_case(Vector3::x(), -100.0 => [0.6428, -0.766, 0.0, 0.0]; "w positive while the trace is")]
    #[test_case(Vector3::z(), -170.0 => [-0.0872, 0.0, 0.0, 0.9962]; "else the largest diagonal's")]
    #[test_case(Vector3::y(), 180.0 => [0.0, 0.0, 1.0, 0.0]; "half a turn")]
    fn quaternion_sign(axis: Vector3<f64>, degrees: f64) -> [f64; 4] {
        let rotation = Isometry3::rotation(axis * degrees.to_radians());
        PalmPose::new(&rotation)
            .orientation
            .map(|c| (c * 10_000.0).round() / 10_000.0)
    }
}
