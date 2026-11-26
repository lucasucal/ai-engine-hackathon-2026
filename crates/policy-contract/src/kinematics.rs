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

const PALM_OFFSET_IN_TOOL0: Vector3<f64> = Vector3::new(0.0, 0.107, 0.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PalmPose {
    pub position: [f64; 3],
    /// `w, x, y, z`, with its largest-magnitude component positive.
    pub orientation: [f64; 4],
}

/// The palm's pose in `base_link`, which is the DH base (`base`) turned 180° about z.
#[must_use]
pub fn palm_pose(joints: &Joints) -> PalmPose {
    let palm = tool0_pose(joints) * tool0_to_palm();

    let position = palm.translation.vector;
    let rotation = palm.rotation;
    let mut orientation = [rotation.w, rotation.i, rotation.j, rotation.k];
    // On a tie the first component, in `w, x, y, z` order, decides the sign; `max_by` returns the
    // last of equals, hence the reversal.
    let largest = orientation
        .iter()
        .rev()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()));
    if largest.is_some_and(|&c| c < 0.0) {
        orientation = orientation.map(|c| -c);
    }
    PalmPose {
        position: [position.x, position.y, position.z],
        orientation,
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

fn tool0_to_palm() -> Isometry3<f64> {
    let palm_axes_in_tool0 = [-Vector3::x(), Vector3::z(), Vector3::y()];
    Isometry3::from_parts(
        PALM_OFFSET_IN_TOOL0.into(),
        UnitQuaternion::from_basis_unchecked(&palm_axes_in_tool0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ARM_JOINTS;
    use approx::assert_abs_diff_eq;
    use nalgebra::{Rotation3, Unit, Vector2, Vector4};
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

    #[test]
    fn bracket_matches_xacro() {
        let arm = urdf_pose(&ARM_URDF, "tool0", "gripper_mount", &[0.0; 6]);
        let gripper = gripper_pose("gripper_mount");

        assert_abs_diff_eq!(
            arm.unwrap().to_homogeneous(),
            gripper.to_homogeneous(),
            epsilon = 1e-6
        );
    }

    /// The palm's axes are the 2F-140's relabelled, as they come out exactly for a gripper bolted
    /// straight onto the flange: its x is the fingers' closing axis and its y the approach.
    #[test]
    fn palm_matches_gripper() {
        let gripper = gripper_pose("robotiq_140_base_link");

        let palm_in_gripper = gripper.inverse() * tool0_to_palm();

        let palm_axes_in_gripper = [Vector3::y(), Vector3::z(), Vector3::x()];
        assert_abs_diff_eq!(
            palm_in_gripper.rotation.to_rotation_matrix(),
            Rotation3::from_basis_unchecked(&palm_axes_in_gripper),
            epsilon = 0.001
        );
        assert_abs_diff_eq!(
            palm_in_gripper.translation.vector.xy(),
            Vector2::zeros(),
            epsilon = 0.001
        );
    }

    #[test]
    #[expect(
        clippy::approx_constant,
        reason = "the contract's example pose, as written"
    )]
    fn contract_example() {
        let pose = palm_pose(&[-0.2055, -0.60, 1.15, -3.60, 1.5708, 0.0]);

        assert_abs_diff_eq!(
            Vector3::from(pose.position),
            Vector3::new(0.8951, -0.0087, 0.2514),
            epsilon = 0.001
        );
        assert_abs_diff_eq!(
            Vector4::from(pose.orientation),
            Vector4::new(-0.0289, -0.6302, 0.7751, 0.0355),
            epsilon = 0.001
        );
    }
}
