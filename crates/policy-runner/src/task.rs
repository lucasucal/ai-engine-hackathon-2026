use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use color_eyre::{
    Result,
    eyre::{WrapErr, bail},
};
use policy_contract::image::{GOAL_SIZE, check_jpeg};
use std::{fs, path::Path};

/// What an episode tells the policy about its task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    pub id: String,
    pub instruction: String,
    /// The goal photo, base64-encoded.
    pub goal: String,
}

impl Task {
    /// Reads `<dir>/<id>.txt`, the instruction, and `<dir>/<id>.jpg`, the goal photo.
    pub fn load(dir: &Path, id: &str) -> Result<Self> {
        let read = |extension: &str| {
            let path = dir.join(format!("{id}.{extension}"));
            fs::read(&path)
                .wrap_err_with(|| format!("cannot read {}", path.display()))
                .map(|contents| (path, contents))
        };
        let (path, instruction) = read("txt")?;
        let instruction = String::from_utf8(instruction)
            .wrap_err_with(|| format!("{} is not UTF-8", path.display()))?;
        let instruction = instruction.trim();
        if instruction.is_empty() {
            bail!("{} is empty", path.display());
        }
        let (path, goal) = read("jpg")?;
        check_jpeg(&goal, GOAL_SIZE).wrap_err_with(|| format!("the goal {}", path.display()))?;
        Ok(Self {
            id: id.to_owned(),
            instruction: instruction.to_owned(),
            goal: BASE64.encode(goal),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use policy_contract::image::BLACK_JPEG;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU32, Ordering},
    };
    use test_case::test_case;

    /// Just enough of a JPEG for its size to be read: a baseline frame header for 640×480.
    pub const GOAL_JPEG: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xE0, 0x02, 0x80, 0x03, 0x01, 0x22, 0x00,
        0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xD9,
    ];

    /// A tasks folder, deleted on drop.
    pub struct Tasks(pub PathBuf);

    impl Tasks {
        /// Holds a valid instruction and goal for each of `ids`.
        pub fn new(ids: &[&str]) -> Self {
            static COUNT: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "policy-runner-tasks-{}-{}",
                std::process::id(),
                COUNT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            let tasks = Self(dir);
            for id in ids {
                tasks.write(&format!("{id}.txt"), b"Lift the can.\n");
                tasks.write(&format!("{id}.jpg"), GOAL_JPEG);
            }
            tasks
        }

        pub fn write(&self, name: &str, contents: &[u8]) {
            fs::write(self.0.join(name), contents).unwrap();
        }
    }

    impl Drop for Tasks {
        fn drop(&mut self) {
            _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn loads() {
        let tasks = Tasks::new(&["lift"]);
        let task = Task::load(&tasks.0, "lift").unwrap();
        assert_eq!(task.instruction, "Lift the can.");
        assert_eq!(task.goal, BASE64.encode(GOAL_JPEG));
    }

    #[test_case(None, "lift.txt" => matches e if e.contains("cannot read"); "no instruction")]
    #[test_case(None, "lift.jpg" => matches e if e.contains("cannot read"); "no goal")]
    #[test_case(Some(b" \n".as_slice()), "lift.txt" => matches e if e.ends_with("is empty"); "empty instruction")]
    #[test_case(Some(BLACK_JPEG), "lift.jpg" => matches e if e.ends_with(": 640×360, not 640×480");
        "goal of the wrong size")]
    fn refuses(contents: Option<&[u8]>, name: &str) -> String {
        let tasks = Tasks::new(&["lift"]);
        match contents {
            Some(contents) => tasks.write(name, contents),
            None => fs::remove_file(tasks.0.join(name)).unwrap(),
        }
        format!("{:#}", Task::load(&tasks.0, "lift").unwrap_err())
    }
}
