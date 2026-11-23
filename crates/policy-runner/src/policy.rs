//! The `/act` contract `v0`, schema version 2: see `CONTRACT.md`.

use crate::kinematics::Joints;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{fmt, time::Duration};

pub const CONTRACT: &str = "v0";
pub const SCHEMA_VERSION: u32 = 2;
pub const MAX_DELTA: f64 = 0.05;
pub const MAX_CHUNK: usize = 16;

pub const ACT_TIMEOUT: Duration = Duration::from_millis(100);
const STATUS_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Serialize)]
pub struct ActRequest<'a> {
    pub episode_id: &'a str,
    pub step: u32,
    pub task_id: &'a str,
    pub schema_version: u32,
    pub images: Images,
    pub proprio: Proprio,
}

#[derive(Debug, Serialize)]
pub struct Images {
    pub top: String,
}

#[derive(Debug, Serialize)]
pub struct Proprio {
    pub joint_pos: Joints,
    pub joint_vel: Joints,
    pub gripper: f64,
    pub ee_pos: [f64; 3],
    pub ee_quat: [f64; 4],
}

/// One validated 50 ms action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Action {
    pub delta: Joints,
    /// The commanded opening: 0 closed, 1 open.
    pub gripper: f64,
}

#[derive(Debug, Deserialize)]
struct Reply {
    episode_id: String,
    step: u64,
    action: ReplyAction,
    #[serde(flatten)]
    unknown: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReplyAction {
    JointDelta(JointDelta),
    ActionChunk {
        actions: Vec<ChunkedAction>,
        chunk_len: Option<usize>,
        #[serde(flatten)]
        unknown: Map<String, Value>,
    },
}

/// A chunk's actions still carry `"type": "joint_delta"`, and can be nothing else.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ChunkedAction {
    JointDelta(JointDelta),
}

#[derive(Debug, Deserialize)]
struct JointDelta {
    values: Vec<f64>,
    gripper: f64,
    #[serde(flatten)]
    unknown: Map<String, Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidReply {
    #[error("HTTP status {0}")]
    Status(u16),
    #[error("malformed reply")]
    Malformed(#[from] serde_json::Error),
    #[error("unknown field {0}")]
    UnknownField(String),
    #[error("{at} has {count} entries, not 6")]
    WrongValueCount { at: String, count: usize },
    #[error("{at} is not finite")]
    NotFinite { at: String },
    #[error("{at} is out of range")]
    OutOfRange { at: String },
    #[error("a chunk of {0} actions, not 1 to {MAX_CHUNK}")]
    ChunkSize(usize),
    #[error("chunk_len is {chunk_len}, but there are {actions} actions")]
    ChunkLenMismatch { chunk_len: usize, actions: usize },
    #[error("{at} is {received:?}, not {expected:?}")]
    WrongEcho {
        at: &'static str,
        expected: String,
        received: String,
    },
}

/// The `X-RC-*` headers of a reply, if present.
#[derive(Debug, Default)]
pub struct EchoHeaders {
    pub episode_id: Option<String>,
    pub step: Option<String>,
}

/// Validates a reply to the request for `episode_id` and `step`, in full, before any of it is
/// applied.
pub fn validate(
    body: &str,
    headers: &EchoHeaders,
    episode_id: &str,
    step: u32,
) -> Result<Vec<Action>, InvalidReply> {
    let reply: Reply = serde_json::from_str(body)?;

    reject_unknown(&reply.unknown, "")?;
    check_echo("episode_id", episode_id, &reply.episode_id)?;
    check_echo("step", &step.to_string(), &reply.step.to_string())?;
    let episode_header = headers.episode_id.as_deref().unwrap_or("missing");
    let step_header = headers.step.as_deref().unwrap_or("missing");
    check_echo("X-RC-Episode-Id", episode_id, episode_header)?;
    check_echo("X-RC-Step", &step.to_string(), step_header)?;

    match reply.action {
        ReplyAction::JointDelta(action) => Ok(vec![joint_delta(&action, "action")?]),
        ReplyAction::ActionChunk {
            actions,
            chunk_len,
            unknown,
        } => {
            reject_unknown(&unknown, "action.")?;
            if !(1..=MAX_CHUNK).contains(&actions.len()) {
                return Err(InvalidReply::ChunkSize(actions.len()));
            }
            if let Some(chunk_len) = chunk_len
                && chunk_len != actions.len()
            {
                return Err(InvalidReply::ChunkLenMismatch {
                    chunk_len,
                    actions: actions.len(),
                });
            }
            actions
                .into_iter()
                .enumerate()
                .map(|(i, ChunkedAction::JointDelta(action))| {
                    joint_delta(&action, &format!("action.actions[{i}]"))
                })
                .collect()
        }
    }
}

fn joint_delta(action: &JointDelta, at: &str) -> Result<Action, InvalidReply> {
    reject_unknown(&action.unknown, &format!("{at}."))?;

    let delta =
        Joints::try_from(action.values.as_slice()).map_err(|_| InvalidReply::WrongValueCount {
            at: format!("{at}.values"),
            count: action.values.len(),
        })?;
    for (i, value) in delta.iter().enumerate() {
        check_number(*value, MAX_DELTA, &format!("{at}.values[{i}]"))?;
    }
    let gripper = action.gripper;
    if !gripper.is_finite() {
        return Err(InvalidReply::NotFinite {
            at: format!("{at}.gripper"),
        });
    }
    if !(0.0..=1.0).contains(&gripper) {
        return Err(InvalidReply::OutOfRange {
            at: format!("{at}.gripper"),
        });
    }
    Ok(Action { delta, gripper })
}

/// Checks that `value` is finite and within `[-limit, limit]`.
fn check_number(value: f64, limit: f64, at: &str) -> Result<(), InvalidReply> {
    if !value.is_finite() {
        return Err(InvalidReply::NotFinite { at: at.to_owned() });
    }
    if value.abs() > limit {
        return Err(InvalidReply::OutOfRange { at: at.to_owned() });
    }
    Ok(())
}

fn reject_unknown(fields: &Map<String, Value>, prefix: &str) -> Result<(), InvalidReply> {
    fields
        .keys()
        .find(|name| !name.starts_with('_'))
        .map_or(Ok(()), |name| {
            Err(InvalidReply::UnknownField(format!("{prefix}{name}")))
        })
}

fn check_echo(at: &'static str, expected: &str, received: &str) -> Result<(), InvalidReply> {
    if expected == received {
        Ok(())
    } else {
        Err(InvalidReply::WrongEcho {
            at,
            expected: expected.to_owned(),
            received: received.to_owned(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ActError {
    /// A timeout or a connection error: worth sending again.
    #[error(transparent)]
    Transient(#[from] ureq::Error),
    #[error("invalid reply")]
    Invalid(#[from] InvalidReply),
}

#[derive(Deserialize)]
struct Readyz {
    ready: bool,
    contract: Option<String>,
    schema_version: Option<u32>,
}

/// `/healthz`, as last polled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    Up,
    Unreachable(String),
    HttpStatus(u16),
}

/// `/readyz`, as last polled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    NotReady,
    Incompatible {
        contract: Option<String>,
        schema_version: Option<u32>,
    },
    Malformed(String),
    Unreachable(String),
    HttpStatus(u16),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyStatus {
    pub health: Health,
    pub readiness: Readiness,
}

impl fmt::Display for Health {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Up => write!(f, "up"),
            Self::Unreachable(reason) => write!(f, "unreachable ({reason})"),
            Self::HttpStatus(status) => write!(f, "HTTP status {status}"),
        }
    }
}

impl fmt::Display for Readiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "ready ({CONTRACT}, schema version {SCHEMA_VERSION})"),
            Self::NotReady => write!(f, "not ready"),
            Self::Incompatible {
                contract,
                schema_version,
            } => {
                let contract = contract.as_deref().unwrap_or("missing");
                let schema_version =
                    schema_version.map_or_else(|| "missing".to_owned(), |v| v.to_string());
                write!(
                    f,
                    "incompatible: contract {contract}, schema version {schema_version}, \
                     not {CONTRACT}, {SCHEMA_VERSION}"
                )
            }
            Self::Malformed(reason) => write!(f, "malformed reply ({reason})"),
            Self::Unreachable(reason) => write!(f, "unreachable ({reason})"),
            Self::HttpStatus(status) => write!(f, "HTTP status {status}"),
        }
    }
}

/// A reply's status and body, or why there was none.
type HttpReply = Result<(u16, String), ureq::Error>;

impl From<HttpReply> for Health {
    fn from(reply: HttpReply) -> Self {
        match reply {
            Ok((200, _)) => Self::Up,
            Ok((status, _)) => Self::HttpStatus(status),
            Err(error) => Self::Unreachable(error.to_string()),
        }
    }
}

impl From<HttpReply> for Readiness {
    fn from(reply: HttpReply) -> Self {
        match reply {
            Ok((200, body)) => match serde_json::from_str::<Readyz>(&body) {
                Ok(Readyz { ready: false, .. }) => Self::NotReady,
                Ok(Readyz {
                    ready: true,
                    contract,
                    schema_version,
                }) => {
                    if contract.as_deref() == Some(CONTRACT)
                        && schema_version == Some(SCHEMA_VERSION)
                    {
                        Self::Ready
                    } else {
                        Self::Incompatible {
                            contract,
                            schema_version,
                        }
                    }
                }
                Err(error) => Self::Malformed(error.to_string()),
            },
            Ok((status, _)) => Self::HttpStatus(status),
            Err(error) => Self::Unreachable(error.to_string()),
        }
    }
}

pub struct Policy {
    agent: ureq::Agent,
    url: String,
}

impl Policy {
    pub fn new(url: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            // A proxy from the environment would sit between us and a policy on the local network.
            .proxy(None)
            .build()
            .new_agent();
        Self {
            agent,
            url: url.trim_end_matches('/').to_owned(),
        }
    }

    /// Polls `/healthz` and `/readyz` once each.
    pub fn status(&self) -> PolicyStatus {
        PolicyStatus {
            health: self.get("/healthz").into(),
            readiness: self.get("/readyz").into(),
        }
    }

    fn get(&self, path: &str) -> HttpReply {
        let mut response = self
            .agent
            .get(format!("{}{path}", self.url))
            .config()
            .timeout_global(Some(STATUS_TIMEOUT))
            .build()
            .call()?;
        let status = response.status().as_u16();
        Ok((status, response.body_mut().read_to_string()?))
    }

    pub fn act(&self, request: &ActRequest<'_>) -> Result<Vec<Action>, ActError> {
        let mut response = self
            .agent
            .post(format!("{}/act", self.url))
            .config()
            .timeout_global(Some(ACT_TIMEOUT))
            .build()
            .send_json(request)?;

        let status = response.status().as_u16();
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        };
        let headers = EchoHeaders {
            episode_id: header("X-RC-Episode-Id"),
            step: header("X-RC-Step"),
        };
        let body = response.body_mut().read_to_string()?;
        if status != 200 {
            return Err(InvalidReply::Status(status).into());
        }
        Ok(validate(&body, &headers, request.episode_id, request.step)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use test_case::test_case;

    type Edit = fn(&mut Value, &mut EchoHeaders);

    fn delta(gripper: f64) -> Value {
        json!({ "type": "joint_delta", "values": [0.05, -0.05, 0.0, 0.01, 0.0, 0.0], "gripper": gripper })
    }

    fn chunk(len: usize) -> Value {
        json!({ "type": "action_chunk", "chunk_len": len, "actions": vec![delta(0.5); len] })
    }

    fn without(mut object: Value, field: &str) -> Value {
        if let Value::Object(fields) = &mut object {
            fields.remove(field);
        }
        object
    }

    fn validate_edited(edit: Edit) -> Result<Vec<Action>, InvalidReply> {
        let mut reply = json!({ "episode_id": "ep", "step": 7, "action": delta(1.0) });
        let mut headers = EchoHeaders {
            episode_id: Some("ep".to_owned()),
            step: Some("7".to_owned()),
        };
        edit(&mut reply, &mut headers);
        validate(&reply.to_string(), &headers, "ep", 7)
    }

    #[test_case(|_, _| () => matches Ok(a) if a.len() == 1; "joint delta")]
    #[test_case(|r, _| r["action"] = chunk(3) => matches Ok(a) if a.len() == 3; "chunk")]
    #[test_case(|r, _| r["action"] = chunk(16) => matches Ok(a) if a.len() == 16; "chunk of 16")]
    #[test_case(|r, _| { r["action"] = without(chunk(2), "chunk_len"); }
        => matches Ok(a) if a.len() == 2; "chunk without chunk_len")]
    #[test_case(|r, _| r["_debug"] = json!({ "anything": 1 }) => matches Ok(_); "underscore at top")]
    #[test_case(|r, _| r["action"]["_debug"] = json!("x") => matches Ok(_); "underscore in action")]
    #[test_case(|r, _| { r["action"] = chunk(2); r["action"]["actions"][1]["_debug"] = json!(null); }
        => matches Ok(_); "underscore in chunk action")]
    #[test_case(|r, _| r["action"]["values"][0] = json!(0.051) => matches Err(InvalidReply::OutOfRange { .. }); "delta above range")]
    #[test_case(|r, _| r["action"]["values"][5] = json!(-0.051) => matches Err(InvalidReply::OutOfRange { .. }); "delta below range")]
    #[test_case(|r, _| r["action"]["gripper"] = json!(1.01) => matches Err(InvalidReply::OutOfRange { .. }); "gripper above range")]
    #[test_case(|r, _| r["action"]["gripper"] = json!(-0.01) => matches Err(InvalidReply::OutOfRange { .. }); "gripper below range")]
    #[test_case(|r, _| { r["action"] = chunk(2); r["action"]["actions"][1]["values"][2] = json!(0.06); }
        => matches Err(InvalidReply::OutOfRange { .. }); "chunk delta out of range")]
    #[test_case(|r, _| r["action"]["values"] = json!([0.0, 0.0, 0.0, 0.0, 0.0])
        => matches Err(InvalidReply::WrongValueCount { count: 5, .. }); "five values")]
    #[test_case(|r, _| r["action"]["values"] = json!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        => matches Err(InvalidReply::WrongValueCount { count: 7, .. }); "seven values")]
    #[test_case(|r, _| r["action"]["values"][0] = json!(null) => matches Err(InvalidReply::Malformed(_)); "null value")]
    #[test_case(|r, _| r["action"]["values"][0] = json!("0.0") => matches Err(InvalidReply::Malformed(_)); "string value")]
    #[test_case(|r, _| { r["action"] = without(delta(1.0), "gripper"); }
        => matches Err(InvalidReply::Malformed(_)); "missing gripper")]
    #[test_case(|r, _| { r["action"] = without(delta(1.0), "values"); }
        => matches Err(InvalidReply::Malformed(_)); "missing values")]
    #[test_case(|r, _| r["extra"] = json!(1) => matches Err(InvalidReply::UnknownField(f)) if f == "extra"; "unknown at top")]
    #[test_case(|r, _| r["action"]["speed"] = json!(1) => matches Err(InvalidReply::UnknownField(f)) if f == "action.speed"; "unknown in action")]
    #[test_case(|r, _| { r["action"] = chunk(2); r["action"]["actions"][0]["x"] = json!(1); }
        => matches Err(InvalidReply::UnknownField(f)) if f == "action.actions[0].x"; "unknown in chunk action")]
    #[test_case(|r, _| r["action"]["type"] = json!("cartesian") => matches Err(InvalidReply::Malformed(_)); "unknown action type")]
    #[test_case(|r, _| { r["action"] = chunk(1); r["action"]["actions"][0] = chunk(1); }
        => matches Err(InvalidReply::Malformed(_)); "chunk in a chunk")]
    #[test_case(|r, _| r["action"] = chunk(0) => matches Err(InvalidReply::ChunkSize(0)); "empty chunk")]
    #[test_case(|r, _| r["action"] = chunk(17) => matches Err(InvalidReply::ChunkSize(17)); "chunk of 17")]
    #[test_case(|r, _| { r["action"] = chunk(3); r["action"]["chunk_len"] = json!(2); }
        => matches Err(InvalidReply::ChunkLenMismatch { chunk_len: 2, actions: 3 }); "wrong chunk_len")]
    #[test_case(|r, _| r["episode_id"] = json!("other") => matches Err(InvalidReply::WrongEcho { at: "episode_id", .. }); "wrong episode_id")]
    #[test_case(|r, _| r["step"] = json!(8) => matches Err(InvalidReply::WrongEcho { at: "step", .. }); "wrong step")]
    #[test_case(|_, h| h.episode_id = Some("other".to_owned())
        => matches Err(InvalidReply::WrongEcho { at: "X-RC-Episode-Id", .. }); "wrong episode header")]
    #[test_case(|_, h| h.step = Some("8".to_owned()) => matches Err(InvalidReply::WrongEcho { at: "X-RC-Step", .. }); "wrong step header")]
    #[test_case(|_, h| h.step = None => matches Err(InvalidReply::WrongEcho { at: "X-RC-Step", .. }); "missing step header")]
    fn reply_validation(edit: Edit) -> Result<Vec<Action>, InvalidReply> {
        validate_edited(edit)
    }

    #[test]
    fn non_finite_number() {
        let body = r#"{"episode_id": "ep", "step": 7,
            "action": {"type": "joint_delta", "values": [1e400, 0, 0, 0, 0, 0], "gripper": 1}}"#;
        let headers = EchoHeaders {
            episode_id: Some("ep".to_owned()),
            step: Some("7".to_owned()),
        };
        std::assert_matches!(
            validate(body, &headers, "ep", 7),
            Err(InvalidReply::Malformed(_))
        );
    }

    fn ok(body: &Value) -> (u16, String) {
        (200, body.to_string())
    }

    #[test_case(Ok(ok(&json!({"ready": true, "contract": "v0", "schema_version": 2}))) => Readiness::Ready; "ready")]
    #[test_case(Ok(ok(&json!({"ready": false}))) => Readiness::NotReady; "not ready")]
    #[test_case(Ok(ok(&json!({"ready": true, "contract": "v1", "schema_version": 2})))
        => Readiness::Incompatible { contract: Some("v1".to_owned()), schema_version: Some(2) };
        "wrong contract")]
    #[test_case(Ok(ok(&json!({"ready": true, "contract": "v0", "schema_version": 3})))
        => Readiness::Incompatible { contract: Some("v0".to_owned()), schema_version: Some(3) };
        "wrong schema version")]
    #[test_case(Ok(ok(&json!({"ready": true})))
        => Readiness::Incompatible { contract: None, schema_version: None };
        "missing fields")]
    #[test_case(Ok((503, String::new())) => Readiness::HttpStatus(503); "non-200")]
    #[test_case(Ok((200, "<html>".to_owned())) => matches Readiness::Malformed(_); "malformed")]
    #[test_case(Err(ureq::Error::ConnectionFailed) => matches Readiness::Unreachable(_); "unreachable")]
    fn readiness_of(reply: HttpReply) -> Readiness {
        Readiness::from(reply)
    }

    #[test_case(Ok(ok(&json!({}))) => Health::Up; "up")]
    #[test_case(Ok((500, String::new())) => Health::HttpStatus(500); "non-200")]
    #[test_case(Err(ureq::Error::ConnectionFailed) => matches Health::Unreachable(_); "unreachable")]
    fn health_of(reply: HttpReply) -> Health {
        Health::from(reply)
    }
}
