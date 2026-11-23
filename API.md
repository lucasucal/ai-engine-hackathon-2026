# Policy API

The policy is an HTTP server. We call `POST /act` once per observation. One observation is the arm and camera at a single moment. The reply says how to move for the next 50 ms, or for the next few 50 ms steps. We apply that, then call again with a new observation. The policy does not pick the rate and does not talk to the arm.

An episode is 300 of those 50 ms steps (15 s) on the current tasks. `episode_id` is a new string each episode and stays the same until it ends. `step` starts at 0 and counts 50 ms steps. The same `episode_id` and `step` may be sent twice. Return a valid action both times.

`GET /healthz` returns `200` and `{}` before weights load. `GET /readyz` returns `200` and `{ready, contract: "v0", schema_version: 2}` after load. Calls start once `ready` is true.

## Request

`POST /act`, `Content-Type: application/json`.

```json
{
  "episode_id": "demo",
  "step": 0,
  "task_id": "bench_lift_can_v0",
  "schema_version": 2,
  "images": { "top": "<base64 of one 640x480 JPEG>" },
  "proprio": {
    "joint_pos": [-0.2055, -0.60, 1.15, -3.60, 1.5708, 0.0],
    "joint_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gripper": 1.0,
    "ee_pos": [0.8951, -0.0087, 0.2514],
    "ee_quat": [-0.0289, -0.6302, 0.7751, 0.0355]
  }
}
```

| field | type | when |
|---|---|---|
| `episode_id` | string | same for every call in one episode |
| `step` | int | the 50 ms step this observation is for, starting at 0 |
| `task_id` | string | which task. `"hidden"` on tasks whose name is not public |
| `schema_version` | int | `2` |
| `images.top` | string | one 640×480 JPEG, standard base64, for this step. This is the only camera |
| `proprio.joint_pos` | 6 floats | joint angles now, radians. Order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3 |
| `proprio.joint_vel` | 6 floats | joint velocity now, rad/s, same order |
| `proprio.gripper` | float | measured opening now. `0.0` closed, `1.0` open |
| `proprio.ee_pos` | 3 floats | palm position now, metres, in the robot base frame |
| `proprio.ee_quat` | 4 floats | palm orientation now, quaternion **wxyz** |

`images` and `proprio` are the state at `step`. They are replaced on the next call.

## Response

JSON. `episode_id` and `step` must be copied from the request. Also send headers `X-RC-Episode-Id` and `X-RC-Step` with those same values.

`action` is one of two objects. No other `action.type` is accepted. There is no Cartesian command.

### `"joint_delta"` — one 50 ms step

This is what the trained policy returns.

```json
{
  "episode_id": "demo",
  "step": 0,
  "action": {
    "type": "joint_delta",
    "values": [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gripper": 1.0
  }
}
```

| field | type | meaning |
|---|---|---|
| `type` | string | `"joint_delta"` |
| `values` | 6 floats | radians to **add** to the joint command over this 50 ms. Each must be in `[-0.05, 0.05]`. Same joint order as `joint_pos` |
| `gripper` | float | commanded opening for this step, in `[0.0, 1.0]`. `0.0` closed, `1.0` open. This is the opening, not a change |

### `"action_chunk"` — several steps from one call

Optional. The trained policy does not send this. `actions[0]` is this `step`, `actions[1]` is the next 50 ms, and so on. Applied in order, with no new observation in between. The next request then has `step` increased by how many actions were applied.

```json
{
  "episode_id": "demo",
  "step": 0,
  "action": {
    "type": "action_chunk",
    "chunk_len": 2,
    "actions": [
      { "type": "joint_delta", "values": [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0], "gripper": 1.0 },
      { "type": "joint_delta", "values": [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0], "gripper": 1.0 }
    ]
  }
}
```

| field | type | meaning |
|---|---|---|
| `type` | string | `"action_chunk"` |
| `actions` | list of 1 to 16 `joint_delta` objects | the steps, in order |
| `chunk_len` | int, optional | if present, must equal `len(actions)` |

Numbers must be finite JSON numbers. Unknown fields are rejected, except names that start with `_`.
