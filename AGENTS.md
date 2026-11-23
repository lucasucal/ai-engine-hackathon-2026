# AGENTS.md

## Comments

Prefer self-documenting code; let names and types carry the explanation. Only
comment on what the code cannot say for itself — the reason behind a non-obvious
choice, an upstream constraint, a workaround. Never restate the next line, and
never narrate your edits.

## Rust dependencies

Every dependency goes in `[workspace.dependencies]`, even one only a single
crate uses, and member crates take it with `<name>.workspace = true`. Do not
specify features there — `default-features = false` is the only exception;
enable features on the member crate that needs them.
