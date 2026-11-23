# AGENTS.md

## Tools

Install a tool with apt when apt has it. Otherwise add it to `mise-tools.toml` with an exact
version, so that image builds are reproducible.

## The image is complete on its own

Everything the container needs is set up in the Dockerfile. Do not add `postCreateCommand` or any
other lifecycle command (`onCreateCommand`, `postStartCommand`, …) to `devcontainer.json`.
