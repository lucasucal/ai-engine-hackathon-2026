# ai-engine-hackathon-2026

Runs a policy, served over HTTP, on a UR12e with a Robotiq 2F-140 gripper and an Orbbec Gemini 336
camera. The policy's API is in [CONTRACT.md](crates/policy-runner/CONTRACT.md).

## Git Configuration

You will need to configure your github to use [ssh keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh), in order to [sign commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) and authenticate yourself. You can use the same public key for authentication and commit signing.

The host computer needs the following git configuration. Run these commands to set up your git config:

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
git config --global user.signingkey "~/.ssh/id_ed25519.pub"
git config --global commit.gpgsign "true"
git config --global gpg.format "ssh"
```

Replace `your-email@example.com` and `Your Name` with your actual email and name.

## Devcontainer setup

We use [devcontainers](https://containers.dev) as development environment to unify everyone's
development setup and also to avoid ROS turning our package installations into unrepairable mess.

The dev container expects a Linux host with an NVIDIA GPU and a Wayland session. GUI apps inside
the container draw directly on the host's Wayland compositor, which runs on the NVIDIA GPU, so the
container needs the host's NVIDIA driver to render on that same GPU. If it renders on another GPU
instead, apps crash with `failed to import supplied dmabufs`.

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host and register it with Docker. The toolkit isn't in Ubuntu's repositories, so we add
NVIDIA's with [`extrepo`](https://manpages.ubuntu.com/manpages/noble/man1/extrepo.1p.html). It is
listed under the `non-free` policy, which `extrepo` doesn't enable by default:

```bash
sudo apt install extrepo
sudo sed -i 's/^# *- non-free/- non-free/' /etc/extrepo/config.yaml
sudo extrepo enable nvidia-docker
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If `extrepo enable` complains about the policy, check that `- non-free` is listed under
`enabled_policies` in `/etc/extrepo/config.yaml`. Restarting Docker stops all running containers.

Then open the workspace in VS Code and run **Dev Containers: Rebuild and Reopen in Container**.
To check that the GPU is visible inside the container, run `nvidia-smi`.

The container sees the host's USB devices, for the Orbbec camera. To be allowed to open the camera,
install Orbbec's udev rules on the host, then plug the camera in again:

```bash
sudo curl -fsSL -o /etc/udev/rules.d/99-obsensor-libusb.rules \
  https://raw.githubusercontent.com/orbbec/OrbbecSDK_ROS2/v2.10.7/orbbec_camera/scripts/99-obsensor-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Inside the container, `lsusb` should then list the camera.

## Running tasks

Tasks are run with [mise](https://mise.jdx.dev). Run `mise tasks` to see all tasks and what they
do. Extra arguments go through to the launch file or the program, e.g. `mise run mock-robot rviz:=true`.

### On mock hardware

In three terminals:

```bash
mise run mock-robot
mise run fake-policy
mise run policy-runner -- --task <task_id>
```

`mock-robot` brings up the UR12e and the gripper on mock hardware. `fake-policy` serves a stand-in
policy on port 8000; `mise run fake-policy -- --help` lists the replies it can send, including late,
chunked and invalid ones. The policy runner is a terminal UI that runs episodes of the policy
against the arm; it sends a black frame unless given `--camera <topic>`.

### On the real robot

```bash
mise run real-robot robot_ip:=<address>
mise run policy-runner -- --task <task_id> --policy <url> --camera /camera/color/image_raw/compressed
```

By default `real-robot` sends the driver's program on connect, which needs the arm in Remote Control
mode; with `headless_mode:=false`, play External Control on the pendant in Local mode instead. It
also starts the Gemini 336 unless given `camera:=false`.

## Python

Python dependencies are managed with [uv](https://docs.astral.sh/uv), in `pyproject.toml`. mise
creates and activates `.venv`; `mise run python:check` formats, lints and type-checks.
