# rl-biped-balance

MuJoCo RL bipedal balance and push recovery.

## Setup

Everything runs in a dedicated conda environment (Python 3.11) so it doesn't
clash with anything else on the machine.

```bash
conda create -n rl-biped python=3.11 -y
conda activate rl-biped
```

If you don't have an NVIDIA GPU (or just want to keep the install small),
grab PyTorch from the CPU-only wheel index first, otherwise pip will pull in
several GB of CUDA packages it won't use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you do have an NVIDIA GPU and want CUDA acceleration, skip that step and
just let the next command install the default GPU-enabled PyTorch build.

```bash
pip install mujoco "gymnasium[mujoco]" stable-baselines3 tensorboard imageio
```

Note for zsh users: extras like `gymnasium[mujoco]` need to be quoted,
otherwise zsh tries to glob-expand the brackets and fails with
"no matches found".

To come back to this environment later:

```bash
conda activate rl-biped
```

### Verifying the install

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "import gymnasium as gym; env = gym.make('Walker2d-v5'); print(env.observation_space, env.action_space)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirmed working versions on this machine: MuJoCo 3.13.0, Gymnasium 1.3.0,
Stable-Baselines3 2.9.0, PyTorch 2.14.0+cpu.

## The model

I'm using the simple `walker2d` MJCF model that
ships with Gymnasium's MuJoCo envs
(`gymnasium/envs/mujoco/assets/walker2d.xml`).
Standard model used for biped benchmarks.
Torso + thigh/shin/foot per side, six motors (hip, knee, ankle x2).

It's planar, so forward/back and pitch only. Cuts out a whole dimension for an initial experimentation.

It's copied it into `models/biped.xml` for adjustment (push force
logic, sensors, domain randomization) without touching the installed
package.

![biped model](assets/images/biped_model.png)

To look at it interactively (drag to orbit, scroll to zoom, ctrl+right-click
drag on a body to shove it around with the mouse):

```bash
python scripts/view_model.py
```

## The task

The goal is purely to stand still and stay on its feet while getting shoved, no walking. Base `Walker2d-v5` rewards walking forward fast, so wrote a custom env instead: `envs/biped_balance_env.py`, registered as `BipedBalance-v0`.

- reward = stay alive + stay upright - control effort - distance strayed
  from the starting spot. Zero reward for moving, and actual displacement
  from start is penalized directly (not just velocity), to avoid slowly moving off the start location.
- episode ends if the torso drops too low or tips past a set pitch angle.
- every so often (random, ~1 in 250 steps) a random horizontal force (our push) hits the torso for a handful of timesteps. Magnitude and
  direction are randomized each time.
- the agent doesn't get to see the push force directly, only feels it
  through the resulting joint and torso motion. A real robot wouldn't have
  a universal "push" sensor either, so no point letting the policy
  cheat with one in sim.

Sanity check that it runs and pushes actually fire:

```bash
python -c "
import envs, gymnasium as gym
env = gym.make('BipedBalance-v0')
obs, info = env.reset(seed=0)
for _ in range(500):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
"
```
