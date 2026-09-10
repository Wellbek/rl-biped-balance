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
