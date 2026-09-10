import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs  # noqa: F401,E402  (registers BipedBalance-v0)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="ppo_biped_balance")
    args = parser.parse_args()

    vec_env = make_vec_env("BipedBalance-v0", n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048 // args.n_envs,
        batch_size=64,
        tensorboard_log="runs",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path="models/checkpoints",
        name_prefix=args.run_name,
    )

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        callback=checkpoint_callback,
    )
    model.save(f"models/{args.run_name}")
    print(f"saved to models/{args.run_name}.zip")


if __name__ == "__main__":
    main()
