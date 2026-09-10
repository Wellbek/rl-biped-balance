from gymnasium.envs.registration import register

from envs.biped_balance_env import BipedBalanceEnv

register(
    id="BipedBalance-v0",
    entry_point="envs.biped_balance_env:BipedBalanceEnv",
    max_episode_steps=1000,
)

__all__ = ["BipedBalanceEnv"]
