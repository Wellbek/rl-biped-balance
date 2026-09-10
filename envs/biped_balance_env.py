import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "biped.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
}


class BipedBalanceEnv(MujocoEnv, utils.EzPickle):
    """Biped that has to stay standing while getting randomly shoved.

    Unlike the stock Walker2d task (reward = walk forward fast), this one
    rewards staying alive, upright, and roughly in place. Every so often a
    random horizontal force gets applied to the torso for a few timesteps
    to simulate a push, and the agent only feels it through the resulting
    joint/torso motion, not through a direct force reading, since a real
    robot wouldn't have that either.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 125,
    }

    def __init__(
        self,
        frame_skip: int = 4,
        healthy_z_range: tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: tuple[float, float] = (-1.0, 1.0),
        ctrl_cost_weight: float = 1e-3,
        action_rate_cost_weight: float = 0.4,
        joint_vel_cost_weight: float = 0.01,
        push_prob: float = 1.0 / 250.0,
        push_force_range: tuple[float, float] = (50.0, 300.0),
        push_duration_steps: int = 5,
        reset_noise_scale: float = 5e-3,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, frame_skip, **kwargs)

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._ctrl_cost_weight = ctrl_cost_weight
        self._action_rate_cost_weight = action_rate_cost_weight
        self._joint_vel_cost_weight = joint_vel_cost_weight
        self._push_prob = push_prob
        self._push_force_range = push_force_range
        self._push_duration_steps = push_duration_steps
        self._reset_noise_scale = reset_noise_scale

        self._push_steps_remaining = 0
        self._current_push_force = 0.0
        self._start_x = 0.0
        self._prev_action = None

        MujocoEnv.__init__(
            self,
            MODEL_PATH,
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        obs_size = self.data.qpos.size - 1 + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self._prev_action = np.zeros(self.model.nu)

    def _get_obs(self):
        # drop the x position (rootx) so the policy can't just memorize an
        # absolute location, only relative pose/velocity matters for balance
        qpos = self.data.qpos.flatten()[1:]
        qvel = self.data.qvel.flatten()
        return np.concatenate([qpos, qvel])

    def _is_healthy(self):
        z = self.data.qpos[1]
        angle = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        return min_z < z < max_z and min_angle < angle < max_angle

    def _maybe_push(self):
        if self._push_steps_remaining > 0:
            self._push_steps_remaining -= 1
            if self._push_steps_remaining == 0:
                self.data.xfrc_applied[1, :] = 0.0
            return

        if self.np_random.random() < self._push_prob:
            magnitude = self.np_random.uniform(*self._push_force_range)
            direction = self.np_random.choice([-1.0, 1.0])
            self.data.xfrc_applied[1, 0] = direction * magnitude
            self._current_push_force = direction * magnitude
            self._push_steps_remaining = self._push_duration_steps

    def step(self, action):
        self._maybe_push()

        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        healthy = self._is_healthy()
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        # penalize jerking the action back and forth between steps, not just
        # its raw magnitude, otherwise buzzing/vibrating in place is free as
        # long as the average torque stays small
        action_rate_cost = self._action_rate_cost_weight * np.sum(
            np.square(action - self._prev_action)
        )
        self._prev_action = np.array(action)
        # penalize raw joint angular velocity directly too, on top of the
        # action-rate cost above, since a stiff joint can still physically
        # bounce/resonate fast even when the commanded action itself is
        # changing smoothly
        joint_vel_cost = self._joint_vel_cost_weight * np.sum(
            np.square(self.data.qvel[3:])
        )
        # penalize actual distance from the starting spot, not just how
        # fast it's currently drifting, otherwise a slow wander off to the
        # side is "free" as long as it's not accelerating
        position_cost = 0.5 * abs(x_after - self._start_x)
        upright_bonus = 1.0 - abs(self.data.qpos[2])

        reward = (
            1.0
            + upright_bonus
            - ctrl_cost
            - action_rate_cost
            - joint_vel_cost
            - position_cost
        )
        terminated = not healthy

        obs = self._get_obs()
        info = {
            "x_position": x_after,
            "push_active": self._push_steps_remaining > 0,
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.data.xfrc_applied[:, :] = 0.0
        self._push_steps_remaining = 0
        self._start_x = qpos[0]
        self._prev_action = np.zeros(self.model.nu)
        return self._get_obs()
