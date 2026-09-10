"""Load a trained policy and watch it balance in a plain window.

Renders offscreen and displays via OpenCV instead of MuJoCo's native GLFW
viewer, since GLFW segfaults on some Wayland setups.

Usage:
    python scripts/watch.py --latest-checkpoint
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OpenCV's bundled Qt doesn't have a Wayland plugin, force it through X11/XWayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2  # noqa: E402
import envs  # noqa: F401,E402
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

JOINTS = [
    "thigh_joint",
    "leg_joint",
    "foot_joint",
    "thigh_left_joint",
    "leg_left_joint",
    "foot_left_joint",
]


def joint_readout(env):
    lines = []
    for name in JOINTS:
        qposadr = env.model.joint(name).qposadr[0]
        qveladr = env.model.joint(name).dofadr[0]
        angle_deg = env.data.qpos[qposadr] * 180.0 / 3.14159265
        vel = env.data.qvel[qveladr]
        lines.append(f"{name:>17s}: {angle_deg:7.1f} deg  {vel:7.2f} rad/s")
    return lines


def draw_overlay(frame, env, info, reward, step_count):
    torso_z = env.data.qpos[1]
    torso_pitch_deg = env.data.qpos[2] * 180.0 / 3.14159265
    push = "PUSH ACTIVE" if info.get("push_active") else ""

    lines = [
        f"step {step_count}  reward {reward:+.2f}  {push}",
        f"torso z: {torso_z:5.2f} m   torso pitch: {torso_pitch_deg:6.1f} deg",
        "",
    ] + joint_readout(env)

    y = 24
    for line in lines:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA
        )
        y += 22
    return frame


def print_console(env, info, reward, step_count):
    torso_z = env.data.qpos[1]
    torso_pitch_deg = env.data.qpos[2] * 180.0 / 3.14159265
    push = "PUSH" if info.get("push_active") else "    "
    joint_str = "  ".join(
        f"{n.split('_joint')[0]}={env.data.qpos[env.model.joint(n).qposadr[0]] * 180.0 / 3.14159265:6.1f}"
        for n in JOINTS
    )
    print(
        f"\rstep {step_count:6d} | z={torso_z:5.2f} pitch={torso_pitch_deg:6.1f} "
        f"r={reward:+.2f} {push} | {joint_str}",
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="ppo_biped_balance_v2")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--latest-checkpoint",
        action="store_true",
        help="load the most recent checkpoint instead of the final saved model",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="don't print live joint values to the terminal",
    )
    args = parser.parse_args()

    if args.latest_checkpoint:
        checkpoints = glob.glob(f"models/checkpoints/{args.run_name}_*_steps.zip")
        checkpoints.sort(key=os.path.getmtime)
        if not checkpoints:
            raise SystemExit("no checkpoints found yet, training needs to run a bit longer")
        model_path = checkpoints[-1]
        print(f"loading {model_path}")
    else:
        model_path = f"models/{args.run_name}.zip"

    model = PPO.load(model_path)
    env = gym.make("BipedBalance-v0", width=1280, height=960).unwrapped

    renderer = mujoco.Renderer(env.model, height=960, width=1280)
    cam = mujoco.MjvCamera()
    cam.lookat = [0, 0, 1.0]
    cam.distance = 3.5
    cam.azimuth = 90
    cam.elevation = -10

    window_name = "biped balance (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)
    obs, _ = env.reset()
    step_count = 0
    for ep in range(args.episodes):
        terminated = truncated = False
        while not (terminated or truncated):
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = draw_overlay(frame_bgr, env, info, reward, step_count)
            cv2.imshow(window_name, frame_bgr)

            if not args.no_console:
                print_console(env, info, reward, step_count)

            time_left = env.dt - (time.time() - step_start)
            wait_ms = max(int(time_left * 1000), 1)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                env.close()
                cv2.destroyAllWindows()
                print()
                return

        step_count = 0
        obs, _ = env.reset()

    env.close()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
