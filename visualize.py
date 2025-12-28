import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from env.swarm_nav_env import SwarmNavEnv


# -------------------------------------------------
# ENV REGISTER
# -------------------------------------------------
def env_creator(config):
    return SwarmNavEnv(config)

register_env("SwarmNavEnv", env_creator)


# -------------------------------------------------
# 1️⃣ LEARNING CURVE
# -------------------------------------------------
def plot_learning_curve(results_dir, save_path="learning_curve.png"):
    csv_files = glob.glob(
        os.path.join(results_dir, "**/progress.csv"),
        recursive=True
    )
    assert len(csv_files) > 0, "progress.csv bulunamadı"

    df = pd.read_csv(csv_files[0])

    plt.figure(figsize=(8, 5))
    plt.plot(df["training_iteration"], df["episode_reward_mean"])
    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO Learning Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✔ Learning curve kaydedildi: {save_path}")


# -------------------------------------------------
# 2️⃣ CHECKPOINT BUL
# -------------------------------------------------
def find_latest_checkpoint(results_dir):
    checkpoints = glob.glob(
        os.path.join(results_dir, "**/checkpoint_*"),
        recursive=True
    )
    checkpoints.sort(key=os.path.getmtime)
    assert len(checkpoints) > 0, "Checkpoint bulunamadı"
    return checkpoints[-1]


# -------------------------------------------------
# 3️⃣ ROLLOUT + VIDEO
# -------------------------------------------------
def rollout_and_record(checkpoint_path, video_path="videos/rollout.mp4"):
    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(checkpoint_path)

    env = SwarmNavEnv({
        "num_agents": 4,
        "arena_size": 10.0,
        "max_steps": 300,
    })

    obs, _ = env.reset()

    frames = []

    done = {"__all__": False}
    while not done["__all__"]:
        actions = {}
        for aid, o in obs.items():
            actions[aid] = algo.compute_single_action(
                o,
                policy_id="shared_policy",
                explore=False
            )

        obs, rewards, done, trunc, info = env.step(actions)

        frame = env.render(mode="rgb_array")
        frames.append(frame)

    ray.shutdown()

    # --- VIDEO ---
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fig = plt.figure()
    im = plt.imshow(frames[0])
    plt.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50
    )

    ani.save(video_path, writer="ffmpeg")
    plt.close()

    print(f"✔ Video kaydedildi: {video_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    RESULTS_DIR = "ray_results"

    plot_learning_curve(
        RESULTS_DIR,
        save_path="learning_curve.png"
    )

    checkpoint = find_latest_checkpoint(RESULTS_DIR)
    print(f"✔ Checkpoint bulundu: {checkpoint}")

    rollout_and_record(
        checkpoint,
        video_path="videos/swarm_rollout.mp4"
    )
