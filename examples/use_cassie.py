import os
import omnisafe
from omnisafe.envs.core import CMDP, make
import gymnasium as gym
import imageio


def test():
    env = make("Cassie-v0", render_mode="rgb_array")
    frames = []

    observation, info = env.reset(seed=42)
    for _ in range(500):
        action = env.action_space.sample()
        state, reward, cost, terminate, truncate, info = env.step(action)

        # print(state)

        frames.append(env.render())

        if terminate or truncate:
            observation, info = env.reset()

    env.close()

    # Save frames as a video using ffmpeg
    # video_path = "output.mp4"
    # imageio.mimsave(video_path, frames, fps=30, codec="libx264")

    # print(f"Video saved to {video_path}")

def train():
    env_id = "Cassie-v0"
    custom_cfgs = {}

    agent = omnisafe.Agent("PPO", env_id, custom_cfgs=custom_cfgs)
    agent.learn()


def evaluate(log_dir: str):
    evaluator = omnisafe.Evaluator(render_mode="rgb_array")
    scan_dir = os.scandir(os.path.join(log_dir, "torch_save"))
    for item in scan_dir:
        if item.is_file() and item.name.split(".")[-1] == "pt":
            evaluator.load_saved(
                render_mode="rgb_array",
                save_dir=log_dir,
                model_name=item.name,
                camera_name="track",
                width=256,
                height=256,
            )
            evaluator.render(num_episodes=1, max_render_steps=300)
            # evaluator.evaluate(num_episodes=1)
    scan_dir.close()


def get_last_run() -> str:
    base_path = "./runs/PPO-{Cassie-v0}"
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    return os.path.join(base_path, max(subfolders))


if __name__ == "__main__":
    # test()

    # Run this first
    # train()

    # Get latest run logs or provide path manually
    log_dir = get_last_run()
    print("Running from", log_dir)
    evaluate(log_dir)
