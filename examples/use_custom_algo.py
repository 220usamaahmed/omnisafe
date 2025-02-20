import os
import omnisafe


def train():
    env_id = "Pendulum-v1"
    custom_cfgs = {}

    agent = omnisafe.Agent("DQN", env_id, custom_cfgs=custom_cfgs)
    agent.learn()


def evaluate(log_dir: str):
    evaluator = omnisafe.Evaluator(render_mode="rgb_array")
    scan_dir = os.scandir(os.path.join(log_dir, "torch_save"))
    for item in scan_dir:
        if item.is_file() and item.name.split(".")[-1] == "pt":
            evaluator.load_saved(
                render_mode="human",
                save_dir=log_dir,
                model_name=item.name,
                camera_name="track",
                width=256,
                height=256,
            )
            # evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=1)
    scan_dir.close()


def get_last_run() -> str:
    base_path = "./runs/DQN-{Pendulum-v1}"
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    return os.path.join(base_path, max(subfolders))


if __name__ == "__main__":
    # Run this first
    train()

    # Get latest run logs or provide path manually
    log_dir = get_last_run()
    evaluate(log_dir)
