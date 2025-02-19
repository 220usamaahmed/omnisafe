import os
import omnisafe


def train():
    env_id = "SafetyPointGoal1-v0"
    custom_cfgs = {
        "train_cfgs": {
            "total_steps": 10000,
            "vector_env_nums": 1,
            "parallel": 1,
        },
        "algo_cfgs": {
            "steps_per_epoch": 1000,
            "update_iters": 1,
        },
        "logger_cfgs": {
            "use_wandb": False,
        },
    }

    # agent = omnisafe.Agent("DDPG", env_id, custom_cfgs=custom_cfgs)
    agent = omnisafe.Agent("DQN", env_id, custom_cfgs=custom_cfgs)
    # agent.learn()


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
            evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=1)
    scan_dir.close()


if __name__ == "__main__":
    train()
    # evaluate(
    #     "/Users/usama/HRL/exploring-omnisafe/runs/DDPG-{Pendulum-v1}/seed-000-2025-02-19-14-47-23"
    # )
