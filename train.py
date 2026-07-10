"""
train.py — Train a DQN agent on ALE/Freeway-v5 using Stable Baselines3.

Team members:
  - Kumi Yunis:        CnnPolicy (10 experiments)  
  - Nformi Modestine:  CnnPolicy (10 experiments)  
  - Josue Byiringiro:  MlpPolicy (10 experiments)  

Saves the best model as dqn_model.zip in the project root.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import importlib
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


# ─── Data Structures ───

@dataclass
class HyperParams:
    lr: float
    gamma: float
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_fraction: float


class EpisodeLoggerCallback(BaseCallback):
    """Logs per-episode reward and length to a CSV file."""
    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.rows.append({
                    "timesteps": int(self.num_timesteps),
                    "reward": float(ep["r"]),
                    "episode_length": int(ep["l"]),
                })
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "reward", "episode_length"])
            writer.writeheader()
            writer.writerows(self.rows)


# ─── Environment Utilities ───

def _try_register_ale_envs() -> bool:
    try:
        ale_py = importlib.import_module("ale_py")
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)
            return True
        if hasattr(ale_py, "register_envs"):
            ale_py.register_envs(gym)
            return True
    except Exception:
        pass
    return False


def resolve_env_id(env_id: str) -> str:
    """Use Freeway-v5, or fall back to a compatible env if needed."""
    candidates = [env_id]
    if env_id.startswith("ALE/") and env_id.endswith("-v5"):
        game = env_id.split("/", 1)[1].replace("-v5", "")
        candidates.append(f"{game}NoFrameskip-v4")
    if env_id.startswith("ALE/"):
        _try_register_ale_envs()
    for candidate in candidates:
        try:
            gym.spec(candidate)
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"Environment '{env_id}' was not found.")


def build_env(env_id, policy, n_envs, seed, frame_stack, render_mode=None):
    """Create a vectorized Atari environment with appropriate wrappers."""
    if policy == "CnnPolicy":
        env = make_atari_env(
            env_id, n_envs=n_envs, seed=seed,
            env_kwargs={"obs_type": "rgb", "render_mode": render_mode},
        )
        env = VecMonitor(env)
        env = VecFrameStack(env, n_stack=frame_stack)
        return env
    env = make_vec_env(
        env_id, n_envs=n_envs, seed=seed,
        env_kwargs={"obs_type": "ram", "render_mode": render_mode},
        wrapper_class=Monitor,
    )
    return VecMonitor(env)


# ─── Training Functions ───

def train_once(env_id, policy, hparams, total_timesteps, n_envs, frame_stack,
               seed, eval_freq, eval_episodes, device, run_dir):
    """Train a single DQN experiment and return metadata dict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    train_env = build_env(env_id, policy, n_envs, seed, frame_stack)
    eval_env = build_env(env_id, policy, 1, seed + 100, frame_stack)

    episode_logger = EpisodeLoggerCallback(run_dir / "episode_log.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq, deterministic=True, render=False,
        n_eval_episodes=eval_episodes,
    )

    model = DQN(
        policy, train_env,
        learning_rate=hparams.lr, gamma=hparams.gamma, batch_size=hparams.batch_size,
        exploration_initial_eps=hparams.epsilon_start,
        exploration_final_eps=hparams.epsilon_end,
        exploration_fraction=hparams.epsilon_fraction,
        buffer_size=100_000, learning_starts=10_000,
        target_update_interval=10_000, train_freq=4, gradient_steps=1,
        verbose=1, seed=seed,
        tensorboard_log=str(run_dir / "tb"), device=device,
    )

    model.learn(total_timesteps=total_timesteps,
                callback=[episode_logger, eval_callback],
                progress_bar=True)
    model.save(run_dir / "dqn_model")

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    metadata = {
        "env_id": env_id, "policy": policy,
        "obs_type": "rgb" if policy == "CnnPolicy" else "ram",
        "hyperparameters": asdict(hparams),
        "mean_reward": float(mean_reward), "std_reward": float(std_reward),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_env.close()
    eval_env.close()
    return metadata


def run_member_experiments(member_name, experiments, policy, output_dir,
                           env_id="ALE/Freeway-v5", total_timesteps=100_000,
                           n_envs=4, frame_stack=4, seed=42,
                           eval_freq=100_000, eval_episodes=3, device="auto"):
    """Run all 10 experiments for one group member and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_env_id(env_id)

    summary_rows = []
    best = None

    for idx, hp in enumerate(experiments, start=1):
        run_dir = output_dir / f"{member_name}_exp{idx}"
        print(f"\n{'='*60}")
        print(f"  {member_name} | Experiment {idx}/10 | {policy}")
        print(f"  lr={hp.lr}, gamma={hp.gamma}, batch={hp.batch_size}, "
              f"eps=[{hp.epsilon_start}->{hp.epsilon_end}], frac={hp.epsilon_fraction}")
        print(f"{'='*60}")

        metadata = train_once(
            env_id=resolved, policy=policy, hparams=hp,
            total_timesteps=total_timesteps, n_envs=n_envs, frame_stack=frame_stack,
            seed=seed + idx, eval_freq=eval_freq, eval_episodes=eval_episodes,
            device=device, run_dir=run_dir,
        )
        summary_rows.append({
            "experiment": idx, "member": member_name, "policy": policy,
            "lr": hp.lr, "gamma": hp.gamma, "batch_size": hp.batch_size,
            "epsilon_start": hp.epsilon_start, "epsilon_end": hp.epsilon_end,
            "epsilon_fraction": hp.epsilon_fraction,
            "mean_reward": metadata["mean_reward"], "std_reward": metadata["std_reward"],
        })
        if best is None or metadata["mean_reward"] > best["mean_reward"]:
            best = {"mean_reward": metadata["mean_reward"], "run_dir": run_dir, "metadata": metadata}

    fieldnames = ["experiment", "member", "policy", "lr", "gamma", "batch_size",
                  "epsilon_start", "epsilon_end", "epsilon_fraction", "mean_reward", "std_reward"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    if best is not None:
        best_model = best["run_dir"] / "dqn_model.zip"
        if best_model.exists():
            (output_dir / "dqn_model.zip").write_bytes(best_model.read_bytes())
        with (output_dir / "best_model_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(best["metadata"], f, indent=2)

    print(f"\n{member_name} done! Best mean reward: {best['mean_reward']:.2f}")
    return summary_rows, best


# ─── Main ───

if __name__ == "__main__":
    # Global config
    ENV_ID = "ALE/Freeway-v5"
    TOTAL_TIMESTEPS = 100_000
    N_ENVS = 4
    FRAME_STACK = 4
    SEED = 42
    EVAL_FREQ = 100_000
    EVAL_EPISODES = 3
    DEVICE = "auto"

    # ─── Kumi Yunis: CnnPolicy (10 experiments) ───
    kumi_experiments = [
        HyperParams(1e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 1: baseline
        HyperParams(5e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 2: higher lr
        HyperParams(2.5e-4, 0.99,  32, 1.0, 0.10, 0.10),   # 3: medium lr
        HyperParams(1e-4,   0.95,  32, 1.0, 0.10, 0.10),   # 4: lower gamma
        HyperParams(1e-4,   0.999, 32, 1.0, 0.10, 0.10),   # 5: higher gamma
        HyperParams(1e-4,   0.99,  64, 1.0, 0.10, 0.10),   # 6: larger batch
        HyperParams(1e-4,   0.99,  32, 1.0, 0.05, 0.10),   # 7: lower eps_end
        HyperParams(1e-4,   0.99,  32, 1.0, 0.10, 0.20),   # 8: longer exploration
        HyperParams(1e-4,   0.99,  32, 1.0, 0.02, 0.25),   # 9: low eps_end + long explore
        HyperParams(5e-5,   0.995, 64, 1.0, 0.05, 0.15),   # 10: conservative combo
    ]

    kumi_rows, kumi_best = run_member_experiments(
        member_name="Kumi_Yunis",
        experiments=kumi_experiments,
        policy="CnnPolicy",
        output_dir="runs_kumi_dqn",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )

    # ─── Nformi Modestine: CnnPolicy (10 experiments) — add later ───
    # nformi_experiments = [
    #     HyperParams(...),  # 1
    #     ...
    # ]
    # nformi_rows, nformi_best = run_member_experiments(
    #     member_name="Nformi_Modestine",
    #     experiments=nformi_experiments,
    #     policy="CnnPolicy",
    #     output_dir="runs_nformi_dqn",
    #     env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
    #     n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 100,
    #     eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    # )

    # ─── Josue Byiringiro: MlpPolicy (10 experiments) — add later ───
    # josue_experiments = [
    #     HyperParams(...),  # 1
    #     ...
    # ]
    # josue_rows, josue_best = run_member_experiments(
    #     member_name="Josue_Byiringiro",
    #     experiments=josue_experiments,
    #     policy="MlpPolicy",
    #     output_dir="runs_josue_dqn",
    #     env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
    #     n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 200,
    #     eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    # )

    # ─── Save best model (update when all members have run) ───
    all_rows = kumi_rows  # add: + nformi_rows + josue_rows
    candidates = [
        ("Kumi_Yunis", kumi_best, "runs_kumi_dqn"),
        # ("Nformi_Modestine", nformi_best, "runs_nformi_dqn"),
        # ("Josue_Byiringiro", josue_best, "runs_josue_dqn"),
    ]

    winner_name, winner_data, winner_dir = max(candidates, key=lambda c: c[1]["mean_reward"])
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {winner_name}")
    print(f"  Mean Reward: {winner_data['mean_reward']:.2f}")
    print(f"{'='*60}")

    winner_out = Path("best_overall_model")
    winner_out.mkdir(parents=True, exist_ok=True)
    src_model = Path(winner_dir) / "dqn_model.zip"
    if src_model.exists():
        shutil.copy2(src_model, winner_out / "dqn_model.zip")
        shutil.copy2(src_model, Path("dqn_model.zip"))
    with (winner_out / "best_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(winner_data["metadata"], f, indent=2)

    with open("all_experiments_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nTraining complete! Best model saved as dqn_model.zip")
