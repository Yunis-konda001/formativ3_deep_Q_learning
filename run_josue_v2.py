"""
run_josue_v2.py — Josue's IMPROVED MlpPolicy experiments (v2).

A follow-up to the baseline runs in train.py (runs_josue_dqn). It reuses the
training machinery from train.py without modifying it, and applies the lessons
from the v1 study:
  - learning-rate grid re-centered on the productive 3e-4..1e-3 band
  - gamma / batch / epsilon tuned ON TOP of a working learning rate
  - longer budget (200k timesteps) to test whether v1's zeros just needed time

Run from the repo root:  python run_josue_v2.py
Output: runs_josue_v2/  (kept alongside the baseline runs_josue_dqn/).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train import HyperParams, run_member_experiments

# Global config — note the longer budget vs the 100k baseline.
ENV_ID = "ALE/Freeway-v5"
TOTAL_TIMESTEPS = 200_000          # v2: doubled from 100k
N_ENVS = 4
FRAME_STACK = 4
SEED = 42
EVAL_FREQ = 200_000
EVAL_EPISODES = 3
DEVICE = "auto"

# v2 experiments: learning rate re-centered on the band that actually learns,
# then gamma / batch / epsilon varied on top of a working lr.
josue_v2_experiments = [
    HyperParams(3e-4,  0.99,  32,  1.0, 0.10, 0.10),   # 1: productive-band baseline
    HyperParams(5e-4,  0.99,  32,  1.0, 0.10, 0.10),   # 2: previous best lr
    HyperParams(1e-3,  0.99,  32,  1.0, 0.10, 0.10),   # 3: aggressive lr
    HyperParams(7e-4,  0.99,  32,  1.0, 0.10, 0.10),   # 4: between 5e-4 and 1e-3
    HyperParams(5e-4,  0.99,  64,  1.0, 0.10, 0.10),   # 5: working lr + larger batch
    HyperParams(5e-4,  0.99, 128,  1.0, 0.10, 0.10),   # 6: working lr + much larger batch
    HyperParams(5e-4,  0.995, 32,  1.0, 0.10, 0.10),   # 7: working lr + higher gamma
    HyperParams(5e-4,  0.99,  32,  1.0, 0.05, 0.10),   # 8: working lr + more greedy end
    HyperParams(5e-4,  0.99,  32,  1.0, 0.10, 0.20),   # 9: working lr + longer exploration
    HyperParams(3e-4,  0.99,  64,  1.0, 0.05, 0.15),   # 10: tuned combo on productive lr
]

if __name__ == "__main__":
    rows, best = run_member_experiments(
        member_name="Josue_Byiringiro_v2",
        experiments=josue_v2_experiments,
        policy="MlpPolicy",
        output_dir="runs_josue_v2",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 200,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )
    print("\n=== JOSUE v2 EXPERIMENTS COMPLETE ===")
    print(f"Best mean reward: {best['mean_reward']:.2f}")
    print("Summary written to runs_josue_v2/summary.csv")
