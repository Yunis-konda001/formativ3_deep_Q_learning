"""
run_josue_v3.py — Josue's multi-seed ROBUSTNESS study (v3).

Motivation: v2 showed the MLP-on-RAM agent is brittle — at the same good lr=5e-4,
single hyperparameter changes collapsed learning to 0.0. This study isolates
*seed sensitivity* by training the SAME best configuration under several different
random seeds, so we can report how reliably it learns (X of N seeds) and the
spread of rewards, instead of trusting a single run.

Reuses train.py's machinery without modifying it.

Run from the repo root:  python run_josue_v3.py
Output: runs_josue_v3/  (summary.csv has one row per seed).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train import HyperParams, run_member_experiments

# Config — longer budget than v2 (200k -> 500k) to give each seed a fairer chance.
ENV_ID = "ALE/Freeway-v5"
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
FRAME_STACK = 4
SEED = 42
EVAL_FREQ = 500_000
EVAL_EPISODES = 3
DEVICE = "auto"

# The best configuration from v1/v2, repeated 5x. run_member_experiments assigns
# seed = base_seed + idx to each, so these 5 identical configs train under 5
# distinct random seeds (base+1 .. base+5).
BEST_CONFIG = HyperParams(5e-4, 0.99, 32, 1.0, 0.10, 0.10)
josue_v3_experiments = [BEST_CONFIG for _ in range(5)]

if __name__ == "__main__":
    rows, best = run_member_experiments(
        member_name="Josue_Byiringiro_v3",
        experiments=josue_v3_experiments,
        policy="MlpPolicy",
        output_dir="runs_josue_v3",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 300,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )
    rewards = [r["mean_reward"] for r in rows]
    learned = [r for r in rewards if r > 0]
    print("\n=== JOSUE v3 MULTI-SEED ROBUSTNESS COMPLETE ===")
    print(f"Per-seed rewards: {[round(x, 2) for x in rewards]}")
    print(f"Learned in {len(learned)}/{len(rewards)} seeds")
    if learned:
        mean = sum(learned) / len(learned)
        print(f"Mean over successful seeds: {mean:.2f}")
    print("Summary written to runs_josue_v3/summary.csv")
