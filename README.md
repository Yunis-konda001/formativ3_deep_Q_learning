
# Deep Q-Learning on Atari — ALE/Freeway-v5

Formative Assignment: Train and evaluate a **Deep Q-Network (DQN)** agent to play an
Atari game using [Stable Baselines3](https://stable-baselines3.readthedocs.io/) and
[Gymnasium](https://gymnasium.farama.org/). The project compares **MlpPolicy** and
**CnnPolicy** architectures and documents a hyperparameter tuning study.

Game video link: (*we have to add the video of our game)*

## The Environment — Freeway

**`ALE/Freeway-v5`** is a classic Atari 2600 game. The agent controls a chicken that
must cross a busy multi-lane highway from bottom to top without being hit by traffic.

- **Goal:** Cross the road as many times as possible within the time limit.
- **Reward:** `+1` each time the chicken reaches the top; `0` otherwise (sparse reward).
- **Actions:** `NOOP`, `UP`, `DOWN` (3 discrete actions).
- **Observations:** RGB image frames (used by `CnnPolicy`) **or** the 128-byte RAM
  state (used by `MlpPolicy`).

The higher the mean reward, the more successful crossings the agent completes.

## Team & Policy Assignments

Each member ran **10 hyperparameter experiments** (10 distinct combinations).

| Member           | Policy        | Observation | Experiments Directory |
| ---------------- | ------------- | ----------- | --------------------- |
| Kumi Yunis       | `CnnPolicy` | RGB frames  | `runs_kumi_dqn/`    |
| Nformi Modestine | `CnnPolicy` | RGB frames  | `runs_nformi_dqn/`  |
| Josue Byiringiro | `MlpPolicy` | RAM state   | `runs_josue_dqn/`   |

## Project Structure

```
formativ3_deep_Q_learning/
├── train.py                     # Trains DQN agents (all members' experiments)
├── play.py                      # Loads a trained model and renders gameplay
├── requirements.txt             # Python dependencies
├── dqn_model.zip                # Best overall model (auto-saved by train.py)
├── all_experiments_summary.csv  # Combined results across all members
├── best_overall_model/          # Best model + metadata
├── runs_kumi_dqn/               # Kumi's 10 runs, logs, summary.csv, best model
├── runs_nformi_dqn/             # Nformi's 10 runs, logs, summary.csv, best model
└── runs_josue_dqn/              # Josue's 10 runs, logs, summary.csv, best model
```

Each per-experiment run directory contains:
`episode_log.csv` (per-episode reward + length), `run_metadata.json`,
`best/` (EvalCallback best model), and `tb/` (TensorBoard logs).

## Setup

```bash
# (recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**Dependencies:** `gymnasium[atari,accept-rom-license]`, `stable-baselines3`,
`ale-py`, `shimmy`, `tensorboard`, `matplotlib`, `pandas`.

## Task 1 — Training (`train.py`)

Trains a DQN agent for every experiment, logs reward trends and episode lengths,
and saves the best-performing model as `dqn_model.zip`.

```bash
python train.py
```

**What it does**

- Builds a vectorized Atari environment with frame stacking (CNN) or RAM
  observations (MLP).
- Runs each member's 10 experiments (`100,000` timesteps each by default).
- Logs per-episode reward and length to `episode_log.csv` and TensorBoard.
- Evaluates each model with `evaluate_policy` and records `mean_reward`/`std_reward`.
- Saves each member's best model, then copies the single best overall model to
  `dqn_model.zip`.

**Key global settings** (top of `__main__` in `train.py`):

| Setting                             | Value   |
| ----------------------------------- | ------- |
| `TOTAL_TIMESTEPS`                 | 100,000 |
| `N_ENVS`                          | 4       |
| `FRAME_STACK`                     | 4       |
| `EVAL_EPISODES`                   | 3       |
| `buffer_size`                     | 100,000 |
| `learning_starts`                 | 10,000  |
| `target_update_interval`          | 10,000  |
| `train_freq` / `gradient_steps` | 4 / 1   |

View training curves with TensorBoard:

```bash
tensorboard --logdir runs_kumi_dqn/Kumi_Yunis_exp2/tb
```

## Task 2 — Playing (`play.py`)

Loads a trained model and plays Freeway using a **Greedy Q-policy**
(`deterministic=True`, i.e. always selects the highest Q-value action), rendering
the game with `env.render()`.

```bash
python play.py            # plays the best overall model (dqn_model.zip)
python play.py kumi       # plays Kumi's best model
python play.py nformi     # plays Nformi's best model
python play.py josue      # plays Josue's best model
```

It runs 5 episodes and prints the reward per episode plus the average.

## Task 3 — Hyperparameter Tuning Study

**Note on naming:** the assignment's `epsilon_decay` corresponds to
`exploration_fraction` in Stable Baselines3 — the fraction of total training
over which ε linearly decays from `epsilon_start` to `epsilon_end`. A larger
value means slower decay / more exploration.

### Kumi Yunis — `CnnPolicy` 

| #  | lr     | gamma | batch | eps_start | eps_end | eps_fraction | Mean Reward     | Std  | Noted Behavior                                           |
| -- | ------ | ----- | ----- | --------- | ------- | ------------ | --------------- | ---- | -------------------------------------------------------- |
| 1  | 1e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.10         | 1.33            | 0.94 | Baseline lr too low — barely learned in 100k steps.     |
| 2  | 5e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.10         | **22.00** | 1.41 | **Best.** Higher lr converged fast and reliably.   |
| 3  | 2.5e-4 | 0.99  | 32    | 1.0       | 0.10    | 0.10         | 20.67           | 1.70 | Medium lr — nearly as strong as 5e-4.                   |
| 4  | 1e-4   | 0.95  | 32    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Low gamma too short-sighted for a sparse reward.         |
| 5  | 1e-4   | 0.999 | 32    | 1.0       | 0.10    | 0.10         | 0.67            | 0.94 | Very high gamma + low lr → almost no learning.          |
| 6  | 1e-4   | 0.99  | 64    | 1.0       | 0.10    | 0.10         | 17.33           | 0.94 | Larger batch stabilized updates, strong improvement.     |
| 7  | 1e-4   | 0.99  | 32    | 1.0       | 0.05    | 0.10         | 15.00           | 0.82 | Lower eps_end (more greedy) helped clearly vs. baseline. |
| 8  | 1e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.20         | 9.00            | 2.83 | Longer exploration helped, but high variance.            |
| 9  | 1e-4   | 0.99  | 32    | 1.0       | 0.02    | 0.25         | 0.00            | 0.00 | Very greedy end + long explore at low lr — failed.      |
| 10 | 5e-5   | 0.995 | 64    | 1.0       | 0.05    | 0.15         | 20.33           | 1.25 | Conservative big-batch combo — surprisingly strong.     |

**Key insights (Kumi):** learning rate mattered most — `5e-4` (Exp 2) reached a mean
reward of **22.0**, with `2.5e-4` (Exp 3, 20.67) close behind. Larger batch size
(Exp 6) and the conservative big-batch combo (Exp 10, 20.33) both did well, while
overly low gamma (Exp 4) and configs pairing a low lr with aggressive exploration
schedules (Exps 5, 9) failed to learn at all within 100k steps.

### Josue Byiringiro — `MlpPolicy` (RAM)

| #  | lr     | gamma | batch | eps_start | eps_end | eps_fraction | Mean Reward     | Std  | Noted Behavior                                                  |
| -- | ------ | ----- | ----- | --------- | ------- | ------------ | --------------- | ---- | --------------------------------------------------------------- |
| 1  | 1e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Baseline lr too low — never learned within 100k steps.         |
| 2  | 5e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.10         | **18.33** | 1.25 | **Best.** Higher lr learned to cross reliably.            |
| 3  | 1e-3   | 0.99  | 32    | 1.0       | 0.10    | 0.10         | 14.00           | 1.41 | Aggressive lr also learned, but slightly less stable than 5e-4. |
| 4  | 1e-4   | 0.90  | 32    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Low lr + short-sighted gamma — no learning.                    |
| 5  | 1e-4   | 0.999 | 32    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Low lr dominates; higher gamma couldn't rescue it.              |
| 6  | 1e-4   | 0.99  | 128   | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Larger batch didn't help while lr stayed too low.               |
| 7  | 1e-4   | 0.99  | 32    | 1.0       | 0.01    | 0.10         | 0.00            | 0.00 | Greedier policy irrelevant with too-slow learning.              |
| 8  | 1e-4   | 0.99  | 32    | 1.0       | 0.10    | 0.30         | 0.00            | 0.00 | Longer exploration wasted — lr still the bottleneck.           |
| 9  | 2.5e-4 | 0.99  | 64    | 1.0       | 0.05    | 0.20         | 0.00            | 0.00 | Medium lr still below the threshold to learn in 100k steps.     |
| 10 | 5e-5   | 0.995 | 64    | 1.0       | 0.05    | 0.15         | 0.00            | 0.00 | Most conservative combo — lowest lr, no learning.              |

**Key insights (Josue):** learning rate is the single decisive hyperparameter for `MlpPolicy` on Freeway's RAM. Only the two highest learning rates learned to cross the road — `5e-4` (Exp 2) reached a mean reward of **18.33**, and `1e-3` (Exp 3) reached 14.0. Every configuration at `lr ≤ 2.5e-4` scored **0.0**: within the 100k-timestep budget (of which the first 10k are random warmup), such rates are too slow to propagate the sparse crossing reward before training ends. Because lr was the bottleneck, the gamma / batch-size / epsilon variations (Exps 4–10) had no chance to show their effect — a clear lesson that you must first get the agent learning at all before finer tuning matters. This mirrors the CNN result, where `1e-4` also scored 0.0 and `5e-4` was strongest.

### Nformi Modestine — `CnnPolicy`

| #  | lr      | gamma | batch | eps_start | eps_end | eps_fraction | Mean Reward     | Std  | Noted Behavior                                                   |
| -- | ------- | ----- | ----- | --------- | ------- | ------------ | --------------- | ---- | ---------------------------------------------------------------- |
| 1  | 1e-4    | 0.99  | 16    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Small batch + low lr — never learned to cross.                  |
| 2  | 1e-4    | 0.99  | 128   | 1.0       | 0.10    | 0.10         | 18.00           | 0.82 | Much larger batch rescued the low lr — strong and stable.       |
| 3  | 3e-4    | 0.99  | 32    | 1.0       | 0.10    | 0.10         | **24.67** | 1.89 | **Best overall.** Moderately higher lr was the sweet spot. |
| 4  | 1e-4    | 0.90  | 32    | 1.0       | 0.10    | 0.10         | 0.00            | 0.00 | Much lower gamma too short-sighted for sparse reward.            |
| 5  | 1e-4    | 0.99  | 32    | 0.5       | 0.10    | 0.10         | 13.00           | 2.45 | Lower initial epsilon (less early exploring) — moderate.        |
| 6  | 1e-4    | 0.99  | 32    | 1.0       | 0.10    | 0.05         | 18.00           | 2.45 | Very short exploration still learned well.                       |
| 7  | 1e-4    | 0.99  | 32    | 1.0       | 0.01    | 0.10         | 5.00            | 1.63 | Very low eps_end (nearly greedy) — weak, under-explored.        |
| 8  | 6.25e-5 | 0.99  | 32    | 1.0       | 0.10    | 0.10         | 6.33            | 4.50 | Lower lr than baseline — slow and highly unstable.              |
| 9  | 1e-4    | 0.99  | 64    | 1.0       | 0.05    | 0.20         | 15.33           | 2.05 | Bigger batch + longer explore + lower eps_end — decent.         |
| 10 | 2.5e-4  | 0.98  | 64    | 1.0       | 0.10    | 0.15         | 22.00           | 2.16 | Combined moderate tweak — second best, very strong.             |

**Key insights (Nformi):** the standout is that a **moderately higher learning rate
was decisive** — `3e-4` (Exp 3) produced the single best result of the whole project
(**24.67**), and the combined tweak with `2.5e-4` (Exp 10, 22.0) came second. Crucially,
Exp 1 (batch 16) scored 0.0 while Exp 2 (batch 128) reached 18.0 at the *same* low lr —
showing a **large batch can compensate for a small learning rate** by giving more stable
gradient estimates. Configs that starved exploration (Exp 7, eps_end 0.01) or lowered
the lr further (Exp 8) were weak/unstable, and low gamma (Exp 4) failed entirely.

## Results

| Member                       | Policy        | Best Mean Reward        | Best Config                                             |
| ---------------------------- | ------------- | ----------------------- | ------------------------------------------------------- |
| **Nformi Modestine**  | `CnnPolicy` | **24.67 ± 1.89** | lr=3e-4, gamma=0.99, batch=32, eps 1.0→0.10, frac=0.10 |
| Kumi Yunis                   | `CnnPolicy` | 22.00 ± 1.41           | lr=5e-4, gamma=0.99, batch=32, eps 1.0→0.10, frac=0.10 |
| Josue Byiringiro             | `MlpPolicy` | 18.33 ± 1.25           | lr=5e-4, gamma=0.99, batch=32, eps 1.0→0.10, frac=0.10 |

**Best overall model: Nformi Modestine's `CnnPolicy` (lr=3e-4), mean reward 24.67.**
It is saved to `dqn_model.zip` and `best_overall_model/`, with its configuration in
`best_overall_model/best_model_metadata.json`. Across all 30 experiments, the two CNN
agents outperformed the best MLP agent (24.67 / 22.00 vs 18.33), indicating that raw
pixel input suits Freeway better than the RAM state — though all three converged on the
same lesson: a **learning rate in the `3e-4`–`5e-4` range is essential**; anything at
`1e-4` or below mostly failed to learn within the 100k-step budget.

## Gameplay Video

We need to add a short recording of `play.py` running with
the agent crossing the road in Freeway.

## Policy Architecture Note (MLP vs CNN)

- **`CnnPolicy`** learns directly from raw pixel frames, giving it richer spatial
  information about traffic positions — well suited to Atari's visual input.
- **`MlpPolicy`** learns from the compact 128-byte RAM state, which is lower
  dimensional and faster per step, but requires the network to infer game structure
  from raw memory values rather than an interpretable image.

Comparing the two on Freeway is the core of this study; results above inform which
architecture generalizes best for this environment.
