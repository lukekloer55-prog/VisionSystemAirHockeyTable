# Parallel league training

Choose **League - parallel candidates + older champions** in `training_ui.py`. Start with **2 candidates** and **2 total simulations**, or 4 total simulations if the CPU and RAM have room. The total simulation budget must be divisible by the candidate count. For example, 4 candidates and 8 simulations means two simulation environments per independently learning policy. Candidate learners and the coordinator also use CPU/memory; the budget counts environments, not every OS process.

Use a new run name. **Start from model** can select a movement, self-play, or league champion with the same geometry/settings. A new league begins with an archive containing that starting model. An existing league resumes its own archived opponents and optimizer states. Candidate count is fixed for a saved run; total simulation workers can change on resume. Earlier arm geometry remains incompatible.

## What happens each round

1. Each candidate starts from the retained champion, including its optimizer state, with a different exploration seed.
2. Candidates train concurrently in separate processes. Their environments choose a frozen opponent from the current champion and saved historical champions at each episode reset. They do not share gradient updates.
3. Freeze all candidates. Evaluate the candidates and champion against one another, then evaluate each against every historical opponent. Matches use matching serve seeds and swapped sides. Evaluation is sequential and does not train policies.
4. Count rally wins as one point and draws as half a point. A candidate is eligible only if it beats the champion head-to-head, has more total points, and has at least the champion's aggregate points against the historical pool. Select the eligible candidate with the most points (then historical points, then stable candidate order). Otherwise retain the champion, including on all-draw rounds. This finite evaluation is a practical rule, not proof of general improvement.
5. Preserve the initial historical model and up to four recent past champions. Discard losing candidate models after the round checkpoint; start the next candidates from the winner. The archive is bounded to limit memory, evaluation time and disk use.

The panel's results/details show phase, candidate steps, completed evaluation matches, standings and retained model. Download results includes the committed candidate models (if mid-round), champion, historical models, state and event logs. Renaming continues to work.

## Windows and headless training

Check **Show every simulation in its own live window** for the actual worker simulations. Windows close/reopen at learning checkpoint boundaries as worker processes restart. A separate window displays evaluation games. Closing a training worker window requests a coordinated stop and save. For maximum speed leave this unchecked: no simulation viewers or rendered frames are created. **Watch saved model** is an independent optional replay.

Worker startup and loading the opponent pool cost time, so tiny training batches can be slower than ordinary training. The default league chunk is 8192 total steps per candidate. Larger worker counts also enlarge PPO rollout batches (256 steps per environment) and can round requested step counts upward. More workers do not guarantee faster learning or stronger policies. The league throughput metric includes worker startup and learning, but excludes evaluation and checkpoint writing.

## Commands

From the repository's `simulation` directory:

```bash
uv run --extra train python training_ui.py
```

Or start a new league directly:

```bash
uv run --extra train python train_league.py --run-dir runs/league-v1 --candidates 2 --num-envs 4
```

Add `--initial-run runs/YOUR-MOVEMENT-RUN` to start from its champion. With custom physical settings, also pass the matching `--config`; the panel handles this automatically. Add `--render` only for live viewing. Default `--rounds 0` continues until stopped; `--rounds 1` is a short round-count limit, not a time limit.

Stop with the panel or Ctrl+C. Resume:

```bash
uv run --extra train python train_league.py --run-dir runs/league-v1 --resume
```

A `STOP` file deliberately prevents training on restart. The panel removes it when you explicitly Resume; for a CLI resume after stopping through the panel, remove only that run's STOP file first.

## Checkpoints and unattended use

Successful learning chunks commit all candidates and their progress together. A cooperative stop saves partial candidate progress after worker shutdown; an in-progress PPO rollout may not yet have applied a gradient update. A failed worker aborts the batch, keeping the previous committed checkpoint available. Completed evaluation matches are checkpointed individually; an interrupted match restarts with the same seed. Reward changes queue until the next league round so candidates share the same scoring rules. A Save request commits at the next learning-chunk or evaluation-match boundary.

After a successful local test, stop the panel-owned trainer and use the existing Linux service helper:

```bash
uv run --extra train python training_service.py start --mode league --run-dir runs/league-v1
```

The service restores the saved worker budget and runs without rendering. If a STOP file remains, remove it intentionally first. Enable user lingering with `loginctl enable-linger "$USER"`, keep the PC awake, and check service logs before leaving it unattended. Stop with:

```bash
uv run --extra train python training_service.py stop --mode league
```

Test helpers: `check_league.py` covers scheduling, promotion conditions and bounded history. `check_training_ui.py` covers league command dispatch, worker budget validation and reward-change boundaries. Short integration tests do not establish week-long reliability.
