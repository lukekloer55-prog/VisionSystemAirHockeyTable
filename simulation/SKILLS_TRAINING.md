# Movement and continuous self-play

These are new simulation-only modes. They use simulator coordinates, not the live camera, and send no commands to the physical servos. The existing `train_policy.py` rally trainer and its sliders remain available. Old 25-observation rally checkpoints cannot be loaded into these new 24-observation policies.

## Install/update

Upload the extracted package contents into the repository's `simulation/` folder. On the Linux simulation PC (not the Jetson):

```bash
cd ~/robotics/airhockey-sim
 git pull
cd simulation
uv sync --extra train
uv run --extra train python check_skills.py
```

If your checkout has a different name, change only the first `cd` path.

## Browser control panel

After updating, stop/restart any old trainer so it loads the new reward-control code. On the simulation PC desktop:

```bash
uv run --extra train python training_ui.py
```

The browser opens automatically. Select Movement or Self-play, enter a new run name, adjust sliders, and click Start training. To continue a saved model, select its run and click Resume training. Stop & save requests a graceful stop; wait for Stopped before switching modes. Use model in new run selects the saved champion as the initial model; it does not overwrite the source run. Training has no default round limit.

Movement sliders include reaching the target, progress, elapsed time, actual joint travel, narrow-link angle/penalty and failure. Self-play additionally exposes goals/concessions, useful contact, pursuit and contact-bonus budget. Click Apply reward changes. Movement changes apply between learning batches; self-play changes wait until the next competition round, before either candidate trains. Queued and applied values are shown separately and applied changes are recorded in checkpoint metadata and event logs. For a stopped run, queued changes take effect on resume. Timing, target tolerance and physical geometry remain in settings files.

Save checkpoint queues a save at the next training boundary; it does not interrupt a match. Download results exports the last committed snapshot (model weights, optimizer states, settings and available event logs). Extracting the downloaded folder inside `runs/` restores it as a selectable run. Watch saved model opens a separate MuJoCo replay window on the simulation PC desktop. Recent metrics refresh after a checkpoint, not every physics step.

Closing the browser tab leaves the trainer running. Closing the dashboard process with Ctrl+C stops and saves trainers it launched. The dashboard is not itself a crash/reboot supervisor: use the Linux service below for the unattended week. It can also edit/stop a running service when both use the same runs folder. In that case closing the dashboard leaves the service alone. Do not launch a second trainer for the same run.

Use `--runs-dir /absolute/path/to/runs` to select another run collection. The dashboard only lists compatible movement/self-play runs directly inside that folder; old rally policies are not compatible. It listens on localhost only. For SSH, launch with `--no-browser --port 8766` on the simulation PC and forward that port with SSH, then open `http://127.0.0.1:8766` locally.

## Train movement first

```bash
uv run --extra train python train_skills.py --mode movement --run-dir runs/movement-v1 --render
```

This has **no round limit**. Ctrl+C or closing the training window saves and stops. To continue the same run:

```bash
uv run --extra train python train_skills.py --mode movement --run-dir runs/movement-v1 --resume
```

A pink sphere marks the target. Each episode samples a random reachable XY point on the robot's half. The paddle must get within 15 mm, slow below 0.12 m/s, and hold there for 0.12 seconds. Success ends that episode; the next episode resets the arm to its starting pose and samples another target. Episodes last at most five simulated seconds.

Rewards:
- +15 for reaching and settling at the target.
- +12 times the reduction in distance, in metres, per step (negative when moving away).
- -0.5 per simulated second, so faster successful moves score better.
- -0.3 per radian of **actual travel summed across all four hinges at physics rate**, including backtracking. This favors efficient arm movement, not simply small control commands.
- A growing penalty when the angle between the two distal links meeting at the paddle falls below 35 degrees. The penalty is `-4 * dt * max(0, 1-angle/35)^2`. Collapse below 5 degrees fails the episode.
- -20 for arm/physics failure. Torque limits and target slew limits still apply.

Target sampling rejects unreachable positions and narrowing-angle singularities. **This does not give full half-table coverage.** The existing provisional +/-0.30 rad command limits and CAD elbow branch constrain the workspace. Its sampled bounding box is printed at startup; the actual reachable region is not rectangular. Joint ranges, masses and rotor inertia remain provisional. Do not widen them solely to increase reward.

Use `--curriculum skills_settings.json` on a NEW run to adjust reward weights, timing, target tolerance or the angle threshold. Resume restores the saved settings. The new `training_ui.py` dashboard provides live reward controls for these modes. The original `reward_ui.py` panel still controls the older rally trainer.

## Two-model competition

Stop movement training first. Then initialize both competing policies from its champion:

```bash
uv run --extra train python train_skills.py --mode selfplay --run-dir runs/selfplay-v1 --initial-run runs/movement-v1
```

Without `--initial-run`, self-play starts from an untrained policy. Movement pretraining is recommended, but does not guarantee game skill. `--initial-model` also accepts a champion ZIP with its adjacent `state.json`.

Each round:
1. Copy the current champion into candidate A. Train A for 8,192 steps against a frozen champion opponent.
2. Independently copy the same champion into candidate B. Train B for 8,192 steps with a different exploration seed against that same frozen opponent.
3. Play an eight-rally match between A and B using deterministic actions. Paired serves reuse the same random seed and swap the players' ends, reducing side/serve bias. Each rally ends on a goal, forfeit, escape or 12-second timeout.
4. A goal or arm forfeit scores one point. Escapes, simultaneous failures and timeouts score neither player. The higher-scoring candidate becomes champion. A tied match retains the previous champion.
5. Both copies start the next round from the winner's parameters and optimizer state, then explore independently again.

The two arms share ONE MuJoCo scene and one physical puck. The far linkage is a 180-degree rotation of the near linkage. Goal openings are 10 inches wide. This is real simulated paddle contact, not two unrelated rollouts compared by reward.

Training rewards include goals/concessions, limited useful-hit bonuses, pursuit, time, joint travel and narrow-angle penalties. Match winners are selected by match points, not shaped training reward. Identical weights alone do not learn: PPO updates and independent exploration happen before every match. A winning candidate can still be weaker overall than a previous champion; this is winner-retains self-play, not a guarantee of monotonic improvement or a historical-opponent league. Frequent 0-0 matches mean training is not yet producing scoring play; inspect before leaving for a week.

Optional new-run flags: `--steps-per-round 16384`, `--match-rallies 16` (must be even), `--checkpoint-steps 2048`. PPO uses 256-step rollouts, so step counts can round up to a full rollout. Resume restores saved schedule values. `--rounds 1` runs one additional movement block or competition round as a smoke test; default `--rounds 0` is unlimited.

## Watch without slowing the trainer

On the Linux desktop, in another terminal:

```bash
uv run --extra train python watch_skills.py --run-dir runs/movement-v1
```

For games, use `runs/selfplay-v1`. The viewer loads the latest saved candidate/champion at each episode boundary. It is a separate replay, not a frame-by-frame view of the active training rollout. Closing it leaves the trainer running. Use the trainer's `--render` option when you want the actual rollout; that paces training and reduces throughput.

## Checkpoints, interruption and recovery

A snapshot is committed after each learning chunk (default 2,048 environment steps) and at match transitions. `CURRENT.json` points to the committed generation containing model ZIPs, optimizer state, task settings, arm geometry, round and phase. Failed/incomplete writes do not replace the pointer. Only the latest three committed generations are retained. Event logs rotate at roughly 10 MB (current + previous). Do not edit/delete snapshots while training.

Ctrl+C, SIGTERM, closing the actual training viewer, or creating `STOP` in the run directory requests a graceful save. Stop from another terminal:

```bash
touch runs/selfplay-v1/STOP
```

Resume after removing that explicit stop request:

```bash
rm runs/selfplay-v1/STOP
uv run --extra train python train_skills.py --mode selfplay --run-dir runs/selfplay-v1 --resume
```

Do not run the `rm` if you stopped with Ctrl+C and no STOP file exists. Resume retains saved candidate learning and round progress, but resets the physical episode. An unfinished PPO rollout may be discarded. A sudden power loss can lose work since the last committed checkpoint; it does not provide exact trajectory/RNG replay. An interrupted match is replayed with the same paired serve seeds. Only one trainer can lock a given run directory.

## A week-long background run on CachyOS

First run at least one complete round interactively and inspect movement and match results. For a short configuration/physics smoke test:

```bash
uv run --extra train python train_skills.py --mode movement --run-dir runs/movement-v1 --rounds 1
```

Use `--resume --rounds 1` if that run already exists. This is a smoke test, not proof of a strong trained policy.

Then install/start a user service that resumes the tested run:

```bash
uv run --extra train python training_service.py start --mode movement --run-dir runs/movement-v1
loginctl enable-linger "$USER"
journalctl --user -u airhockey-movement.service -f
```

For competition, create/test `runs/selfplay-v1` with `--rounds 1` first, then use:

```bash
uv run --extra train python training_service.py start --mode selfplay --run-dir runs/selfplay-v1
```

Prefer one training service at a time on this PC. The service runs headlessly, resumes after process failure, and is enabled at boot; lingering keeps the user manager available after logout/at boot. `loginctl` may require administrator authorization on your installation. Keep the PC plugged in and disable automatic sleep in desktop power settings; screen blanking is fine. The service does not change sleep settings or install operating-system updates. Do not change code, geometry or dependencies while a run is active.

Stop and disable automatic startup:

```bash
uv run --extra train python training_service.py stop --mode selfplay
```

Use `--mode movement` for movement training. Ctrl+C in `journalctl -f` only exits the log viewer. The service helper uses a maximum of three rapid restart attempts within five minutes, avoiding an endless crash loop. A STOP file still causes normal exit and prevents training on reboot until removed. Check service status after starting it:

```bash
uv run --extra train python training_service.py status --mode selfplay
```

Service reference: [systemd service restart and stop behavior](https://www.freedesktop.org/software/systemd/man/257/systemd.service.html). This package was locally tested on Windows with MuJoCo/PPO; the Linux service must be checked on your CachyOS PC before an unattended week. No week-long reliability or learning-quality run has been completed here.


### Rename a saved run
Select a saved run in the training panel, enter its **Run display name** under Saved results, and click **Rename run**. The display name is saved in `run_details.json` and included in exported results. The folder name, checkpoints, model weights and background service paths stay unchanged.

To continue movement training in self-play, use **Use model in new run**, choose self-play and a new folder name, and start. The new run records the source training stage in its checkpoint history. Rename the new run to a descriptive name such as `Robot v1 - movement + self-play`. Existing runs do not gain retroactive history.


### Parallel CPU simulations and live demonstration
Set **Parallel CPU simulations** in the panel to 2 or 4 to start. Each spawned process owns an independent MuJoCo simulation, with distinct random seeds; their experience trains one PPO policy. Self-play candidates still train against the same frozen champion and are compared using paired, side-swapped matches. Match evaluation remains sequential.

Check **Show every simulation in its own live window** to open one native window per worker on the training desktop. Arrange these windows side by side. They show the actual training workers, not checkpoint replays. Workers pause during PPO updates and winner matches; a separate match window may open during evaluation. Closing any worker window requests a save and stops all workers.

For speed, leave the checkbox unchecked: no viewer is created and no frames are rendered. Stop/save and resume to switch rendering or worker count. The background service resumes the saved worker count and always omits rendering. `Watch saved model` is a separate optional replay.

CLI demonstration:
```bash
uv run --extra train python train_skills.py --mode movement --run-dir runs/new-arms-movement --num-envs 4 --render
```
After stopping, resume without rendering:
```bash
uv run --extra train python train_skills.py --mode movement --run-dir runs/new-arms-movement --resume --num-envs 4
```
Use a new run with the updated geometry; older-arm checkpoints are intentionally incompatible. The default without a worker setting is one worker (or the saved count on resume). `--rounds 0`, the default, runs until stopped. Counts are total experience across workers, not steps per worker. PPO collects full 256-step rollouts per worker, so checkpoint and round step limits can be rounded upward to that batch size. Stop requests can save partial progress earlier. Increasing workers also increases rollout batch size and changes optimization behavior; speed and learning improvement are not guaranteed to scale together.

Each worker uses one PyTorch CPU thread; the learner defaults to one thread. Compare the logged **Training throughput: ... total steps/s** for 1, 2 and 4 workers with rendering off. Startup, checkpoint writes and winner matches add overhead outside that measured learning-block rate. Leave CPU/memory headroom and avoid exceeding available logical CPUs. No GPU is required for these workers.


## Parallel league mode
The panel now offers **League - parallel candidates + older champions**. Separate candidates learn concurrently, then undergo frozen head-to-head and historical-opponent evaluation. Start with two candidates and two or four total simulations. Leave live windows unchecked for headless training. See [LEAGUE_TRAINING.md](LEAGUE_TRAINING.md) for promotion, checkpoint, archive and background-service behavior.
