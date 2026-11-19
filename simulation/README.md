# Air hockey simulation and training

New: **random-target movement training and continuous two-arm self-play**. See [SKILLS_TRAINING.md](SKILLS_TRAINING.md) for setup, the separate viewer, checkpoint recovery and the Linux background service. These modes use `train_skills.py` and have no default round limit.

Launch the movement/self-play browser control panel with `uv run --extra train python training_ui.py`. It includes reward sliders, mode selection, start/resume, stop/save, model reuse and result downloads.

This is the CPU MuJoCo simulation for the direct-drive PlotClock arm. Start with TRAINING_SETUP.md for installation, live reward controls and saved-policy playback.

From the simulation repository folder:

```bash
uv run --extra train python check_training.py
uv run --extra train python train_policy.py --reward-ui --render --steps 300000
```

The second command opens a MuJoCo training window and a browser with reward sliders. Click **Apply at next batch** to change scoring. Output is saved in a new timestamped `runs/rally-...` folder printed in the terminal. No existing run is overwritten.

The full-rally task requires a new policy. Old 20-observation defender checkpoints cannot be resumed in this 25-observation task. Keep those files; they are not deleted.

For the original arm movement demonstration, see PLOTCLOCK_SETUP.md and run `uv run python plotclock_demo.py`. For the table/puck demonstration, run `uv run python table_demo.py`. Neither demo starts training.

This package does not modify the camera tracker or send real motor commands.


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


## Experimental MJX-JAX benchmark
See [MJX_BENCHMARK.md](MJX_BENCHMARK.md) for a separate CUDA 12 environment and headless physics benchmark. This does not enable GPU league training. The original cylinder/box collision model is unsupported by MJX-JAX; an explicitly requested planar approximation is provided for validation only.


## Whole-half movement targets
Movement mode now samples the whole robot half, including near the robot end, without inverse-kinematics or motor-range filtering. Paddle boundary clearance remains. The arm still obeys its simulated limits; unreached targets time out. Existing same-geometry movement runs can resume with this broader target distribution.
