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
