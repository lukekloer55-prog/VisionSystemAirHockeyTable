# Air hockey simulation and training

Start with **TRAINING_SETUP.md** for the new CPU reinforcement-learning setup, commands, assumptions, and saved-policy playback.

```bash
uv run --extra train python check_training.py
uv run --extra train python train_policy.py --steps 300000 --run-dir runs/defender-v1
uv run --extra train python play_policy.py --model runs/defender-v1/latest.zip
```

For the original arm motion demonstration, see PLOTCLOCK_SETUP.md and run:

```bash
uv run python plotclock_demo.py
```

For the original table/puck demo:

```bash
uv run python table_demo.py
```

Training never starts automatically from either demonstration. It starts only when you run train_policy.py.
