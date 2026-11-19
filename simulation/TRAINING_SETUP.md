# Train the PlotClock rally player

## Install and start

On the separate CachyOS simulation computer, use the existing simulation repository at `~/robotics/airhockey-sim`. Upload the extracted update files into the same simulation folder on GitHub, commit there, then run:

```bash
cd ~/robotics/airhockey-sim
git pull
uv run --extra train python check_training.py
uv run --extra train python train_policy.py --reward-ui --render --steps 300000
```

The package has not been pushed to GitHub for you. Upload the extracted files, not just the ZIP. Keep any existing vision files in the repository.

The training command opens the MuJoCo simulation and a local browser reward panel. If the browser does not open, copy the printed `http://127.0.0.1:PORT` address into a browser on that same computer. The port changes each run.

Training uses CPU PyTorch, configured in pyproject.toml, avoiding the GTX 1070 Ti CUDA incompatibility. uv uses Python 3.12 and installs dependencies on first launch. Do not run this inside the old Isaac Lab environment.

Omitting --run-dir creates a new timestamped runs/rally-... folder. The terminal prints its path. An explicitly named --run-dir must not already exist. PPO collects full 1,024-step batches and can exceed --steps slightly.

## Reward sliders

Each slider also has a numeric entry. Penalties are positive magnitudes that are subtracted from reward. Edit one or several values, then click **Apply at next batch**. Editing alone does not apply changes. The panel distinguishes applied and queued values.

Updates apply before the next PPO rollout, after the previous batch has been used for learning. At real-time rendering speed, 1,024 decisions take about 20.48 seconds, plus neural-network updates and evaluation. A rally can span batches and therefore scoring revisions; each learning batch uses one revision. Avoid changing weights constantly while comparing learning progress.

The panel displays training steps, scored/conceded goals, finished rallies and the last rally's reward breakdown. **Set sliders to defaults** fills the controls but still requires Apply. **Download reward JSON** exports the currently displayed slider values, which may not yet be applied.

| Control | Default | Meaning |
| --- | ---: | --- |
| Opponent goal | 30 | Entire puck crosses the opponent goal plane inside the opening |
| Goal conceded | 30 penalty | Entire puck enters the robot goal |
| Center crossing | 3 | Entire puck crosses toward opponent after a qualifying paddle impact |
| Useful new hit | 0.5 | A distinct hit increases forward puck velocity |
| Defensive save | 1 | A useful hit turns an incoming puck forward |
| Bank goal | 3 extra | Goal after a side-wall bounce since the latest paddle contact |
| Pursuit/recovery shaping | 2 | Potential-based shaping toward a reachable puck target or defensive home position |
| Behind paddle | 2 per second penalty | Entire puck is behind the paddle |
| Timeout | 5 penalty | Rally reaches 15 seconds without a terminal event |
| Invalid/outside state | 40 penalty | Illegal paddle position, puck escape or invalid physics |
| Time | 0.05 per second penalty | Encourages resolving the rally |
| Effort | 0.05 coefficient | Time-scaled sum of squared normalized motor torques |
| Minor event budget | 8 | Shared maximum for hit, save and crossing bonuses in one rally |

A hit or center crossing does not end the episode. The robot can chase and hit again after a rebound. Distinct hits have a 0.25-second cooldown; another hit rearms crossing credit. Event bonuses share a budget to discourage tapping forever. Changing the budget preserves the amount already paid in the current rally. Changing pursuit strength resets its reward baseline instead of giving a fake one-step bonus.

These are initial weights, not proven optimal values. Keep goals more valuable than minor events. Increasing penalties too far can teach the policy to avoid attempting useful actions. Compare goal rates under fixed evaluation settings, not just reward totals across changing weights.

## Viewing, stopping and saved files

--render shows the actual training rollout. Add --render-speed 4 to view four times faster. Omit --render to train headlessly while still using --reward-ui. The simulation may pause during learning updates and separate evaluation episodes.

Closing the browser does not stop training. Closing the MuJoCo window or pressing Ctrl+C stops training and saves latest.zip. The local panel shuts down when training ends. Unapplied queued changes are not saved as active settings.

Each run includes:

- latest.zip and latest.settings.json: final policy and its active settings.
- settings.json, task_schema.json and plotclock_geometry.json: run configuration.
- reward_settings.json and reward_changes.jsonl after applying changes: reusable overrides and change history with training step numbers.
- checkpoints/: periodic policies every 10,000 new steps, each with a matching .settings.json.
- best/: best mean evaluation-reward policy and its settings, from evaluations of ten episodes every 10,000 steps.
- CSV training, episode and evaluation logs.

Changing weights resets the best-reward comparison for future evaluations. The existing best model remains until a new evaluation replaces it; its settings sidecar preserves its scoring revision. Reward totals before/after changes are not directly comparable.

## Playback and resume

Replace the example path with the run folder printed by your training command:

```bash
uv run --extra train python play_policy.py --model runs/rally-YOUR-RUN/latest.zip
uv run --extra train python play_policy.py --model runs/rally-YOUR-RUN/latest.zip --headless --episodes 100
uv run --extra train python train_policy.py --resume runs/rally-YOUR-RUN/latest.zip --reward-ui --render --steps 300000
```

Resume restores policy and optimizer state, starts fresh episodes and saves in a new folder. Keep the settings sidecar, task schema and geometry with each policy. Old 20-observation defender policies are intentionally rejected: this full-rally task uses 25 observations including contact and bonus state. Start a new run for this update.

For fixed reward overrides without the UI:

```bash
uv run --extra train python train_policy.py --reward-config reward_settings.json --steps 300000
```

Use --config training_settings.json to supply physical parameters for a NEW run. Resume uses the saved physical settings. Reward overrides and sliders do not change link lengths, motor limits or physics.

## Physical model and limits

The model keeps the measured 23-inch proximal and 28.04874-inch distal pivot spacings, 18-inch servo spacing, and servo-pivot line 9.5 inches behind the inside near table edge. The two direct-drive motors have +/-2.4 N.m torque limits. The initial controller has conservative angle and target-speed limits. This mechanism does not cover the entire half-table.

The full table is 41.3125 inches wide and 77 inches long, with centered 10-inch goals and a 2.823-inch puck. Coordinates use metres and seconds, with positive y toward the opponent. The agent sees exact simulator state and controls two motor target angles at 50 Hz; physics steps at 1,000 Hz.

Unmeasured defaults remain estimates: paddle diameter 100 mm, paddle mass 150 g, each link 400 g and puck 20 g. Link geometry uses simplified box shapes based on CAD pivots. Rotor inertia, friction, restitution and servo response need physical identification. Arm-link collision geometry is disabled; paddle/puck/rail collisions are enabled.

This covers goals, returns, repeated contact, saves, bank goals, pursuit, recovery, timeout and invalid states. It is not every possible real-game scenario: there is no opposing paddle, self-play, camera noise/latency or real motor interface yet. A useful trained policy and safe hardware behavior have not been established by the short software checks.
