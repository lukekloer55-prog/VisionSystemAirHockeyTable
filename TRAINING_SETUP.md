# Train the PlotClock air-hockey defender

This package adds a real reinforcement-learning loop to the measured PlotClock mechanism. It keeps the four 23-inch links, 18-inch servo spacing, and a servo-pivot line 4 inches behind the table. It is simulation only, with no connection to real motor drivers or the Jetson camera.

## Install and start

On the separate CachyOS simulation computer, extract the package and open a terminal in the folder containing pyproject.toml. If updating GitHub, upload the extracted files together into the existing simulation folder, then git pull on the simulation computer. Do not upload the ZIP as a substitute for the files. This package has not been pushed to GitHub.

Run the checks:

```bash
uv run --extra train python check_training.py
```

Start training:

```bash
uv run --extra train python train_policy.py --steps 300000 --run-dir runs/defender-v1
```

This command ACTUALLY TRAINS. It opens no window; progress and episode statistics appear in the terminal. It uses a small PPO neural network on the CPU. The pyproject config explicitly selects CPU PyTorch wheels, avoiding the GTX 1070 Ti CUDA compatibility problem. The first run downloads dependencies. Use the Python 3.12 environment selected by uv.

A run directory must be new. If defender-v1 already exists, use a new name or omit --run-dir for an automatically named folder. PPO collects full 1024-step batches, so it can slightly exceed the requested step count.

Ctrl+C saves current weights as latest.zip. Periodic checkpoints are saved every 10,000 new steps. Evaluation uses ten separate episodes every 10,000 steps; the checkpoint with best mean evaluation reward is saved as best/best_model.zip. Best reward is not necessarily best success rate; compare held-out playback outcomes too.

## Watch or evaluate the saved policy

After training completes or is interrupted:

```bash
uv run --extra train python play_policy.py --model runs/defender-v1/latest.zip
```

Watch the best periodically evaluated checkpoint (only exists after an evaluation):

```bash
uv run --extra train python play_policy.py --model runs/defender-v1/best/best_model.zip
```

Evaluate 100 new episodes without a window:

```bash
uv run --extra train python play_policy.py --model runs/defender-v1/latest.zip --headless --episodes 100
```

Continue an existing policy for another 300,000 steps, saving a new run:

```bash
uv run --extra train python train_policy.py --resume runs/defender-v1/latest.zip --steps 300000 --run-dir runs/defender-v2
```

Resume restores policy and optimizer state; it starts fresh episodes, rather than resuming halfway through a physical episode. Keep settings.json and plotclock_geometry.json beside each checkpoint. Do not load model files from untrusted sources.

## What the learner does

- Observations: 20 values: puck XY position and XY velocity; four hinge angles and four angular velocities; paddle XY position and XY velocity; two commanded motor angles; remaining episode fraction; and whether the puck has already contacted the paddle.
- Units: metres, seconds, radians, kg, and N*m. Coordinates use the table center as origin, X right, Y toward the opponent. These are exact simulated states, not camera images or measured noisy estimates.
- Actions: two numbers in [-1,1], mapped to motor targets in [-0.30,+0.30] radians relative to the CAD assembly pose. Targets slew at up to 0.75 rad/s. A simulated PD loop produces torque limited to +/-2.4 N*m at each direct-drive joint.
- Timing: policy decisions at 50 Hz, physics at 1000 Hz. Target slew is not a hard limit on actual joint speed.
- Task: block an incoming puck and return it past Y=+0.15 m with positive Y velocity. This is an initial return task, not opponent-goal scoring or a full match.
- Reset: the arm starts with the pin centered in X. Pucks start at Y=+0.15 m with X in [0.04,0.28] m, at 0.60-0.90 m/s, aimed at the robot goal. This narrow, right-biased training strip stays reachable under the provisional joint range. It does not cover the whole half-table.
- Rewards: +1 for the first real paddle contact, +5 for a completed return, -10 for conceding a goal, -5 for other terminal failures, -1 for timeout, a small motor-effort cost, and potential-based alignment shaping.
- Termination: return, conceded goal, paddle outside its allowed half-table area, escaped puck or invalid physics. A six-second timeout truncates the episode so PPO can bootstrap correctly.

Do not judge training by contact alone: a successful return must come after a real paddle-puck impact. Episode outcomes and return rate are printed by playback. The training directory also contains progress.csv and train/eval monitor logs.

## Placeholder physical values

`training_settings.json` contains editable defaults. The normal training command uses the same defaults; to load your edited file explicitly:

```bash
uv run --extra train python train_policy.py --config training_settings.json --run-dir runs/measured-v1
```

Current placeholders are a 100 mm diameter / 150 g paddle, 400 g per arm link, and 20 g puck. Puck diameter is the supplied 2.823 inches. Arm mounting height, rotor inertia, joint friction, actual travel limits and contact damping remain provisional. Longer links were not substituted.

Paddle and puck collisions, side rails, and centered 10-inch goal openings are modeled. Arm-link self-collisions and link/rail collisions are disabled, as in the mechanism demo. The paddle attaches rigidly below the pin; its mount is not a detailed CAD part. Contact parameters are not measured restitution. These assumptions must be resolved before treating the simulation as a model of real hardware.

There is no trained real-world controller, domain randomization, vision latency/noise model, live camera input, or hardware output in this version. Replacing simulated puck state with vision measurements and validating the physical mechanism are later steps.

## Files

- airhockey_env.py: Gymnasium environment, physics, observations, rewards and resets.
- train_policy.py: CPU PPO training, checkpoints, evaluation and resume.
- play_policy.py: saved-policy viewer and headless evaluation.
- check_training.py: environment API and meaningful physics tests.
- training_settings.json: optional editable physical/task settings.
- plotclock_demo.py: original scripted arm movement demo, still available.
- plotclock_geometry.json: measured dimensions from your STEP assembly.
- table_demo.py: original independent puck/table demo, still available.

Implementation references:
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#equality-connect
