# Validation results

Verified locally on CPU with MuJoCo 3.17.0, Gymnasium 1.2.3, Stable-Baselines3 2.7.1 and PyTorch 2.7.1+cpu.

- Gymnasium/SB3 environment API check passed.
- Seeded resets are repeatable.
- 50 generated initial shots intersect the defense strip within the action limits.
- 600 randomized policy actions preserved finite states, closed-link constraints and motor torque caps.
- A puck crossing the robot goal terminated as goal_conceded.
- Side-rail impact reversed the puck velocity.
- A real paddle impact registered and produced a completed return.
- Episode timeout used truncation rather than terminal success/failure.
- A 2,048-step CPU PPO run trained and saved latest.zip.
- Resuming that checkpoint for 10,240 additional steps completed and saved new weights.
- Periodic checkpoint and best-model evaluation paths executed.
- The saved model reloaded for headless held-out playback.

The short test run is a software validation only. Its periodic deterministic evaluation had zero successful returns in ten episodes. This does not establish a useful learned defender; more training and physically measured parameters are needed. No untrained/test checkpoint is included as a finished policy.

The graphical scene was rendered offscreen and inspected. Interactive viewer behavior on your CachyOS desktop has not been tested locally.

Live-view update: callback timing and window-close behavior passed automated tests. An actual PPO run with a mocked viewer stopped and saved latest.zip on close. The native window on CachyOS still requires user-side verification.
