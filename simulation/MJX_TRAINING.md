# Experimental MJX movement training

Run the rail and paddle tests in MJX_BENCHMARK.md first. This optional trainer batches movement physics on MJX/JAX. It is not yet GPU league training or a dashboard backend. The normal CPU trainer is unchanged.

## Install on the Linux simulation PC

From the simulation folder, with the separate .venv-mjx already created:

```bash
uv pip install --python .venv-mjx/bin/python -r requirements-mjx-training.txt
.venv-mjx/bin/python train_mjx_movement.py --run-dir runs/mjx-movement-test --num-envs 16 --steps 8192
```

GPU detection is mandatory by default; there is no silent CPU fallback. Initial JIT compilation can take minutes. Start with 16 worlds, then compare 32 or 64. More worlds can exhaust memory or be slower.

After the short test, resume indefinitely:

```bash
.venv-mjx/bin/python train_mjx_movement.py --run-dir runs/mjx-movement-test --num-envs 16 --resume
```

Ctrl+C or a file named STOP inside the run folder requests stopping and saving. The request is handled after the current compiled step returns. Explicit --resume clears STOP. Checkpoints are written at roughly 8192 experience steps (rounded to complete PPO rollouts), with atomic replacement of latest.zip. A process lock prevents simultaneous writers. A hard kill can lose work since the last checkpoint. Do not use this experimental path for unattended week-long training until stability and recovery have been tested on your GPU.

## What runs where

- GPU: multiple MuJoCo worlds, 20 physics substeps per action, PD torque control, arm-travel accumulation.
- CPU: reward/observation calculation, independent episode resets, SB3 PPO policy inference and updates, checkpoint writing.
- No rendering or viewer is created.
- Whole-half target sampling, rewards, hold time, motor limits and closing penalties reuse the CPU movement logic.
- The temporary model uses the benchmark's coplanar sphere/frictionless-contact proxy. Movement puck collisions are disabled. Actual game collision validation is still required before league integration.

Host/device transfers and CPU forward-kinematics/rewards may erase the GPU speed gain. Measure completed training steps per second after compilation. This is a compatibility-first bridge, not a promise of faster training or an all-GPU PPO implementation.

## Checkpoints and limits

Runs have a separate experimental manifest and latest.zip; they are not registered with the existing dashboard/league run store. --resume only accepts this trainer's runs with matching geometry. It does not import existing CPU run folders. The ZIP uses standard SB3 PPO format, but automatic cross-backend migration is not implemented. Existing saved CPU models are not modified.

`--device cpu` is for developer tests only. It is not a GPU performance result. Default physical settings and default movement rewards are used; the dashboard reward sliders are not connected to this experimental trainer yet.
