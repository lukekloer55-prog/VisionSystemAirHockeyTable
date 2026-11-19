# MJX-JAX compatibility experiment for the GTX 1070 Ti

This is a headless physics benchmark, not a GPU training mode. It does not load or update policies, change training files, or replace the CPU league trainer. MJX-JAX is explicitly selected; Warp is not used. JAX must actually detect and execute a kernel on the requested GPU; there is no silent CPU fallback.

## Install separately on the Linux simulation PC

From `simulation`:

```bash
uv venv --python 3.12 .venv-mjx
uv pip install --python .venv-mjx/bin/python -r requirements-mjx-cuda12.txt
```

Use CUDA **12** for this experiment with the GTX 1070 Ti. Do not switch these requirements to CUDA 13. JAX documents support for SM 5.2+ with CUDA 12 and SM 7.5+ with CUDA 13. A compatible NVIDIA driver is required. The pip extras supply user-space CUDA libraries; the driver's `nvidia-smi` CUDA version is not the installed wheel runtime version.

## Important collision limitation

Our current model uses cylinder puck/paddle shapes and box rails. MJX-JAX does not support cylinder-box collision pairs. Running without `--planar-proxy` attempts the original model and should report the incompatibility rather than remove contacts silently.

The explicit `--planar-proxy` option changes only the temporary benchmark model: the puck and both paddles use spheres with their original radii, and their collision centers are aligned at z=6mm. Contacts also use `condim=1` (explicitly frictionless normal contact), rather than three-dimensional contacts with zero friction coefficients. This avoided a large MJX/CPU discrepancy in the first test. Original compiled body mass/inertia, planar joints, rails, goals and linkage constraints are preserved. The altered contact formulation can still change bounce dynamics. The table surface already has collisions disabled. This approximates horizontal circular contacts under planar motion; it is not full cylinder physics and is not yet approved as a replacement training model.

## Run a small test first

```bash
.venv-mjx/bin/python benchmark_mjx.py --planar-proxy --batch-sizes 1 16 --output mjx-rail.json
```

Then test paddle contact:

```bash
.venv-mjx/bin/python benchmark_mjx.py --planar-proxy --scenario paddle --batch-sizes 16 --output mjx-paddle.json
```

Only after these succeed, try larger batches:

```bash
.venv-mjx/bin/python benchmark_mjx.py --planar-proxy --batch-sizes 64 256 --output mjx-large.json
```

These are test sizes, not guaranteed capacities. First compilation may take minutes and is reported separately. An out-of-memory error means try fewer worlds. No viewers or rendering are used. To test code on a machine without CUDA, explicitly pass `--device cpu`; those results are **not GPU measurements**.

## Interpret the report

- `device_kind`: verify it identifies the intended NVIDIA GPU.
- `physics_steps_per_second`: total 1ms physics steps across all worlds, with synchronized timing after compilation.
- `control_steps_equivalent_per_second`: physics throughput divided by 20, since training uses 20 substeps per action. This is only a conversion, **not measured PPO throughput**.
- `peak_closure_gap_m`: linkage gap over the rollout; the benchmark fails above 2mm.
- `finite_all_steps`: rejects non-finite positions or velocities.
- CPU/MJX position and velocity differences: compare the same fixed-target trajectory in world zero. Coordinates include both metres and radians, so the maximum coordinate difference is not purely a distance. The short comparison fails if maximum position-coordinate error exceeds 0.005 or velocity-coordinate error exceeds 0.1; passing those checks is not complete physics validation.
- CPU reference timing includes validation bookkeeping. No CPU multi-core scaling or policy/reward/reset overhead is benchmarked.

Every repeat resets to the same initial batch. Other worlds receive small velocity differences. Default runs last 200 physics steps per world (0.2 simulated seconds), sufficient for the deliberately nearby collision but not long-term stability. The model uses the existing solver settings; reducing iterations for speed would require new accuracy checks. Benchmarks preserve these settings.

Successful completion validates only this short physics experiment. Before GPU league training we still need longer collision/constraint validation, equivalent rewards/resets, batched policies and a measured training-speed comparison. Existing SB3 ZIP checkpoints are not automatically converted to a JAX trainer.

Sources: [MuJoCo MJX features](https://mujoco.readthedocs.io/en/stable/mjx.html) and [JAX installation requirements](https://docs.jax.dev/en/latest/installation.html).

Optional Warp import warnings can appear in this JAX-only environment. The benchmark explicitly selects JAX; use the final JSON status and any fatal error to judge the test. No Warp installation is needed for this experiment.

Experimental next step: see MJX_TRAINING.md for batched MJX movement physics feeding CPU PPO. This is headless and uses a separate run format, not the dashboard or league trainer.
