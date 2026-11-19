# GPU and shared CPU + GPU movement training

See HYBRID_TRAINING.md for webpage controls, CPU-model import, shared experience, and live rendering.

Install the separate environment from the simulation directory:

```bash
uv venv --python 3.12 .venv-mjx
uv pip install --python .venv-mjx/bin/python -r requirements-mjx-training.txt
```

Do not recreate an environment that already works. PyTorch is pinned to the CPU wheel so it does not conflict with JAX CUDA libraries. Physics runs on the GPU by default, with no silent CPU fallback.

Command-line example (two CPU workers plus 64 GPU worlds):

```bash
.venv-mjx/bin/python train_mjx_movement.py --run-dir runs/shared-movement-v1 --num-envs 64 --cpu-envs 2
```

Append `--initial-run runs/your-cpu-movement-run` only when creating a new run. Stop/save the source first. To continue an existing shared run use `--resume` with the desired GPU and CPU counts. `--steps 0` (default) continues until stopped; Ctrl+C saves. First compilation can take minutes. World counts alter rollout batch size and may affect learning as well as speed.

All worlds contribute to one shared PPO policy/optimizer. Policy and reward calculations still run on CPU. CPU simulation subprocesses run concurrently with GPU physics. No rendering occurs unless the live-view control is used. Self-play/league remain separate CPU trainers.

Checkpoint replacement is atomic; a hard kill may lose work since the previous save. GPU numerical checks and benchmark results do not prove long-term stability. Test the combined configuration before unattended training.
