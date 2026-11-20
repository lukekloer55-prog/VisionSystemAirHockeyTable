# Shared CPU + GPU movement training

Open the standard panel with `uv run --extra train python training_ui.py`, then click **Open CPU + GPU shared movement training**.

## Use a previously trained CPU movement model

1. Stop and save its CPU run.
2. Select New run on the shared training page and give it a new name.
3. Choose the CPU run in **Start from trained model**. The source remains unchanged. Physical settings are copied and geometry/schema must match. Its reward settings populate the sliders; you can adjust them before starting.
4. Set GPU worlds and CPU worlds independently. Try your working GPU count with 1 or 2 CPU worlds first.
5. Start training. All worlds collect experience for **one PPO policy and optimizer**. Each update learns from the combined rollout, then all worlds use that updated policy.

A CPU count of zero keeps the earlier GPU-only physics mode. Resume preserves learned weights; world counts can change. Standard CPU self-play and league remain on their original page. Shared game/league training is not implemented.

## Count and performance display

The page shows requested GPU / CPU / total counts and the saved/running run's actual counts. One CPU world runs per spawned process; the GPU worlds are vectorized in JAX. CPU stepping is dispatched before GPU stepping and both results join the same on-policy rollout. This is concurrent simulation, not adding model weights or separate independent training jobs.

CPU workers can contend with reward calculation and PPO, so more workers do not necessarily mean faster training. The log and progress file report aggregate experience. Compare steady-state throughput before choosing the count for a long session. The synchronous rollout waits for the slower group.

## Visualize

**Watch live training** opens native MuJoCo windows on the simulation PC for up to two GPU and two CPU worlds. Each window labels its backend. These are sampled copies of actual training states, refreshed up to five times per second; accelerated training can appear jumpy. Rendering never drives the physics or supplies rewards. Close any viewer window to close the live-view process and disable snapshot requests. The remaining hundreds of worlds do not render.

**Replay saved model** runs a separate, real-time CPU visualization of the latest policy, loading new checkpoints at episode boundaries. It is useful for judging movement quality, but is not an actual training rollout.

Start / Resume runs indefinitely. Stop & save requests a checkpoint and shutdown; Save checkpoint leaves training running. Sliders apply to both CPU and GPU environments at the next rollout boundary. Result downloads include the shared policy, manifest, physical settings, and curriculum. First compilation can take minutes.

## Physics and validation scope

The CPU workers use the same planar collision proxy as MJX, and the same target/reward/termination code. MJX is still an experimental solver backend; matching data schemas does not guarantee identical learned behavior. Short compatibility tests do not establish week-long stability. Run short training and replay checks before leaving the computer unattended.


Joint-travel update: see JOINT_TRAVEL.md. Create a new stage with the expand-travel checkbox enabled to migrate a restricted model. Ordinary resume preserves its saved travel.
