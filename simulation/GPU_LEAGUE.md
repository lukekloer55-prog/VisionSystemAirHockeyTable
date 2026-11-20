# GPU self-play, league and unattended operation

## Start from the webpage

After updating the simulation files, restart `uv run --extra train python training_ui.py`. Open the CPU + GPU page. Select New run and choose Movement, Self-play tournament, or League with historical opponents. Choose a source model if desired, GPU worlds and CPU worlds, then Start. Modes cannot change on Resume: use a new run to move from movement to play.

GPU self-play trains two candidates against a frozen champion, then evaluates the candidates and champion with paired serves and swapped ends. League also samples historical champions during learning and evaluates against retained history. Candidates run sequentially on one reusable GPU/CPU pool rather than launching competing JAX processes. Within each active candidate, GPU and CPU experience trains one policy. Across candidates, the tournament retains the winning candidate's policy/optimizer; it does not average their weights.

A candidate replaces the champion only after beating it head-to-head, scoring more total tournament points and doing at least as well against historical opponents. Ties retain the current champion. These finite matches are a selection heuristic, not proof of general improvement.

The arena shows the champion's persistent heroic display name, latest completed rankings, current round/phase and champion history. The initial champion is Julius Caesar. Promotions receive names such as Athena and Leonidas, with numeric suffixes after the name list cycles. These names do not change checkpoint paths. Saved rankings retain the identities used in that tournament.

Watch live training displays sampled active training worlds; during tournament evaluation it may show Waiting for next state. Replay saved model shows the current champion playing a copy of itself. The viewers run on the simulation PC desktop, not inside the browser. Close them for unattended runs.

## Performance

```bash
.venv-mjx/bin/python training_ui.py --performance
```

Or use `train_gpu_league.py --performance`, `train_mjx_movement.py --performance`, or `python performance.py`. Reports system CPU utilization sampled over one second, system RAM and NVIDIA GPU utilization, VRAM, temperature and power. This is whole-system utilization, not exclusively training. The webpage's CPU / GPU performance button samples the same data. Missing sensors are reported rather than treated as zero.

## Week-long service setup (Linux simulation PC)

First run a short league session and verify it reaches a completed round. Stop & save and wait until the panel says Stopped. Replace YOUR-RUN with the actual folder name:

```bash
cd ~/robotics/airhockey-sim/simulation
rm -f runs/YOUR-RUN/STOP
loginctl enable-linger "$USER"
.venv-mjx/bin/python training_service.py start --mode league --run-dir runs/YOUR-RUN
```

Use `--mode selfplay` for a self-play run. This setup is done once; the training then runs unlimited rounds with no prompts. It continues after the browser closes and resumes after logout/reboot when user lingering is enabled. Disable automatic suspend in desktop power settings and keep the PC powered. A system configured to require interactive disk unlock cannot boot unattended through that prompt.

Monitor:

```bash
.venv-mjx/bin/python training_service.py status --mode league
journalctl --user -u airhockey-league.service -n 40 --no-pager
```

Stop and disable automatic startup:

```bash
.venv-mjx/bin/python training_service.py stop --mode league
```

The service restarts failures after 60 seconds. A 15-minute watchdog restarts a GPU trainer that stops reporting progress. Child CPU workers are stopped with the service. Intentional STOP files prevent automatic restart, including at reboot; remove STOP only when explicitly restarting. Only one service per mode is installed; do not run independent services concurrently against the same GPU unless you have separately benchmarked that arrangement.

## Recovery and bounded storage

League checkpoints commit models and training phase atomically. Training batches and each completed evaluation match are checkpointed, retaining the three latest generations. A crash resumes the last committed batch or match; work since that boundary can be lost. Historical opponents retain the initial anchor and at most four later champions. Display history retains at most 100 promotions. Event logs rotate at 10 MB. Normal webpage output logs rotate at 10 MB; service output uses the system journal's configured retention. At least 256 MiB free disk is required before committing league snapshots; leave substantially more space for models and system logs.

The webpage can still read status and send STOP/SAVE/reward requests to a service-owned run through its files. Service output is in journalctl; the webpage also reads the last 60 journal lines while a service-owned run is active.

## Scope of validation

GPU game simulations use the experimental planar sphere/frictionless contact proxy from the benchmark, not full cylinder collisions. Shared CPU game workers use the same proxy. Long horizons, expanded joint travel, contacts and learned policies need hardware-specific validation. Local tests cover the MJX game adapter on JAX CPU, short PPO league rounds, restart from checkpoints, promotion rules and service command generation. A seven-day endurance test has not been performed. No software can guarantee recovery from power loss, hardware failure or an exhausted disk without adequate host configuration.

## Browser performance and unattended controls
Open **Performance & unattended operation** at the top of either training page. Performance samples CPU/RAM/GPU usage every five seconds. After a short successful training test, Stop & save, select the saved run on this page, and choose Start unattended operation. Enable logout/reboot persistence with the separate button (Linux may request authentication). The service status is displayed below. Stop, save & disable restart gracefully stops that run and disables its service. Disable automatic suspend in the desktop power settings. Systemd actions are Linux-only; the browser remains local HTTP on 127.0.0.1, not a publicly hosted HTTPS service.
