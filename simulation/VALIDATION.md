# Validation results

Tested locally using MuJoCo 3.17.0, Gymnasium 1.2.3, Stable-Baselines3 2.7.1 and PyTorch 2.7.1+cpu.

- Gymnasium/SB3 API, seeded resets and 25-element observations passed.
- Fifty generated initial shots intersected the reachable defense strip.
- Six hundred randomized actions preserved finite states, link closure and motor torque limits.
- Both goals, non-goal escapes, side rebounds, bank-goal bonus, actual paddle impacts and repeated hits in the same rally passed targeted checks.
- Center-line reward requires paddle-contact credit and does not end the rally; minor event rewards respect the shared budget.
- Behind-paddle penalty, pursuit after contact and timeout truncation checks passed.
- Local HTTP panel served HTML/state and accepted complete authenticated reward changes. Invalid requests were rejected.
- Queued updates left settings unchanged before the batch boundary, then updated training/evaluation together, refreshed the cached observation and preserved rewards already paid.
- Applied settings and change history were written successfully.
- A 10,240-step CPU PPO run completed, including periodic checkpoint and best-model evaluation paths.

These are software checks, not evidence of a trained competition-ready policy. Native MuJoCo interaction and browser appearance on CachyOS require user-side verification. No smoke-test model is included as a finished policy.

Additional checks passed: resumed the full-rally checkpoint for 1,024 steps with the live panel callback, applied a queued goal-reward change at step 10,240, and verified the new checkpoint settings while older checkpoint settings remained unchanged. Headless playback loaded the resumed policy for three held-out rallies (all timed out; zero goals). Python files and browser JavaScript passed syntax checks.

## Movement/self-play update

- Six `check_skills.py` tests passed: reachable targets/24-value observations, target success and cost terms, mirrored four-actuator scene and both goals, physical far-paddle rebound, side-balanced winner/draw/stop behavior, atomic save failure and bounded retention.
- Stable-Baselines3 environment checks passed for both movement and game tasks.
- Real CPU PPO smoke runs completed: movement update, resume, independent A/B updates and a two-rally match, and movement-checkpoint initialization of self-play. Short game tests intentionally used 0.2-second rallies and ended in draws; they verify orchestration, not playing skill.
- A running unlimited PPO process responded to a STOP file, saved its partial candidate, resumed and completed the round. Snapshots stayed bounded at three generations.
- Existing full-rally physics/reward regression suite passed after the optional XML-export refactor in training_model.
- Both scenes were rendered offscreen and visually inspected. Native live-viewer interaction and Linux systemd execution were not exercised on this Windows test host.
- No trained champion, week-long soak run or physical hardware validation is supplied. The package contains source code and configuration; it does not upload itself to GitHub.

## Training dashboard update

Four API/transaction tests pass: reward bounds and field validation, atomic application/persistence, deferral until a fair self-play round boundary, lock polling without file growth, save requests, localhost/token checks, path traversal rejection and snapshot ZIP export. The six movement/self-play regression tests still pass.

Browser verification started a real PPO movement run, changed time_cost from 0.5 to 1.25 using the UI, observed the applied checkpoint value, requested a checkpoint, stopped cleanly after 6,657 steps, selected its saved model for a new self-play run, and verified mode-specific controls. The training controller was shut down after testing. Native replay-window launch was not retested in this dashboard pass; prior offscreen scene validation remains applicable. Linux service execution remains untested on this Windows host.


Run display-name update: controller tests passed for renaming while locked, preserving checkpoint pointer and weights, export metadata, invalid names and path traversal. Existing UI polling/new-run JavaScript tests passed. A 256-step PPO movement transfer completed and persisted its source training lineage. Linux GUI and background service execution were not retested for this metadata-only change.


Assembly3 and parallel workers: updated distal pivot spacing 28.04874066 inches, servo-axis setback 9.5 inches. Twelve-second linkage test passed (2.60 mm maximum tracking error, effectively zero closure gap). Six skills tests and old rally regression passed. Two spawned workers completed movement training and both self-play candidates plus a paired match. Resume from two workers to one completed. Separate checks confirmed distinct targets, reward propagation, finite observations, zero viewer allocation headless, two live native viewers when enabled, and process cleanup. UI script syntax/polling checks passed. These are short Windows tests, not a week-long Linux endurance test.

Parallel PPO follows the SubprocVecEnv architecture described at https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html . Local short runs measured about 728 total steps/s with two workers versus 504 with one; these are illustrative learning-block measurements, not a controlled benchmark or a prediction for the user's CPU.


Parallel league update: five unit checks passed for balanced schedules, all-draw retention, head-to-head requirements, historical nonregression, archive anchoring/retention and reward-field compatibility. Six dashboard/controller checks passed including league dispatch, worker budget validation, reward update boundaries and existing rename/export checks. JavaScript syntax and new-run polling tests passed. Real PPO tests completed a two-candidate round, all six frozen evaluations, cooperative stop during a worker batch, and resume with four simulation workers. A second rendered run completed two native training viewers and evaluation. Short test games tied and correctly retained the existing champion; promotion itself was tested deterministically. These Windows checks do not validate week-long uptime or Linux service execution. Startup overhead is significant for short batches; no speedup guarantee is made.


MJX experiment: MuJoCo 3.17.0, mujoco-mjx 3.17.0 and JAX 0.12.1 were exercised on CPU. The strict model correctly failed on cylinder-box contacts. The explicit sphere proxy plus condim=1 passed short rail and paddle scenarios; MJX/CPU proxy coordinate differences were about 9.7e-6 and 3.6e-6, with linkage gaps below 0.5 micrometres. A two-world JAX-vmap run completed. CPU original-versus-proxy coordinate differences reached roughly 0.0036 in these tests, so the approximation is not asserted equivalent. The benchmark now rejects nonfinite state, excessive linkage gap and CPU/MJX disagreement. No GTX 1070 Ti execution, CUDA package installation, GPU throughput, long-duration physics or GPU policy training was tested here. CPU training files were not switched to MJX.


Whole-half sampling update: 4,000 seeded samples populated every cell of a 4x4 partition and the bottom six-inch strip; inverse kinematics was mocked to raise during sampling, confirming it is not called. Paddle clearance and timeout behavior passed. Six skills regression tests and UI JavaScript checks passed. Game pursuit reachability remains unchanged.

Experimental MJX movement bridge: local JAX CPU two-world step and automatic episode reset passed; a small PPO update, atomic-format ZIP save/load and resumed update passed. Six CPU skills tests and two whole-half sampling tests passed. GPU speed/stability and league integration are unverified.

Shared CPU/GPU movement update: tested two MJX worlds on the JAX CPU backend plus one real spawned CPU worker; combined PPO updates, independent resets, reward changes, live-state extraction and save/resume passed. CPU checkpoint import copied physical settings, continued timesteps and left source ZIP unchanged. Native MuJoCo viewer opening and backend text overlays passed locally. Six CPU skills regressions passed. Actual concurrent GTX GPU speed and long-duration stability remain user-machine tests.
