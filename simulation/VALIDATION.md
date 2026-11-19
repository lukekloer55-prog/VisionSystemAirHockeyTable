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
