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
