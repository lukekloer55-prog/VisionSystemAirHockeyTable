# Expanded joint travel

The old motor command range was +/-0.30 rad (17.2 degrees), with physical stops at +/-0.35 rad. These were provisional commissioning restrictions, not measurements from the CAD assembly.

New default stages allow base commands and simulated base travel up to +/-pi radians (180 degrees) relative to the assembled home pose. Passive elbows have no explicit angle stops. This is a broad simulation envelope, not a measured hardware stop specification. The inverse-kinematics helper no longer rejects commands above 0.35 rad.

The narrow paddle-link angle still incurs the configured reward penalty, but an angle below 5 degrees no longer automatically terminates the episode. Finite-state, linkage-closure and paddle/table boundary failures remain. Torque, damping, target slew and physical dimensions remain as before. Link self-collisions are still disabled: geometric reach is not proof of a collision-free, dynamically feasible hardware trajectory.

## Update an existing movement model

Stop and save the old run. In the shared CPU/GPU page choose **New run**, enter a different name, choose the old model in **Start from trained model**, and leave **New stage: allow base motors +/-180 degrees from home** checked. Start the new run.

Resume alone preserves the saved range; it does not expand old checkpoints silently. New stage import rescales the Gaussian actor mean and standard deviation by old_range/new_range to preserve its initial physical commands. Adam state for those changed action-head tensors is reset; other learned parameters remain. Subsequent training may explore the enlarged range. A fresh untrained model can also be selected.

The saved-run status shows its command travel in degrees. Existing CPU runs with explicit action_angle_rad keep that setting. Updated default physical settings use pi radians. The checkbox belongs to the shared training page.

## Verification

A 21x21 grid across the paddle-clearance robot half had 431/441 valid geometric poses. The other 10 exceeded one or both chains' link-length reach. The largest required base angle among valid points was 99.33 degrees. Static closure errors were below 1e-7 metres. Tests also verified legacy +/-0.35 physical limits and preservation of imported deterministic physical commands after action-head rescaling. This is not a full dynamic or collision coverage guarantee.
