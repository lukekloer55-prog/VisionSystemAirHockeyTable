# PlotClock arm simulation

This adds `plotclock_demo.py` to the existing air-hockey MuJoCo project. It is a simulation-only mechanism check, not a trained controller and not a real servo interface.

## Install on the separate simulation computer

Extract this ZIP into a folder, then open a terminal in that folder. It contains pyproject.toml, table_demo.py, plotclock_demo.py and plotclock_geometry.json.

```bash
uv run python plotclock_demo.py --check
uv run python plotclock_demo.py
```

If you copy the files into the previous project instead, run those commands from that project directory. No nano editing is required. The archive has not been published to GitHub.

The default demo moves the end pin around a small 20 mm radius circle every eight seconds. Blue is the left chain; orange is the right chain; green is the controlled pin; magenta is the target. Close the viewer to stop. Drag the view to inspect the mechanism.

Hold the original pose:

```bash
uv run python plotclock_demo.py --hold
```

Run the original free-puck demo separately:

```bash
uv run python table_demo.py
```

## What comes from your STEP model

- Component3: fixed base, 457.2 mm (18 inches) between the motor pivot axes.
- Component2 and Component5: left and right motor-driven links.
- Component4 and Component1: left and right passive distal links.
- Each of the four links: 584.2 mm (23 inches) between pivot axes.
- Component6: controlled end pin, approximately 12.665 mm diameter.
- Link section modeled as 25.4 x 25.4 mm.
- The initial planar pivot coordinates are recovered from the assembled STEP hole axes, then rotated so the base is horizontal. Geometry is converted from millimetres to metres.

This is an analytical reconstruction with simple box links, not a tessellation of the exact CAD solids. Hole recesses, fasteners and detailed mesh surfaces are omitted. The original STEP is included for later mesh import. Distal links are shown on separate vertical layers. Four planar hinge joints plus a tip connection equality constraint form the closed mechanism.

## What is provisional

- Two identical direct-drive 1:1 actuators, with a conservative +/-2.4 N*m cap per motor based on the supplied nominal torque.
- The 3000 RPM rating is not used as a commanded arm speed or assumed maximum speed. Peak torque and the full torque-speed curve are not modeled.
- 0.4 kg per link, 0.02 kg end pin, rotor inertia 0.0001 kg*m^2 per motor, damping, position-loop gains, and +/-0.35 rad base-joint travel around the CAD pose are estimates.
- No actual T3DL electrical model, encoder delay, motor thermal model, hardware communication, or hardware safety controller is included.
- The base servo-pivot line sits 101.6 mm (4 inches) behind the near edge of the 77 x 41.3125 inch table, as requested. The setback is measured to the pivot line, not the nearest face of the bar. The arm plane remains provisionally 100 mm above the table.
- The right elbow extends past the side rail in the original pose. Links have collisions disabled for this mechanism-only test. This does not establish physical table/rail clearance.
- There is no paddle collider or puck in the arm demo yet. The green pin is the controlled attachment point. Paddle dimensions/mass, link masses, actual joint limits and mounting position are needed before useful strike or reachability tests.

The controller uses inverse kinematics for both two-link chains, preserving the CAD elbow branches. A torque-clipped PD loop follows the target. The 12-second test checks finite physics, linkage closure below 0.5 mm, tracking below 20 mm, actual motor torque within caps, motion, and absence of MuJoCo warnings.

Validated locally with MuJoCo 3.17.0: closure gap below display precision (0.0000 mm), maximum target tracking error 2.78 mm, peak motor torque 0.022 N*m for the slow test. These are simulation checks under estimated masses, not predictions of real-machine performance.

MuJoCo connection constraint reference: https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#equality-connect

## Reach limits

Each chain has two 23-inch links, so its maximum straightened reach is 46 inches from its servo axis. With the 4-inch setback and 18-inch servo spacing, the theoretical centerline reach is sqrt(46^2 - 9^2) - 4 = 41.11 inches into the table. Thus it cannot cover the full 77-inch table. The robot-half far corners are also outside the two-chain reach intersection: their distance from the opposite servo is about 51.82 inches, exceeding 46 inches. These are geometric upper bounds; singularities, actual joint limits, paddle radius and collision clearance further restrict usable reach. The demo retains its small provisional joint range and does not claim full-half-table coverage.
