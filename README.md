# Air hockey in MuJoCo

A 3D table and moving yellow puck. Runs on the simulation PC, separate from the Jetson. Standard MuJoCo uses CPU physics; CUDA is not required.

## Install and run

Install Git and uv on CachyOS if needed: `sudo pacman -Syu --needed git uv`.

Clone this repository using its GitHub Code button's HTTPS URL, then open a terminal in the downloaded repository folder.

```bash
# Install Python and project dependencies into .venv automatically.
uv sync
# Launch the table and moving puck.
uv run python table_demo.py
```

Press Space in the viewer to pause/resume. Close the window to quit. Run the launch command again to restart. No environment activation is needed, including in fish shell.

## Update

```bash
git pull
uv sync
uv run python table_demo.py
```

## Model

- Full length: 77 inches, assuming two equal 38.5-inch halves.
- Width: 41 5/16 inches.
- Puck diameter: 2.823 inches.
- Coordinates: table centre is (0, 0); robot's half is negative Y.
- Model units: metres, seconds, kilograms. Camera millimetres need conversion and a coordinate transform.
- All four rails are closed until goal dimensions are supplied.
- Puck thickness (10 mm), mass (20 g), rail dimensions and contact response are provisional.
- Puck is constrained to X/Y translation at fixed height. No spin, air flow, arm, controller or trained AI yet.

The next stage is adding the PlotClock linkage using measured pivot spacing, link lengths, paddle size and motor specifications. Contact behaviour must be calibrated before training for hardware.

## Check physics without opening a window

```bash
uv run python table_demo.py --check
```

This checks model compilation, finite positions, planar movement and approximate containment during a short rollout. It does not validate real-world accuracy or graphics support.
