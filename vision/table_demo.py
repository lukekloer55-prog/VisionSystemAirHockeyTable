"""A measured table and a planar puck; start with `uv run python table_demo.py`."""

import argparse  # Reads the optional --check command-line flag.
import math  # Checks that simulated positions remain finite.

import mujoco  # Compiles the scene and advances its physics.

# Convert the user's measurements from inches to metres.
WIDTH = 41.3125 * 0.0254
LENGTH = 77.0 * 0.0254  # Two equal halves, each 38.5 inches long.
RADIUS = 2.823 * 0.0254 / 2


def build_scene():
    """Describe the table in MJCF, MuJoCo's XML scene format."""
    # Box geometry uses half sizes, not full dimensions.
    half_w, half_l = WIDTH / 2, LENGTH / 2
    # These four stationary boxes form closed rails with 30 mm thickness.
    rails = [
        ("left", -half_w - 0.015, 0, 0.015, half_l),
        ("right", half_w + 0.015, 0, 0.015, half_l),
        ("far", 0, half_l + 0.015, half_w + 0.03, 0.015),
        ("near", 0, -half_l - 0.015, half_w + 0.03, 0.015),
    ]
    # Turn each rail description into one XML geometry element.
    rail_xml = "\n".join(
        f'<geom name="{name}" type="box" pos="{x} {y} 0.025" '
        f'size="{sx} {sy} 0.025" rgba="0.15 0.25 0.45 1"/>'
        for name, x, y, sx, sy in rails
    )
    return f"""
<mujoco model="Air hockey table">
  <!-- Fixed-height planar motion approximates an air-supported puck. -->
  <option timestep="0.001" gravity="0 0 0"/>
  <statistic center="0 0 0" extent="2.5"/>
  <default>
    <!-- Provisional underdamped contact; not measured restitution. -->
    <geom friction="0 0 0" solref="0.004 0.2"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <!-- Surface and centre line are visual only; joints fix puck height. -->
    <geom name="surface" type="box" pos="0 0 -0.02"
          size="{half_w} {half_l} 0.02" rgba="0.92 0.92 0.92 1"
          contype="0" conaffinity="0"/>
    <geom name="centre_line" type="box" pos="0 0 0.0005"
          size="{half_w} 0.002 0.0005" rgba="0.15 0.15 0.15 1"
          contype="0" conaffinity="0"/>
    {rail_xml}
    <body name="puck" pos="0 0 0.006">
      <!-- These joints allow translation along X and Y only. -->
      <joint name="puck_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="puck_y" type="slide" axis="0 1 0" damping="0"/>
      <!-- Temporary mass 20 g and full thickness 10 mm. -->
      <geom name="puck_shape" type="cylinder" size="{RADIUS} 0.005"
            mass="0.020" rgba="1 0.85 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_simulation():
    """Compile the scene and give the puck its initial position and velocity."""
    model = mujoco.MjModel.from_xml_string(build_scene())
    data = mujoco.MjData(model)
    # Start in the robot half and launch diagonally toward the opposite end.
    data.joint("puck_y").qpos[0] = -0.5
    data.joint("puck_x").qvel[0] = 0.7  # Metres per second rightward.
    data.joint("puck_y").qvel[0] = 1.2  # Metres per second toward far end.
    mujoco.mj_forward(model, data)  # Update geometry before displaying it.
    return model, data


def check_physics(model, data):
    """Exercise collisions for ten seconds without requiring a graphics window."""
    contact_seen = False
    for _ in range(10000):
        mujoco.mj_step(model, data)
        # Reject unstable numeric output and escaped pucks.
        x, y = float(data.qpos[0]), float(data.qpos[1])
        assert math.isfinite(x) and math.isfinite(y), "Non-finite puck state"
        assert abs(x) < WIDTH / 2 and abs(y) < LENGTH / 2, "Puck escaped"
        contact_seen = contact_seen or data.ncon > 0
    assert contact_seen, "Expected at least one rail collision"
    print("PASS: ten-second physics rollout, finite state, containment and contact.")


def main():
    # Offer a headless check as well as the normal interactive viewer.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Test physics without a window")
    args = parser.parse_args()
    model, data = make_simulation()
    if args.check:
        check_physics(model, data)
    else:
        # Import graphics only when a viewer is requested.
        import mujoco.viewer
        print("Space: pause/resume. Close window: quit. Rerun script: relaunch puck.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
