# Full-table vision shot planner

Copy `airhockey_shot_tracker.py`, `shot_planner.py`, and `shot_overlay.py` together into your existing Jetson vision repository. Keep the original `airhockey_calibrated.py` as a working fallback. This update requires the same OpenCV/NumPy installation and CSI camera setup as the old tracker. No new pip packages are needed.

## Run on the Jetson

```bash
cd ~/VisionSystemAirHockeyTable
python3 airhockey_shot_tracker.py --self-test
python3 airhockey_shot_tracker.py
```

## Full-table calibration is required

## Resizable windows and camera modes

All three image windows now allow border dragging and maximize, with OpenCV aspect preservation requested. Some behaviour depends on the Jetson's GUI backend. Mouse clicks use image coordinates supplied by OpenCV; check a corner selection after resizing before capturing all four.

The default `--camera-mode full30` uses this Jetson's reported sensor mode 3: 1640x1232 at 30 FPS, converted to 820x616 for processing. This keeps the exact 205:154 IMX219 sensor ratio (slightly different from 4:3) and selects the full-aspect binned mode. It should reveal more table than the cropped 720p mode; actual optical coverage still depends on camera mounting/lens and driver crop settings.

```bash
# Default: full sensor aspect, 30 FPS.
python3 airhockey_shot_tracker.py --camera-mode full30

# Full native-resolution capture at 21 FPS, same 820x616 processing size.
python3 airhockey_shot_tracker.py --camera-mode full21 --config full21_calibration.json

# Previous cropped 16:9 mode for 60 FPS; use its own calibration file.
python3 airhockey_shot_tracker.py --camera-mode fast60 --config fast60_calibration.json
```

Changing camera mode invalidates corner calibration, including between modes with equal processing size. Older calibration files without a mode tag must also be recalibrated once. Merely resizing windows does not invalidate calibration or alter the captured frame.

The raw camera view preserves camera aspect. The top-down table intentionally preserves the measured TABLE aspect, plus a label panel. Forcing that view to the camera aspect would distort distances and angles. Letterboxing/unused window space is expected when the window proportions differ.

The listed rates are requested sensor rates, not guaranteed application throughput. Full30 doubles the frame interval versus 60 FPS; assess fast-puck blur/tracking after changing modes. No new camera hardware test was possible on the development machine.

## Capturing full-table corners

The new program uses `full_table_calibration.json`; it does not reinterpret the old half-table calibration. Aim the camera at the entire playing area. Default width is 1049.3375 mm, full length is 1955.8 mm (assuming two equal 38.5-inch halves), and puck diameter is 71.7042 mm.

Far end means opponent goal and is Y=0; near end is Y=1955.8. X increases to the right; Y increases toward the robot. Move the puck into each physical corner touching both rails, click its centre in the raw calibration window, hold still, then press 1 far-left, 2 far-right, 3 near-right, 4 near-left. Enter applies and saves all four. If a corner is not physically accessible, don't substitute an arbitrary point.

The opponent goal defaults to a centred 254 mm (10-inch) opening:
- Left inside post X=397.66875 mm.
- Right inside post X=651.66875 mm.
- Goal line Y=0.
- With puck radius and default extra 5 mm margin, safe centre crossing X=438.52085..610.81665 mm.

## Controls and display

- H cycles DIRECT, LEFT bank, RIGHT bank. The planner aims at the safe goal midpoint.
- G optionally overrides the goal: click inside left then right goal-post edges in the top-down view. Goal must be at the far edge; Y snaps to zero. Esc cancels. Overrides save in `shot_goal.json` and invalidate when calibration changes.
- C recaptures table corners; S saves calibration/HSV settings; Q quits.
- Magenta X: proposed paddle contact point on the measured puck circumference.
- Magenta radius: contact normal. White line: tangent at that contact.
- Orange path: requested post-impact puck-centre trajectory; magenta dot: one-wall bounce; green cross: goal target.
- Blue/cyan incoming arrow: measured travel. Yellow arrow: signed required paddle normal velocity.
- Existing cyan forecast and red short-horizon marker still show UNSTRUCK motion.
- Angle labels use degrees, positive clockwise from screen-right because image Y points down. Tangent orientation is modulo 180 degrees.

Paddle-relative incoming/departure angles are measured from the contact normal. They are equal only for the ideal elastic case, not generally for a driven paddle with restitution less than one.

## Shot settings

```bash
# Requested outgoing puck speed, not a measured or commanded motor speed.
python3 airhockey_shot_tracker.py --shot-speed-mm-s 1500
```

Optional `--paddle-diameter-mm` accepts your actual measured paddle diameter. Without it, contact X and tangent still work, but there is no paddle-centre circle or rail-clearance check for the paddle. With it, a yellow paddle circle and centre coordinates appear; overlapping rail/end configurations are flagged.

`--paddle-restitution` defaults to an UNCALIBRATED 0.8; `--goal-margin-mm` defaults to 5; `--goal-width-mm` defaults to 254. A saved clicked goal overrides the default goal width. Use a different `--goal-config` path to start with the default again.

## What the model means

Let v be measured incoming velocity and w be desired launch velocity along the selected path. The contact normal n is `(w-v)/|w-v|`; contact point is `puck_center - puck_radius*n`. The tangent is perpendicular to n. This preserves the tangential puck velocity in a frictionless impact.

For a paddle whose velocity is prescribed and unaffected by impact, restitution e gives required normal paddle speed `u_n = v dot n + |w-v|/(1+e)`. The yellow arrow shows this signed normal motion. It does not assume the puck simply reflects off a stationary paddle.

Banks reflect at X=puck_radius or width-puck_radius. Wall collisions are ideal elastic and lossless. Goal clearance is checked conservatively at both the radius-offset end line and the goal line.

This is a CURRENT-POSITION aiming overlay, not a timed reachable interception controller. No motors are commanded. Incoming velocity must be available from fresh measured frames; lost/estimated-only frames suppress the shot overlay. Full robot reach, travel time, acceleration, opponents, obstacles, spin, friction, wall energy loss and camera/control latency are not modeled. A marked shot is not a guarantee that a real paddle can execute it or score.

## Validation

`--self-test` runs the existing reflection cases plus independent shot geometry and moving-paddle impact reconstruction checks. Live CSI capture, GUI mouse interaction and real impacts must be checked on the Jetson.
