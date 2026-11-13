# Calibrated air-hockey tracker

This is a separate app. Keep your working `puck_tracker.py` unchanged.

## Your dimensions

Defaults are prefilled from your measurements:

| Measurement | Inches | Millimetres |
|---|---:|---:|
| Playing-area width | 41 5/16 | 1049.3375 |
| Playing-area length | 38.5 | 977.9 |
| Puck diameter | 2.823 | 71.7042 |

The program treats this as the calibrated playing rectangle. If this describes only half the table, the prediction ends at that rectangle's end line; it does not know the other half exists. This calibration method assumes all four recorded puck centers are half a puck diameter inward from the rectangle's boundaries. For an imaginary boundary such as the center line, mark that inset explicitly rather than putting the puck center directly on the boundary. Rounded physical corners can prevent the puck touching two straight rails simultaneously; use marked equivalent center positions in that case.

## Install and run

On the Jetson desktop, outside the NanoOWL container:

```bash
sudo apt install python3-opencv python3-numpy
mkdir -p ~/airhockey
```

Copy `airhockey_calibrated.py` into `~/airhockey` using Files or a USB drive. Do not type the script manually. Close the old tracker and other camera apps.

```bash
cd ~/airhockey
python3 airhockey_calibrated.py
```

First run asks for width, length, and diameter in millimetres. Press Enter three times to accept the prefilled measurements. Subsequent runs load the saved file automatically.

## Four-corner calibration

Mount the camera rigidly, with all four puck positions fully visible. Choose an orientation: the far end becomes the TOP of the corrected image; left and right are the side rails when facing that far end. The raw camera image need not already have this orientation.

Use the SAME yellow puck at each corner, one position at a time:

1. Place it at the **far-left corner**, with its rim just touching both straight boundary rails. Click the puck's center in **Camera - calibration**, keep it still until a green outline appears and about half a second has passed, then press **1**.
2. Repeat at **far-right**, click the puck, and press **2**.
3. Repeat at **near-right**, click the puck, and press **3**.
4. Repeat at **near-left**, click the puck, and press **4**.
5. Press **Enter** to apply and save. This creates `table_calibration.json` beside the script.

Each number stores the average detected center over 12 frames. If another yellow object is selected, click nearer the puck. A key only records when the selected region is stable. You can recapture any numbered corner before pressing Enter.

In **Yellow mask**, the puck should be solid white. Tune the HSV sliders if needed. H=yellow color range; S=color strength; V=brightness. Lower bounds must not exceed upper bounds. Press **S** to save changed yellow settings.

### Keys

| Key | Action |
|---|---|
| 1 / 2 / 3 / 4 | Record a corner while calibrating |
| Enter | Apply and save all four corners |
| C | Start new corner calibration with the current dimensions |
| S | Save calibration and current HSV thresholds |
| Q | Quit |

Click an OpenCV video window first so it receives keyboard input. To change physical dimensions too, run `python3 airhockey_calibrated.py --recalibrate`.

## What the display means

- Green: visually measured puck and center.
- Yellow: current position estimated through a missing detection, for at most 50 ms.
- Red cross: predicted center 30 ms ahead of the latest processed observation.
- Cyan line: a separate longer path preview, up to 500 ms. It is not the red marker's horizon and becomes less reliable farther ahead.
- Magenta dots: predicted side-wall contacts.
- X/Y: millimetres from the corrected table's far-left physical boundary intersection; X rightward, Y toward the near end.
- Vx/Vy: millimetres per second, measured AFTER perspective correction.
- Angles: heading from +X, positive toward +Y (clockwise on screen). A displayed 30 -> 150 degrees is a right-side reflection, not a 120-degree incidence angle.

Prediction stops when the puck center reaches an end limit. Goals and end-wall reflections are not implemented. The reflection model includes the puck radius, reverses horizontal velocity at a side hit, and preserves vertical velocity and speed. It assumes an ideal bounce: no friction, spin, impact loss, paddle collisions, or curved corners.

Shorter cyan preview:

```bash
python3 airhockey_calibrated.py --horizon-ms 30 --preview-ms 150
```

## Validate before trusting predictions

1. Hold the puck stationary at several locations. The corrected image should show a roughly circular puck, and velocity should settle near zero.
2. Return to all four calibration positions. The displayed centers should be about 35.85 mm from the adjacent boundaries.
3. Move horizontally and vertically. Confirm correct axes and plausible speeds. The top-down playing area is slightly wider than long with your dimensions; that is intentional.
4. Slide slowly toward a side rail. The cyan path should turn at the puck-center limit, one radius from the physical wall.
5. Compare actual rebounds with the predicted path. After a real bounce, the next observations update the estimate.
6. Test faster motion only after this works. Blur, camera buffering, rolling shutter, lighting, and lost detections all affect accuracy.

Recalibrate whenever the camera moves, rotates, zooms, or changes capture mode. Four-point perspective correction assumes a plane. It cannot remove lens distortion or recover hidden objects, and it is only approximate for thick 3D objects. A steep view magnifies errors at the far end. This app uses frame-read times, not sensor exposure timestamps; the red cross is not a fully latency-compensated physical prediction.

## Mathematical tests

```bash
python3 airhockey_calibrated.py --self-test
```

These test straight travel, radius-offset rails, left/right hits, multiple bounces, exact-wall starts, stationary pucks, and stopping at end lines. Camera operation, HSV settings, homography accuracy, and speed still require validation on your Jetson.

## Learning the code

- `make_transform`: converts four camera points into radius-inset top-down points using OpenCV's `getPerspectiveTransform`.
- `warpPerspective`: rectifies each image before detection, so both axes share one physical scale.
- `detect`: finds yellow regions and filters their size and shape.
- `predict_path`: computes time to each next boundary and advances/refects the velocity.
- The main loop measures movement, draws labels, and processes input once per frame.

For a wall normal vector n, reflection is v_new = v - 2*(v dot n)*n. A vertical wall has a horizontal normal, so vx becomes -vx while vy stays unchanged.

Official perspective-transform tutorial: https://docs.opencv.org/4.13.0/da/d6e/tutorial_py_geometric_transformations.html
