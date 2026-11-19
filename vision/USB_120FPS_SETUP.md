# USB camera support for the Jetson vision tracker

For the newer optional 1080p nvv4l2/BGRx path and independent display refresh, see USB_PERFORMANCE_SETUP.md. Software decoding remains the default for usb_tracker.py.

## Install this update

Upload the extracted files together into the repository's `vision/` folder. Keep existing camera/calibration JSON files; none are included or overwritten by this package. This update is for the Jetson vision system, not the simulation PC, and has not been pushed to GitHub.

On the Jetson:

```bash
cd ~/VisionSystemAirHockeyTable
git pull
cd vision
python3 check_usb.py
python3 usb_tracker.py --self-test
```

Use your existing Jetson system Python with OpenCV and NumPy. Do not replace its OpenCV with a pip wheel: many wheels lack GStreamer support. If Linux utilities/plugins are missing, install the Ubuntu packages:

```bash
sudo apt update
sudo apt install v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

## Test the camera first

Close other camera apps. Connect the ELP camera to a USB 3 port. The shortcut defaults to /dev/video1, MJPG, 1280x720 at a REQUESTED 120 FPS, with 960x540 BGR output for processing.

```bash
python3 usb_tracker.py --test-camera --test-camera-seconds 20 --camera-report usb_camera_test.json
```

This measures decoded/resized BGR delivery through the same capture path the tracker uses. It includes JPEG decoding, resizing, color conversion and OpenCV reads. It excludes detection, warping, shot planning and window rendering. The tracker separately displays Loop FPS.

The driver accepting 120 FPS is not proof of delivering 120 usable, unique frames. The report does not measure sensor timestamps or exposure-to-display latency. Your earlier raw V4L2 test reached about 120 FPS, while OpenCV decoding was around 31 FPS. The code cannot guarantee 120 tracker FPS or repair camera firmware/USB payload problems.

If the test stalls, Ctrl+C may not interrupt an OpenCV native read immediately. From another terminal, close the test process normally, or run a bounded diagnostic test instead:

```bash
timeout -s INT -k 5s 30s python3 usb_tracker.py --test-camera
```

## Run tracking

```bash
python3 usb_tracker.py
```

The launcher selects USB automatically. The main `airhockey_shot_tracker.py` still defaults to CSI, so old CSI launch commands continue to work. All normal tracker arguments work with the USB launcher too.

First launch asks for table and puck dimensions, then opens calibration. Defaults remain the full 77-inch length, 41.3125-inch width and 2.823-inch puck diameter. Confirm the actual full-table measurements. Place the puck against BOTH rails at each corner, click its center, hold still and press:

1. Far-left (opponent end)
2. Far-right
3. Near-right (robot end)
4. Near-left

Press Enter after all four; calibration is applied and saved. C recalibrates, S saves, G selects the goal, H changes shot mode, and Q quits. Window borders remain resizable. See SHOT_PLANNER_SETUP.md for the goal/contact overlay.

USB saves `usb_table_calibration.json` separately from CSI. Changing capture mode or processing resolution invalidates incompatible corner coordinates. Recalibrate after moving/replacing a camera even if the mode matches. Existing goal settings are retained; review them after changing the view.

## Select the actual video device

```bash
v4l2-ctl --list-devices
python3 usb_tracker.py --usb-device /dev/video0 --test-camera
```

Replace video0 with the camera's capture node. A camera may expose a separate metadata node that cannot provide images. This version accepts /dev/videoN paths, not arbitrary GStreamer strings.

## Resolution and processing cost

Request 1920x1080 MJPG at 120 FPS:

```bash
python3 usb_tracker.py --usb-width 1920 --usb-height 1080 --test-camera
```

Request the advertised 4:3 1280x960 mode:

```bash
python3 usb_tracker.py --usb-width 1280 --usb-height 960 --test-camera
```

Use only combinations advertised under the selected pixel format by `v4l2-ctl -d /dev/video1 --list-formats-ext`. Different sensor modes may crop differently; inspect the image before calibrating. Software processing keeps all pixels of the selected camera frame, with no additional crop or letterbox border, and uses an exact aspect-ratio resize.

Reduce processing size while retaining the requested camera capture mode:

```bash
python3 usb_tracker.py --usb-process-width 640 --test-camera
python3 usb_tracker.py --usb-process-width 640
```

Use --usb-process-width 0 for native-resolution BGR output. Smaller processing images reduce conversion, copying and tracking work; software JPEG still decodes the source JPEG. This is not a promise of a particular throughput. Lower resolution also reduces detection detail.

## Backend choices

Default GStreamer pipeline: v4l2src -> selected camera caps -> bounded leaky queue -> jpegparse/jpegdec for MJPG -> aspect-preserving resize -> BGR conversion -> one-frame dropping appsink. Old queued frames may be discarded to prevent growing backlog; discarded frames are not counted as decoded deliveries. Arrival-time velocity estimates are not hardware-timestamped measurements.

The hardware nvjpegdec path is deliberately not enabled: it stalled in the earlier tests. Software jpegdec completed those tests. No silent decoder/backend fallback occurs.

Explicitly test the OpenCV V4L2 backend:

```bash
python3 usb_tracker.py --usb-backend v4l2 --test-camera
```

It validates reported resolution, FPS, FOURCC and image shape before tracking. For both backends the actual benchmark is more useful than the requested rate. MJPG corrupt-data messages are not suppressed; inspect resulting images and report them if they persist.

YUYV is supported explicitly, but your previous YUYV 120 FPS tests returned no frames. Treat it as unverified on this camera/Jetson combination:

```bash
python3 usb_tracker.py --usb-fourcc YUYV --usb-width 640 --usb-height 480 --usb-fps 30 --test-camera
```

## Optional manual exposure

Only if desired, request 2 milliseconds using the verified control names from your ELP camera:

```bash
python3 usb_tracker.py --usb-exposure-us 2000 --test-camera
python3 usb_tracker.py --usb-exposure-us 2000
```

This explicitly sets UVC manual exposure and verifies readback (20 units of 100 microseconds). Without this flag, the program does not change exposure. Manual settings may remain in the device after the program exits. Restore auto exposure with:

```bash
v4l2-ctl -d /dev/video1 --set-ctrl=auto_exposure=3
```

At 120 FPS the frame interval is about 8.33 ms. Short exposure reduces motion blur but requires enough light. Exposure control alone did not resolve your earlier YUYV timeout, so it is not presented as that fix.

## Collect a diagnostic report

```bash
python3 usb_diagnostics.py --device /dev/video1
```

This saves usb_diagnostics.txt containing device/mode/control listings, USB link topology, GStreamer element availability and OpenCV build information. It reads configuration without changing camera settings. Send that file and usb_camera_test.json if capture fails or delivered FPS remains low. A USB 3 negotiated link is useful evidence, not proof of successful decoding.
