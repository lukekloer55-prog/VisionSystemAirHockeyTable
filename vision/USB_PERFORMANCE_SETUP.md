# 1080p BGRx capture and tracker profiling

The isolated Jetson tests completed 2,400 source buffers in 37.36 s with BGR output, versus 21.69 s with BGRx output (about 64.2 versus 110.6 buffers/s overall). These include startup and do not certify unique decoded exposures, latency or tracker FPS. They motivate avoiding the final full-frame CPU BGRx-to-BGR conversion.

## Update and test

Put all files from this package together inside GitHub's vision folder, replacing matching names. Existing calibration JSON files are not included. On the Jetson:

```bash
cd ~/VisionSystemAirHockeyTable
git pull
cd vision
python3 check_usb.py
python3 usb_1080p.py --test-camera --test-camera-seconds 20
```

This new shortcut explicitly requests /dev/video0, 1920x1080 MJPG at 120 FPS, native-size four-channel BGRx output, nvv4l2decoder and nvvidconv. It includes bounded queues and OpenCV appsink delivery; the earlier fakesink test did not include those OpenCV reads. Compare the Delivered/decoded result, not the requested FPS. Expected output shape is (1080, 1920, 4).

Then run:

```bash
python3 usb_1080p.py
```

Existing calibration is reused when image geometry and camera signature match. Inspect corner alignment because decoder/color paths can differ. Recalibrate with C if it is wrong. Keep the camera at its calibrated position.

Software fallback at the same capture and processing dimensions:

```bash
python3 usb_1080p.py --usb-decoder software
```

The older usb_tracker.py launcher keeps its software-decoding defaults. No silent fallback occurs. Do not replace the Jetson system OpenCV with pip OpenCV; the Jetson needs its GStreamer-enabled build.

## What changed

The optional path is:

v4l2src io-mode=2 -> MJPG caps -> bounded queue -> jpegparse -> nvv4l2decoder mjpeg=1 -> nvvidconv -> system-memory BGRx -> appsink.

It does NOT use the nvjpegdec plugin that stalled, and does NOT perform a full-frame videoconvert to BGR. The tracker accepts four channels and converts BGRx directly to HSV for yellow masking; the unused fourth byte does not affect HSV. This still copies frames into CPU memory, and is not a zero-copy GPU tracker.

Windows redraw at up to 30 FPS, independently of the tracking loop. Measurement and velocity updates continue on every delivered tracking frame. Visualization-only forecasts, shot-overlay drawing and camera-view copies happen on display frames. The shot proposals are display outputs, not motor commands. Keyboard/mouse events are serviced at the display cadence.

Use --display-fps 15 to reduce GUI work further, or --display-fps 0 to display every processed frame. The display setting does not request a different camera FPS. Loop FPS also appears on the calibration camera window now.

Native 1080p capture is retained, but the existing detector still uses the rectified table image, whose longest dimension is approximately 800 pixels. This update does not change that detection resolution or implement raw-image ROI tracking. More sensor pixels do not automatically mean every sensor pixel is used for contour localization.

## Read the profiling output

Every approximately three seconds the terminal prints PROFILE, with measured loop FPS and average milliseconds per processed frame:

- capture_wait: time spent waiting for/receiving the next OpenCV image; the GStreamer pipeline runs asynchronously, so this is not an isolated decoder execution time.
- prepare: frame validation, timing and occasional display copy.
- warp: perspective transform, during calibrated tracking.
- detect: HSV conversion, thresholding and contour extraction.
- tracking: candidate selection, velocity update and detection markers.
- overlay_display: visualization/shot rendering and some window-submission work. During calibration it also includes corner-selection bookkeeping.
- gui: camera-window submission and event handling.
- keys_other: remaining key handling and calibration actions.

These are averages across processed frames. Display-only work is included only when performed. PROFILE loop FPS is interval frame count divided by wall time; the on-image label is a smoothed loop estimate. Neither measures unique sensor exposures or exposure-to-actuation latency.

Send both the camera benchmark result and several PROFILE lines while calibrated and tracking a moving puck. That distinguishes acquisition from warp/detection/display limits. No 120 FPS performance guarantee is made by this update.

## Compatibility and validation

The OpenCV 4.6 GStreamer source supports BGRx as four-channel input: https://raw.githubusercontent.com/opencv/opencv/4.6.0/modules/videoio/src/cap_gstreamer.cpp . Vendor builds still need the actual test. If opening fails, retain the error output and use the explicit software fallback.

Twelve camera-independent tests passed, including the accelerated caps/shape contract and display/profiling scheduling. Real OpenCV 4.10 image tests confirmed identical BGR/BGRx HSV results before and after warping with varying unused fourth-channel values. The actual calibrated tracking loop ran on synthetic four-channel images using real OpenCV image operations with mocked camera/GUI calls, including reduced display frequency and cleanup. Existing 87 trajectory and 81 shot/collision tests passed. Native Jetson appsink capture, live GUI and performance remain hardware-side verification.
