# USB camera capture at 120 FPS

This version adds USB support to `airhockey_shot_tracker.py` (the earlier screenshot showed only CSI options). Copy all the Python files together into the vision repository; `usb_camera.py` is a new required file for USB operation. Existing CSI support remains the default.

The camera's mode list in your photo includes 1280x960 at 120 FPS. Its FOURCC heading was not visible. MJPG below is a trial setting, not a verified camera format. If the 120 FPS entry is under YUYV, replace `MJPG` with `YUYV` in both commands. Use the ELP's actual video node; `/dev/video0` is an example. `/dev/v4l/by-id/...` paths are also accepted.

## First check decoded frame delivery

```bash
cd ~/VisionSystemAirHockeyTable
git pull
python3 airhockey_shot_tracker.py --camera-source usb --usb-device /dev/video0 --usb-width 1280 --usb-height 960 --usb-fps 120 --usb-fourcc MJPG --test-camera
```

The test requires no calibration and opens no video windows. It requests the mode, prints driver readback, checks a decoded frame, then measures decoded delivery over approximately ten seconds. It stops on mode mismatch rather than silently falling back. The test does not prove unique frames, exposure rate, capture latency, or tracker performance.

## Start the tracker

```bash
python3 airhockey_shot_tracker.py --camera-source usb --usb-device /dev/video0 --usb-width 1280 --usb-height 960 --usb-fps 120 --usb-fourcc MJPG
```

The shorter equivalent uses those USB defaults:

```bash
python3 airhockey_shot_tracker.py --camera-source usb
```

First launch requires recalibrating all four corners. USB uses `usb_table_calibration.json` separately from the CSI calibration. Format, device, resolution and requested frame-rate changes invalidate saved corner calibration. Moving the camera/lens always requires `--recalibrate` even if settings are unchanged.

Capture is 1280x960; processing downsamples to 960x720 with the same 4:3 aspect and no software crop. The top-down view preserves table proportions. Windows remain resizable.

## Read the rates correctly

- `USB requested` is what the program asked for.
- `USB driver reports` is negotiated metadata, not a measured rate.
- `Delivered/decoded` is the camera-only test rate.
- `Loop FPS` is the actual tracker-loop delivery/processing/display rate.

120 FPS is not guaranteed by requesting it. USB connection bandwidth, exposure, format decoding, warping and GUI rendering can limit delivery. Exposure longer than about 8.33 ms cannot represent 120 independent exposures per second. No exposure controls are changed automatically. A one-frame buffer is requested but may be unsupported. This update does not yet use threaded capture or GPU decoding.

If mode negotiation fails, run `v4l2-ctl -d /dev/video0 --list-formats-ext` and check the FOURCC heading above 1280x960. If the driver accepts 120 but the test is slower, share the printed rates and `lsusb -t` output before changing settings.

## Checks

`python3 airhockey_shot_tracker.py --self-test` checks trajectory and shot math without camera dependencies. The development machine cannot test your ELP/Jetson: mode negotiation was checked with mocked driver responses, including rejected FPS, dimensions and FOURCC.
