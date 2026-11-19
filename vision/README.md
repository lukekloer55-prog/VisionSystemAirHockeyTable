# Jetson air-hockey vision

USB camera shortcut (ELP, /dev/video1, MJPG, requested 1280x720 at 120 FPS):

```bash
python3 usb_tracker.py --test-camera
python3 usb_tracker.py
```

See USB_120FPS_SETUP.md for installation, calibration, 1080p/4:3 modes, exposure, actual-rate measurement and troubleshooting. See SHOT_PLANNER_SETUP.md for the puck contact and bank-shot overlay.

Camera-free checks:

```bash
python3 check_usb.py
python3 usb_tracker.py --self-test
```

CSI remains available through `python3 airhockey_shot_tracker.py`. This project does not train policies or command motors. Actual 120 FPS tracking has not been verified on the Jetson.
