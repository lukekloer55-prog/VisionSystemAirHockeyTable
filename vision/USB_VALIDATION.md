# USB update validation

Camera-free tests passed locally:

- Exact aspect-ratio sizing for 16:9, 4:3 and IMX219 ratios, plus native-size mode.
- GStreamer MJPG/YUYV pipeline generation, bounded queues, resize caps and software JPEG selection.
- Rejection of invalid/injected device paths, invalid FPS and incompatible output aspect ratios.
- V4L2 property order and rejection/cleanup for mismatched FPS, resolution, FOURCC and non-finite readback.
- GStreamer open failure, empty frames and mismatched output shapes release the camera.
- Explicit V4L2 processing resize detects a changed source shape before resizing it.
- Benchmark frame counts, JSON output and resource cleanup on failure.
- Manual exposure conversion and verified readback, using mocked Linux control commands.
- End-to-end USB launcher benchmark invocation with an explicit 1080p override, mocked OpenCV capture and saved JSON.
- Existing 87 trajectory cases and 81 shot/collision cases passed.

The tests use fake capture objects and mocked controls. They establish software behavior, not Jetson driver or camera compatibility. No ELP camera or Jetson is attached to this workspace. Real GStreamer negotiation, decoded image quality, actual camera/tracker FPS, GUI calibration and exposure changes require the user's hardware test. Earlier screenshots demonstrated raw USB frame receipt near 120 FPS, not this updated full tracker running at 120 FPS.
