"""Strict USB/V4L2 mode negotiation; never silently claim 120 FPS."""
import math
import time


def fourcc_text(code):
    """Decode the driver's FOURCC integer, such as MJPG or YUYV."""
    value = int(code)
    return ''.join(chr((value >> (8*i)) & 255) for i in range(4))


def open_usb(cv, device, width, height, fps, fourcc):
    """Request a mode and check driver readback plus the first decoded frame."""
    camera = cv.VideoCapture(device, cv.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Cannot open USB camera {device}. Check device path and close other camera apps.')
        # Format first, dimensions next, interval last: drivers may reset the
        # frame interval when format or size changes.
        camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))
        camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv.CAP_PROP_FPS, fps)
        # Ask for a small queue; some drivers do not support this property.
        if not camera.set(cv.CAP_PROP_BUFFERSIZE, 1):
            print('Note: driver did not accept a one-frame capture buffer.')
        aw = camera.get(cv.CAP_PROP_FRAME_WIDTH)
        ah = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        afps = camera.get(cv.CAP_PROP_FPS)
        afmt = fourcc_text(camera.get(cv.CAP_PROP_FOURCC))
        print(f'USB requested: {width}x{height} {fps:g} FPS {fourcc}')
        print(f'USB driver reports: {aw:g}x{ah:g} {afps:g} FPS {afmt!r}')
        if not all(math.isfinite(v) for v in (aw, ah, afps)):
            raise RuntimeError('Driver returned invalid mode information.')
        if (round(aw) != width or round(ah) != height or afmt != fourcc
                or abs(afps-fps) > max(0.5, fps*0.01)):
            raise RuntimeError('USB mode was not accepted. Match width, height, FPS and FOURCC '
                               'to v4l2-ctl --list-formats-ext. No lower-rate fallback was used.')
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError('USB camera opened but returned no image.')
        if frame.shape[:2] != (height, width) or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise RuntimeError(f'Expected a decoded BGR image, received shape {frame.shape}.')
        return camera
    except Exception:
        camera.release()
        raise


def benchmark(camera, seconds=10):
    """Measure delivered decoded frames without the tracker or display workload."""
    try:
        for _ in range(10):  # Discard startup frames before timing.
            if not camera.read()[0]:
                raise RuntimeError('Camera read failed during warmup.')
        start = time.monotonic()
        frames = 0
        while time.monotonic()-start < seconds:
            if not camera.read()[0]:
                raise RuntimeError('Camera read failed during benchmark.')
            frames += 1
        elapsed = time.monotonic()-start
        print(f'Delivered/decoded: {frames/elapsed:.1f} FPS over {elapsed:.1f} s.')
        print('This does not measure exposure latency, unique sensor frames, or tracker FPS.')
    finally:
        camera.release()
