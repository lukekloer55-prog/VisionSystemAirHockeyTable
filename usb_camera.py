"""Strict USB/V4L2 mode negotiation; never silently claim 120 FPS."""
import math
import time
from fractions import Fraction
import re


def fourcc_text(code):
    """Decode the driver's FOURCC integer, such as MJPG or YUYV."""
    value = int(code)
    return ''.join(chr((value >> (8*i)) & 255) for i in range(4))


def open_usb(cv, device, width, height, fps, fourcc, backend='gstreamer'):
    """Choose the requested capture backend without silently falling back."""
    if backend == 'gstreamer':
        return open_gstreamer(cv, device, width, height, fps, fourcc)
    return open_v4l2(cv, device, width, height, fps, fourcc)


def build_pipeline(device, width, height, fps, fourcc):
    """Build a USB pipeline; decode MJPEG in software, then convert to BGR."""
    # Accept Linux video nodes only so a path cannot inject pipeline elements.
    if not re.fullmatch(r'/dev/video[0-9]+', device):
        raise RuntimeError('Use a device path such as /dev/video1.')
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError('Camera dimensions and FPS must be positive.')
    rate = Fraction(str(fps)).limit_denominator(10000)
    caps = f'width={width},height={height},framerate={rate.numerator}/{rate.denominator}'
    if fourcc == 'MJPG':
        # This is the software decoder verified by the standalone camera test.
        decode = f'image/jpeg,{caps} ! jpegparse ! jpegdec ! '
    elif fourcc == 'YUYV':
        decode = f'video/x-raw,format=YUY2,{caps} ! '
    else:
        raise RuntimeError('Supported formats are MJPG and YUYV.')
    # Keep only the newest decoded frame when the tracker falls behind.
    # Dropping frames avoids an accumulating delay; it does not speed up tracking.
    return (f'v4l2src device={device} ! ' + decode +
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink max-buffers=1 drop=true sync=false')


def open_gstreamer(cv, device, width, height, fps, fourcc):
    """Open the pipeline and verify that it delivers a correctly sized BGR frame."""
    pipeline = build_pipeline(device, width, height, fps, fourcc)
    print(f'USB GStreamer requested: {width}x{height} {fps:g} FPS {fourcc}')
    camera = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open USB GStreamer pipeline. Check OpenCV GStreamer support, '
                               'camera mode, device path, and that no other app uses the camera.')
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError('GStreamer opened but returned no camera image.')
        if len(frame.shape) != 3 or frame.shape != (height, width, 3):
            raise RuntimeError(f'Expected {width}x{height} BGR, received {frame.shape}.')
        print('BGR frames verified. Requested FPS is not a measured delivery rate; use --test-camera.')
        return camera
    except Exception:
        camera.release()
        raise


def open_v4l2(cv, device, width, height, fps, fourcc):
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
