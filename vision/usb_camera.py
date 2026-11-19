"""Strict USB/V4L2 mode negotiation; never silently claim 120 FPS."""
import math
import time
from fractions import Fraction
import re
import json
import subprocess
from pathlib import Path


def fourcc_text(code):
    """Decode the driver's FOURCC integer, such as MJPG or YUYV."""
    if not math.isfinite(code):
        raise RuntimeError('Driver returned invalid FOURCC information.')
    value = int(code)
    return ''.join(chr((value >> (8*i)) & 255) for i in range(4))


def open_usb(cv, device, width, height, fps, fourcc, backend='gstreamer', output_size=None, decoder='software'):
    """Choose the requested capture backend without silently falling back."""
    validate_mode(device,width,height,fps,fourcc,output_size)
    if decoder not in ('software','nvv4l2'):
        raise RuntimeError('Decoder must be software or nvv4l2.')
    if decoder=='nvv4l2' and (backend!='gstreamer' or fourcc!='MJPG'):
        raise RuntimeError('nvv4l2 requires GStreamer and MJPG.')
    if backend == 'gstreamer':
        return open_gstreamer(cv, device, width, height, fps, fourcc,output_size,decoder)
    if backend != 'v4l2':
        raise RuntimeError('USB backend must be gstreamer or v4l2.')
    camera=open_v4l2(cv, device, width, height, fps, fourcc)
    return ResizedCapture(cv,camera,output_size,(width,height)) if output_size else camera


def build_pipeline(device, width, height, fps, fourcc, output_size=None, decoder="software"):
    """Build a USB pipeline; decode MJPEG in software, then convert to BGR."""
    validate_mode(device,width,height,fps,fourcc,output_size)
    rate = Fraction(str(fps)).limit_denominator(10000)
    caps = f'width={width},height={height},framerate={rate.numerator}/{rate.denominator}'
    if decoder=='nvv4l2':
        if fourcc!='MJPG':
            raise RuntimeError('nvv4l2 requires MJPG.')
        ow,oh=output_size or (width,height)
        return (f'v4l2src device={device} io-mode=2 ! image/jpeg,{caps} ! '
                'queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                'jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! '
                f'video/x-raw,format=BGRx,width={ow},height={oh},pixel-aspect-ratio=1/1 ! '
                'appsink max-buffers=1 drop=true sync=false')
    if decoder!='software':
        raise RuntimeError('Decoder must be software or nvv4l2.')
    if fourcc == 'MJPG':
        # This is the software decoder verified by the standalone camera test.
        decode = f'image/jpeg,{caps} ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! jpegparse ! jpegdec ! '
    elif fourcc == 'YUYV':
        decode = f'video/x-raw,format=YUY2,{caps} ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
    else:
        raise RuntimeError('Supported formats are MJPG and YUYV.')
    # Keep only the newest decoded frame when the tracker falls behind.
    # Dropping frames avoids an accumulating delay; it does not speed up tracking.
    resize = (f'videoscale add-borders=false ! video/x-raw,width={output_size[0]},height={output_size[1]},pixel-aspect-ratio=1/1 ! '
              if output_size and tuple(output_size)!=(width,height) else '')
    return (f'v4l2src device={device} ! ' + decode + resize +
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink max-buffers=1 drop=true sync=false')


def open_gstreamer(cv, device, width, height, fps, fourcc, output_size=None, decoder="software"):
    """Open the pipeline and verify that it delivers a correctly sized BGR frame."""
    pipeline = build_pipeline(device, width, height, fps, fourcc,output_size,decoder)
    print(f'USB GStreamer requested: {width}x{height} {fps:g} FPS {fourcc}')
    print('Pipeline: '+pipeline)
    camera = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open USB GStreamer pipeline. Check OpenCV GStreamer support, '
                               'camera mode, device path, and that no other app uses the camera.')
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError('GStreamer opened but returned no camera image.')
        ow,oh=output_size or (width,height)
        channels=4 if decoder=='nvv4l2' else 3
        color='BGRx' if channels==4 else 'BGR'
        if len(frame.shape) != 3 or frame.shape != (oh,ow,channels):
            raise RuntimeError(f'Expected {ow}x{oh} {color} output, received {frame.shape}. '
                               'This OpenCV build may not support BGRx appsink input; use --usb-decoder software.')
        print(f'{color} frames verified. Requested FPS is not measured delivery; use --test-camera.')
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


def benchmark(camera, seconds=10, report_path=None, requested_fps=None):
    """Measure delivered decoded frames without the tracker or display workload."""
    try:
        if not math.isfinite(seconds) or seconds<=0:
            raise RuntimeError('Benchmark duration must be positive and finite.')
        shape=None
        for _ in range(10):  # Discard startup frames before timing.
            ok,frame=camera.read()
            if not ok or frame is None:
                raise RuntimeError('Camera read failed during warmup.')
            shape=frame.shape
        start = time.monotonic()
        frames = 0
        while time.monotonic()-start < seconds:
            ok,frame=camera.read()
            if not ok or frame is None:
                raise RuntimeError('Camera read failed during benchmark.')
            if frame.shape!=shape:
                raise RuntimeError('Camera output dimensions changed during benchmark.')
            frames += 1
        elapsed = time.monotonic()-start
        report=dict(frames=frames,elapsed_seconds=elapsed,decoded_fps=frames/elapsed,
                    requested_fps=requested_fps,output_shape=list(shape),
                    scope='Decoded image delivery only; not unique sensor frames, latency or tracker FPS')
        print(f'Delivered/decoded: {frames/elapsed:.1f} FPS over {elapsed:.1f} s.; image shape {shape}')
        if requested_fps and frames/elapsed < requested_fps*.9:
            print('Below requested rate: check camera timing, decoder/conversion cost and USB diagnostics.')
        if report_path:
            Path(report_path).expanduser().write_text(json.dumps(report,indent=2)+'\n')
            print(f'Saved benchmark: {report_path}')
        print('This does not measure exposure latency, unique sensor frames, or tracker FPS.')
        return report
    finally:
        camera.release()


def processing_size(width,height,max_width=960):
    """Keep the complete frame and its exact integer aspect ratio; zero is native."""
    if width<=0 or height<=0 or max_width<0:
        raise RuntimeError('Dimensions must be positive; process width must be nonnegative.')
    divisor=math.gcd(width,height)
    uw,uh=width//divisor,height//divisor
    factor=divisor if max_width==0 else max(1,min(divisor,max_width//uw))
    return (uw*factor,uh*factor)


def validate_mode(device,width,height,fps,fourcc,output_size=None):
    if not re.fullmatch(r'/dev/video[0-9]+',device):
        raise RuntimeError('Use a device path such as /dev/video1.')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (width,height)):
        raise RuntimeError('Camera dimensions must be positive integers.')
    if not math.isfinite(fps) or fps<=0 or fourcc not in ('MJPG','YUYV'):
        raise RuntimeError('Use positive finite FPS and MJPG or YUYV format.')
    if output_size is not None:
        if len(output_size)!=2 or any(not isinstance(v,int) or v<=0 for v in output_size):
            raise RuntimeError('Output dimensions must be positive integers.')
        ow,oh=output_size
        if ow*height!=oh*width or ow>width or oh>height:
            raise RuntimeError('Output must preserve capture aspect ratio without upscaling.')


class ResizedCapture:
    """Match GStreamer output sizing when explicitly selecting V4L2."""
    def __init__(self,cv,camera,size,capture_size):
        self.cv=cv;self.camera=camera;self.size=tuple(size);self.capture_size=tuple(capture_size)
    def isOpened(self):return self.camera.isOpened()
    def release(self):self.camera.release()
    def read(self):
        ok,frame=self.camera.read()
        if ok and frame is not None and frame.shape!=(self.capture_size[1],self.capture_size[0],3):
            raise RuntimeError('USB capture shape changed; recalibrate for the new mode.')
        if ok and frame is not None and frame.shape[:2]!=(self.size[1],self.size[0]):
            frame=self.cv.resize(frame,self.size,interpolation=self.cv.INTER_AREA)
        return ok,frame


def set_manual_exposure(device,microseconds):
    """Explicit opt-in UVC control, in the standard 100 microsecond units."""
    if not re.fullmatch(r'/dev/video[0-9]+',device):
        raise RuntimeError('Use a device path such as /dev/video1.')
    if microseconds<=0 or microseconds%100:
        raise RuntimeError('Exposure must be a positive multiple of 100 microseconds.')
    units=microseconds//100
    try:
        for control in ('auto_exposure=1',f'exposure_time_absolute={units}'):
            subprocess.run(['v4l2-ctl','-d',device,'--set-ctrl='+control],check=True,
                           capture_output=True,text=True,timeout=5)
        result=subprocess.run(['v4l2-ctl','-d',device,'--get-ctrl=auto_exposure,exposure_time_absolute'],
                              check=True,capture_output=True,text=True,timeout=5)
        if not re.search(r'auto_exposure:\s*1\b',result.stdout) or not re.search(r'exposure_time_absolute:\s*'+str(units)+r'\b',result.stdout):
            raise RuntimeError('Exposure readback did not match the request: '+result.stdout)
        print(f'Manual exposure verified: {microseconds} us. '+result.stdout.strip())
    except (OSError,subprocess.SubprocessError) as error:
        detail=getattr(error,'stderr','') or str(error)
        raise RuntimeError('Could not set UVC exposure. Install v4l-utils and check supported controls. '+detail) from error
