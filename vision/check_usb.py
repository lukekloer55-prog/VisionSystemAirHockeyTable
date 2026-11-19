#!/usr/bin/env python3
"""USB capture contract tests: no camera, OpenCV or Linux utilities required."""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import usb_camera as usb


class Camera:
    def __init__(self,shape=(720,1280,3),override=None,opened=True,ok=True):
        self.frame=SimpleNamespace(shape=shape);self.override=override or {}
        self.opened=opened;self.ok=ok;self.values={};self.calls=[];self.released=False
    def isOpened(self):return self.opened
    def set(self,k,v):self.calls.append(k);self.values[k]=v;return True
    def get(self,k):return self.override.get(k,self.values[k])
    def read(self):return self.ok,self.frame if self.ok else None
    def release(self):self.released=True


class CV:
    CAP_V4L2=200;CAP_GSTREAMER=1800;CAP_PROP_FOURCC=6
    CAP_PROP_FRAME_WIDTH=3;CAP_PROP_FRAME_HEIGHT=4;CAP_PROP_FPS=5;CAP_PROP_BUFFERSIZE=38;INTER_AREA=3
    def __init__(self,camera):self.camera=camera;self.open_args=None
    def VideoCapture(self,*args):self.open_args=args;return self.camera
    @staticmethod
    def VideoWriter_fourcc(*chars):return sum(ord(c)<<(8*i) for i,c in enumerate(chars))
    @staticmethod
    def resize(frame,size,interpolation):return SimpleNamespace(shape=(size[1],size[0],3))


class USBTests(unittest.TestCase):
    def setUp(self):
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__()
    def tearDown(self):self.output.__exit__(None,None,None)
    def test_exact_aspect(self):
        for w,h in [(1920,1080),(1280,960),(1280,720),(1640,1232),(640,480)]:
            ow,oh=usb.processing_size(w,h)
            self.assertEqual(ow*h,oh*w);self.assertLessEqual(ow,w)
            self.assertEqual(usb.processing_size(w,h,0),(w,h))
        self.assertEqual(usb.processing_size(1920,1080),(960,540))
    def test_pipeline(self):
        p=usb.build_pipeline('/dev/video1',1920,1080,120,'MJPG',(960,540))
        for part in ('image/jpeg,width=1920,height=1080,framerate=120/1','leaky=downstream','jpegdec','videoscale','format=BGR','drop=true'):
            self.assertIn(part,p)
        self.assertNotIn('nvjpegdec',p)
        self.assertNotIn('jpegdec',usb.build_pipeline('/dev/video1',640,480,120,'YUYV'))
        self.assertNotIn('videoscale',usb.build_pipeline('/dev/video1',640,480,120,'MJPG',(640,480)))
    def test_reject_invalid_mode(self):
        for values in [('/dev/video1 ! fakesink',640,480,120,'MJPG'),('/dev/video1',640,480,float('nan'),'MJPG'),('/dev/video1',640,480,120,'BAD!')]:
            with self.assertRaises(RuntimeError):usb.build_pipeline(*values)
        with self.assertRaises(RuntimeError):usb.build_pipeline('/dev/video1',640,480,120,'MJPG',(960,540))
    def test_v4l2_mode_contract(self):
        cam=Camera();cv=CV(cam)
        self.assertIs(usb.open_usb(cv,'/dev/video1',1280,720,120,'MJPG','v4l2'),cam)
        self.assertEqual(cam.calls[:4],[6,3,4,5]);self.assertFalse(cam.released)
    def test_v4l2_rejection_cleanup(self):
        for change in ({5:60},{3:640},{6:CV.VideoWriter_fourcc(*'YUYV')},{6:float('nan')}):
            cam=Camera(override=change)
            with self.assertRaises(RuntimeError):usb.open_usb(CV(cam),'/dev/video1',1280,720,120,'MJPG','v4l2')
            self.assertTrue(cam.released)
    def test_gstreamer_shape_and_cleanup(self):
        cam=Camera(shape=(540,960,3));cv=CV(cam)
        self.assertIs(usb.open_usb(cv,'/dev/video1',1280,720,120,'MJPG',output_size=(960,540)),cam)
        self.assertEqual(cv.open_args[1],cv.CAP_GSTREAMER)
        for cam in (Camera(),Camera(opened=False),Camera(ok=False)):
            with self.assertRaises(RuntimeError):usb.open_usb(CV(cam),'/dev/video1',1280,720,120,'MJPG',output_size=(960,540))
            self.assertTrue(cam.released)
    def test_v4l2_resize_and_changed_source(self):
        cam=Camera();cv=CV(cam)
        capture=usb.open_usb(cv,'/dev/video1',1280,720,120,'MJPG','v4l2',(960,540))
        self.assertEqual(capture.read()[1].shape,(540,960,3))
        cam.frame.shape=(480,640,3)
        with self.assertRaises(RuntimeError):capture.read()
        capture.release();self.assertTrue(cam.released)
    def test_benchmark_report_and_cleanup(self):
        cam=Camera()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder,patch.object(usb.time,'monotonic',side_effect=[0,.01,.02,.03,.04,.05,.06]):
            path=Path(folder)/'report.json';report=usb.benchmark(cam,.04,path,120)
            self.assertEqual(report['frames'],3)
            self.assertEqual(json.loads(path.read_text())['requested_fps'],120)
            self.assertEqual(report['output_shape'],[720,1280,3])
        self.assertTrue(cam.released)
        cam=Camera(ok=False)
        with self.assertRaises(RuntimeError):usb.benchmark(cam)
        self.assertTrue(cam.released)
    def test_launcher_benchmark_end_to_end(self):
        cam=Camera(shape=(540,960,3));cv=CV(cam)
        launcher=Path(__file__).with_name('usb_tracker.py')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'cli.json'
            argv=[str(launcher),'--test-camera','--test-camera-seconds','.04',
                  '--camera-report',str(path),'--usb-width','1920','--usb-height','1080']
            with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'cv2':cv,'numpy':SimpleNamespace()}),patch.object(usb.time,'monotonic',side_effect=[0,.01,.02,.03,.04,.05,.06]):
                runpy.run_path(str(launcher),run_name='__main__')
            self.assertIn('device=/dev/video1',cv.open_args[0])
            self.assertIn('width=1920,height=1080,framerate=120/1',cv.open_args[0])
            self.assertEqual(json.loads(path.read_text())['output_shape'],[540,960,3])
            self.assertTrue(cam.released)

    def test_accelerated_bgrx_contract(self):
        p=usb.build_pipeline('/dev/video0',1920,1080,120,'MJPG',decoder='nvv4l2')
        self.assertIn('nvv4l2decoder mjpeg=1',p)
        self.assertIn('io-mode=2',p);self.assertIn('format=BGRx',p)
        self.assertNotIn('videoconvert',p);self.assertNotIn('nvjpegdec',p)
        cam=Camera(shape=(1080,1920,4))
        self.assertIs(usb.open_usb(CV(cam),'/dev/video0',1920,1080,120,'MJPG',decoder='nvv4l2'),cam)
        cam=Camera(shape=(1080,1920,3))
        with self.assertRaises(RuntimeError):usb.open_usb(CV(cam),'/dev/video0',1920,1080,120,'MJPG',decoder='nvv4l2')
        self.assertTrue(cam.released)
        with self.assertRaises(RuntimeError):usb.open_usb(CV(cam),'/dev/video0',1920,1080,120,'MJPG',backend='v4l2',decoder='nvv4l2')
        with self.assertRaises(RuntimeError):usb.build_pipeline('/dev/video0',1920,1080,120,'YUYV',decoder='nvv4l2')
    def test_display_schedule_and_profile(self):
        from vision_timing import DisplaySchedule,FrameProfiler
        schedule=DisplaySchedule(30)
        self.assertTrue(schedule.due(0));self.assertFalse(schedule.due(.008));self.assertTrue(schedule.due(.034))
        every=DisplaySchedule(0)
        self.assertTrue(every.due(0));self.assertTrue(every.due(.001))
        ticks=iter([0,0,.01,.02,.03,.03])
        profiler=FrameProfiler(True,.02,clock=lambda:next(ticks))
        profiler.begin();profiler.mark('capture');profiler.mark('detect');profiler.end()
        self.assertEqual(profiler.frames,0)

    def test_exposure_readback(self):
        result=SimpleNamespace(stdout='auto_exposure: 1 (Manual Mode)\nexposure_time_absolute: 20\n')
        with patch.object(usb.subprocess,'run',return_value=result) as run:
            usb.set_manual_exposure('/dev/video1',2000)
            self.assertEqual(run.call_args_list[1].args[0][-1],'--set-ctrl=exposure_time_absolute=20')
        with patch.object(usb.subprocess,'run',return_value=SimpleNamespace(stdout='auto_exposure: 3\n')):
            with self.assertRaises(RuntimeError):usb.set_manual_exposure('/dev/video1',2000)
        with self.assertRaises(RuntimeError):usb.set_manual_exposure('/dev/video1',2030)


if __name__=='__main__':unittest.main(verbosity=2)
