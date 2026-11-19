#!/usr/bin/env python3
"""Jetson 1080p MJPG tracking with the optional nvv4l2/BGRx capture path."""
import sys
from airhockey_shot_tracker import main

if __name__=='__main__':
    sys.argv[1:1]=['--camera-source','usb','--usb-device','/dev/video0',
                  '--usb-backend','gstreamer','--usb-decoder','nvv4l2',
                  '--usb-width','1920','--usb-height','1080','--usb-process-width','0',
                  '--usb-fps','120','--usb-fourcc','MJPG','--display-fps','30','--profile']
    main()
