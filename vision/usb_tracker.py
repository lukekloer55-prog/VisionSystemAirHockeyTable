#!/usr/bin/env python3
"""Shortcut for the ELP USB camera; all normal tracker arguments remain available."""
import sys
from airhockey_shot_tracker import main

if __name__=='__main__':
    # Later explicit flags override these defaults through argparse.
    sys.argv[1:1]=['--camera-source','usb','--usb-device','/dev/video1',
                  '--usb-backend','gstreamer','--usb-width','1280','--usb-height','720',
                  '--usb-fps','120','--usb-fourcc','MJPG']
    main()
