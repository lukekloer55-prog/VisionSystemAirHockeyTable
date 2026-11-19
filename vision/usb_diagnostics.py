#!/usr/bin/env python3
"""Read-only Linux USB camera diagnostics; no capture modes or controls are changed."""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='/dev/video1')
    parser.add_argument('--output',default='usb_diagnostics.txt')
    args=parser.parse_args()
    if not re.fullmatch(r'/dev/video[0-9]+',args.device):
        parser.error('Use a device such as /dev/video1.')
    commands=[['uname','-r'],['lsusb','-t'],['v4l2-ctl','--list-devices'],
              ['v4l2-ctl','-d',args.device,'--all'],
              ['v4l2-ctl','-d',args.device,'--list-formats-ext'],
              ['v4l2-ctl','-d',args.device,'--list-ctrls-menus']]
    commands += [['gst-inspect-1.0',name] for name in
                 ('v4l2src','jpegparse','jpegdec','videoscale','videoconvert','appsink')]
    sections=[]
    for command in commands:
        print('Checking: '+' '.join(command),flush=True)
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            content=f'Exit {result.returncode}\n'+result.stdout+result.stderr
        except (OSError,subprocess.SubprocessError) as error:
            content=str(error)
        sections.append('$ '+' '.join(command)+'\n'+content)
    try:
        import cv2
        sections.append('OpenCV '+cv2.__version__+'\n'+cv2.getBuildInformation())
    except ImportError as error:
        sections.append('OpenCV unavailable: '+str(error))
    Path(args.output).expanduser().write_text('\n\n'.join(sections))
    print('Saved '+str(Path(args.output).expanduser().resolve()))
    print('This reports advertised modes and software support, not measured camera FPS.')

if __name__=='__main__':main()
