"""Install/control a resumable user service on the Linux simulation PC."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def unit_text(python,script,working,mode,run):
    def quote(value):return json.dumps(str(value).replace('%','%%').replace('$','$$'))
    command=' '.join(quote(v) for v in [python,'-u',script,'--mode',mode,'--run-dir',run,'--resume'])
    return f'''[Unit]
Description=Air hockey {mode} simulation training
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory={quote(working)}
ExecStart={command}
Restart=on-failure
RestartSec=15
TimeoutStopSec=120
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','stop','status'])
    p.add_argument('--mode',choices=['movement','selfplay'],required=True)
    p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    if not sys.platform.startswith('linux'):p.error('This helper is for the Linux simulation PC')
    unit=f'airhockey-{args.mode}.service'
    if args.action=='start':
        if not args.run_dir:p.error('start needs --run-dir with a successfully tested checkpoint')
        run=args.run_dir.resolve()
        from train_skills import Store
        state,_=Store(run).load()
        if state['mode']!=args.mode:p.error('Run mode differs')
        if (run/'STOP').exists():p.error('Remove the STOP file before starting again')
        # A running service must be stopped before its definition is changed.
        if subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:p.error('Service already running; stop it first')
        root=Path(__file__).resolve().parent
        path=Path.home()/'.config/systemd/user'/unit;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(unit_text(sys.executable,root/'train_skills.py',root,args.mode,run))
        subprocess.run(['systemctl','--user','daemon-reload'],check=True)
        subprocess.run(['systemctl','--user','reset-failed',unit],check=False)
        subprocess.run(['systemctl','--user','enable','--now',unit],check=True)
        print(f'Started {unit}. Check: journalctl --user -u {unit} -f')
        print('For logout/reboot persistence, enable lingering for your user with loginctl enable-linger "$USER".')
    elif args.action=='stop':
        subprocess.run(['systemctl','--user','disable','--now',unit],check=True)
    else:subprocess.run(['systemctl','--user','status',unit,'--no-pager'],check=False)

if __name__=='__main__':main()
