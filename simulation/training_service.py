"""Install/control a resumable user service on the Linux simulation PC."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def unit_text(python,script,working,mode,run,gpu=False,counts=None):
    def quote(value):return json.dumps(str(value).replace('%','%%').replace('$','$$'))
    parts=[python,'-u',script,'--run-dir',run,'--resume']
    if not gpu or mode!='movement':parts+=['--mode',mode]
    if gpu and counts:parts+=['--num-envs',str(counts[0]),'--cpu-envs',str(counts[1])]
    command=' '.join(quote(v) for v in parts)
    return f'''[Unit]
Description=Air hockey {mode} simulation training
StartLimitIntervalSec=0

[Service]
Type={'notify' if gpu else 'simple'}
{'NotifyAccess=main' if gpu else ''}
{'WatchdogSec=900' if gpu else ''}
ExecCondition=/usr/bin/test ! -e {quote(Path(run)/'STOP')}
WorkingDirectory={quote(working)}
ExecStart={command}
Restart=on-failure
RestartSec=60
TimeoutStartSec=900
TimeoutStopSec=180
KillMode=control-group
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','stop','status'])
    p.add_argument('--mode',choices=['movement','selfplay','league'],required=True)
    p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    if not sys.platform.startswith('linux'):p.error('This helper is for the Linux simulation PC')
    unit=f'airhockey-{args.mode}.service'
    if args.action=='start':
        if not args.run_dir:p.error('start needs --run-dir with a successfully tested checkpoint')
        run=args.run_dir.resolve()
        from train_skills import Store
        gpu=(run/'mjx_run.json').exists()
        if gpu:
            state=json.loads((run/'mjx_run.json').read_text());state.setdefault('mode','movement')
            if not (run/'latest.zip').exists():p.error('Run must have a tested checkpoint')
        else:state,_=Store(run).load()
        if state['mode']!=args.mode:p.error('Run mode differs')
        if (run/'STOP').exists():p.error('Remove the STOP file before starting again')
        # A running service must be stopped before its definition is changed.
        if subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:p.error('Service already running; stop it first')
        root=Path(__file__).resolve().parent
        path=Path.home()/'.config/systemd/user'/unit;path.parent.mkdir(parents=True,exist_ok=True)
        python=root/'.venv-mjx/bin/python' if gpu else Path(sys.executable)
        script=root/('train_mjx_movement.py' if args.mode=='movement' else 'train_gpu_league.py') if gpu else root/('train_league.py' if args.mode=='league' else 'train_skills.py')
        if gpu and not python.exists():p.error('Missing .venv-mjx/bin/python')
        path.write_text(unit_text(python,script,root,args.mode,run,gpu,(state.get('num_envs',64),state.get('cpu_envs',0))))
        from skills_controls import atomic_json
        atomic_json(run/'service.json',{'unit':unit})
        subprocess.run(['systemctl','--user','daemon-reload'],check=True)
        subprocess.run(['systemctl','--user','reset-failed',unit],check=False)
        subprocess.run(['systemctl','--user','enable','--now','--no-block',unit],check=True)
        print(f'Started {unit}. Check: journalctl --user -u {unit} -f')
        print('For logout/reboot persistence, enable lingering for your user with loginctl enable-linger "$USER".')
    elif args.action=='stop':
        subprocess.run(['systemctl','--user','disable','--now','--no-block',unit],check=True)
    else:subprocess.run(['systemctl','--user','status',unit,'--no-pager'],check=False)

if __name__=='__main__':main()
