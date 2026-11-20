"""Local dashboard operations; subprocesses use fixed arguments, never shell text."""
import json
import os
import subprocess
import sys

MODES = ('movement', 'selfplay', 'league')

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        raise ValueError('System service command timed out. Refresh status before retrying.')
    if p.returncode:
        raise ValueError((p.stderr or p.stdout or 'System command failed').strip())
    return p.stdout.strip()

def state(c):
    rows = []
    for p in c.root.iterdir():
        if not p.is_dir() or p.is_symlink(): continue
        try:
            c.path(p.name)
            if (p/'mjx_run.json').exists():
                m = json.loads((p/'mjx_run.json').read_text())
                saved = (p/'latest.zip').exists()
            else:
                m, _ = c.snapshot(p); saved = True
            rows.append({'name':p.name,'mode':m.get('mode','movement'), 'running':c.running(p),'checkpoint':saved})
        except (ValueError, OSError, KeyError): continue
    services = []
    linux = sys.platform.startswith('linux')
    linger = 'Unavailable'
    if linux:
        for mode in MODES:
            unit = f'airhockey-{mode}.service'
            try:
                info = command(['systemctl','--user','show',unit,'--property=ActiveState,SubState,UnitFileState'])
            except (ValueError,OSError) as e: info = str(e)
            services.append({'mode':mode,'status':info})
        try: linger = command(['loginctl','show-user',str(os.getuid()),'--property=Linger','--value'])
        except (ValueError,OSError) as e: linger = str(e)
    return {'runs':rows,'linux':linux,'services':services,'linger':linger}

def action(c, kind, data):
    if not sys.platform.startswith('linux'): raise ValueError('Unattended services run on the Linux training PC.')
    with c.lock:
        if kind == 'linger':
            command(['loginctl','enable-linger',str(os.getuid())])
            return {'message':'Logout/reboot persistence enabled for this user.'}
        if kind not in ('start','stop'): raise ValueError('Unknown operation')
        p = c.path(data.get('name',''))
        if (p/'mjx_run.json').exists():
            m=json.loads((p/'mjx_run.json').read_text())
            if not (p/'latest.zip').exists(): raise ValueError('Save a checkpoint first.')
        else: m,_=c.snapshot(p)
        mode=m.get('mode','movement')
        if mode not in MODES: raise ValueError('Invalid run mode')
        if kind == 'start':
            if any(c.running(x) for x in c.root.iterdir() if x.is_dir()):
                raise ValueError('Click Stop & save on the training page and wait for Stopped before enabling unattended operation.')
            for service_mode in MODES:
                status=command(['systemctl','--user','show',f'airhockey-{service_mode}.service','--property=ActiveState','--value'])
                if status in ('active','activating','deactivating','reloading'):
                    raise ValueError('An unattended service is active or changing state. Stop it and wait before starting another.')
            # Explicit start is authorization to resume a previously stopped run.
            (p/'STOP').unlink(missing_ok=True)
        else:
            # Do not stop another run using the same mode's service.
            record=json.loads((p/'service.json').read_text()) if (p/'service.json').exists() else {}
            unit=f'airhockey-{mode}.service'
            if record.get('unit') != unit: raise ValueError('This run has no unattended service. Use its training page to stop it.')
            invocation=command(['systemctl','--user','show',unit,'--property=ExecStart','--value'])
            if str(p) not in invocation: raise ValueError('That service belongs to another run.')
            (p/'STOP').touch()
        message=command([sys.executable,str(c.code/'training_service.py'),kind,'--mode',mode,'--run-dir',str(p)])
        return {'message':message or 'Stop & save requested; service disabled. Wait for training to finish saving.'}
