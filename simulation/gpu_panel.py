"""GPU movement dashboard helpers; shares process ownership with CPU panel."""
import json,os,subprocess,threading,io,zipfile,sys
from pathlib import Path
from dataclasses import asdict
from skills_env import Curriculum
from skills_controls import validate,atomic_json,controls

def state(c):
    runs=[]
    for path in c.root.iterdir():
        if not path.is_dir() or path.is_symlink() or not (path/'mjx_run.json').exists():continue
        try:
            c.path(path.name)
            row={'name':path.name,'running':c.running(path),'checkpoint':(path/'latest.zip').exists()}
            row.update(json.loads((path/'mjx_run.json').read_text()))
            if (path/'progress.json').exists():row.update(json.loads((path/'progress.json').read_text()))
            log=path.with_name(path.name+'.trainer.log')
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-10000));row['log']=f.read().decode(errors='replace')
            if c.owned_run==path and c.process is not None:row['exit_code']=c.process.poll()
            runs.append(row)
        except (ValueError,OSError):continue
    sources=[]
    for row in c.state()['runs']:
        if row.get('mode')=='movement' and row.get('checkpoint'):sources.append({'name':row['name'],'label':'CPU: '+row.get('display_name',row['name']),'curriculum':row.get('curriculum',{})})
    sources += [{'name':r['name'],'label':'GPU/hybrid: '+r['name'],'curriculum':r.get('curriculum',{})} for r in runs if r.get('checkpoint')]
    return {'sources':sources,'cpu_count':min(32,os.cpu_count() or 1),'runs':runs,'controls':controls('movement'),'defaults':asdict(Curriculum())}

def action(c,kind,data):
    with c.lock:
        path=c.path(data['name'])
        if kind=='start':
            if (c.process is not None and c.process.poll() is None) or any(c.running(p) for p in c.root.iterdir() if p.is_dir()):raise ValueError('Stop and save the active trainer first')
            count=data.get('num_envs',64)
            if type(count)!=int or not 1<=count<=256:raise ValueError('Use 1..256 simulations')
            python=c.code/'.venv-mjx'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            if not python.exists():raise ValueError('Install .venv-mjx and requirements-mjx-training.txt first')
            cpu=data.get('cpu_envs',0)
            if type(cpu)!=int or not 0<=cpu<=min(32,os.cpu_count() or 1):raise ValueError('CPU simulations must fit available cores (max 32)')
            args=[str(python),'-u',str(c.code/'train_mjx_movement.py'),'--run-dir',str(path),'--num-envs',str(count),'--cpu-envs',str(cpu)]
            if data.get('resume'):
                if not (path/'mjx_run.json').exists() or not (path/'latest.zip').exists():raise ValueError('No saved GPU checkpoint to resume')
                args+=['--resume']
            else:
                if path.exists():raise ValueError('Name already exists; select Resume or use a new name')
                values=validate('movement',data['values']);config=c.root/'.ui-configs'/(path.name+'.mjx.json')
                config.parent.mkdir(exist_ok=True);atomic_json(config,{**asdict(Curriculum()),**values})
                args+=['--curriculum',str(config)]
                if data.get('initial_run'):
                    source=c.path(data['initial_run'])
                    if c.running(source):raise ValueError('Stop and save the source run before copying it')
                    args+=['--initial-run',str(source)]
            opts={'cwd':str(c.code),'stdin':subprocess.DEVNULL,'stdout':subprocess.PIPE,'stderr':subprocess.STDOUT}
            if os.name=='nt':opts['creationflags']=subprocess.CREATE_NO_WINDOW
            c.process=subprocess.Popen(args,**opts);c.owned_run=path
            c.log_thread=threading.Thread(target=c._logs,args=(c.process,path),daemon=True);c.log_thread.start()
            return {'message':'Starting GPU movement. Compilation can take minutes; training continues until Stop & save.'}
        if not (path/'mjx_run.json').exists():raise ValueError('Select a GPU run')
        if kind in ('watch','replay'):
            if kind=='replay' and not (path/'latest.zip').exists():raise ValueError('Save a checkpoint first')
            if kind=='watch' and not c.running(path):raise ValueError('Start training for live view, or use Replay saved model')
            c.viewers=[v for v in c.viewers if v.poll() is None]
            if c.viewers:raise ValueError('Close the existing viewer windows first')
            python=c.code/'.venv-mjx'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            command=[str(python),str(c.code/'watch_hybrid.py'),'--run-dir',str(path)]
            if kind=='watch':command+=['--live']
            opts={'cwd':str(c.code),'stdin':subprocess.DEVNULL}
            if os.name=='nt':opts['creationflags']=subprocess.CREATE_NO_WINDOW
            c.viewers.append(subprocess.Popen(command,**opts))
            return {'message':'Opening native simulation viewer windows on the training PC. Close them to disable live sampling.'}
        if kind in ('stop','save'):
            if c.running(path):(path/('STOP' if kind=='stop' else 'SAVE')).touch()
            return {'message':'Requested '+kind+'. Wait for the next training boundary.'}
        if kind=='rewards':
            values=validate('movement',data['values']);atomic_json(path/'pending_rewards.json',{'values':values})
            return {'message':'Rewards queued for the next rollout, or resume.'}
        raise ValueError('Unknown GPU action')

def export(c,name):
    path=c.path(name)
    if not (path/'mjx_run.json').exists() or not (path/'latest.zip').exists():raise ValueError('No saved GPU checkpoint yet')
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('latest.zip','mjx_run.json','progress.json','curriculum.json'):
            p=path/name
            if p.exists():z.write(p,path.name+'/'+name)
    return b.getvalue()
