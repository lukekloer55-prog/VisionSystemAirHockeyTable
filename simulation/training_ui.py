"""Local browser control panel for movement and self-play training."""
import argparse
import gpu_panel
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit,parse_qs
import uuid
import webbrowser
import zipfile
from skills_env import Curriculum,SCHEMA
from skills_controls import controls,validate,atomic_json
from train_skills import Store,RunLock


class TrainingController:
 def __init__(self,root):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
  self.lock=threading.RLock();self.process=None;self.owned_run=None;self.log_thread=None
  self.code=Path(__file__).resolve().parent;self.viewers=[]
 def path(self,name):
  if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):raise ValueError('Use a run name with letters, digits, hyphens or underscores')
  path=self.root/name
  if path.resolve().parent!=self.root:raise ValueError('Run must be directly inside the runs folder')
  return path
 def running(self,path):
  if self.process is not None and self.process.poll() is None and self.owned_run==path:return True
  if not (path/'trainer.lock').exists():return False
  try:
   with RunLock(path/'trainer.lock'):return False
  except RuntimeError:return True
 def snapshot(self,path):
  for _ in range(3):
   try:return Store(path).load()
   except FileNotFoundError:time.sleep(.02)
  raise ValueError('No complete checkpoint yet')
 def state(self):
  with self.lock:
   runs=[]
   for path in sorted(self.root.iterdir()):
    if not path.is_dir() or path.is_symlink() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',path.name):continue
    if (path/'mjx_run.json').exists():continue
    record={'name':path.name,'running':self.running(path)}
    try:
     state,folder=self.snapshot(path)
     if state.get('schema')!=SCHEMA:continue
     record.update(state);record['checkpoint']=folder.name
     try:record['display_name']=json.loads((path/'run_details.json').read_text(encoding='utf-8'))['display_name']
     except (OSError,ValueError,KeyError,TypeError):record['display_name']=path.name
    except (ValueError,KeyError,OSError,json.JSONDecodeError):record['status']='Starting or no checkpoint'
    for file,key in [('pending_rewards.json','queued_rewards'),('applying_rewards.json','applying_rewards')]:
     try:record[key]=json.loads((path/file).read_text())
     except (OSError,ValueError):pass
    record['save_pending']=(path/'SAVE').exists()
    record['stop_pending']=(path/'STOP').exists()
    if path==self.owned_run and self.process is not None:record['exit_code']=self.process.poll()
    runs.append(record)
   return {'runs':runs,'root':str(self.root),'cpu_count':min(32,os.cpu_count() or 1),'defaults':asdict(Curriculum()),'controls':{m:controls(m) for m in ('movement','selfplay','league')}}
 def _logs(self,process,path):
  log=path.with_name(path.name+'.trainer.log')
  with process.stdout:
   for line in iter(process.stdout.readline,b''):
    if log.exists() and log.stat().st_size>10_000_000:os.replace(log,log.with_suffix('.previous.log'))
    with log.open('ab') as stream:stream.write(line)
 def start(self,data):
  with self.lock:
   name=data['name'];path=self.path(name);mode=data['mode']
   if (path/'mjx_run.json').exists():raise ValueError('Open the GPU movement page for this run')
   if mode not in ('movement','selfplay','league'):raise ValueError('Select a valid mode')
   if (self.process is not None and self.process.poll() is None) or any(self.running(p) for p in self.root.iterdir() if p.is_dir()):raise ValueError('Stop the active trainer and wait for its save before starting another mode')
   args=[sys.executable,'-u',str(self.code/('train_league.py' if mode=='league' else 'train_skills.py')),'--mode',mode,'--run-dir',str(path)]
   workers=data.get('num_envs',1)
   if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=min(32,os.cpu_count() or 1):raise ValueError('Choose a worker count within available CPU cores (maximum 32)')
   args+=['--num-envs',str(workers)]
   resume=data.get('resume',False)
   if mode=='league':
    candidates=data.get('candidates',2)
    if isinstance(candidates,bool) or not isinstance(candidates,int) or not 2<=candidates<=8 or workers<candidates or workers%candidates:raise ValueError('League needs 2..8 candidates, and total workers must be divisible by candidates')
    if resume and self.snapshot(path)[0]['candidates']!=candidates:raise ValueError('Resume keeps the saved candidate count')
    args+=['--candidates',str(candidates)]
   if resume:
    state,_=self.snapshot(path)
    if state['mode']!=mode:raise ValueError('A saved run cannot change modes; create a new run using its model')
    if (path/'STOP').exists():(path/'STOP').unlink()
    args+=['--resume']
   else:
    if path.exists():raise ValueError('That run already exists; choose Resume or a new name')
    values=validate(mode,data['values'])
    settings={**asdict(Curriculum()),**values}
    Curriculum(**settings).validate()
    configs=self.root/'.ui-configs';configs.mkdir(exist_ok=True)
    config=configs/(name+'.json');atomic_json(config,settings)
    args+=['--curriculum',str(config)]
    initial=data.get('initial_run')
    if initial:
     initial_path=self.path(initial);source,_=self.snapshot(initial_path)
     if source['schema']!=SCHEMA:raise ValueError('Incompatible initial model')
     physical=configs/(name+'.settings.json');atomic_json(physical,source['settings'])
     args+=['--initial-run',str(initial_path),'--config',str(physical)]
   if data.get('render'):args+=['--render']
   options={'cwd':str(self.code),'stdin':subprocess.DEVNULL,'stdout':subprocess.PIPE,'stderr':subprocess.STDOUT}
   if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
   self.process=subprocess.Popen(args,**options);self.owned_run=path
   self.log_thread=threading.Thread(target=self._logs,args=(self.process,path),daemon=True);self.log_thread.start()
   return {'message':'Training starting. Check status below; closing this browser tab does not stop it.'}
 def action(self,action,data):
  if action=='start':return self.start(data)
  with self.lock:
   path=self.path(data['name'])
   if not path.exists():raise ValueError('Run does not exist')
   if action=='rename':
    self.snapshot(path)
    label=data.get('display_name')
    if not isinstance(label,str) or not 1<=len(label.strip())<=100 or any(ord(c)<32 or ord(c)==127 for c in label):raise ValueError('Enter a display name of 1-100 characters without control characters')
    label=label.strip()
    atomic_json(path/'run_details.json',{'display_name':label})
    return {'message':'Renamed to '+label+'. Checkpoints, learned weights and the folder path are unchanged.'}
   if action=='stop':
    if self.running(path):(path/'STOP').touch()
    return {'message':'Stop requested. Wait for Running to change to Stopped; the trainer saves before exiting.'}
   if action=='save':
    if self.running(path):
     (path/'SAVE').touch();return {'message':'Checkpoint requested at the next training boundary; a running match finishes first.'}
    self.snapshot(path);return {'message':'This stopped run already has its last committed checkpoint.'}
   if action=='rewards':
    state,_=self.snapshot(path);values=validate(state['mode'],data['values'])
    atomic_json(path/'pending_rewards.json',{'revision':uuid.uuid4().hex,'values':values})
    return {'message':'Queued. Movement applies at the next learning batch; self-play and league apply at the next competition round. A stopped run applies on resume.'}
   if action=='watch':
    self.snapshot(path)
    options={'cwd':str(self.code),'stdin':subprocess.DEVNULL}
    if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
    self.viewers=[p for p in self.viewers if p.poll() is None]
    if self.viewers:raise ValueError('Close the existing policy viewer before opening another')
    self.viewers.append(subprocess.Popen([sys.executable,str(self.code/'watch_skills.py'),'--run-dir',str(path)],**options))
    return {'message':'Opening a MuJoCo replay window on the training PC desktop. It is separate from the active rollout.'}
   raise ValueError('Unknown action')
 def export(self,name):
  path=self.path(name)
  for attempt in range(3):
   state,folder=self.snapshot(path);buffer=io.BytesIO()
   try:
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
     z.writestr(name+'/CURRENT.json',json.dumps({'snapshot':folder.name}))
     for p in folder.iterdir():
      if p.is_file():z.write(p,name+'/'+folder.name+'/'+p.name)
     for filename in ['events.jsonl','events.previous.jsonl','run_details.json']:
      if (path/filename).exists():z.write(path/filename,name+'/'+filename)
    return buffer.getvalue()
   except FileNotFoundError:
    if attempt==2:raise
 def close(self):
  if self.process is not None and self.process.poll() is None:
   # The child can still be creating its run directory.
   for _ in range(100):
    if self.owned_run.exists() or self.process.poll() is not None:break
    time.sleep(.1)
   if self.owned_run.exists():(self.owned_run/'STOP').touch()
   try:self.process.wait(timeout=120)
   except subprocess.TimeoutExpired:print('Trainer is still saving; inspect it before closing the computer.',flush=True)


class Dashboard:
 def __init__(self,controller,port=0):
  self.controller=controller;self.token=secrets.token_urlsafe(24);dashboard=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def send(self,status,data,kind='application/json',filename=None):
    content=data if isinstance(data,bytes) else data.encode() if isinstance(data,str) else json.dumps(data,allow_nan=False).encode()
    self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(content)))
    self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
    if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
    self.end_headers();self.wfile.write(content)
   def local(self):
    allowed={f'localhost:{dashboard.server.server_port}',f'127.0.0.1:{dashboard.server.server_port}'}
    return self.headers.get('Host') in allowed and self.headers.get('Origin') in (None,*['http://'+h for h in allowed])
   def do_GET(self):
    if not self.local():return self.send(403,{'error':'Local requests only'})
    route=urlsplit(self.path)
    try:
     if route.path=='/':return self.send(200,Path(__file__).with_name('training_panel.html').read_text(encoding='utf-8').replace('__TOKEN__',dashboard.token),'text/html; charset=utf-8')
     if route.path=='/gpu':return self.send(200,Path(__file__).with_name('gpu_panel.html').read_text(encoding='utf-8').replace('__TOKEN__',dashboard.token),'text/html; charset=utf-8')
     if route.path=='/api/gpu/state':return self.send(200,gpu_panel.state(controller))
     if route.path=='/api/gpu/export':return self.send(200,gpu_panel.export(controller,parse_qs(route.query).get('name',[''])[0]),'application/zip','gpu-results.zip')
     if route.path=='/api/state':return self.send(200,controller.state())
     if route.path=='/api/export':
      name=parse_qs(route.query).get('name',[''])[0]
      return self.send(200,controller.export(name),'application/zip',name+'-results.zip')
     return self.send(404,{'error':'Not found'})
    except (ValueError,OSError,KeyError) as e:return self.send(400,{'error':str(e)})
   def do_POST(self):
    if not self.local() or self.headers.get('X-Panel-Token')!=dashboard.token:return self.send(403,{'error':'Unauthorized'})
    try:
     size=int(self.headers.get('Content-Length','0'))
     if not 0<size<=16384:raise ValueError('Invalid request size')
     data=json.loads(self.rfile.read(size));action=self.path.removeprefix('/api/')
     if not isinstance(data,dict):raise ValueError('Expected an object')
     if action.startswith('gpu/'):return self.send(200,gpu_panel.action(controller,action.split('/',1)[1],data))
     return self.send(200,controller.action(action,data))
    except (ValueError,OSError,KeyError,TypeError) as e:return self.send(400,{'error':str(e)})
  self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
  self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}'
 def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runs-dir',type=Path,default=Path('runs'))
 p.add_argument('--port',type=int,default=0)
 p.add_argument('--no-browser',action='store_true')
 args=p.parse_args()
 if not 0<=args.port<=65535:p.error('port must be 0..65535')
 controller=TrainingController(args.runs_dir);dashboard=Dashboard(controller,args.port)
 print('Training control panel:',dashboard.url,flush=True)
 print('Closing the browser leaves training running. Ctrl+C here stops and saves trainers launched by this panel.',flush=True)
 if not args.no_browser:webbrowser.open(dashboard.url)
 try:
  while True:time.sleep(.5)
 except KeyboardInterrupt:pass
 finally:dashboard.close();controller.close()

if __name__=='__main__':main()
