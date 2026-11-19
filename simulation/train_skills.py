"""Continuous movement learning or winner-retains self-play, with resumable snapshots."""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid
import numpy as np
from skills_env import SkillsEnv, Curriculum, SCHEMA
from airhockey_env import Settings


class RunLock:
    """OS lock is released on process death; stale lock files do not block resume."""
    def __init__(self,path):self.path=path
    def __enter__(self):
        self.file=open(self.path,'a+b');self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise RuntimeError('Another trainer owns this run directory')
        return self
    def __exit__(self,*exc):self.file.close()


class Store:
    """Commit model ZIPs and manifest together, then atomically switch CURRENT."""
    def __init__(self,path):self.path=Path(path)
    def remove_snapshot(self, path):
        if path.resolve().parent != self.path.resolve():
            raise ValueError('Snapshot cleanup escaped run directory')
        shutil.rmtree(path)
    def load(self):
        name=json.loads((self.path/'CURRENT.json').read_text())['snapshot']
        if Path(name).name!=name:raise ValueError('Invalid snapshot name')
        folder=self.path/name
        return json.loads((folder/'state.json').read_text()),folder
    def save(self,state,models):
        name='snapshot-'+str(time.time_ns())+'-'+uuid.uuid4().hex[:6]
        tmp=self.path/('.'+name);tmp.mkdir()
        try:
            for key,model in models.items():model.save(tmp/(key+'.zip'))
            (tmp/'state.json').write_text(json.dumps(state,indent=2))
            # Flush model and metadata contents before publishing the snapshot.
            for p in tmp.iterdir():
                with p.open('r+b') as f:os.fsync(f.fileno())
            folder=self.path/name;tmp.rename(folder)
            pointer=self.path/'CURRENT.tmp';pointer.write_text(json.dumps({'snapshot':name}))
            with pointer.open('r+b') as f:os.fsync(f.fileno())
            os.replace(pointer,self.path/'CURRENT.json')
            if os.name!='nt':
                fd=os.open(self.path,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            for old in sorted(self.path.glob('snapshot-*'),reverse=True)[3:]:self.remove_snapshot(old)
            return folder
        finally:
            if tmp.exists():self.remove_snapshot(tmp)
    def event(self,event):
        path=self.path/'events.jsonl'
        if path.exists() and path.stat().st_size>10_000_000:os.replace(path,self.path/'events.previous.jsonl')
        with path.open('a') as f:f.write(json.dumps({'time':datetime.now().isoformat(),**event})+'\n')


def play_match(env,a,b,seed,rallies=8,stop=lambda:False,render=False):
    """Paired serve seeds with swapped ends; only goals/arm forfeits score points."""
    scores=[0,0];outcomes={}
    for index in range(rallies):
        swap=index%2;players=[b,a] if swap else [a,b]
        env.reset(seed=seed+index//2)
        while True:
            if stop():return None
            act=[players[s].predict(env.observation(s),deterministic=True)[0] for s in (0,1)]
            _,_,term,trunc,info=env.step_pair(*act)
            if render:
                env.render()
                if not env.viewer.is_running():return None
            if term or trunc:break
        outcomes[info['outcome']]=outcomes.get(info['outcome'],0)+1
        if info['winner'] is not None:scores[info['winner']^swap]+=1
    winner=0 if scores[0]>scores[1] else 1 if scores[1]>scores[0] else None
    return {'scores':scores,'winner':winner,'outcomes':outcomes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['movement','selfplay'],default='movement')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--initial-model',type=Path,help='Movement champion ZIP for a new self-play run')
    p.add_argument('--initial-run',type=Path,help='Use the champion from a stopped movement run')
    p.add_argument('--config',type=Path,help='Physical Settings JSON, new runs only')
    p.add_argument('--curriculum',type=Path,help='Movement/game reward and angle settings JSON')
    p.add_argument('--rounds',type=int,default=0,help='0 means keep training until stopped')
    p.add_argument('--steps-per-round',type=int,default=8192,help='PPO steps per candidate before a match')
    p.add_argument('--checkpoint-steps',type=int,default=2048,help='Checkpoint after this many learning steps')
    p.add_argument('--match-rallies',type=int,default=8,help='Even number, balanced serve/side pairs')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--render',action='store_true')
    p.add_argument('--render-speed',type=float,default=1.)
    args=p.parse_args()
    if args.rounds<0 or args.steps_per_round<=0 or args.checkpoint_steps<=0 or args.threads<=0 or args.match_rallies<2 or args.match_rallies%2:p.error('Positive step/thread counts and an even match-rallies >=2 are required; rounds >=0')
    if not np.isfinite(args.render_speed) or args.render_speed<=0:p.error('render-speed must be finite and positive')
    if args.resume and (args.initial_model or args.initial_run or args.config or args.curriculum):p.error('Resume restores saved configuration; omit config/curriculum/initial-model')
    if args.initial_run and args.initial_model:p.error('Choose initial-run or initial-model, not both')
    if args.initial_run:args.initial_model=Store(args.initial_run).load()[1]/'champion.zip'
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(args.threads)
    run=args.run_dir
    if args.resume:
        if not (run/'CURRENT.json').exists():p.error('No committed checkpoint in that run directory')
    else:run.mkdir(parents=True,exist_ok=False)
    stopped=[False]
    def stop_signal(*_):stopped[0]=True
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    def stop():return stopped[0] or (run/'STOP').exists()
    class Callback(BaseCallback):
        def __init__(self,env):super().__init__();self.env=env;self.last=time.monotonic();self.history=deque(maxlen=100)
        def _on_step(self):
            for done,info in zip(self.locals.get('dones',[]),self.locals.get('infos',[])):
                if done:self.history.append({'outcome':info.get('outcome'),'success':bool(info.get('is_success')) if self.env.mode=='movement' else info.get('outcome')=='goal' and info.get('winner')==0,'travel_rad':info.get('joint_travel_rad'), 'reward':info.get('episode',{}).get('r')})
            if args.render:
                if self.env.viewer is not None and not self.env.viewer.is_running():stopped[0]=True;return False
                delay=self.env.settings.control_dt/args.render_speed-(time.monotonic()-self.last)
                if delay>0:time.sleep(delay)
                self.env.render();self.last=time.monotonic()
            return not stop()
    geometry=json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
    with RunLock(run/'trainer.lock'):
        store=Store(run);models={};envs=[]
        try:
            if args.resume:
                state,folder=store.load()
                if state['schema']!=SCHEMA or state['geometry']!=geometry:p.error('Checkpoint geometry or observation schema differs')
                if state['mode']!=args.mode:p.error('Use the same --mode as the saved run')
                settings=Settings(**state['settings']).validate();curr=Curriculum(**state['curriculum']).validate()
            else:
                settings=Settings.load(args.config);curr=Curriculum(**json.loads(args.curriculum.read_text())).validate() if args.curriculum else Curriculum()
                state=dict(schema=SCHEMA,geometry=geometry,settings=asdict(settings),curriculum=asdict(curr),mode=args.mode,
                           round=0,phase='train_a',trained=0,seed=args.seed,steps_per_round=args.steps_per_round,
                           checkpoint_steps=args.checkpoint_steps,match_rallies=args.match_rallies,total_steps=0)
            mode='movement' if args.mode=='movement' else 'game'
            env=SkillsEnv(mode,settings,curr,'human' if args.render else None);envs.append(env)
            print('Reachable target region (metres):',env.workspace.min(0), 'to',env.workspace.max(0),flush=True)
            if args.resume:
                for key in ('champion','a','b'):
                    if (folder/(key+'.zip')).exists():models[key]=PPO.load(folder/(key+'.zip'),device='cpu')
            else:
                if args.initial_model:
                    initial_state=args.initial_model.parent/'state.json'
                    if not initial_state.exists():p.error('initial-model needs its adjacent state.json from this trainer')
                    source=json.loads(initial_state.read_text())
                    if source['schema']!=SCHEMA or source['geometry']!=geometry or source['settings']!=asdict(settings):p.error('Initial model geometry/schema/physical settings do not match')
                    models['champion']=PPO.load(args.initial_model,device='cpu')
                else:models['champion']=PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=0,n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,gamma=settings.gamma,policy_kwargs={'net_arch':[64,64]})
                folder=store.save(state,models)
            print(f'Continuous {args.mode} training: {run.resolve()}',flush=True)
            print('Ctrl+C, SIGTERM, closing the viewer, or creating STOP in the run folder saves and stops.',flush=True)
            print('Snapshots keep the latest three generations. No physical motor commands.',flush=True)
            starting_round=state['round'];callback=Callback(env)
            while not stop() and (args.rounds==0 or state['round']-starting_round<args.rounds):
                if state['phase'].startswith('train_'):
                    key=state['phase'][-1]
                    if key not in models:
                        # Separate optimizer states and exploration seeds; both begin at the champion.
                        models[key]=PPO.load(folder/'champion.zip',device='cpu')
                    learner=models[key];learner.set_env(env)
                    env.opponent=models['champion'] if mode=='game' else None
                    learner.set_random_seed(state['seed']+state['round']*100003+(0 if key=='a' else 1009)+state['trained'])
                    remaining=state['steps_per_round']-state['trained']
                    steps=min(state['checkpoint_steps'],remaining)
                    before=learner.num_timesteps
                    if steps>0:
                        learner.learn(total_timesteps=steps,callback=callback,reset_num_timesteps=False)
                    advanced=learner.num_timesteps-before;state['trained']+=advanced;state['total_steps']+=advanced
                    if not stop() and state['trained']>=state['steps_per_round']:
                        state['trained']=0
                        if mode=='movement':
                            models={'champion':learner};state['round']+=1;state['phase']='train_a'
                        else:state['phase']='train_b' if key=='a' else 'match'
                    state['recent_episodes']=list(callback.history)
                    folder=store.save(state,models)
                    if callback.history:
                        metric='target success' if mode=='movement' else 'opponent-goal scoring rate'
                        print(f"Recent episodes={len(callback.history)} {metric}={sum(x['success'] for x in callback.history)/len(callback.history):.1%}",flush=True)
                    print(f"Saved round={state['round']} phase={state['phase']} total_steps={state['total_steps']}",flush=True)
                    store.event({'event':'checkpoint','round':state['round'],'phase':state['phase'],'total_steps':state['total_steps']})
                else:
                    result=play_match(env,models['a'],models['b'],state['seed']+state['round']*997,state['match_rallies'],stop,args.render)
                    if result is None:stopped[0]=True;break
                    chosen=('a','b')[result['winner']] if result['winner'] is not None else 'champion'
                    models={'champion':models[chosen]};state['round']+=1;state['phase']='train_a';state['trained']=0
                    state['last_match']=result
                    folder=store.save(state,models)
                    store.event({'event':'match','round':state['round'],**result})
                    print(f"Match {state['round']}: A {result['scores'][0]} - B {result['scores'][1]}; retained {chosen}",flush=True)
            print('Stopped. Resume with the same mode/run-dir plus --resume. Remove STOP first if present.',flush=True)
        finally:
            for e in envs:e.close()

if __name__=='__main__':main()

