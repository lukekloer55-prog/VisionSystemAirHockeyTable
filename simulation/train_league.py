"""Parallel candidate learning and frozen historical-opponent league evaluation."""
import argparse
from dataclasses import asdict
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import signal
import tempfile
import time
import traceback
from skills_env import SkillsEnv, Curriculum, SCHEMA
from airhockey_env import Settings
from parallel_training import WorkerEnv
from train_skills import Store,RunLock,play_match

class LeagueEnv(WorkerEnv):
    def set_pool(self,paths):
        from stable_baselines3 import PPO
        self.pool=[PPO.load(path,device='cpu') for path in paths]
    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        if getattr(self,'pool',None):self.opponent=self.pool[int(self.np_random.integers(len(self.pool)))]
        return observation,info

def make_env(settings,curriculum,paths,render):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    env=LeagueEnv('game',Settings(**settings),Curriculum(**curriculum),'human' if render else None)
    env.set_pool(paths)
    return Monitor(env)

def learn_candidate(job,stop_event):
    """One independent optimizer; subordinate simulation workers share only its policy."""
    env=None
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
        factories=[partial(make_env,job['settings'],job['curriculum'],job['opponents'],job['render']) for _ in range(job['workers'])]
        env=SubprocVecEnv(factories,start_method='spawn') if len(factories)>1 else DummyVecEnv(factories)
        learner=PPO.load(job['model'],env=env,device='cpu')
        learner.set_random_seed(job['seed'])
        class Callback(BaseCallback):
            def _on_step(self):
                if stop_event.is_set() or Path(job['run'],'STOP').exists():return False
                if job['render']:
                    if not all(env.env_method('preview')):stop_event.set();return False
                    time.sleep(job['settings']['control_dt'])
                return True
        before=learner.num_timesteps
        if not stop_event.is_set():learner.learn(total_timesteps=job['steps'],callback=Callback(),reset_num_timesteps=False)
        learner.save(Path(job['output'])/'candidate.zip')
        (Path(job['output'])/'result.json').write_text(json.dumps({'steps':learner.num_timesteps-before}))
    except BaseException:
        stop_event.set()
        (Path(job['output'])/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:env.close()

def schedule(candidates,history):
    players=['champion',*candidates]
    return [(a,b) for i,a in enumerate(players) for b in players[i+1:]]+[(a,b) for a in players for b in history]

def promotion(candidates,history,results,rallies):
    players=['champion',*candidates];points={k:0. for k in players};historic={k:0. for k in players};head={}
    for result in results:
        a,b=result['pair'];sa,sb=result['scores'];draw=(rallies-sa-sb)/2
        if a in points:points[a]+=sa+draw
        if b in points:points[b]+=sb+draw
        if b in history:historic[a]+=sa+draw
        if a=='champion' and b in candidates:head[b]=sb>sa
    eligible=[k for k in candidates if head.get(k,False) and points[k]>points['champion'] and historic[k]>=historic['champion']]
    winner=max(eligible,key=lambda k:(points[k],historic[k],-candidates.index(k))) if eligible else 'champion'
    return {'retained':winner,'points':points,'historical_points':historic,'matches':results}

def retain_champion(state,models,chosen):
    """Keep the initial anchor and at most four later historical champions."""
    if chosen!='champion':
        old=f"history_{state['champion_id']:06d}"
        if old not in state['history']:
            state['history'].append(old);models[old]=models['champion']
        state['history']=[state['history'][0],*state['history'][1:][-4:]]
        models['champion']=models[chosen];state['champion_id']=state['round']+1
    return {k:models[k] for k in ['champion',*state['history']]}

def train_batch(run,folder,state,candidates,stop,render):
    ctx=mp.get_context('spawn');event=ctx.Event();processes=[];jobs=[]
    with tempfile.TemporaryDirectory(prefix='.league-batch-',dir=run) as temp:
        try:
            for index,key in enumerate(candidates):
                remaining=state['steps_per_round']-state['candidate_steps'][key]
                if remaining<=0:continue
                out=Path(temp)/key;out.mkdir()
                model=folder/(key+'.zip')
                job=dict(model=str(model if model.exists() else folder/'champion.zip'),output=str(out),run=str(run.resolve()),
                         settings=state['settings'],curriculum=state['curriculum'],workers=state['num_envs']//state['candidates'],
                         opponents=[str(folder/(k+'.zip')) for k in ['champion',*state['history']]],render=render,
                         steps=min(remaining,state['checkpoint_steps']),seed=state['seed']+state['round']*100003+index*1009+state['candidate_steps'][key])
                proc=ctx.Process(target=learn_candidate,args=(job,event));proc.start();processes.append(proc);jobs.append((key,out))
            while any(p.is_alive() for p in processes):
                if stop():event.set()
                for proc in processes:proc.join(timeout=.05)
            errors=[(out/'error.txt').read_text() for _,out in jobs if (out/'error.txt').exists()]
            if errors or any(p.exitcode!=0 for p in processes):raise RuntimeError('Candidate worker failed; last committed batch is resumable.\n'+'\n'.join(errors))
            from stable_baselines3 import PPO
            learned={};counts={}
            for key,out in jobs:
                learned[key]=PPO.load(out/'candidate.zip',device='cpu');counts[key]=json.loads((out/'result.json').read_text())['steps']
            return learned,counts,event.is_set()
        finally:
            event.set()
            for proc in processes:
                proc.join(timeout=30)
                if proc.is_alive():proc.terminate();proc.join(timeout=5)


def main():
    import sys
    if '--performance' in sys.argv:
        from performance import main as performance_main
        performance_main();return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['league'],default='league')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--initial-run',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--curriculum',type=Path)
    p.add_argument('--num-envs',type=int,default=None,help='Total simulation worker budget, divisible by candidate count')
    p.add_argument('--candidates',type=int,default=None)
    p.add_argument('--steps-per-round',type=int,default=8192);p.add_argument('--checkpoint-steps',type=int,default=8192)
    p.add_argument('--match-rallies',type=int,default=8);p.add_argument('--rounds',type=int,default=0)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.candidates is not None and not 2<=args.candidates<=8:p.error('candidates must be 2..8')
    if args.num_envs is not None and not 2<=args.num_envs<=32:p.error('num-envs must be 2..32')
    if args.resume and (args.initial_run or args.config or args.curriculum):p.error('Resume restores configuration; omit initial/config/curriculum')
    if args.rounds<0 or min(args.steps_per_round,args.checkpoint_steps)<=0 or args.match_rallies<2 or args.match_rallies%2:p.error('Positive step counts, nonnegative rounds and even match-rallies >=2 required')
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    geometry=json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
    run=args.run_dir.resolve();stopped=[False]
    def stop_signal(*_):stopped[0]=True
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    def stop():return stopped[0] or (run/'STOP').exists()
    if not args.resume:run.mkdir(parents=True,exist_ok=False)
    with RunLock(run/'trainer.lock'):
        store=Store(run);env=None
        try:
            if args.resume:
                state,folder=store.load()
                if state['mode']!='league' or state['schema']!=SCHEMA or state['geometry']!=geometry:p.error('Run mode, geometry or schema differs')
                if args.candidates is not None and args.candidates!=state['candidates']:p.error('Candidate count cannot change on resume')
                models={key:PPO.load(path,device='cpu') for path in folder.glob('*.zip') for key in [path.stem]}
            else:
                settings=Settings.load(args.config);curr=Curriculum(**json.loads(args.curriculum.read_text())).validate() if args.curriculum else Curriculum()
                state=dict(mode='league',schema=SCHEMA,geometry=geometry,settings=asdict(settings),curriculum=asdict(curr),
                           round=0,phase='league_start',seed=args.seed,candidates=args.candidates or 2,num_envs=args.num_envs or 2,
                           steps_per_round=args.steps_per_round,checkpoint_steps=args.checkpoint_steps,match_rallies=args.match_rallies,
                           total_steps=0,history=['history_000000'],champion_id=0,candidate_steps={},evaluation=[])
                env=SkillsEnv('game',settings,curr)
                if args.initial_run:
                    source,source_folder=Store(args.initial_run).load()
                    if source['schema']!=SCHEMA or source['geometry']!=geometry or source['settings']!=state['settings']:p.error('Initial model geometry/schema/settings differs')
                    champion=PPO.load(source_folder/'champion.zip',device='cpu')
                    state['lineage']=[*source.get('lineage',[]),{'run':args.initial_run.name,'mode':source['mode'],'steps':source.get('total_steps',0)}]
                else:champion=PPO('MlpPolicy',env,device='cpu',seed=args.seed,n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,gamma=settings.gamma,policy_kwargs={'net_arch':[64,64]})
                models={'champion':champion,'history_000000':champion}
            state['num_envs']=args.num_envs if args.num_envs is not None else state['num_envs']
            if not 2<=state['candidates']<=8 or not state['candidates']<=state['num_envs']<=32 or state['num_envs']%state['candidates']:p.error('Use 2..8 candidates and a total worker budget <=32 divisible by candidates')
            if env is not None:env.close()
            env=SkillsEnv('game',Settings(**state['settings']),Curriculum(**state['curriculum']),'human' if args.render else None)
            folder=store.save(state,models)
            start_round=state['round'];keys=[f'candidate_{i}' for i in range(state['candidates'])]
            print(f"League: {len(keys)} candidates / {state['num_envs']} total simulations. Rendering: {args.render}",flush=True)
            while not stop() and (not args.rounds or state['round']-start_round<args.rounds):
                if state['phase']=='league_start':
                    from skills_controls import apply_pending
                    updated=apply_pending(env,state,models,store)
                    if updated:folder=updated
                    state['candidate_steps']={k:0 for k in keys};state['evaluation']=[];state['phase']='league_train'
                    folder=store.save(state,models)
                if state['phase']=='league_train':
                    began=time.monotonic();learned,counts,halt=train_batch(run,folder,state,keys,stop,args.render)
                    models.update(learned)
                    for key,count in counts.items():state['candidate_steps'][key]+=count;state['total_steps']+=count
                    state['training_steps_per_second']=sum(counts.values())/max(time.monotonic()-began,1e-9)
                    if all(n>=state['steps_per_round'] for n in state['candidate_steps'].values()):state['phase']='league_evaluate'
                    folder=store.save(state,models)
                    print(f"Saved league {state['round']}: {state['candidate_steps']}; {state['training_steps_per_second']:.1f} total steps/s including worker startup",flush=True)
                    if halt:stopped[0]=True
                else:
                    pairs=schedule(keys,state['history'])
                    for index in range(len(state['evaluation']),len(pairs)):
                        a,b=pairs[index]
                        result=play_match(env,models[a],models[b],state['seed']+state['round']*997,state['match_rallies'],stop,args.render)
                        if result is None:stopped[0]=True;break
                        state['evaluation'].append({'pair':[a,b],**result});folder=store.save(state,models)
                        print(f"Evaluation {index+1}/{len(pairs)}: {a} {result['scores']} {b}",flush=True)
                    if stop():break
                    summary=promotion(keys,state['history'],state['evaluation'],state['match_rallies']);chosen=summary['retained']
                    models=retain_champion(state,models,chosen)
                    state['last_league']=summary;state['round']+=1;state['phase']='league_start'
                    folder=store.save(state,models);store.event({'event':'league','round':state['round'],**summary})
                    print(f"League round {state['round']}: retained {chosen}; points {summary['points']}",flush=True)
            print('League saved and stopped. Resume with --resume. Default rounds=0 runs until stopped.',flush=True)
        finally:
            if env is not None:env.close()

if __name__=='__main__':main()
