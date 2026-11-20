"""Resumable MJX/hybrid self-play and historical league. Default: unlimited rounds."""
import argparse,json,os,signal,time,shutil,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

def heartbeat(run):
    now=time.monotonic()
    if now-getattr(heartbeat,'last',0)<2:return
    heartbeat.last=now
    (run/'heartbeat').touch()
    address=os.environ.get('NOTIFY_SOCKET')
    if address:
        import socket
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as sock:sock.sendto(b'READY=1\nWATCHDOG=1',('\0'+address[1:]) if address.startswith('@') else address)
        except OSError:pass

def evaluate(env,a,b,seed,rallies,stop):
    """Paired serves and swapped sides; every rally counted once before auto-reset."""
    scores=[0,0];outcomes={};finished=0
    while finished<rallies:
        n=min(env.num_envs,rallies-finished)
        env._seeds=[seed+(finished+i)//2 for i in range(env.num_envs)]
        obs=env.reset();env.pool=[b,a];env.assignment=np.arange(env.num_envs)%2;done=np.zeros(env.num_envs,bool)
        for _ in range(round(env.envs[0].curriculum.rally_seconds/env.envs[0].settings.control_dt)+2):
            if stop():return None
            actions=np.zeros((env.num_envs,2),np.float32)
            for role,policy in enumerate((a,b)):
                ids=np.arange(role,env.num_envs,2);actions[ids]=policy.predict(obs[ids],deterministic=True)[0]
            obs,_,ends,infos=env.step(actions)
            # Resetting worlds must keep their assigned evaluator, not sample the pool.
            env.assignment=np.arange(env.num_envs)%2
            for i in range(n):
                if ends[i] and not done[i]:
                    info=infos[i];outcomes[info['outcome']]=outcomes.get(info['outcome'],0)+1
                    if info['winner'] is not None:scores[info['winner']^(i%2)]+=1
                    done[i]=True
            if done[:n].all():break
        if not done[:n].all():raise RuntimeError('Evaluation exceeded bounded rally duration')
        finished+=n
    return {'scores':scores,'winner':0 if scores[0]>scores[1] else 1 if scores[1]>scores[0] else None,'outcomes':outcomes}

def main():
    if '--performance' in sys.argv:
        from performance import main as perf
        perf();return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['selfplay','league'],default='league');p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--initial-run',type=Path);p.add_argument('--curriculum',type=Path)
    p.add_argument('--num-envs',type=int);p.add_argument('--cpu-envs',type=int);p.add_argument('--candidates',type=int,default=2)
    p.add_argument('--steps-per-round',type=int,default=8192);p.add_argument('--match-rallies',type=int,default=8)
    p.add_argument('--rounds',type=int,default=0);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--seed',type=int,default=1)
    args=p.parse_args()
    if not 2<=args.candidates<=8 or args.steps_per_round<1 or args.match_rallies<2 or args.match_rallies%2 or args.rounds<0:p.error('2..8 candidates, positive steps, even rallies >=2, rounds >=0')
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    torch.set_num_threads(1)
    from skills_env import SkillsEnv,Curriculum,SCHEMA
    from airhockey_env import Settings
    from skills_controls import atomic_json,validate
    from train_skills import Store,RunLock
    from train_league import schedule,promotion,retain_champion
    from league_names import hero,rankings
    from mjx_movement import MJXMovement
    from hybrid_movement import HybridMovement
    geometry=json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
    run=args.run_dir.resolve();halt=[False];env=None;gpu=None;arena=None
    def stop_signal(*_):halt[0]=True
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    def stop():heartbeat(run);return halt[0] or (run/'STOP').exists()
    if not args.resume:run.mkdir(parents=True,exist_ok=False)
    with RunLock(run/'trainer.lock'):
        store=Store(run)
        try:
            if args.resume:
                state,folder=store.load()
                if state.get('backend')!='mjx-league-v1' or state['mode']!=args.mode or state['geometry']!=geometry:p.error('Saved backend, mode or geometry mismatch')
                models={x.stem:PPO.load(x,device='cpu') for x in folder.glob('*.zip')}
            else:
                settings=Settings();curr=Curriculum(**json.loads(args.curriculum.read_text())).validate() if args.curriculum else Curriculum();source_model=None;lineage=None
                if args.initial_run:
                    src=args.initial_run
                    if (src/'CURRENT.json').exists():
                        source,sf=Store(src).load()
                        if source.get('geometry')!=geometry or source.get('schema')!=SCHEMA:p.error('Initial model geometry/schema mismatch')
                        settings=Settings(**source['settings']);source_model=sf/'champion.zip'
                    else:
                        import hashlib
                        source=json.loads((src/'mjx_run.json').read_text())
                        if source.get('geometry_hash')!=hashlib.sha256(Path(__file__).with_name('plotclock_geometry.json').read_bytes()).hexdigest():p.error('Initial model geometry mismatch')
                        settings=Settings(**source.get('settings',{'action_angle_rad':.3}));source_model=src/'latest.zip'
                    lineage={'run':str(src.resolve()),'checkpoint':str(source_model)}
                settings.validate();dummy=SkillsEnv('game',settings,curr)
                try:champ=PPO.load(source_model,device='cpu') if source_model else PPO('MlpPolicy',dummy,device='cpu',n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,seed=args.seed,policy_kwargs={'net_arch':[64,64]})
                finally:dummy.close()
                state=dict(backend='mjx-league-v1',schema=SCHEMA,mode=args.mode,geometry=geometry,settings=asdict(settings),curriculum=asdict(curr),lineage=lineage,
                    round=0,phase='train',seed=args.seed,candidates=2 if args.mode=='selfplay' else args.candidates,steps_per_round=args.steps_per_round,match_rallies=args.match_rallies,
                    num_envs=args.num_envs or 16,cpu_envs=args.cpu_envs or 0,total_steps=0,history=['history_000000'],champion_id=0,candidate_steps={},evaluation=[],champion_name=hero(0),hero_serial=0,names={'champion':hero(0),'history_000000':hero(0)},recent_champions=[])
                models={'champion':champ,'history_000000':champ}
            state['num_envs']=args.num_envs if args.num_envs is not None else state['num_envs'];state['cpu_envs']=args.cpu_envs if args.cpu_envs is not None else state['cpu_envs']
            if not 1<=state['num_envs']<=256 or not 0<=state['cpu_envs']<=min(32,os.cpu_count() or 1):p.error('Invalid GPU/CPU world counts')
            if (run/'STOP').exists():(run/'STOP').unlink()
            keys=[f'candidate_{i}' for i in range(state['candidates'])]
            def publish():
                nonlocal folder
                if shutil.disk_usage(run).free<256*1024*1024:raise RuntimeError('Less than 256 MiB disk space remains; previous committed checkpoint retained')
                folder=store.save(state,models)
                shutil.copyfile(folder/'champion.zip',run/'latest.pending');os.replace(run/'latest.pending',run/'latest.zip')
                atomic_json(run/'mjx_run.json',{**state,'total_envs':state['num_envs']+state['cpu_envs'],'joint_travel_degrees':state['settings']['action_angle_rad']*180/np.pi})
                atomic_json(run/'progress.json',{'total_steps':state['total_steps'],'updated':time.time(),'phase':state['phase'],'round':state['round'],'gpu_envs':state['num_envs'],'cpu_envs':state['cpu_envs'],'total_envs':state['num_envs']+state['cpu_envs'],'experience_steps_per_second':state.get('training_steps_per_second',0)})
                heartbeat(run)
            publish()
            settings=Settings(**state['settings']);curr=Curriculum(**state['curriculum'])
            gpu=MJXMovement(state['num_envs'],args.device,state['seed'],settings,mode='game');env=gpu
            for e in gpu.envs:e.set_curriculum(curr)
            if state['cpu_envs']:env=HybridMovement(gpu,state['cpu_envs'],settings,curr,state['seed'])
            arena=MJXMovement(min(8,state['match_rallies']),args.device,state['seed'],settings,mode='game')
            for e in arena.envs:e.set_curriculum(curr)
            start_round=state['round'];last_live=[0.]
            class Callback(BaseCallback):
                def _on_step(self):
                    heartbeat(run)
                    if (run/'WATCH').exists() and time.monotonic()-last_live[0]>.2:
                        last_live[0]=time.monotonic();frames=[(e.data.qpos.copy(),e.data.qvel.copy(),e.target.copy()) for e in gpu.envs[:2]];labels=['GPU '+str(i+1) for i in range(len(frames))]
                        if state['cpu_envs']:frames+=env.cpu.env_method('live_state',indices=list(range(min(2,state['cpu_envs']))));labels+=['CPU '+str(i+1) for i in range(min(2,state['cpu_envs']))]
                        with (run/'live.pending').open('wb') as f:np.savez(f,qpos=np.array([f[0] for f in frames]),qvel=np.array([f[1] for f in frames]),target=np.array([f[2] for f in frames]),labels=np.array(labels),updated=time.time())
                        os.replace(run/'live.pending',run/'live.npz')
                    return not stop()
            print(f"{args.mode}: {state['num_envs']} GPU + {state['cpu_envs']} CPU worlds per active candidate; candidates train in sequence to avoid competing for VRAM. Shared experience within each candidate.",flush=True)
            while not stop() and (not args.rounds or state['round']-start_round<args.rounds):
                if state['phase']=='train':
                    if not state['candidate_steps']:
                        pending=run/'pending_rewards.json'
                        if pending.exists():
                            values=validate(state['mode'],json.loads(pending.read_text())['values']);curr=Curriculum(**{**state['curriculum'],**values}).validate();state['curriculum']=asdict(curr);pending.unlink()
                            for e in gpu.envs+arena.envs:e.set_curriculum(curr)
                            if state['cpu_envs']:env.cpu.env_method('set_curriculum',curr)
                        state['names'].update({k:'Challenger '+str(i+1)+' / round '+str(state['round']+1) for i,k in enumerate(keys)});publish()
                    pool=['champion']+(state['history'] if state['mode']=='league' else [])
                    env.set_pool([str(folder/(k+'.zip')) for k in pool])
                    for index,key in enumerate(keys):
                        if stop():break
                        completed=state['candidate_steps'].get(key,0)
                        if completed>=state['steps_per_round']:continue
                        src=folder/(key+'.zip') if (folder/(key+'.zip')).exists() else folder/'champion.zip'
                        learner=PPO.load(src,env=env,device='cpu');learner.set_random_seed(state['seed']+state['round']*100003+index*1009+completed)
                        began=time.monotonic();before=learner.num_timesteps
                        learner.learn(total_timesteps=min(8192,state['steps_per_round']-completed),reset_num_timesteps=False,callback=Callback())
                        count=learner.num_timesteps-before;state['candidate_steps'][key]=completed+count;state['total_steps']+=count;models[key]=learner
                        state['training_steps_per_second']=count/max(time.monotonic()-began,.001);publish()
                        print(f"Saved round {state['round']+1} {key}: {state['candidate_steps'][key]} steps; {state['training_steps_per_second']:.1f} experience/s",flush=True)
                    if all(state['candidate_steps'].get(k,0)>=state['steps_per_round'] for k in keys):state['phase']='evaluate';publish()
                else:
                    history=state['history'] if state['mode']=='league' else []
                    pairs=schedule(keys,history)
                    for index in range(len(state['evaluation']),len(pairs)):
                        a,b=pairs[index];result=evaluate(arena,models[a],models[b],state['seed']+state['round']*997,state['match_rallies'],stop)
                        if result is None:break
                        state['evaluation'].append({'pair':[a,b],**result});publish();print(f'Match {a} {result["scores"]} {b}',flush=True)
                    if stop():break
                    summary=promotion(keys,history,state['evaluation'],state['match_rallies']);chosen=summary['retained']
                    names=dict(state['names']);summary['rankings']=rankings(summary,names)
                    if chosen!='champion':
                        state['names'][f"history_{state['champion_id']:06d}"]=state['champion_name'];state['hero_serial']+=1;state['champion_name']=hero(state['hero_serial']);state['names']['champion']=state['champion_name']
                        for entry in summary['rankings']:
                            if entry['model']==chosen:entry['name']=state['champion_name']
                        state['recent_champions']=(state['recent_champions']+[{'round':state['round']+1,'name':state['champion_name'],'candidate':chosen}])[-100:]
                    models=retain_champion(state,models,chosen);summary['champion_name']=state['champion_name'];state['last_league']=summary;state['round']+=1;state['phase']='train';state['candidate_steps']={};state['evaluation']=[]
                    state['names']={k:v for k,v in state['names'].items() if k in models};publish();store.event({'event':'gpu_league','round':state['round'],**summary})
                    print(f"Champion: {state['champion_name']} | completed round {state['round']}",flush=True)
            publish();print('Saved and stopped. Resume continues the committed phase.',flush=True)
        finally:
            if arena is not None:arena.close()
            if env is not None:env.close()
if __name__=='__main__':main()
