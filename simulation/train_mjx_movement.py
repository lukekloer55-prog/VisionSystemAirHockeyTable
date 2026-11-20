"""Shared-policy CPU + GPU movement training, optional external live viewers."""
import argparse,json,os,signal,time,hashlib,math
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

def main():
    import sys
    if '--performance' in sys.argv:
        from performance import main as performance_main
        performance_main();return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=64,help='GPU worlds')
    p.add_argument('--cpu-envs',type=int,default=0)
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--initial-run',type=Path)
    p.add_argument('--expand-joint-travel',action='store_true',help='New stage only: use +/-180 degree commands; adapt imported Gaussian actor scale')
    p.add_argument('--steps',type=int,default=0)
    p.add_argument('--curriculum',type=Path)
    p.add_argument('--seed',type=int,default=1)
    args=p.parse_args()
    if not 1<=args.num_envs<=256 or not 0<=args.cpu_envs<=min(32,os.cpu_count() or 1) or args.steps<0:p.error('GPU worlds 1..256; CPU worlds 0..available cores (max 32)')
    if args.resume and args.expand_joint_travel:p.error('Create a new stage to expand joint travel; existing runs retain their settings')
    if args.resume and args.initial_run:p.error('Resume and initial-run are mutually exclusive')
    from dataclasses import asdict
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from train_skills import RunLock,Store
    from skills_env import Curriculum,SCHEMA
    from airhockey_env import Settings
    from skills_controls import atomic_json,validate
    from mjx_movement import MJXMovement
    from hybrid_movement import HybridMovement
    torch.set_num_threads(1)
    geometry_file=Path(__file__).with_name('plotclock_geometry.json')
    geometry_hash=hashlib.sha256(geometry_file.read_bytes()).hexdigest()
    settings=Settings();initial=None;lineage=None;run=args.run_dir
    if args.resume:
        meta=json.loads((run/'mjx_run.json').read_text())
        if meta.get('geometry_hash')!=geometry_hash or meta.get('schema')!='mjx-movement-experimental-v1':p.error('Geometry/schema mismatch')
        initial=run/'latest.zip'
        if not initial.exists():p.error('No checkpoint to resume')
        settings=Settings(**meta.get('settings',{'action_angle_rad':.30})).validate()
    else:
        if args.initial_run:
            src=args.initial_run
            if (src/'mjx_run.json').exists():
                info=json.loads((src/'mjx_run.json').read_text())
                if info.get('geometry_hash')!=geometry_hash:p.error('Source geometry mismatch')
                initial=src/'latest.zip';settings=Settings(**info.get('settings',{'action_angle_rad':.30})).validate()
            else:
                info,folder=Store(src).load()
                if info.get('schema')!=SCHEMA or info.get('mode')!='movement' or info.get('geometry')!=json.loads(geometry_file.read_text()):p.error('Source must be a movement run with matching geometry')
                initial=folder/'champion.zip';settings=Settings(**info['settings']).validate()
            if not initial.exists():p.error('Source checkpoint missing')
            lineage={'source_run':str(src.resolve()),'source_checkpoint':str(initial)}
        run.mkdir(parents=True,exist_ok=False)
        meta={'schema':'mjx-movement-experimental-v1','targets':'whole_robot_half_v1','experimental':True,'geometry_hash':geometry_hash,'lineage':lineage}
    original_action_scale=settings.action_angle_rad
    if args.expand_joint_travel:settings.action_angle_rad=math.pi
    settings.validate()
    meta['joint_travel_degrees']=math.degrees(settings.action_angle_rad)
    with RunLock(run/'trainer.lock'):
        if args.resume and (run/'STOP').exists():(run/'STOP').unlink()
        curriculum_file=run/'curriculum.json'
        curriculum=Curriculum(**json.loads(curriculum_file.read_text())) if curriculum_file.exists() else Curriculum(**json.loads(args.curriculum.read_text())) if args.curriculum else Curriculum()
        curriculum.validate();atomic_json(curriculum_file,asdict(curriculum))
        meta.update(num_envs=args.num_envs,cpu_envs=args.cpu_envs,total_envs=args.num_envs+args.cpu_envs,settings=asdict(settings),curriculum=asdict(curriculum),backend='Shared PPO: MJX GPU + CPU MuJoCo' if args.cpu_envs else 'MJX physics / CPU PPO')
        atomic_json(run/'mjx_run.json',meta)
        from train_gpu_league import heartbeat
        heartbeat(run)
        stop=[False];env=None;model=None;gpu=None;last_view=[0.];began=time.monotonic();start_steps=[0]
        def request_stop(*_):stop[0]=True
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        def progress():
            elapsed=time.monotonic()-began
            atomic_json(run/'progress.json',{'total_steps':model.num_timesteps,'updated':time.time(),'gpu_envs':args.num_envs,'cpu_envs':args.cpu_envs,'total_envs':args.num_envs+args.cpu_envs,'experience_steps_per_second':(model.num_timesteps-start_steps[0])/max(elapsed,.001)})
        def save():
            model.save(run/'pending.zip');os.replace(run/'pending.zip',run/'latest.zip');progress()
            if (run/'SAVE').exists():(run/'SAVE').unlink()
        class Callback(BaseCallback):
            def _on_step(self):
                heartbeat(run)
                now=time.monotonic()
                if now-last_view[0]>.2:
                    last_view[0]=now
                    if (run/'WATCH').exists():
                        frames=[];labels=[]
                        for i,e in enumerate(gpu.envs[:2]):frames.append((e.data.qpos.copy(),e.data.qvel.copy(),e.target.copy()));labels.append('GPU '+str(i+1))
                        if args.cpu_envs:
                            frames+=env.cpu.env_method('live_state',indices=list(range(min(2,args.cpu_envs))))
                            labels+=['CPU '+str(i+1) for i in range(min(2,args.cpu_envs))]
                        with (run/'live.pending').open('wb') as f:np.savez(f,qpos=np.array([x[0] for x in frames]),qvel=np.array([x[1] for x in frames]),target=np.array([x[2] for x in frames]),labels=np.array(labels),updated=time.time())
                        os.replace(run/'live.pending',run/'live.npz')
                if self.n_calls%50==0:progress()
                if (run/'SAVE').exists():save()
                return not (stop[0] or (run/'STOP').exists())
        try:
            gpu=MJXMovement(args.num_envs,args.device,args.seed,settings)
            env=gpu
            for e in gpu.envs:e.set_curriculum(curriculum)
            if args.cpu_envs:env=HybridMovement(gpu,args.cpu_envs,settings,curriculum,args.seed)
            model=PPO.load(initial,env=env,device='cpu') if initial else PPO('MlpPolicy',env,device='cpu',n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,seed=args.seed,verbose=1,policy_kwargs={'net_arch':[64,64]})
            if initial and args.expand_joint_travel:
                from joint_travel import rescale_actor
                rescale_actor(model,original_action_scale,settings.action_angle_rad)
                meta['action_scale_migration']={'from_rad':original_action_scale,'to_rad':settings.action_angle_rad}
                atomic_json(run/'mjx_run.json',meta)
            # Loading preserves the policy and optimizer; rollout buffers are rebuilt for the combined world count.
            start_steps[0]=model.num_timesteps
            print(f'Shared model: GPU={args.num_envs}, CPU={args.cpu_envs}, total={env.num_envs}. Every world contributes to the same PPO update.',flush=True)
            while not stop[0] and not (run/'STOP').exists():
                pending=run/'pending_rewards.json'
                if pending.exists():
                    values=validate('movement',json.loads(pending.read_text())['values']);curriculum=Curriculum(**{**asdict(curriculum),**values}).validate()
                    if args.cpu_envs:env.set_curriculum(curriculum)
                    else:
                        for e in gpu.envs:e.set_curriculum(curriculum)
                    atomic_json(curriculum_file,asdict(curriculum));meta['curriculum']=asdict(curriculum);atomic_json(run/'mjx_run.json',meta);pending.unlink()
                remaining=args.steps-(model.num_timesteps-start_steps[0]) if args.steps else 8192
                if remaining<=0:break
                model.learn(total_timesteps=min(8192,remaining),reset_num_timesteps=False,callback=Callback());save()
            save()
        finally:
            if env is not None:env.close()
if __name__=='__main__':main()
