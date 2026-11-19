"""Experimental headless movement PPO with MJX batched physics. Ctrl+C saves."""
import argparse,json,os,signal,time,hashlib
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--steps',type=int,default=0,help='Additional steps; zero runs until stopped')
    p.add_argument('--seed',type=int,default=1)
    args=p.parse_args()
    if not 1<=args.num_envs<=256 or args.steps<0:p.error('Use 1..256 environments and nonnegative steps')
    from mjx_movement import MJXMovement
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    torch.set_num_threads(1)
    from train_skills import RunLock
    geometry_hash=hashlib.sha256(Path(__file__).with_name("plotclock_geometry.json").read_bytes()).hexdigest()
    run=args.run_dir
    if args.resume:
        if not (run/'latest.zip').is_file():p.error('No latest.zip to resume')
        meta=json.loads((run/'mjx_run.json').read_text())
        if meta.get('geometry_hash')!=geometry_hash:p.error('Geometry changed; create a new run')
        if meta.get('schema')!='mjx-movement-experimental-v1':p.error('Incompatible run')
    else:
        run.mkdir(parents=True,exist_ok=False)
        (run/'mjx_run.json').write_text(json.dumps({'schema':'mjx-movement-experimental-v1','backend':'MJX physics / CPU PPO and rewards','targets':'whole_robot_half_v1','experimental':True,'geometry_hash':geometry_hash},indent=2))
    lock=RunLock(run/"trainer.lock");lock.__enter__()
    if args.resume and (run/"STOP").exists():(run/"STOP").unlink()
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    env=None;model=None
    def save():
        model.save(run/'pending.zip');os.replace(run/'pending.zip',run/'latest.zip')
        (run/'progress.json').write_text(json.dumps({'total_steps':model.num_timesteps,'updated':time.time()}))
    class SaveCallback(BaseCallback):
        def _on_step(self):
            if stop[0] or (run/'STOP').exists():return False
            return True
    try:
        env=MJXMovement(args.num_envs,args.device,args.seed)
        model=PPO.load(run/'latest.zip',env=env,device='cpu') if args.resume else PPO('MlpPolicy',env,device='cpu',n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,seed=args.seed,verbose=1,policy_kwargs={'net_arch':[64,64]})
        print('Experimental MJX movement. Headless; CPU rewards/PPO. Ctrl+C or STOP file saves. First compilation may take minutes.',flush=True)
        start=model.num_timesteps
        while not stop[0] and not (run/'STOP').exists():
            remaining=args.steps-(model.num_timesteps-start) if args.steps else 8192
            if remaining<=0:break
            model.learn(total_timesteps=min(8192,remaining),reset_num_timesteps=False,callback=SaveCallback())
            save()
        save()
    finally:
        if env is not None:env.close()
        lock.__exit__(None,None,None)
if __name__=='__main__':main()
