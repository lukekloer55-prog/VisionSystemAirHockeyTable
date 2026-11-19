"""Exercise combined GPU/CPU observations, updates, save/load and live samples."""
import argparse,tempfile
from pathlib import Path
import numpy as np
from dataclasses import asdict

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','gpu'],default='gpu');args=p.parse_args()
    from mjx_movement import MJXMovement
    from hybrid_movement import HybridMovement
    from skills_env import Curriculum
    from airhockey_env import Settings
    from stable_baselines3 import PPO
    gpu=MJXMovement(2,args.device);env=HybridMovement(gpu,1,Settings(),Curriculum(),123)
    try:
        obs=env.reset();assert obs.shape==(3,24)
        obs,r,d,infos=env.step(np.zeros((3,2)));assert obs.shape==(3,24) and len(infos)==3 and np.isfinite(r).all()
        assert not np.array_equal(obs[0,:2],obs[2,:2])
        samples=env.cpu.env_method('live_state');assert len(samples)==1 and len(samples[0])==3
        env.set_curriculum(Curriculum(movement_seconds=.02,target_hold_seconds=.01))
        obs,r,d,infos=env.step(np.zeros((3,2)));assert d.all() and all('terminal_observation' in i for i in infos)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            model=PPO('MlpPolicy',env,n_steps=2,batch_size=6,n_epochs=1,device='cpu',policy_kwargs={'net_arch':[8,8]})
            model.learn(6);model.save(Path(tmp)/'model.zip');model=PPO.load(Path(tmp)/'model.zip',env=env,device='cpu');model.learn(6,reset_num_timesteps=False);assert model.num_timesteps==12
        print('PASS mixed world order, independent targets, live samples, reward propagation, resets and shared PPO save/resume',flush=True)
    finally:env.close()
if __name__=='__main__':main()
