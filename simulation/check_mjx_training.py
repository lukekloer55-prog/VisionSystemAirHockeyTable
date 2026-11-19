"""Short integration test; explicit CPU option for developers."""
import argparse,tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        from mjx_movement import MJXMovement
        import numpy as np
        v=MJXMovement(2,args.device);o=v.reset();assert o.shape==(2,24)
        o,r,d,i=v.step(np.zeros((2,2)));assert np.isfinite(o).all() and np.isfinite(r).all()
        print('PASS batched movement:',o.shape,r,d,flush=True)
        for e in v.envs:e.curriculum.movement_seconds=.02
        o,r,d,i=v.step(np.zeros((2,2)));assert d.all();assert all('terminal_observation' in x for x in i)
        print('PASS independent automatic resets',flush=True)
        from stable_baselines3 import PPO
        model=PPO('MlpPolicy',v,n_steps=2,batch_size=4,n_epochs=1,device='cpu',policy_kwargs={'net_arch':[8,8]})
        model.learn(4)
        model.save(str(Path(tmp)/"policy.zip"))
        loaded=PPO.load(str(Path(tmp)/"policy.zip"),env=v,device='cpu')
        assert loaded.num_timesteps==4
        loaded.learn(4,reset_num_timesteps=False)
        assert loaded.num_timesteps==8
        print('PASS PPO update, save, load and resumed update',flush=True)
        v.close()

if __name__=='__main__':main()
