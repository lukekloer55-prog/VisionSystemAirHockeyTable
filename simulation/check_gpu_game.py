"""Short GPU game adapter, contact and paired-evaluation integration check."""
import argparse,numpy as np,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='gpu');args=p.parse_args()
 from mjx_movement import MJXMovement
 from skills_env import Curriculum
 from train_gpu_league import evaluate
 class Hold:
  def predict(self,obs,deterministic=True):return np.zeros((len(obs),2),np.float32),None
 env=MJXMovement(2,args.device,mode='game')
 try:
  for e in env.envs:e.set_curriculum(Curriculum(rally_seconds=.04))
  obs=env.reset();env.pool=[Hold()];env.assignment[:]=0
  obs,r,d,i=env.step(np.zeros((2,2)));assert obs.shape==(2,24) and np.isfinite(r).all()
  for _ in range(3):
   obs,r,d,i=env.step(np.zeros((2,2)))
   if d.all():break
  assert d.all() and all('episode' in x for x in i)
  result=evaluate(env,Hold(),Hold(),123,2,lambda:False);assert result['scores']==[0,0] and sum(result['outcomes'].values())==2
  from hybrid_movement import HybridMovement
  from airhockey_env import Settings
  from stable_baselines3 import PPO
  hybrid=HybridMovement(env,1,Settings(),Curriculum(rally_seconds=.04),123)
  try:
   with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
    policy=PPO('MlpPolicy',env.envs[0],n_steps=2,batch_size=2,device='cpu');policy.save(Path(tmp)/'opponent.zip')
    hybrid.set_pool([str(Path(tmp)/'opponent.zip')]);obs=hybrid.reset();assert obs.shape==(3,24)
    obs,reward,done,info=hybrid.step(np.zeros((3,2)));assert np.isfinite(reward).all() and len(info)==3
  finally:hybrid.close()
  print('PASS game integration, opponent actions, GPU episode records, terminal reset and paired evaluation',flush=True)
 finally:env.close()
if __name__=='__main__':main()
