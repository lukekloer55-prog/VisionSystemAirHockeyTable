"""Exercise real PPO league coordinator/checkpoint recovery using CPU-backed test worlds."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from skills_env import SkillsEnv,Curriculum,SCHEMA
from airhockey_env import Settings
from hybrid_movement import apply_proxy
from train_skills import Store
class TestWorlds(DummyVecEnv):
 def __init__(self,count,device,seed,settings,mode='movement'):
  self.mode=mode;self.pool=[];self.assignment=np.zeros(count,int)
  def make():
   e=SkillsEnv(mode,settings);apply_proxy(e);return e
  super().__init__([make for _ in range(count)])
 def set_pool(self,paths):self.pool=[PPO.load(p,device='cpu') for p in paths]
 def step_async(self,actions):
  if self.pool:
   for i,e in enumerate(self.envs):e.opponent=self.pool[self.assignment[i]%len(self.pool)]
  super().step_async(actions)
def main():
 import train_gpu_league
 from league_names import hero
 with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
  root=Path(tmp);src=root/'source';src.mkdir();run=root/'league';settings=Settings();curr=Curriculum(rally_seconds=.04)
  e=SkillsEnv('movement',settings);policy=PPO('MlpPolicy',e,n_steps=2,batch_size=4,n_epochs=1,device='cpu',policy_kwargs={'net_arch':[8,8]})
  state={'schema':SCHEMA,'mode':'movement','settings':asdict(settings),'curriculum':asdict(curr),'geometry':json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())}
  Store(src).save(state,{'champion':policy});config=root/'curr.json';config.write_text(json.dumps(asdict(curr)))
  args=['train_gpu_league.py','--run-dir',str(run),'--mode','league','--num-envs','2','--steps-per-round','4','--match-rallies','2','--rounds','1']
  with patch('mjx_movement.MJXMovement',TestWorlds),patch.object(sys,'argv',args+['--initial-run',str(src),'--curriculum',str(config)]):train_gpu_league.main()
  first,_=Store(run).load();assert first['round']==1 and first['last_league']['rankings'] and first['champion_name']==hero(0)
  with patch('mjx_movement.MJXMovement',TestWorlds),patch.object(sys,'argv',args+['--resume']):train_gpu_league.main()
  second,_=Store(run).load();assert second['round']==2 and second['total_steps']>first['total_steps'];assert len(list(run.glob('snapshot-*')))<=3
  assert (run/'latest.zip').exists() and (run/'heartbeat').exists()
  print('PASS real PPO candidates, paired matches, rankings, bounded snapshots and league resume',flush=True);e.close()
if __name__=='__main__':main()
