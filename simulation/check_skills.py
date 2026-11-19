"""Tests for movement rewards, physical opponents, match fairness and durable saves."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from skills_env import SkillsEnv,Curriculum
from train_skills import Store,play_match
from table_demo import LENGTH,RADIUS

class Tests(unittest.TestCase):
 def test_targets_and_schema(self):
  e=SkillsEnv();o,_=e.reset(seed=4)
  self.assertEqual(o.shape,(24,))
  for _ in range(100):
   t=e.sample_target();self.assertTrue(e.target_valid(t));self.assertLess(t[1],0)
  e.close()
 def test_reward_and_success(self):
  e=SkillsEnv();e.reset(seed=3)
  action=e.commands[0]/e.settings.action_angle_rad
  e.target=e.paddle(0)[0].copy()
  for _ in range(30):
   _,r,t,tr,info=e.step(action)
   if t or tr:break
  self.assertTrue(info['is_success']);self.assertGreater(info['reward_components']['target'],0)
  e.reset(seed=3)
  with patch.object(e,'tip_angle',return_value=15.):
   _,_,_,_,i=e.step(np.zeros(2))
  self.assertLess(i['reward_components']['closing'],0)
  self.assertLess(i['reward_components']['joint_travel'],0)
  self.assertLess(i['reward_components']['time'],0)
  e.close()
 def test_two_arms_and_goals(self):
  e=SkillsEnv('game');e.reset(seed=1)
  self.assertEqual(e.model.nu,4)
  np.testing.assert_allclose(e.paddle(0)[0],e.paddle(1)[0],atol=1e-10)
  for side,sign in [(0,1),(1,-1)]:
   e.reset(seed=1)
   e.data.joint('puck_x').qpos[0]=0
   e.data.joint('puck_y').qpos[0]=sign*(LENGTH/2+RADIUS+.01)
   e.data.joint('puck_x').qvel[0]=0;e.data.joint('puck_y').qvel[0]=0
   hold=e.commands[0]/e.settings.action_angle_rad
   _,_,term,_,info=e.step_pair(hold,hold)
   self.assertTrue(term);self.assertEqual(info['winner'],side);self.assertEqual(info['outcome'],'goal')
  e.close()
 def test_opponent_paddle_rebounds_puck(self):
  e=SkillsEnv('game');e.reset(seed=1)
  far=-e.paddle(1)[0]
  e.data.joint('puck_x').qpos[0]=far[0]
  e.data.joint('puck_y').qpos[0]=far[1]-.13
  e.data.joint('puck_x').qvel[0]=0;e.data.joint('puck_y').qvel[0]=.8
  hold=e.commands[0]/e.settings.action_angle_rad
  bounced=False
  for _ in range(30):
   _,_,t,tr,info=e.step_pair(hold,hold)
   if e.puck()[1][1]<0:bounced=True;break
   if t or tr:break
  self.assertTrue(bounced,'Far paddle must physically return the puck')
  e.close()
 def test_match_side_swap_and_draw(self):
  class Policy:
   def __init__(self,value):self.value=value
   def predict(self,*a,**kw):return [self.value,0],None
  class Arena:
   def reset(self,seed):pass
   def observation(self,side):return [side]
   def step_pair(self,a,b):
    winner=0 if a[0]>b[0] else 1 if b[0]>a[0] else None
    return None,0,True,False,{'winner':winner,'outcome':'goal' if winner is not None else 'timeout'}
  result=play_match(Arena(),Policy(2),Policy(1),1,8)
  self.assertEqual(result['scores'],[8,0]);self.assertEqual(result['winner'],0)
  self.assertIsNone(play_match(Arena(),Policy(1),Policy(1),1,8)['winner'])
  self.assertIsNone(play_match(Arena(),Policy(1),Policy(1),1,8,lambda:True))
 def test_atomic_store_and_retention(self):
  class Fake:
   def save(self,p):Path(p).write_bytes(b'model')
  class Bad:
   def save(self,p):raise RuntimeError('interrupted write')
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
   store=Store(tmp);store.save({'step':1},{'champion':Fake()})
   with self.assertRaises(RuntimeError):store.save({'step':2},{'champion':Bad()})
   self.assertEqual(store.load()[0]['step'],1)
   for i in range(5):store.save({'step':i},{'champion':Fake()})
   self.assertEqual(len(list(Path(tmp).glob('snapshot-*'))),3)
   self.assertEqual(store.load()[0]['step'],4)

if __name__=='__main__':unittest.main()

