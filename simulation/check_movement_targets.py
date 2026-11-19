"""Whole-half target sampling, independent of motor reach and closing-angle limits."""
from unittest.mock import patch
import unittest
import numpy as np
from dataclasses import replace
from skills_env import SkillsEnv
from table_demo import WIDTH,LENGTH

class TargetTests(unittest.TestCase):
 def test_all_regions_and_no_ik_filter(self):
  e=SkillsEnv()
  try:
   e.reset(seed=91);r=e.settings.paddle_diameter_m/2
   with patch('skills_env.motor_targets',side_effect=AssertionError('sampling must not use inverse kinematics')):
    points=np.array([e.sample_target() for _ in range(4000)])
   self.assertTrue(np.all(np.abs(points[:,0])<WIDTH/2-r))
   self.assertTrue(np.all(points[:,1]>-LENGTH/2+r));self.assertTrue(np.all(points[:,1]<-r))
   self.assertGreater(np.sum(points[:,1]<-LENGTH/2+.1524),200) # bottom six inches
   bins,_,_=np.histogram2d(points[:,0],points[:,1],bins=4,range=[[-WIDTH/2+r,WIDTH/2-r],[-LENGTH/2+r,-r]])
   self.assertGreater(bins.min(),150)
   e.set_curriculum(replace(e.curriculum,min_tip_angle_deg=120))
   self.assertTrue(e.target_valid([0,-LENGTH/2+r+.001]))
  finally:e.close()
 def test_unreached_target_times_out(self):
  e=SkillsEnv()
  try:
   e.reset(seed=92);e.target=np.array([0.,-LENGTH/2+e.settings.paddle_diameter_m/2+.001])
   action=e.commands[0]/e.settings.action_angle_rad
   for _ in range(300):
    _,_,terminated,truncated,info=e.step(action)
    if terminated or truncated:break
   self.assertTrue(truncated);self.assertEqual(info['outcome'],'timeout');self.assertFalse(info['is_success'])
  finally:e.close()
if __name__=='__main__':unittest.main()
