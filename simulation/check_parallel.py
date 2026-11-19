"""Parallel worker smoke checks, including optional native viewer demonstration."""
import argparse
from dataclasses import asdict,replace
import numpy as np
from airhockey_env import Settings
from skills_env import Curriculum
from parallel_training import make_parallel_env

def main():
 p=argparse.ArgumentParser();p.add_argument('--viewers',action='store_true');args=p.parse_args()
 env=make_parallel_env('movement',Settings(),Curriculum(),2,args.viewers)
 try:
  env.seed(5);obs=env.reset();assert obs.shape==(2,24)
  assert not np.array_equal(obs[0,:2],obs[1,:2]),'workers need distinct seeded targets'
  if not args.viewers:assert env.get_attr('viewer')==[None,None]
  changed=asdict(replace(Curriculum(),time_cost=1.2))
  env.env_method('configure_training',changed,None)
  assert all(c.time_cost==1.2 for c in env.get_attr('curriculum'))
  env.reset()
  for _ in range(40):
   obs,reward,done,info=env.step(np.zeros((2,2)))
   assert np.isfinite(obs).all()
   if args.viewers:assert all(env.env_method('preview'))
  if not args.viewers:
   assert all(env.env_method('preview'))
   assert env.get_attr('viewer')==[None,None], 'headless preview must not allocate a viewer'
  print('PASS: independent workers, curriculum propagation, finite steps, '+('two native viewers' if args.viewers else 'no viewers in headless mode'))
 finally:env.close()
 assert all(not proc.is_alive() for proc in env.processes)
 print('PASS: worker processes exited')
if __name__=='__main__':main()
