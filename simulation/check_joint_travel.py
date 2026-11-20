import math,json
import numpy as np
import mujoco
from skills_env import SkillsEnv
from airhockey_env import Settings
from plotclock_demo import motor_targets,chain,P,BASE_Y,wrap
from table_demo import WIDTH,LENGTH
from stable_baselines3 import PPO
from joint_travel import rescale_actor

e=SkillsEnv();e.reset(seed=1);r=e.settings.paddle_diameter_m/2
counts={'tested':0,'reachable':0,'outside_link_lengths':0};max_angle=0.;gap=0.
for x in np.linspace(-WIDTH/2+r+.001,WIDTH/2-r-.001,21):
 for y in np.linspace(-LENGTH/2+r+.001,-r-.001,21):
  counts['tested']+=1;point=np.array([x,y]);target=point-[0,BASE_Y]
  try:q=motor_targets(target)
  except ValueError:counts['outside_link_lengths']+=1;continue
  max_angle=max(max_angle,float(np.max(np.abs(q))))
  for i,side in enumerate(('left','right')):
   l,_,home,elbow_home=chain(side);theta=home+q[i];elbow=P[side+'_base']+l*np.array([math.cos(theta),math.sin(theta)])
   e.data.joint(side+'_motor').qpos[0]=q[i];e.data.joint(side+'_elbow').qpos[0]=wrap(math.atan2(*(target-elbow)[::-1])-theta-elbow_home)
  mujoco.mj_forward(e.model,e.data)
  err=np.linalg.norm(e.data.site('left_tip').xpos[:2]-point);gap=max(gap,np.linalg.norm(e.data.site('left_tip').xpos-e.data.site('right_tip').xpos))
  assert err<1e-7
  counts['reachable']+=1
assert max_angle>.35 and gap<1e-7
for side in ('left','right'):
 assert np.allclose(e.model.joint(side+'_motor').range,[-math.pi,math.pi])
old=SkillsEnv(settings=Settings(action_angle_rad=.3));old.reset(seed=2)
assert np.allclose(old.model.joint('left_motor').range,[-.35,.35])
model=PPO('MlpPolicy',old,n_steps=2,batch_size=2,device='cpu');obs,_=old.reset(seed=3)
a=model.predict(obs,deterministic=True)[0]*.3
rescale_actor(model,.3,math.pi);b=model.predict(obs,deterministic=True)[0]*math.pi
assert np.allclose(a,b,atol=1e-6)
print(json.dumps({**counts,'largest_base_command_deg':math.degrees(max_angle),'max_static_closure_gap_m':float(gap)},indent=2))
print('PASS expanded joints, legacy limits, static reach and action migration')
e.close();old.close()
