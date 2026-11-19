"""Validated reward controls shared by the dashboard and training process."""
from dataclasses import asdict,replace
import json
import math
import os
from pathlib import Path
import uuid
from skills_env import Curriculum

COMMON={
 'time_cost':('Elapsed time penalty / second',0,5,.05),
 'joint_travel_cost':('Arm travel penalty / radian',0,5,.05),
 'closing_cost':('Narrow-link penalty strength',0,30,.25),
 'min_tip_angle_deg':('Start narrow-link penalty below (degrees)',5,120,1),
 'failure_cost':('Arm / physics failure penalty',0,100,1),
}
MOVEMENT={
 'reach_reward':('Reach and settle at target',0,100,1),
 'progress_weight':('Progress toward target / metre',0,50,.5),
}
GAME={
 'goal_reward':('Goal scored reward / conceded penalty',0,100,1),
 'contact_reward':('Useful contact reward',0,5,.1),
 'pursuit_cost':('Distance from reachable puck penalty',0,5,.05),
 'hit_bonus_budget':('Maximum contact bonuses per rally',0,20,.5),
}

def controls(mode):
 if mode not in ('movement','selfplay'):raise ValueError('Unknown training mode')
 return {**COMMON,**(MOVEMENT if mode=='movement' else GAME)}

def validate(mode,values):
 fields=controls(mode)
 if not isinstance(values,dict) or set(values)!=set(fields):raise ValueError('Send every visible reward field together')
 for name,value in values.items():
  _,low,high,_=fields[name]
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{name} must be between {low} and {high}')
 return values

def atomic_json(path,data):
 path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('w') as f:
   json.dump(data,f,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()

def apply_pending(env,state,models,store):
 """Training-thread transaction. Self-play waits until BOTH candidates can use new weights."""
 if state['mode']=='selfplay' and not (state['phase']=='train_a' and state['trained']==0 and 'a' not in models):return None
 pending=store.path/'pending_rewards.json';processing=store.path/'applying_rewards.json'
 if not processing.exists():
  if not pending.exists():return None
  os.replace(pending,processing)
 change=json.loads(processing.read_text())
 if state.get('reward_update',{}).get('revision')==change.get('revision'):
  processing.unlink();return None
 previous=env.curriculum
 try:
  values=validate(state['mode'],change['values'])
  new=replace(previous,**values).validate()
  env.set_curriculum(new)
  state['curriculum']=asdict(new)
  state['reward_update']={'revision':change['revision'],'status':'applied','values':values,'steps':state['total_steps'],'round':state['round']}
 except (ValueError,KeyError,TypeError) as error:
  env.curriculum=previous
  state['reward_update']={'revision':change.get('revision'),'status':'rejected','message':str(error)}
 folder=store.save(state,models)
 store.event({'event':'reward_update',**state['reward_update']})
 processing.unlink()
 return folder
