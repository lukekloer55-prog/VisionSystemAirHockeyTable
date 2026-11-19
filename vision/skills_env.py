"""Movement curriculum and two physical PlotClock players. Simulation only."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from airhockey_env import AirHockeyEnv, Settings, training_model
from plotclock_demo import P, BASE_Y, chain, motor_targets, wrap, TORQUE
from table_demo import WIDTH, LENGTH, RADIUS

SCHEMA = 'plotclock-movement-selfplay-v1-obs24'

@dataclass
class Curriculum:
    target_tolerance_m: float = .015
    target_hold_seconds: float = .12
    target_speed_limit_m_s: float = .12
    movement_seconds: float = 5.0
    rally_seconds: float = 12.0
    reach_reward: float = 15.0
    progress_weight: float = 12.0
    time_cost: float = .5
    joint_travel_cost: float = .3
    closing_cost: float = 4.0
    min_tip_angle_deg: float = 35.0
    failure_cost: float = 20.0
    goal_reward: float = 30.0
    contact_reward: float = .5
    def validate(self):
        weights={'reach_reward','progress_weight','time_cost','joint_travel_cost','closing_cost','failure_cost','goal_reward','contact_reward'}
        for k,v in asdict(self).items():
            if not math.isfinite(v) or v<0 or (v==0 and k not in weights):
                raise ValueError(f'{k} must be finite and nonnegative (timing/geometry must be positive)')
        if self.target_hold_seconds>self.movement_seconds:raise ValueError('Target hold must fit within a movement episode')
        if not 5<=self.min_tip_angle_deg<=150: raise ValueError('Tip angle must be 5..150 degrees')
        return self


def arena_model(settings, mode):
    root=ET.fromstring(training_model(settings, return_xml=True));world=root.find('worldbody')
    if mode=='game':
        # Rotate the complete second linkage 180 degrees, preserving its handedness.
        wrapper=ET.SubElement(world,'body',name='opponent_frame',quat='0 0 0 1')
        def clone(element):
            element=deepcopy(element)
            for node in element.iter():
                for attr in ('name','joint','site1','site2','body1','body2'):
                    if attr in node.attrib: node.set(attr,'opp_'+node.get(attr))
            return element
        for name in ('left_upper','right_upper'):
            wrapper.append(clone(world.find(f"body[@name='{name}']")))
        wrapper.append(clone(world.find("geom[@name='fixed_base']")))
        for section in ('actuator','equality'):
            for node in list(root.find(section)):root.find(section).append(clone(node))
    else:
        puck=world.find("body[@name='puck']/geom")
        puck.set('contype','0');puck.set('conaffinity','0');puck.set('rgba','0 0 0 0')
    marker=ET.SubElement(world,'body',name='movement_target',mocap='true',pos='0 0 .025')
    ET.SubElement(marker,'site',name='movement_target_site',type='sphere',size='.015',rgba='1 0 .8 1' if mode=='movement' else '0 0 0 0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class SkillsEnv(gym.Env):
    metadata={'render_modes':['human'],'render_fps':50}
    def __init__(self, mode='movement', settings=None, curriculum=None, render_mode=None):
        if mode not in ('movement','game'):raise ValueError('mode must be movement or game')
        self.mode=mode;self.settings=(settings or Settings()).validate()
        self.curriculum=(curriculum or Curriculum()).validate()
        self.render_mode=render_mode;self.viewer=None;self.opponent=None
        self.model=arena_model(self.settings,mode);self.data=mujoco.MjData(self.model)
        self.action_space=spaces.Box(-1,1,(2,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(24,),np.float32)
        self.workspace=AirHockeyEnv._workspace_samples(self)
        # Remove singular targets and sample uniformly in table XY, not motor angle space.
        self.workspace=np.array([p for p in self.workspace if self.target_valid(p)])
        if len(self.workspace)<2:raise ValueError('No usable target workspace at this angle threshold')
        self.commands=np.zeros((2,2));self.target=np.zeros(2);self.done=True
        self.joints=['left_motor','left_elbow','right_motor','right_elbow']
        self.active=[False,False];self.hit_budget=[4.,4.];self.hold=0.

    def target_valid(self, point):
        try:q=motor_targets(np.asarray(point)-[0,BASE_Y])
        except ValueError:return False
        if np.max(np.abs(q))>self.settings.action_angle_rad:return False
        elbows=[]
        for i,side in enumerate(('left','right')):
            length,_,home,_=chain(side)
            elbows.append(P[side+'_base']+length*np.array([math.cos(home+q[i]),math.sin(home+q[i])])+[0,BASE_Y])
        a,b=[e-point for e in elbows]
        angle=math.degrees(math.acos(float(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
        return self.curriculum.min_tip_angle_deg<angle<175

    def _prefix(self,side):return '' if side==0 else 'opp_'
    def _q(self,side,velocity=False):
        return np.array([getattr(self.data.joint(self._prefix(side)+n),'qvel' if velocity else 'qpos')[0] for n in self.joints])
    def paddle(self,side):
        site=self.model.site(self._prefix(side)+'left_tip').id
        jac=np.zeros((3,self.model.nv));mujoco.mj_jacSite(self.model,self.data,jac,None,site)
        sign=1 if side==0 else -1
        return self.data.site_xpos[site,:2].copy()*sign,(jac@self.data.qvel)[:2]*sign
    def tip_angle(self,side):
        prefix=self._prefix(side);tip=self.data.site(prefix+'left_tip').xpos[:2]
        vectors=[self.data.body(prefix+s+'_lower').xpos[:2]-tip for s in ('left','right')]
        a,b=vectors;den=np.linalg.norm(a)*np.linalg.norm(b)
        return math.degrees(math.acos(float(np.clip(a@b/max(den,1e-12),-1,1))))
    def puck(self,side=0):
        sign=1 if side==0 else -1
        return np.array([self.data.joint('puck_'+a).qpos[0] for a in ('x','y')])*sign,np.array([self.data.joint('puck_'+a).qvel[0] for a in ('x','y')])*sign
    def observation(self,side=0):
        p,v=(self.target,np.zeros(2)) if self.mode=='movement' else self.puck(side)
        pad,padv=self.paddle(side)
        if self.mode=='game':
            other,otherv=self.paddle(1-side);other=-other;otherv=-otherv
        else:other=otherv=np.zeros(2)
        duration=self.curriculum.movement_seconds if self.mode=='movement' else self.curriculum.rally_seconds
        return np.asarray([*p,*v,*self._q(side),*self._q(side,True),*pad,*padv,*self.commands[side],*other,*otherv,max(0,1-self.data.time/duration),float(self.mode=='movement')],np.float32)

    def sample_target(self):
        lo=self.workspace.min(axis=0);hi=self.workspace.max(axis=0);pad,_=self.paddle(0)
        r=self.settings.paddle_diameter_m/2
        for _ in range(10000):
            p=self.np_random.uniform(lo,hi)
            if abs(p[0])<WIDTH/2-r and -LENGTH/2+r<p[1]<-r and self.target_valid(p) and np.linalg.norm(p-pad)>.04:return p
        raise RuntimeError('Cannot sample reachable movement target')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetData(self.model,self.data)
        target=np.array([0.,P['tip'][1]]);angles=motor_targets(target)
        for side in range(2 if self.mode=='game' else 1):
            for i,name in enumerate(('left','right')):
                l,_,home,home_elbow=chain(name);theta=home+angles[i]
                elbow=P[name+'_base']+l*np.array([math.cos(theta),math.sin(theta)])
                self.data.joint(self._prefix(side)+name+'_motor').qpos[0]=angles[i]
                self.data.joint(self._prefix(side)+name+'_elbow').qpos[0]=wrap(math.atan2(*(target-elbow)[::-1])-theta-home_elbow)
            self.commands[side]=angles
        self.active=[False,False];self.hit_budget=[4.,4.];self.hold=0.;self.done=False
        self.travel=0.;self.steps=0
        # Seeded serve direction is balanced by swapping players during match evaluation.
        self.data.joint('puck_x').qpos[0]=self.np_random.uniform(-.10,.10)
        self.data.joint('puck_y').qpos[0]=0
        vx=self.np_random.uniform(-.12,.12);vy=self.np_random.choice([-1,1])*self.np_random.uniform(.6,.9)
        self.data.joint('puck_x').qvel[0]=vx if self.mode=='game' else 0
        self.data.joint('puck_y').qvel[0]=vy if self.mode=='game' else 0
        mujoco.mj_forward(self.model,self.data)
        if self.mode=='movement':
            self.target=self.sample_target();self.data.mocap_pos[0]=[*self.target,.025]
            mujoco.mj_forward(self.model,self.data)
        return self.observation(),{}

    def step(self,action):
        other=np.zeros(2)
        if self.mode=='game' and self.opponent is not None:other=self.opponent.predict(self.observation(1),deterministic=True)[0]
        return self.step_pair(action,other)

    def step_pair(self,action,other):
        if self.done:raise RuntimeError('reset before stepping')
        actions=np.asarray([action,other],float)
        if actions.shape!=(2,2) or not np.isfinite(actions).all():raise ValueError('Actions must be finite 2-vectors')
        sides=2 if self.mode=='game' else 1;dt=self.settings.control_dt
        before=[self._q(s) for s in range(sides)];old_dist=np.linalg.norm(self.paddle(0)[0]-self.target)
        desired=np.clip(actions,-1,1)*self.settings.action_angle_rad
        self.commands+=np.clip(desired-self.commands,-self.settings.target_slew_rad_s*dt,self.settings.target_slew_rad_s*dt)
        hits=[False,False];travel=0.
        for _ in range(round(dt/self.model.opt.timestep)):
            for side in range(sides):
                for i,name in enumerate(('left','right')):
                    j=self.data.joint(self._prefix(side)+name+'_motor')
                    self.data.ctrl[side*2+i]=np.clip(25*(self.commands[side,i]-j.qpos[0])-4*j.qvel[0],-TORQUE,TORQUE)
            mujoco.mj_step(self.model,self.data)
            # Sum actual travel at physics rate; back-and-forth motion cannot cancel.
            q=self._q(0);travel+=float(np.abs(q-before[0]).sum());before[0]=q
            for side in range(sides):
                ids={self.model.geom(self._prefix(side)+'paddle').id,self.model.geom('puck_shape').id}
                contact=any({c.geom1,c.geom2}==ids for c in self.data.contact[:self.data.ncon])
                hits[side]|=contact and not self.active[side];self.active[side]=contact
        mujoco.mj_forward(self.model,self.data);self.steps+=1;self.travel+=travel
        pad,speed=self.paddle(0);angle=self.tip_angle(0)
        invalid=not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or any(w.number for w in self.data.warning)
        failures=[]
        for side in range(sides):
            pp,_=self.paddle(side);prefix=self._prefix(side);r=self.settings.paddle_diameter_m/2
            gap=np.linalg.norm(self.data.site(prefix+'left_tip').xpos-self.data.site(prefix+'right_tip').xpos)
            failures.append(bool(gap>.002 or abs(pp[0])>WIDTH/2-r or not -LENGTH/2+r<=pp[1]<=-r or self.tip_angle(side)<5))
        parts={'time':-self.curriculum.time_cost*dt,'joint_travel':-self.curriculum.joint_travel_cost*travel,
               'closing':-self.curriculum.closing_cost*dt*max(0,1-angle/self.curriculum.min_tip_angle_deg)**2}
        terminated=bool(invalid or any(failures));winner=None;outcome='running';success=False
        if self.mode=='movement':
            dist=np.linalg.norm(pad-self.target)
            parts['progress']=self.curriculum.progress_weight*(old_dist-dist)
            self.hold=self.hold+dt if dist<=self.curriculum.target_tolerance_m and np.linalg.norm(speed)<=self.curriculum.target_speed_limit_m_s else 0.
            success=self.hold>=self.curriculum.target_hold_seconds
            if success:parts['target']=self.curriculum.reach_reward;terminated=True;outcome='target_reached'
            duration=self.curriculum.movement_seconds
        else:
            p,v=self.puck();duration=self.curriculum.rally_seconds
            if abs(p[0])+RADIUS<=.127 and abs(p[1])>LENGTH/2+RADIUS:
                winner=0 if p[1]>0 else 1;terminated=True;outcome='goal'
                parts['goal']=self.curriculum.goal_reward*(1 if winner==0 else -1)
            elif abs(p[0])>WIDTH/2+RADIUS or abs(p[1])>LENGTH/2+RADIUS:
                terminated=True;outcome='escape';parts['escape']=-self.curriculum.failure_cost
            if hits[0] and v[1]>.05:
                paid=min(self.hit_budget[0],self.curriculum.contact_reward);parts['hit']=paid;self.hit_budget[0]-=paid
            # Bounded pursuit encourages play before goals are learned.
            desired=p if p[1]<0 else np.array([0.,BASE_Y+P['tip'][1]])
            reachable=self.workspace[np.argmin(np.sum((self.workspace-desired)**2,axis=1))]
            parts['pursuit']=-.2*dt*float(np.linalg.norm(pad-reachable))
        if invalid or any(failures):
            outcome='invalid_physics' if invalid else 'arm_failure';terminated=True;success=False
            winner=None if invalid or all(failures) or self.mode=='movement' else (1 if failures[0] else 0)
            parts['failure']=-self.curriculum.failure_cost if invalid or failures[0] else self.curriculum.failure_cost
        truncated=bool(not terminated and self.data.time>=duration)
        if truncated:outcome='timeout'
        self.done=terminated or truncated
        obs=np.nan_to_num(self.observation(),nan=0,posinf=0,neginf=0)
        reward=float(np.nan_to_num(sum(parts.values()),nan=-self.curriculum.failure_cost))
        return obs,reward,terminated,truncated,{'outcome':outcome,'winner':winner,'is_success':success,'tip_angle_deg':angle,'joint_travel_rad':self.travel,'target_distance_m':float(np.linalg.norm(pad-self.target)) if self.mode=='movement' else None,'reward_components':parts}

    def render(self):
        if self.render_mode!='human':return
        if self.viewer is None:
            import mujoco.viewer as mv
            self.viewer=mv.launch_passive(self.model,self.data)
            self.viewer.cam.lookat[:]=[0,0,0];self.viewer.cam.distance=3.;self.viewer.cam.azimuth=90;self.viewer.cam.elevation=-70
        self.viewer.sync()
    def close(self):
        if self.viewer is not None:self.viewer.close();self.viewer=None
