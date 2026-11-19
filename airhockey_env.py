"""Gymnasium environment for the direct-drive PlotClock air-hockey defender.

Simulation only. Observations use simulator state, not live camera images.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from plotclock_demo import make_model, motor_targets, chain, P, BASE_Y, ARM_Z, TORQUE, wrap
from table_demo import WIDTH, LENGTH, RADIUS


@dataclass
class Settings:
    # Editable placeholders: replace these with measurements when available.
    paddle_diameter_m: float = 0.100
    paddle_mass_kg: float = 0.150
    link_mass_kg: float = 0.400
    puck_mass_kg: float = 0.020
    # Begin with slower incoming shots in the reachable central region.
    min_puck_speed_m_s: float = 0.60
    max_puck_speed_m_s: float = 0.90
    spawn_half_width_m: float = 0.12
    spawn_center_x_m: float = 0.16
    episode_seconds: float = 6.0
    # Policy acts at 50 Hz; physics runs at 1000 Hz.
    control_dt: float = 0.02
    target_slew_rad_s: float = 0.75
    action_angle_rad: float = 0.30
    # Must match the PPO discount for the potential-based reward term.
    gamma: float = 0.99

    def validate(self):
        for name,value in asdict(self).items():
            if not math.isfinite(value) or (name != 'spawn_center_x_m' and value <= 0):
                raise ValueError(f'{name} must be positive and finite.')
        if not self.min_puck_speed_m_s <= self.max_puck_speed_m_s <= 2.0:
            raise ValueError('Use an ordered puck speed range up to 2 m/s for this initial task.')
        if not .02 <= self.paddle_diameter_m <= .20:
            raise ValueError('Paddle diameter must be between 0.02 and 0.20 m.')
        if abs(self.spawn_center_x_m)+self.spawn_half_width_m > WIDTH/2-RADIUS-.01:
            raise ValueError('Puck spawn range crosses a side rail.')
        if self.action_angle_rad > .30 or self.target_slew_rad_s > 2:
            raise ValueError('Initial environment supports at most 0.30 rad commands and 2 rad/s target slew.')
        if self.gamma >= 1 or not .001 <= self.control_dt <= .05:
            raise ValueError('Require gamma < 1 and control_dt between .001 and .05 s.')
        if abs(round(self.control_dt/.001)*.001-self.control_dt)>1e-9:
            raise ValueError('control_dt must be a multiple of the 1 ms physics timestep.')
        return self

    @classmethod
    def load(cls,path=None):
        return cls(**json.loads(Path(path).read_text())).validate() if path else cls().validate()

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2)+'\n')


def training_model(settings):
    """Add a physical paddle and puck, and split the end rails around 10-inch goals."""
    _,_,xml=make_model()
    root=ET.fromstring(xml);world=root.find('worldbody')
    world.remove(world.find("body[@name='target']"))
    root.find('option').set('timestep','.001')
    for side in ('left','right'):
        # Preserve CAD link dimensions, replacing only the estimated mass.
        for body_name in (side+'_upper',side+'_lower'):
            body=root.find(f".//body[@name='{body_name}']")
            body.find("geom[@type='box']").set('mass',str(settings.link_mass_kg))
    lower=root.find(".//body[@name='left_lower']")
    length=chain('left')[1]
    # The paddle hangs from the controlled pin down to the puck plane.
    # Planar joints keep its cylinder vertical; its contact is real MuJoCo contact.
    ET.SubElement(lower,'geom',name='paddle',type='cylinder',
        pos=f'{length} 0 {.012-ARM_Z}',size=f'{settings.paddle_diameter_m/2} .012',
        mass=str(settings.paddle_mass_kg),rgba='.2 .75 .25 1',contype='1',conaffinity='1',
        friction='0 0 0',solref='.004 .5')
    puck=ET.SubElement(world,'body',name='puck',pos='0 0 .006')
    for axis,direction in [('x','1 0 0'),('y','0 1 0')]:
        ET.SubElement(puck,'joint',name='puck_'+axis,type='slide',axis=direction,damping='0')
    ET.SubElement(puck,'geom',name='puck_shape',type='cylinder',size=f'{RADIUS} .005',
        mass=str(settings.puck_mass_kg),rgba='1 .85 0 1',friction='0 0 0',solref='.004 .5')
    # Both ends have centered, 254 mm openings. Side rails are unchanged.
    goal_half=.254/2
    for end,sign in [('near',-1),('far',1)]:
        world.remove(world.find(f"geom[@name='{end}']"))
        inner,outer=goal_half,WIDTH/2+.03
        for side in (-1,1):
            ET.SubElement(world,'geom',name=f'{end}_post_{side}',type='box',
                pos=f'{side*(inner+outer)/2} {sign*(LENGTH/2+.015)} .025',
                size=f'{(outer-inner)/2} .015 .025',rgba='.15 .25 .45 1',
                friction='0 0 0',solref='.004 .5')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class AirHockeyEnv(gym.Env):
    """Learn two motor angle targets; a torque-clipped PD loop drives the joints."""
    metadata={'render_modes':['human'],'render_fps':50}

    def __init__(self,settings=None,render_mode=None):
        super().__init__()
        self.settings=(settings or Settings()).validate()
        if render_mode not in (None,'human'):
            raise ValueError('Supported render_mode is human or None.')
        self.render_mode=render_mode;self.viewer=None
        self.model=training_model(self.settings);self.data=mujoco.MjData(self.model)
        self.frame_skip=round(self.settings.control_dt/self.model.opt.timestep)
        self.action_space=spaces.Box(-1,1,shape=(2,),dtype=np.float32)
        # puck XY/VXY (4), all hinge positions/speeds (8), paddle XY/VXY (4),
        # commanded motor angles (2), remaining episode fraction (1), prior contact (1).
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(20,),dtype=np.float32)
        self.joint_names=['left_motor','left_elbow','right_motor','right_elbow']
        self.paddle_id=self.model.geom('paddle').id
        self.puck_id=self.model.geom('puck_shape').id
        self.tip_id=self.model.site('left_tip').id
        self.defense_y=BASE_Y+P['tip'][1]
        self.command=np.zeros(2)
        self.episode_steps=0;self.touched=False;self.done=True

    def puck_state(self):
        """Return puck coordinates and velocity in metres and metres per second."""
        pos=np.array([self.data.joint('puck_'+a).qpos[0] for a in ('x','y')])
        vel=np.array([self.data.joint('puck_'+a).qvel[0] for a in ('x','y')])
        return pos,vel

    def paddle_state(self):
        """Use the pin Jacobian to get linear velocity of the attached paddle."""
        jac=np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,jac,None,self.tip_id)
        return self.data.site('left_tip').xpos[:2].copy(),(jac@self.data.qvel)[:2]

    def _obs(self):
        p,v=self.puck_state();pad,padv=self.paddle_state()
        q=[self.data.joint(n).qpos[0] for n in self.joint_names]
        dq=[self.data.joint(n).qvel[0] for n in self.joint_names]
        remaining=max(0,1-self.data.time/self.settings.episode_seconds)
        return np.asarray([*p,*v,*q,*dq,*pad,*padv,*self.command,remaining,float(self.touched)],dtype=np.float32)

    def _potential(self):
        """Small shaping signal for lining up with a straight incoming trajectory."""
        p,v=self.puck_state();pad,_=self.paddle_state()
        t=np.clip((self.defense_y-p[1])/v[1],0,2) if v[1]<-.05 else 0
        predicted_x=float(np.clip(p[0]+v[0]*t,-.30,.30))
        return -2*float(np.linalg.norm(pad-[predicted_x,self.defense_y]))

    def reset(self,*,seed=None,options=None):
        """Restore exact closed-link geometry and launch a newly randomized shot."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        # Center the pin in X, preserving its initial distance ahead of the base.
        target=np.array([0.,P['tip'][1]])
        angles=motor_targets(target)
        for i,side in enumerate(('left','right')):
            length1,_,home,home_elbow=chain(side)
            theta=home+angles[i]
            elbow=P[side+'_base']+length1*np.array([math.cos(theta),math.sin(theta)])
            forearm=math.atan2(*(target-elbow)[::-1])
            self.data.joint(side+'_motor').qpos[0]=angles[i]
            self.data.joint(side+'_elbow').qpos[0]=wrap(forearm-theta-home_elbow)
        self.command=angles.copy()
        # Aim toward the usable goal opening, making incoming shots genuinely threatening.
        x=self.settings.spawn_center_x_m+self.np_random.uniform(-self.settings.spawn_half_width_m,self.settings.spawn_half_width_m)
        goal_x=self.np_random.uniform(-.07,.07)
        position=np.array([x,.15]);direction=np.array([goal_x,-LENGTH/2])-position
        velocity=direction/np.linalg.norm(direction)*self.np_random.uniform(
            self.settings.min_puck_speed_m_s,self.settings.max_puck_speed_m_s)
        for i,a in enumerate(('x','y')):
            self.data.joint('puck_'+a).qpos[0]=position[i]
            self.data.joint('puck_'+a).qvel[0]=velocity[i]
        self.episode_steps=0;self.touched=False;self.done=False
        mujoco.mj_forward(self.model,self.data)
        self.previous_potential=self._potential()
        return self._obs(),{'outcome':'running','is_success':False}

    def step(self,action):
        """Apply an action, advance contacts, then reward a return or penalize a miss."""
        if self.done:
            raise RuntimeError('Call reset() before stepping a completed episode.')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must be two finite numbers.')
        desired=np.clip(action,-1,1)*self.settings.action_angle_rad
        max_change=self.settings.target_slew_rad_s*self.settings.control_dt
        self.command+=np.clip(desired-self.command,-max_change,max_change)
        contact_now=False
        for _ in range(self.frame_skip):
            for i,side in enumerate(('left','right')):
                joint=self.data.joint(side+'_motor')
                self.data.ctrl[i]=np.clip(25*(self.command[i]-joint.qpos[0])-4*joint.qvel[0],-TORQUE,TORQUE)
            mujoco.mj_step(self.model,self.data)
            # Check every physics step so brief puck/paddle impacts are not missed.
            for contact in self.data.contact[:self.data.ncon]:
                if {contact.geom1,contact.geom2}=={self.paddle_id,self.puck_id}:
                    contact_now=True
        mujoco.mj_forward(self.model,self.data)
        self.episode_steps+=1
        p,v=self.puck_state();pad,_=self.paddle_state()
        first_touch=contact_now and not self.touched
        self.touched=self.touched or contact_now
        gap=np.linalg.norm(self.data.site('left_tip').xpos-self.data.site('right_tip').xpos)
        # Unstable physics is a failed episode, not an exploitable success.
        invalid=(not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel))
                 or gap>.002 or any(w.number for w in self.data.warning))
        pad_r=self.settings.paddle_diameter_m/2
        outside=abs(pad[0])>WIDTH/2-pad_r or pad[1]<-LENGTH/2+pad_r or pad[1]>-.02
        goal=p[1]<-LENGTH/2-RADIUS
        returned=self.touched and p[1]>.15 and v[1]>.1
        # Crossing the whole end plane is a goal; a puck leaving sideways is invalid.
        escaped=abs(p[0])>WIDTH/2+RADIUS or p[1]>LENGTH/2+RADIUS
        terminated=bool(invalid or outside or goal or returned or escaped)
        truncated=bool(not terminated and self.data.time>=self.settings.episode_seconds)
        outcome=('invalid_physics' if invalid else 'paddle_outside' if outside else 'goal_conceded'
                 if goal else 'returned' if returned else 'puck_escaped' if escaped else
                 'timeout' if truncated else 'running')
        # Contact bonus can happen only once. Success requires a subsequent real return.
        reward=1.0 if first_touch else 0.0
        reward+=5.0 if outcome=='returned' else -10.0 if outcome=='goal_conceded' else -5.0 if terminated else 0.0
        if truncated:
            reward-=1.0
        potential=0.0 if terminated else self._potential()
        reward+=self.settings.gamma*potential-self.previous_potential
        self.previous_potential=potential
        reward-=.001*float(np.sum((self.data.ctrl/TORQUE)**2))
        self.done=terminated or truncated
        obs=self._obs()
        # Only numerical-failure termination may sanitize observations for the learner.
        if invalid:
            obs=np.nan_to_num(obs,nan=0,posinf=0,neginf=0)
        info={'outcome':outcome,'is_success':outcome=='returned','paddle_contact':contact_now,
              'closure_error_m':float(gap),'torque_nm':self.data.actuator_force.copy()}
        return obs,float(reward),terminated,truncated,info

    def render(self):
        """Launch a viewer for explicit human rendering; headless runs never open it."""
        if self.render_mode!='human':
            return
        if self.viewer is None:
            import mujoco.viewer as viewer_module
            self.viewer=viewer_module.launch_passive(self.model,self.data)
            self.viewer.cam.lookat[:]=[0,-.40,.05]
            self.viewer.cam.distance=2.6;self.viewer.cam.azimuth=90;self.viewer.cam.elevation=-65
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close();self.viewer=None
