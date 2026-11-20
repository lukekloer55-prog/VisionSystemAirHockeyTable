"""Experimental SB3 adapter: batched MJX movement physics, CPU rewards/policy."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
import mujoco
from stable_baselines3.common.vec_env import VecEnv
from skills_env import SkillsEnv
from plotclock_demo import TORQUE, ARM_Z

class MJXMovement(VecEnv):
    def __init__(self,count=16,device='gpu',seed=1,settings=None,mode='movement'):
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        self.jax,self.jp,self.mjx=jax,jp,mjx
        devices=jax.devices(device)
        if not devices:raise RuntimeError('Requested device unavailable; no CPU fallback')
        self.device=devices[0]
        print('MJX physics device:',self.device.device_kind,flush=True)
        self.mode=mode;self.pool=[];self.assignment=np.zeros(count,dtype=int)
        self.envs=[SkillsEnv(mode,settings=settings) for _ in range(count)]
        self.initial_seed=seed
        from hybrid_movement import apply_proxy
        for env in self.envs:apply_proxy(env)
        model=self.envs[0].model
        with jax.default_device(self.device):
            self.mx=mjx.put_model(model,impl='jax')
        qids=np.array([model.joint(n).qposadr[0] for n in self.envs[0].joints])
        motorq=model.jnt_qposadr[model.actuator_trnid[:,0]]
        motorv=model.jnt_dofadr[model.actuator_trnid[:,0]]
        paddles=[model.geom(pre+'paddle').id for pre in (['','opp_'] if mode=='game' else [''])]
        puck=model.geom('puck_shape').id
        substeps=round(self.envs[0].settings.control_dt/model.opt.timestep)
        def advance(data,command,active):
            def body(_,carry):
                d,travel,active,hits=carry
                ctrl=jp.clip(25*(command-d.qpos[motorq])-4*d.qvel[motorv],-TORQUE,TORQUE)
                nxt=mjx.step(self.mx,d.replace(ctrl=ctrl))
                contact=jp.stack([jp.any((((nxt.contact.geom[:,0]==pad)&(nxt.contact.geom[:,1]==puck))|((nxt.contact.geom[:,1]==pad)&(nxt.contact.geom[:,0]==puck)))&(nxt.contact.dist<=0)) for pad in paddles]) if mode=='game' else jp.zeros(1,dtype=bool)
                return nxt,travel+jp.sum(jp.abs(nxt.qpos[qids]-d.qpos[qids])),contact,hits|(contact&~active)
            return jax.lax.fori_loop(0,substeps,body,(data,jp.array(0.,dtype=data.qpos.dtype),active,jp.zeros(len(paddles),dtype=bool)))
        self.advance=jax.jit(jax.vmap(advance))
        super().__init__(count,self.envs[0].observation_space,self.envs[0].action_space)
    def reset(self):
        observations=[];states=[]
        with self.jax.default_device(self.device):
            for i,e in enumerate(self.envs):
                seed=self._seeds[i] if self._seeds[i] is not None else self.initial_seed+i
                obs,_=e.reset(seed=seed);observations.append(obs)
                states.append(self.mjx.put_data(e.model,e.data,impl='jax'))
            self.data=self.jax.tree.map(lambda *x:self.jp.stack(x),*states)
        self.assignment=np.array([e.np_random.integers(len(self.pool)) if self.pool else 0 for e in self.envs]);self.returns=np.zeros(self.num_envs);self.lengths=np.zeros(self.num_envs,dtype=int)
        self._reset_seeds();return np.array(observations)
    def step_async(self,actions):self.actions=np.asarray(actions)
    def step_wait(self):
        if self.actions.shape!=(self.num_envs,2) or not np.isfinite(self.actions).all():raise ValueError('Invalid actions')
        distances=[]
        other=np.zeros_like(self.actions)
        if self.mode=='game' and self.pool:
            for index,policy in enumerate(self.pool):
                ids=np.flatnonzero(self.assignment==index)
                if len(ids):other[ids]=policy.predict(np.array([self.envs[i].observation(1) for i in ids]),deterministic=True)[0]
        for e,a,b in zip(self.envs,self.actions,other):
            distances.append(np.linalg.norm(e.paddle(0)[0]-e.target))
            dt=e.settings.control_dt;limit=e.settings.target_slew_rad_s*dt
            e.commands[0]+=np.clip(np.clip(a,-1,1)*e.settings.action_angle_rad-e.commands[0],-limit,limit)
            if self.mode=='game':e.commands[1]+=np.clip(np.clip(b,-1,1)*e.settings.action_angle_rad-e.commands[1],-limit,limit)
        with self.jax.default_device(self.device):
            self.data,travel,active,hits=self.advance(self.data,self.jp.asarray([e.commands[:2 if self.mode=='game' else 1].reshape(-1) for e in self.envs]),self.jp.asarray([e.active[:2 if self.mode=='game' else 1] for e in self.envs]))
        q,v,t,travel,active,hits=self.jax.device_get((self.data.qpos,self.data.qvel,self.data.time,travel,active,hits))
        obs=[];rewards=[];dones=[];infos=[]
        for i,e in enumerate(self.envs):
            if not np.isfinite(q[i]).all() or not np.isfinite(v[i]).all():
                raise RuntimeError('Non-finite MJX physics; stopping rather than saving corrupted experience')
            e.data.qpos[:]=q[i];e.data.qvel[:]=v[i];e.data.time=float(t[i])
            o,r,terminated,truncated,info=e.finish_step(distances[i],float(travel[i]),hits[i])
            e.active=list(active[i]);self.returns[i]+=r;self.lengths[i]+=1
            done=terminated or truncated;info['TimeLimit.truncated']=truncated and not terminated
            if done:
                info['episode']={'r':float(self.returns[i]),'l':int(self.lengths[i]),'t':float(e.data.time)}
                self.returns[i]=0;self.lengths[i]=0
                info['terminal_observation']=o.copy()
                o,self.reset_infos[i]=e.reset()
                if self.pool:self.assignment[i]=e.np_random.integers(len(self.pool))
                with self.jax.default_device(self.device):
                    reset=self.mjx.put_data(e.model,e.data,impl='jax')
                    self.data=self.jax.tree.map(lambda batch,single:batch.at[i].set(single),self.data,reset)
            obs.append(o);rewards.append(r);dones.append(done);infos.append(info)
        return np.array(obs),np.array(rewards,np.float32),np.array(dones),infos
    def set_pool(self,paths):
        from stable_baselines3 import PPO
        self.pool=[PPO.load(p,device='cpu') for p in paths]
        self.assignment=np.array([e.np_random.integers(len(self.pool)) if self.pool else 0 for e in self.envs])
    def close(self):
        for e in self.envs:e.close()
    def get_attr(self,name,indices=None):return [getattr(self.envs[i],name) for i in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):
        for i in self._get_indices(indices):setattr(self.envs[i],name,value)
    def env_method(self,name,*args,indices=None,**kwargs):return [getattr(self.envs[i],name)(*args,**kwargs) for i in self._get_indices(indices)]
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
