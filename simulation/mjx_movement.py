"""Experimental SB3 adapter: batched MJX movement physics, CPU rewards/policy."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
import mujoco
from stable_baselines3.common.vec_env import VecEnv
from skills_env import SkillsEnv
from plotclock_demo import TORQUE, ARM_Z

class MJXMovement(VecEnv):
    def __init__(self,count=16,device='gpu',seed=1,settings=None):
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        self.jax,self.jp,self.mjx=jax,jp,mjx
        devices=jax.devices(device)
        if not devices:raise RuntimeError('Requested device unavailable; no CPU fallback')
        self.device=devices[0]
        print('MJX physics device:',self.device.device_kind,flush=True)
        self.envs=[SkillsEnv('movement',settings=settings) for _ in range(count)]
        self.initial_seed=seed
        from hybrid_movement import apply_proxy
        for env in self.envs:apply_proxy(env)
        model=self.envs[0].model
        with jax.default_device(self.device):
            self.mx=mjx.put_model(model,impl='jax')
        qids=np.array([model.joint(n).qposadr[0] for n in self.envs[0].joints])
        motorq=qids[[0,2]]
        motorv=np.array([model.joint(n).dofadr[0] for n in ('left_motor','right_motor')])
        substeps=round(self.envs[0].settings.control_dt/model.opt.timestep)
        def advance(data,command):
            def body(_,carry):
                d,travel=carry
                ctrl=jp.clip(25*(command-d.qpos[motorq])-4*d.qvel[motorv],-TORQUE,TORQUE)
                nxt=mjx.step(self.mx,d.replace(ctrl=ctrl))
                return nxt,travel+jp.sum(jp.abs(nxt.qpos[qids]-d.qpos[qids]))
            return jax.lax.fori_loop(0,substeps,body,(data,jp.array(0.,dtype=data.qpos.dtype)))
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
        self._reset_seeds();return np.array(observations)
    def step_async(self,actions):self.actions=np.asarray(actions)
    def step_wait(self):
        if self.actions.shape!=(self.num_envs,2) or not np.isfinite(self.actions).all():raise ValueError('Invalid actions')
        distances=[]
        for e,a in zip(self.envs,self.actions):
            distances.append(np.linalg.norm(e.paddle(0)[0]-e.target))
            dt=e.settings.control_dt;limit=e.settings.target_slew_rad_s*dt
            e.commands[0]+=np.clip(np.clip(a,-1,1)*e.settings.action_angle_rad-e.commands[0],-limit,limit)
        with self.jax.default_device(self.device):
            self.data,travel=self.advance(self.data,self.jp.asarray([e.commands[0] for e in self.envs]))
        q,v,t,travel=self.jax.device_get((self.data.qpos,self.data.qvel,self.data.time,travel))
        obs=[];rewards=[];dones=[];infos=[]
        for i,e in enumerate(self.envs):
            if not np.isfinite(q[i]).all() or not np.isfinite(v[i]).all():
                raise RuntimeError('Non-finite MJX physics; stopping rather than saving corrupted experience')
            e.data.qpos[:]=q[i];e.data.qvel[:]=v[i];e.data.time=float(t[i])
            o,r,terminated,truncated,info=e.finish_step(distances[i],float(travel[i]),[False,False])
            done=terminated or truncated;info['TimeLimit.truncated']=truncated and not terminated
            if done:
                info['terminal_observation']=o.copy()
                o,self.reset_infos[i]=e.reset()
                with self.jax.default_device(self.device):
                    reset=self.mjx.put_data(e.model,e.data,impl='jax')
                    self.data=self.jax.tree.map(lambda batch,single:batch.at[i].set(single),self.data,reset)
            obs.append(o);rewards.append(r);dones.append(done);infos.append(info)
        return np.array(obs),np.array(rewards,np.float32),np.array(dones),infos
    def close(self):
        for e in self.envs:e.close()
    def get_attr(self,name,indices=None):return [getattr(self.envs[i],name) for i in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):
        for i in self._get_indices(indices):setattr(self.envs[i],name,value)
    def env_method(self,name,*args,indices=None,**kwargs):return [getattr(self.envs[i],name)(*args,**kwargs) for i in self._get_indices(indices)]
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
