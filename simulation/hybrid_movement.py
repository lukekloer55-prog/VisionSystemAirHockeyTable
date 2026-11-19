"""One PPO rollout composed of concurrent GPU and spawned CPU worlds."""
import numpy as np
import mujoco
from functools import partial
from stable_baselines3.common.vec_env import VecEnv,SubprocVecEnv
from skills_env import SkillsEnv,Curriculum
from airhockey_env import Settings
from plotclock_demo import ARM_Z

def apply_proxy(env):
    env.model.geom_condim[:]=1
    for name in ('puck_shape','paddle'):
        g=env.model.geom(name);r=float(g.size[0])
        g.type=mujoco.mjtGeom.mjGEOM_SPHERE;g.size[:]=[r,0,0]
        if name=='paddle':g.pos[2]=.006-ARM_Z
        env.model.geom_rbound[g.id]=r;env.model.geom_aabb[g.id]=[0,0,0,r,r,r]

class CPUWorld(SkillsEnv):
    def live_state(self):return self.data.qpos.copy(),self.data.qvel.copy(),self.target.copy()

def worker(settings,curriculum):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    env=CPUWorld('movement',Settings(**settings),Curriculum(**curriculum));apply_proxy(env)
    return Monitor(env)

class HybridMovement(VecEnv):
    def __init__(self,gpu,cpu_count,settings,curriculum,seed):
        from dataclasses import asdict
        self.gpu=gpu;self.envs=gpu.envs
        self.cpu=SubprocVecEnv([partial(worker,asdict(settings),asdict(curriculum)) for _ in range(cpu_count)],start_method='spawn')
        self.cpu.seed(seed+100000)
        super().__init__(gpu.num_envs+cpu_count,gpu.observation_space,gpu.action_space)
    def reset(self):
        if self._seeds[0] is not None:
            self.gpu.seed(self._seeds[0]);self.cpu.seed(self._seeds[0]+100000)
        self._reset_seeds()
        return np.concatenate([self.gpu.reset(),self.cpu.reset()])
    def step_async(self,actions):
        # CPU workers start before GPU integration, so both execute concurrently.
        self.cpu.step_async(actions[self.gpu.num_envs:]);self.gpu.step_async(actions[:self.gpu.num_envs])
    def step_wait(self):
        a=self.gpu.step_wait();b=self.cpu.step_wait()
        return (*[np.concatenate([a[i],b[i]]) for i in range(3)],list(a[3])+list(b[3]))
    def set_curriculum(self,c):
        for e in self.gpu.envs:e.set_curriculum(c)
        self.cpu.env_method('set_curriculum',c)
    def close(self):
        try:self.cpu.close()
        finally:self.gpu.close()
    def _call(self,method,*args,indices=None,**kwargs):
        out=[]
        for i in self._get_indices(indices):
            env=self.gpu if i<self.gpu.num_envs else self.cpu
            index=i if env is self.gpu else i-self.gpu.num_envs
            result=getattr(env,method)(*args,indices=[index],**kwargs)
            if result is not None:out.extend(result)
        return out
    def get_attr(self,name,indices=None):return self._call('get_attr',name,indices=indices)
    def set_attr(self,name,value,indices=None):return self._call('set_attr',name,value,indices=indices)
    def env_method(self,name,*args,indices=None,**kwargs):return self._call('env_method',name,*args,indices=indices,**kwargs)
    def env_is_wrapped(self,cls,indices=None):return self._call('env_is_wrapped',cls,indices=indices)

