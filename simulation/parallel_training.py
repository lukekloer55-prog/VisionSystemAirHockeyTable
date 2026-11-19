"""Spawned CPU simulation workers; all rollouts train one shared PPO policy."""
from functools import partial
from dataclasses import asdict
from skills_env import SkillsEnv, Curriculum
from airhockey_env import Settings

class WorkerEnv(SkillsEnv):
    def preview(self):
        if self.render_mode!='human':return True
        if self.viewer is not None and not self.viewer.is_running():return False
        self.render()
        return self.viewer.is_running()

    def configure_training(self, curriculum, opponent_path):
        if asdict(self.curriculum)!=curriculum:self.set_curriculum(Curriculum(**curriculum))
        if opponent_path != getattr(self,'_opponent_path',None):
            from stable_baselines3 import PPO
            self.opponent=PPO.load(opponent_path,device='cpu') if opponent_path else None
            self._opponent_path=opponent_path

def make_worker(mode,settings,curriculum,render):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    return Monitor(WorkerEnv(mode,Settings(**settings),Curriculum(**curriculum),'human' if render else None))

def make_parallel_env(mode,settings,curriculum,count,render=False):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    return SubprocVecEnv([partial(make_worker,mode,asdict(settings),asdict(curriculum),render) for _ in range(count)],start_method='spawn')
