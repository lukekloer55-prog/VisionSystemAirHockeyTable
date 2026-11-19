"""Reward-panel integration and per-checkpoint configuration snapshots."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

from stable_baselines3.common.callbacks import BaseCallback
from airhockey_env import TASK_SCHEMA


def snapshot(folder,settings,model_name=None):
    """Keep each checkpoint's reward settings, even if sliders change later."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    name=Path(model_name).stem+'.settings.json' if model_name else 'settings.json'
    settings.save(folder/name)
    (folder/'task_schema.json').write_text(json.dumps(TASK_SCHEMA,indent=2))
    shutil.copyfile(Path(__file__).with_name('plotclock_geometry.json'),folder/'plotclock_geometry.json')


class SnapshotCheckpoint(BaseCallback):
    def __init__(self,env,folder,every=10000):
        super().__init__();self.env=env;self.folder=Path(folder);self.every=every
    def _on_step(self):
        if self.n_calls%self.every==0:
            path=self.folder/f'policy_{self.num_timesteps}_steps.zip'
            path.parent.mkdir(parents=True,exist_ok=True)
            self.model.save(path)
            snapshot(self.folder,self.env.settings,path.name)
        return True


class BestSnapshot(BaseCallback):
    def __init__(self,env,folder):
        super().__init__();self.env=env;self.folder=folder
    def _on_step(self):
        snapshot(self.folder,self.env.settings,'best_model.zip')
        return True


class RewardPanelCallback(BaseCallback):
    def __init__(self,panel,env,eval_env,run,evaluation=None):
        super().__init__();self.panel=panel;self.env=env;self.eval_env=eval_env;self.run=Path(run);self.evaluation=evaluation
    def _on_rollout_start(self):
        change=self.panel.take_pending()
        if change:
            for env in (self.env,self.eval_env):
                env.settings=replace(env.settings,**change['values']).validate()
                env.bonus_remaining=max(0,env.settings.event_bonus_budget-env.bonus_paid)
                # Avoid an artificial shaping reward just because the weight changed.
                env.previous_potential=env._potential()
            # Budget is part of the observation; refresh PPO's cached observation.
            self.model._last_obs[0]=self.env._obs()
            if self.evaluation is not None:
                self.evaluation.best_mean_reward=float("-inf")
            snapshot(self.run,self.env.settings)
            self.panel.applied(change,self.num_timesteps)
            print(f'Reward sliders applied at step {self.num_timesteps}.')
        elif self.panel.status=='Waiting for the first training batch':
            with self.panel.lock:
                self.panel.status='Training active. Edit sliders and click Apply.'
    def _on_step(self):
        self.panel.update_metrics(self.num_timesteps,self.locals.get('infos',[]))
        return True
