"""Write per-rally reward components so shaping and real goals can be distinguished."""
from stable_baselines3.common.callbacks import BaseCallback


class RallyMetricsCallback(BaseCallback):
    def _on_step(self):
        # Monitor adds 'episode' only when the rally finishes or times out.
        for info in self.locals.get('infos',[]):
            if 'episode' in info:
                for name,value in info.get('episode_rewards',{}).items():
                    self.logger.record_mean('rally_reward/'+name,value)
                self.logger.record_mean('rally/useful_hits',info.get('hits',0))
                self.logger.record_mean('rally/center_crossings',info.get('crossings',0))
        return True
