"""Display the actual training environment, without running a second policy."""
import time
from stable_baselines3.common.callbacks import BaseCallback


class LiveTrainingCallback(BaseCallback):
    """Pace training rollouts so the user can follow the puck and arm."""

    def __init__(self, env, speed=1.0):
        super().__init__()
        self.env = env
        self.frame_seconds = env.settings.control_dt / speed
        self.last_frame = None

    def _on_training_start(self):
        # PPO has reset the environment before this hook, so the scene is ready.
        self.env.render()
        self.last_frame = time.monotonic()

    def _on_step(self):
        # Closing the viewer asks PPO to stop; train_policy.py then saves latest.zip.
        if not self.env.viewer.is_running():
            print('Training window closed; stopping and saving current weights.')
            return False
        remaining = self.frame_seconds - (time.monotonic() - self.last_frame)
        if remaining > 0:
            time.sleep(remaining)
        self.env.render()
        self.last_frame = time.monotonic()
        return self.env.viewer.is_running()
