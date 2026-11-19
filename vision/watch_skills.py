"""Watch saved movement or self-play models without slowing the training process."""
import argparse
import json
from pathlib import Path
import time
from skills_env import SkillsEnv, Curriculum, SCHEMA
from airhockey_env import Settings
from train_skills import Store


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--speed',type=float,default=1.)
    args=p.parse_args()
    import math
    if not math.isfinite(args.speed) or args.speed<=0:p.error('speed must be positive and finite')
    from stable_baselines3 import PPO
    geometry=json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
    env=None;episode=0
    try:
        while True:
            # Trainer can publish/prune while the viewer reads. Retry a vanished generation.
            for attempt in range(3):
                try:
                    state,folder=Store(args.run_dir).load()
                    if state['schema']!=SCHEMA or state['geometry']!=geometry:raise ValueError('Geometry/schema mismatch')
                    a=PPO.load(folder/('a.zip' if (folder/'a.zip').exists() else 'champion.zip'),device='cpu')
                    b=PPO.load(folder/('b.zip' if (folder/'b.zip').exists() else 'champion.zip'),device='cpu')
                    break
                except FileNotFoundError:
                    if attempt==2:raise
            if env is None:env=SkillsEnv('movement' if state['mode']=='movement' else 'game',Settings(**state['settings']),Curriculum(**state['curriculum']),'human')
            env.reset(seed=episode);episode+=1
            print(f"Snapshot round {state['round']}, {state['phase']}; display episode {episode}",flush=True)
            while True:
                start=time.monotonic()
                action=a.predict(env.observation(),deterministic=True)[0]
                other=b.predict(env.observation(1),deterministic=True)[0] if env.mode=='game' else [0,0]
                _,_,term,trunc,info=env.step_pair(action,other)
                if env.viewer is not None and not env.viewer.is_running():return
                env.render()
                delay=env.settings.control_dt/args.speed-(time.monotonic()-start)
                if delay>0:time.sleep(delay)
                if term or trunc:
                    print(info['outcome'],flush=True);break
    except KeyboardInterrupt:pass
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
