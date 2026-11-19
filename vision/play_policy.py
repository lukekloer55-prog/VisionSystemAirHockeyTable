"""Play or evaluate a saved policy, using the same physical settings as training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from airhockey_env import AirHockeyEnv, Settings, TASK_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True,help='Saved policy ZIP')
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=20000,help='Separate from training/evaluation seeds')
    parser.add_argument('--headless',action='store_true',help='Measure outcomes without opening a viewer')
    args=parser.parse_args()
    if args.episodes<=0:
        parser.error('episodes must be positive')
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(2)
    folder=args.model.parent
    schema=folder/'task_schema.json'
    if not schema.exists() or json.loads(schema.read_text())!=TASK_SCHEMA:
        parser.error('This policy is not a full-rally policy. Use the earlier package to play an earlier checkpoint.')
    specific=args.model.with_suffix('.settings.json')
    settings=Settings.load(specific if specific.exists() else folder/'settings.json')
    current=json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
    saved=json.loads((folder/'plotclock_geometry.json').read_text())
    if current!=saved:
        parser.error('Arm geometry differs from the model training geometry. Restore the original geometry.')
    env=AirHockeyEnv(settings,render_mode=None if args.headless else 'human')
    model=PPO.load(args.model,device='cpu')
    outcomes=Counter();returns=[]
    try:
        for episode in range(args.episodes):
            obs,_=env.reset(seed=args.seed+episode)
            total=0.0
            while True:
                start=time.monotonic()
                action,_=model.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                total+=reward
                if not args.headless:
                    env.render()
                    if not env.viewer.is_running():
                        return
                    time.sleep(max(0,settings.control_dt-(time.monotonic()-start)))
                if terminated or truncated:
                    outcomes[info['outcome']]+=1;returns.append(total)
                    print(f'Episode {episode+1}: {info["outcome"]}; reward {total:.2f}')
                    break
        print(f'Outcomes: {dict(outcomes)}; mean reward: {np.mean(returns):.2f}')
        print(f'Goal rate: {outcomes["goal_scored"] / args.episodes:.1%}')
    finally:
        env.close()


if __name__=='__main__':
    main()
