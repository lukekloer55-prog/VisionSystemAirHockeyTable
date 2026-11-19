"""Train a CPU PPO policy in the simulation; no real motor commands are sent."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

from airhockey_env import AirHockeyEnv, Settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=300000,help='Additional environment steps to train')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--config',type=Path,help='JSON file with measured physical parameters')
    parser.add_argument('--run-dir',type=Path,help='New output folder; default is a timestamped runs/ folder')
    parser.add_argument('--resume',type=Path,help='Checkpoint ZIP to continue; uses its adjacent settings.json')
    parser.add_argument('--threads',type=int,default=2,help='CPU threads for the small policy network')
    args=parser.parse_args()
    if args.steps<=0 or args.threads<=0:
        parser.error('steps and threads must be positive')
    # Imports here keep the basic movement demo independent of PyTorch.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(args.threads)
    if args.resume and args.config:
        parser.error('Resume uses the original settings; do not also specify --config.')
    if args.resume:
        candidates=[args.resume.parent/'settings.json',args.resume.parent.parent/'settings.json']
        config_path=next((p for p in candidates if p.exists()),None)
        if config_path is None:
            parser.error('Cannot find settings.json beside the checkpoint or its parent folder.')
        settings=Settings.load(config_path)
        saved_geometry=config_path.with_name('plotclock_geometry.json')
        current_geometry=Path(__file__).with_name('plotclock_geometry.json')
        if not saved_geometry.exists() or json.loads(saved_geometry.read_text())!=json.loads(current_geometry.read_text()):
            parser.error('Cannot resume: arm geometry differs from the original training run.')
    else:
        settings=Settings.load(args.config)
    run=args.run_dir or Path('runs')/datetime.now().strftime('defender-%Y%m%d-%H%M%S-%f')
    # Never overwrite an existing training run or checkpoint directory.
    run.mkdir(parents=True,exist_ok=False)
    settings.save(run/'settings.json')
    # Save the geometry used by this run to detect accidental incompatible playback.
    geometry=Path(__file__).with_name('plotclock_geometry.json')
    shutil.copyfile(geometry,run/'plotclock_geometry.json')
    (run/'run.json').write_text(json.dumps({'seed':args.seed,'requested_steps':args.steps,
        'algorithm':'PPO','device':'cpu','resume':str(args.resume) if args.resume else None},indent=2))
    env=Monitor(AirHockeyEnv(settings),str(run/'train.monitor.csv'))
    eval_env=Monitor(AirHockeyEnv(settings),str(run/'eval.monitor.csv'))
    eval_env.reset(seed=args.seed+10000)
    model=None
    try:
        if args.resume:
            model=PPO.load(args.resume,env=env,device='cpu')
        else:
            model=PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=1,
                n_steps=1024,batch_size=64,n_epochs=5,learning_rate=3e-4,
                gamma=settings.gamma,policy_kwargs={'net_arch':[64,64]})
        model.set_logger(configure(str(run),['stdout','csv']))
        callbacks=[CheckpointCallback(save_freq=10000,save_path=str(run/'checkpoints'),name_prefix='policy'),
            EvalCallback(eval_env,best_model_save_path=str(run/'best'),log_path=str(run/'evaluation'),
                         eval_freq=10000,n_eval_episodes=10,deterministic=True)]
        # Best/checkpoint folders carry settings too, so playback remains self-contained.
        for directory in ('best','checkpoints'):
            (run/directory).mkdir(exist_ok=True)
            settings.save(run/directory/'settings.json')
            shutil.copyfile(geometry,run/directory/'plotclock_geometry.json')
        print(f'Training on CPU. Results: {run.resolve()}')
        print('Ctrl+C stops training and saves latest.zip. No window opens during training.')
        try:
            model.learn(total_timesteps=args.steps,callback=callbacks,reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print('Interrupted; saving current weights.')
        model.save(run/'latest.zip')
        print(f'Saved: {run / "latest.zip"}')
        print(f'Watch: uv run --extra train python play_policy.py --model "{run / "latest.zip"}"')
    finally:
        env.close();eval_env.close()


if __name__=='__main__':
    main()
