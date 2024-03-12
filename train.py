import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import argparse
import os
from datetime import datetime

import envs.EledenGym  # import EledenGym, must be imported
from alg.EldenPPO import EledenFeatureExtractor, EldenCallback
from utils.utils_start import setup_game


parser = argparse.ArgumentParser()
parser.add_argument("--keep_checkpoint", help="path to pre-trained model, continue trained if provided", type=str, default="")
args = parser.parse_args()


if __name__ == "__main__":
    config = {
        'learning_rate': 1e-4,
        'n_steps': 128,
        'batch_size': 16,
        'n_epochs': 10,
        'gamma': 0.95,
    }
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_path = os.path.join("./models/", now)
    os.makedirs(log_dir_path, exist_ok=True)

    model_callbacks = EldenCallback(log_dir=log_dir_path)
    # setup_game()
    env = gym.make('EledenGym-v0')

    # check if using pre-trained model
    if args.keep_checkpoint:
        try:
            model = PPO.load(args.keep_checkpoint, env=env)
            print(f"[INFO] successfully loaded pretrained model: {args.keep_checkpoint}, continue training")
        except FileNotFoundError:
            print(f"[INFO] faied to load pretrained model: {args.keep_checkpoint}, stop training")
            exit()
        
    else:
        print(f"[INFO] no pretrained model provided, start training from scratch")
        model = PPO(ActorCriticPolicy, env, policy_kwargs={
                        'features_extractor_class': EledenFeatureExtractor,
                        'features_extractor_kwargs': {'features_dim': 256}},
                    verbose=1,
                    tensorboard_log = log_dir_path,
                    learning_rate = config['learning_rate'],
                    n_steps = config['n_steps'],
                    batch_size = config['batch_size'],
                    n_epochs = config['n_epochs'],
                    gamma = config['gamma']
                    )

    setup_game()
    model.learn(total_timesteps=10000, callback=model_callbacks)
    model.save("ppo_custom_feature_extractor")
