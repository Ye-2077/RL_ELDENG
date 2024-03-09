import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import envs.EledenGym # import EledenGym, must be imported
from alg.EldenPPO import EledenFeatureExtractor, EldenCallback

from utils.utils_start import setup_game

if __name__ == "__main__":

    config = {
        'learning_rate': 2e-4,
        'n_steps': 256,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
    }
    model_callbacks = EldenCallback(log_dir = "./models")

    setup_game()
    env = gym.make('EledenGym-v0')
    model = PPO(ActorCriticPolicy, env, policy_kwargs={
                    'features_extractor_class': EledenFeatureExtractor,
                    'features_extractor_kwargs': {'features_dim': 256}},
                verbose=1,
                tensorboard_log="./logs/",
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gamma=config['gamma']
                )
    model.learn(total_timesteps=10000, callback=model_callbacks)
    model.save("ppo_custom_feature_extractor")