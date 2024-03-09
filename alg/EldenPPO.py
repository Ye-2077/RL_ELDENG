import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

class EledenFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim: int = 256
                 ):
        
        super(EledenFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)
        
        # cnn for 64x64 image input
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # output: [32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output: [32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output: [64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output: [64, 16, 16]
            nn.Flatten(),  # flatten
        )
        
        # caculate the output size after the CNN
        # two maxpooling layers for 64 channel 16x16 image, the output size is 64*16*16
        cnn_output_size = 64 * 16 * 16
        
        # fully connected layer, combined the CNN output and state_values
        # extra 2 for the state_values
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size + 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        vision_input = observations["vision_image"].float() / 255.0 # regularize the image
        state_values = observations["state_values"].float()

        vision_features = self.cnn(vision_input)  # cnn process
        combined_features = torch.cat([vision_features, state_values], dim=1)  # concatenate the vision features and state values
        return self.linear(combined_features)  # fully connected layer


class EldenCallback(BaseCallback):
    def __init__(self, 
                 check_freq: int =10,
                 best_check_freq: int=100, 
                 log_dir: str = None, 
                 verbose=1):
        
        super(EldenCallback, self).__init__(verbose)

        self.check_freq = check_freq
        self.best_check_freq = best_check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float('inf')
        self.episode_counter = 1
        
    
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]

        if self.n_calls % 10 == 0:

            print(f"---------- Episode {self.episode_counter} / Step {self.num_timesteps} ----------")

            print(f"- Episode Total Reward: {info['total_reward']}")
            print(f"- Action Taken: {info['action_taken']}")
            print(f"- Boss Health: {info['boss_health']}")
            print(f"- Player Health: {info['player_health']}")
        
        # if self.n_calls % self.best_check_freq == 0:
        #     x, y = self.model.episode_reward_history[-1]
        #     if y > self.best_mean_reward:
        #         self.best_mean_reward = y
        #         self.model.save(self.log_dir + 'best_model')
        #         print(f"Best mean reward updated: {self.best_mean_reward}")

        if 'done' in self.locals and self.locals['done']:
            self.episode_counter += 1

            if self.episode_counter % self.check_freq == 0:
                model_path = f"{self.log_dir}/model_{self.episode_counter}.zip"
                self.model.save(model_path)
                print(f"Model saved: {model_path}")
            

        
        return True