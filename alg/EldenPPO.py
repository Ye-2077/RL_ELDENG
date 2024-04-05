import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

class ResBlock(nn.Module):
    """
    A simple residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # Skip connection
        return self.relu(out)

class EledenFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim: int = 256
                 ):
        
        super(EledenFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)
        
        # cnn for 64x64 image input
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # output: [32, 64, 64]
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # output: [32, 32, 32]
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output: [64, 32, 32]
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # output: [64, 16, 16]
        #     nn.Flatten(),  # flatten
        # )
         # Adjust the calculation of cnn_output_size according to the added layers and max pooling
        # cnn_output_size = 128 * (16//4) * (16//4)  # Assuming additional max pooling reduces size

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(64),  # Adding a residual block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Increasing depth
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Further reducing dimensions
            nn.Flatten(),
        )
        cnn_output_size = 8192  # 128 * 16 * 16

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # output: [32, 128, 128]
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # output: [32, 64, 64]
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output: [64, 64, 64]
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # output: [64, 32, 32]
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 新增的卷积层, output: [128, 32, 32]
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # 新增的池化层, output: [128, 16, 16]
        #     nn.Flatten(),  # flatten
        # )
        
        # caculate the output size after the CNN
        # two maxpooling layers for 64 channel 16x16 image, the output size is 64*16*16
        # cnn_output_size = 128 * 16 * 16
        
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
                 check_freq: int =5,
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

        # print(f"---------- Episode {self.episode_counter} / Step {self.num_timesteps} ----------")

        # print(f"Action Taken: {info['action_taken']}")
        # print(f"Boss Health: {info['boss_health']} | Player Health: {info['player_health']}")
        # print(f"Boss Damage: {info['boss_damage']} | Player Damage: {info['player_damage']}")
        # print(f"Health Keep Count: {info['health_keep_count']}")

        # print(f"Episode Total Reward: {info['total_reward']}")        
        # # print(f"- reward_damage: {info['rew_damage']}")
        # # print(f"- reward_action_cost: {info['rew_action_cost']}")
        # # print(f"- reward_health_keep: {info['rew_health_keep']}")
        # # print(f"- reward_boss_low_health: {info['rew_boss_low_health']}")
        
        # print(f"---------------------------------")
        
        # if self.n_calls % self.best_check_freq == 0:
        #     x, y = self.model.episode_reward_history[-1]
        #     if y > self.best_mean_reward:
        #         self.best_mean_reward = y
        #         self.model.save(self.log_dir + 'best_model')
        #         print(f"Best mean reward updated: {self.best_mean_reward}")

        # if 'done' in self.locals and self.locals['done']:
        #     self.episode_counter += 1

        #     if self.episode_counter % self.check_freq == 0:
        #         model_path = f"{self.log_dir}/model_{self.episode_counter}.zip"
        #         self.model.save(model_path)
        #         print(f"Model saved: {model_path}")
        
        if self.n_calls % 500 == 0:
            model_path = f"{self.log_dir}/model_{self.n_calls}.zip"
            self.model.save(model_path)

        
        return True