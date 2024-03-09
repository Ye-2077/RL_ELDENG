import cv2
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import utils.utils_actions as actions
import utils.utils_start as start
import utils.utils_vision as vision


class EledenGym(gym.Env):
    
    def __init__(self):
        super(EledenGym, self).__init__()
        
        self.vision_size = 64
        self.action_dict = {
            0: actions.light_attack,
            1: actions.defend,
            2: actions.dodge_right,
            3: actions.go_right,
            4: actions.do_nothing
        }

        self.observation_space = spaces.Dict({
            "state_values": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            "vision_image": spaces.Box(low=0, high=255, shape=(1, self.vision_size, self.vision_size), dtype=np.uint8)
        })
        self.action_space = spaces.Discrete(len(self.action_dict))

        self.player_blood_region = (80,60,500,71)
        self.boss_blood_region = (245,500,800,505)
        self.state_window = (315,100,715,500)

        self.state = None
        self.initial_player_health = None
        self.initial_boss_health = None
        self.previous_blood = {"player_blood": 100, "boss_blood": 100}
        self.action_history = []
        self.total_reward = 0
        self.action_rewards = {}

    def reset(self, seed=None, options=None):
        if self.state is not None:
            start.restart_game()
        self.previous_blood = {"player_blood": 100, "boss_blood": 100}
        self.action_history = []
        self.initial_boss_health = None
        self.initial_player_health = None
        self.total_reward = 0
        self.action_rewards = {}

        self.state = self._get_observation()
        info = {}

        # state = self.state
        # print(type(state))
        # print(type(state["state_values"]))
        # print(type(state["vision_image"]))
        
        return self.state, info


    def step(self, action):
        # action = action[0] if isinstance(action, np.ndarray) else action
        
        action = self._step_action(action) # step the action
        state = self._get_observation() # get the observation
        done = self._check_done() # check if the game is done
        reward = self._get_reward() # get the reward

        self.state = state
        self.total_reward += reward

        info = {
            "action_taken": self.action_dict[action].__name__,
            "boss_health": self.state["state_values"][1],
            "player_health": self.state["state_values"][0],
            "total_reward": self.total_reward
        }

        # print(type(state))
        # print(type(state["state_values"]))
        # print(type(state["vision_image"]))

        return state, reward, done, False, info


    def render(self, mode='human'):
        pass


    def close(self):
        pass


    def _step_action(self, action):

        if action in self.action_dict:
            action_func = self.action_dict[action]
            action_func()
        else:
            print(f"[ERROR] Action {action} not found in action_dict")

        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        print(f"Action: {action_func.__name__}")
        return action


    def _get_observation(self):

        state_window = vision.grab_screen2gray(self.state_window)
        player_blood = vision.grab_screen2hsv(self.player_blood_region)
        boss_blood = vision.grab_screen2hsv(self.boss_blood_region)

        vision_image = self._preprocess_image(state_window)
        player_blood, _ = vision.grab_blood(player_blood)
        boss_blood, _ = vision.grab_blood(boss_blood)

        if self.initial_player_health is None or self.initial_boss_health is None:
            self.initial_player_health = player_blood
            self.initial_boss_health = boss_blood
            # print(f"Initial player health: {self.initial_player_health}")
            # print(f"Initial boss health: {self.initial_boss_health}")

        player_blood_percent = int(player_blood / self.initial_player_health * 100)
        boss_blood_percent = int(boss_blood / self.initial_boss_health * 100)

        # state_values = np.array([player_blood_percent, boss_blood_percent])
        # state_values = torch.tensor(state_values, dtype=torch.float32)
        # state_values = state_values.unsqueeze(0) # add batch dimension
        
        state_values = np.array([player_blood_percent, boss_blood_percent], dtype=np.float32)
        vision_image = vision_image.numpy()

        observation = {
            "state_values": state_values,
            "vision_image": vision_image
        }

        # observation = observation.numpy()

        return observation

    
    def _preprocess_image(self, image):
        # resize
        image_resize = cv2.resize(image, (self.vision_size, self.vision_size))

        # convert to tensor
        image_tensor = torch.tensor(image_resize, dtype=torch.uint8)
        # image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.unsqueeze(0) # add batch dimension

        return image_tensor


    def _check_done(self):
        state_values = self.state["state_values"]
        if state_values[0] <= 0 or state_values[1] <= 0:
            return True
        else:
            return False


    def _get_reward(self):
        reward = 0
        current_blood = self.state["state_values"]

        # boss get hit
        boss_damage = self.previous_blood["boss_blood"] - current_blood[1]
        if boss_damage > 0:
            reward += 1

        # player get hit
        player_damage = self.previous_blood["player_blood"] - current_blood[0]
        if player_damage > 0:
            reward -= 1
        else:
            reward += 0.1

        # player dodge successfully
        if len(self.action_history) >= 2 and self.action_history[-2:] in [[actions.light_attack, actions.dodge_left], [actions.light_attack, actions.dodge_right]] and player_damage == 0:
            reward += 1
            if boss_damage >= 0:
                reward += 1
        if self.action_history[-1] in [actions.dodge_right, actions.dodge_left] and player_damage == 0:
            reward += 1

        # player defend successfully
        if self.action_history[-1] == actions.defend and player_damage <= 20:
            reward += 1

        if self.action_history[-1] == actions.light_attack:
            if boss_damage == 0:
                reward -= 0.1
        
        if self.action_history[-1] == actions.do_nothing:
            reward -= 0.15

        # player attack after dodge
        # if len(self.action_history) >= 2 and self.action_history[-2:] == [actions.dodge_right, actions.dodge_left, actions.light_attack]:
        #     if boss_damage > 0:
        #         reward += 1
        #     else:
        #         reward -= 0.5
        
        # if self.action_history[-1] == actions.special_attack:
        #     if boss_damage >= 5:
        #         reward += 1
        #     else:
        #         reward -= 0.1
        
        self.previous_blood = {"player_blood": current_blood[0], "boss_blood": current_blood[1]}

        return reward
    

register(
    id='EledenGym-v0',
    entry_point='envs.EledenGym:EledenGym',
)