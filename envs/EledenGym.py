import cv2
import time
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
            1: actions.go_backward,
            2: actions.dodge_right,
            3: actions.go_right,
            4: actions.go_left,
            5: actions.dodge_left,
            6: actions.special_attack,
            7: actions.go_backward,
        }

        self.observation_space = spaces.Dict({
            "state_values": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            "vision_image": spaces.Box(low=0, high=255, shape=(1, self.vision_size, self.vision_size), dtype=np.uint8)
        })
        self.action_space = spaces.Discrete(len(self.action_dict))

        self.player_blood_region = (80,60,500,71)
        self.boss_blood_region = (245,500,800,505)
        self.state_window = (200,90,850,500) #(315,100,715,500)

        self.state = None
        self.initial_player_health = None
        self.initial_boss_health = None
        self.previous_blood = {"player_blood": 100, "boss_blood": 100}

        self.total_reward = 0
        self.rew_damage = 0
        self.rew_action_cost = 0
        self.rew_health_keep = 0
        self.rew_boss_low_health = 0
        self.health_keep_count = 0


        self.flag_boss_blood_lower_than_20 = False
        self.action_history = []
        self.boss_damage_history = []  # store the boss damage history for the last 15 actions
        self.player_damage_history = []  # store the player damage history for the last 15 actions

    def reset(self, seed=None, options=None):
        if self.state is not None:
            start.restart_game()
        self.previous_blood = {"player_blood": 100, "boss_blood": 100}
        self.initial_boss_health = None
        self.initial_player_health = None
        self.total_reward = 0
        self.rew_damage = 0
        self.rew_action_cost = 0
        self.rew_health_keep = 0
        self.rew_boss_low_health = 0
        self.health_keep_count = 0

        self.action_history = []
        self.boss_damage_history = []
        self.player_damage_history = [] 

        self.flag_boss_blood_lower_than_20 = False
        self.health_keep_count = 0

        self.state = self._get_observation()
        info = {}
        
        return self.state, info


    def step(self, action):
        
        action = self._step_action(action) # step the action
        state = self._get_observation() # get the observation
        done = self._check_done() # check if the game is done
        reward = self._get_reward(action) # get the reward

        self.state = state
        self.total_reward += reward

        info = {
            "action_taken": self.action_dict[action].__name__,
            "boss_health": self.state["state_values"][1],
            "player_health": self.state["state_values"][0],
            "health_keep_count": self.health_keep_count,
            "total_reward": self.total_reward,
            "- rew_damage": self.rew_damage,
            "- rew_action_cost": self.rew_action_cost,
            "- rew_health_keep": self.rew_health_keep,
            "- rew_boss_low_health": self.rew_boss_low_health
        }

        return state, reward, done, False, info


    def render(self, mode='human'):
        pass


    def close(self):
        pass


    def _step_action(self, action):

        if action in self.action_dict:
            action_func = self.action_dict[action]
            action_func()
            # time.sleep(0.05)
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

        player_blood_percent = int(player_blood / self.initial_player_health * 100)
        boss_blood_percent = int(boss_blood / self.initial_boss_health * 100)
        
        state_values = np.array([player_blood_percent, boss_blood_percent], dtype=np.float32)
        vision_image = vision_image.numpy()

        observation = {
            "state_values": state_values,
            "vision_image": vision_image
        }

        return observation

    
    def _preprocess_image(self, image):
        image_resize = cv2.resize(image, (self.vision_size, self.vision_size))
        image_tensor = torch.tensor(image_resize, dtype=torch.uint8)
        image_tensor = image_tensor.unsqueeze(0) # add batch dimension

        return image_tensor


    def _check_done(self):
        state_values = self.state["state_values"]
        if state_values[0] <= 0 or state_values[1] <= 0:
            return True
        else:
            return False


    def _get_reward(self,action):
        if action in self.action_dict:
            action = self.action_dict[action]
        
        reward = 0
        
        current_blood = self.state["state_values"]
        player_damage = self.previous_blood["player_blood"] - current_blood[0]
        boss_damage = self.previous_blood["boss_blood"] - current_blood[1]
        
        self.health_keep_count = 0.5 + self.health_keep_count if player_damage == 0 else 0
        self.player_damage_history = (self.player_damage_history + [player_damage])[-15:]
        self.boss_damage_history = (self.boss_damage_history + [boss_damage])[-15:]
        total_boss_damage = sum(self.boss_damage_history)
        total_player_damage = sum(self.player_damage_history)

        self.rew_damage = self._rew_damage(player_damage, boss_damage, total_player_damage, total_boss_damage) # rew_damage
        self.rew_action_cost = self._rew_action_cost(action) # rew_action_cost
        self.rew_health_keep = self._rew_health_keep() # rew_health_keep
        self.rew_boss_low_health = self._rew_boss_low_health(current_blood) # rew_boss_low_health

        reward = self.rew_damage + self.rew_action_cost + self.rew_health_keep + self.rew_boss_low_health       
        
        self.previous_blood = {"player_blood": current_blood[0], "boss_blood": current_blood[1]}
        return reward
    

    def _rew_damage(self,player_damage, boss_damage, total_player_damage, total_boss_damage):
        rew_damage = 0
        rew_damage += boss_damage * 0.5
        rew_damage -= player_damage * 0.5

        if total_boss_damage > 20 and total_player_damage < 10:
            rew_damage += 10
        return rew_damage
    

    def _rew_action_cost(self,action):
        action_cost_mapping = {
            actions.light_attack: -1,
            actions.special_attack: -0.5,
            actions.defend: -0.5,
            actions.dodge_left: -0.1,
            actions.dodge_right: -0.1,
            actions.go_backward: -0.1,
            actions.go_forward: -0.1,
            actions.go_left: -0.1,
            actions.go_right: -0.1,
        }
        rew_action_cost = action_cost_mapping.get(action, 0)
        if rew_action_cost == 0 and action not in action_cost_mapping:
            print(f"[WARNING] Action '{action.__name__}' not found in action_cost_mapping. Default cost 0 applied.")
        
        return rew_action_cost


    def _rew_health_keep(self):
        rew_health_keep = 0
        if self.health_keep_count % 15 == 0 and self.health_keep_count != 0:
            rew_health_keep += 5
        return rew_health_keep


    def _rew_boss_low_health(self, current_blood):
        rew_boss_low_health = 0
        if current_blood[1] < 20 and not self.flag_boss_blood_lower_than_20:
            rew_boss_low_health += 60
            if current_blood[0] > 50:
                rew_boss_low_health += 40
        self.flag_boss_blood_lower_than_20 = True
        return rew_boss_low_health


register(
    id='EledenGym-v0',
    entry_point='envs.EledenGym:EledenGym',
)