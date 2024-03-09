from envs.EledenGym import EledenGym
from utils.utils_start import setup_game
import numpy as np
import torch

if __name__ == "__main__":
    setup_game()
    env = EledenGym()
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()

        print(f"Action: {action}")
        next_obs, reward, done, _ = env.step(action)

        total_reward += reward
        obs = next_obs
        print("------------------------")
        print(obs)
        print(obs['state_values'].shape)
        print(obs['vision_image'].shape)

    print("Episode finished! Total reward:", total_reward)

    env.close()
