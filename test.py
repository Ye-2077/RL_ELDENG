import gymnasium as gym
from stable_baselines3 import PPO
from alg.EldenPPO import EledenFeatureExtractor
import envs.EledenGym # import EledenGym, must be imported
from utils.utils_start import setup_game

obs_list = []

def test_model(env_id, model_path, num_episodes=10):

    setup_game()
    env = gym.make(env_id)
    model = PPO.load(model_path, env=env)
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=False)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            obs_list.append(obs)
            # env.render()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    
    env_id = 'EledenGym-v0'
    model_path = ''
    test_model(env_id, model_path, num_episodes=10)
