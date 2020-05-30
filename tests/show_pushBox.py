from env_suite.envs import pushBox
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
import time

model = DQN.load("../data/pretrained_models/pushBox/DQN")

env = pushBox()
n_episodes = 10

for episode in range(n_episodes):
  obs = env.reset()
  env.render()
  done = False
  sum_reward = 0
  while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    sum_reward += reward
    time.sleep(0.5)
  print(f"Episode reward: {sum_reward}")