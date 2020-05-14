from env_suite.env_suite import pushBox
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

env = DummyVecEnv([lambda: pushBox()])
modelname = 'DQN_pushBox'

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("../custom_models/" + modelname)