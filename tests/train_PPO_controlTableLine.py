from env_suite.envs import controlTableLine
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines import PPO2
import datetime, os

logdir = os.path.join(os.getcwd(),"logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

eval_env = controlTableLine()
eval_callback = EvalCallback(eval_env, best_model_save_path='/gdrive/My Drive/Code/RL/controlTableLine/best_model/', eval_freq=100000, deterministic=True, render=False)

env = make_vec_env(controlTableLine, n_envs=10)

modelname = 'PPO2_controlTableLine'

model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=5000000, callback=eval_callback)
model.save("../custom_models/" + modelname)