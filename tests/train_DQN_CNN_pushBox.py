from env_suite.envs import pushBox
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2

env_kwargs = {
    "mode": "image",
}
env = make_vec_env(pushBox, n_envs=64, env_kwargs=env_kwargs)

modelname = 'PPO2_CNN_pushBox'

# model = PPO2('CnnPolicy', env, verbose=1)
model = PPO2.load("custom_models/" + modelname, env)
model.learn(total_timesteps=5000000)
model.save("custom_models/" + modelname)