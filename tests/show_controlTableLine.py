from env_suite.env_suite import controlTableLine
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

mode = 'show'

if mode == 'video':
    video_folder = '../data/videos'
    video_length = 5000
  
    env = DummyVecEnv([lambda: controlTableLine()])
    obs = env.reset()
  
    env = VecVideoRecorder(env, video_folder,
                            record_video_trigger=lambda x: x == 0, video_length = video_length,
                            name_prefix="PPO_controlTableLine", )
    env.reset()

    model = PPO2.load('../data/pretrained_models/controlTableLine/PPO')
  
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    env.close()

elif mode == 'gif':
    import imageio
    from stable_baselines.common.cmd_util import make_vec_env

    images = []
    env = make_vec_env(controlTableLine, n_envs=1)
    model = PPO2.load('../data/pretrained_models/controlTableLine/PPO', env)
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    for i in range(1200):
        images.append(img)
        action, _ = model.predict(obs, deterministic = True)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
    imageio.mimsave('../data/videos/PPO_controlTableLine.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

else:
    show_env = controlTableLine()
  
    model = PPO2.load('../data/pretrained_models/controlTableLine/PPO')
  
    n_episodes = 20
  
    for episode in range(n_episodes):
        obs = show_env.reset()
        show_env.render()
        done = False
        sum_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = show_env.step(action)
            show_env.render()
            sum_reward += reward
        print(f"Episode reward: {sum_reward}")