import gym
import env_suite

if __name__ == "__main__":
    penv = gym.make('pushBox-v0')
    penv.seed(0)
    penv.reset()
    penv.step(0)

    cenv=gym.make('controlTableLine-v0')
    cenv.seed(0)
    cenv.reset()
    cenv.step([0.1, -0.4])

    print('No errors!')