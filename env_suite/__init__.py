from gym.envs.registration import register

register(
    id='pushBox-v0',
    entry_point='env_suite.envs:pushBox'
)

register(
    id='controlTableLine-v0',
    entry_point='env_suite.envs:controlTableLine'
)