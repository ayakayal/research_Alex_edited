from gym.envs.registration import register

register(
    id='keygrid-v0',
    entry_point='gym_keygrid.envs:KeyGridSparse',
)

register(
    id='keygrid-v1',
    entry_point='gym_keygrid.envs:KeyGrid2d',
)
