from gym.envs.registration import register

register(
    id='BinPack3D-v0',
    entry_point='gym_BinPack3D.envs:PackingGame',
)
