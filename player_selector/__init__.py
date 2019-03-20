import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TeamCreator-v0', 
    entry_point='player_selector.envs:TeamCreatorEnv', 
    max_episode_steps=200,
)

register(
    id='PlayerSelector-v0', 
    entry_point='player_selector.envs:PlayerSelectorEnv', 
    max_episode_steps=200,
)

register(
    id='PlayerSelector2-v0', 
    entry_point='player_selector.envs:PlayerSelector2Env', 
    max_episode_steps=200,
)

register(
    id='PlayerSelector3-v0', 
    entry_point='player_selector.envs:PlayerSelector3Env', 
    max_episode_steps=200,
)
