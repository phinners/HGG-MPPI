import gym

from .fetch.franka_pick_dyn_labyrinth import FrankaFetchPickDynLabyrinthEnv
from .fetch.franka_pick_dyn_sqr_obstacle import FrankaFetchPickDynSqrObstacleEnv
from .fetch.pick_dyn_door_obstacles import FetchPickDynDoorObstaclesEnv
from .fetch.pick_dyn_labyrinth import FetchPickDynLabyrinthEnv
from .fetch.pick_dyn_lifted_obstacles import FetchPickDynLiftedObstaclesEnv
from .fetch.pick_dyn_obstacles import FetchPickDynObstaclesEnv
from .fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2
from .fetch.pick_dyn_obstacles_max import FetchPickDynObstaclesMaxEnv
from .fetch.pick_dyn_sqr_obstacle import FetchPickDynSqrObstacleEnv
from .fetch.pick_static_sqr_obstacle import FetchPickStaticSqrObstacleEnv


def register_custom_envs():
    gym.envs.register(
        id='FetchPickStaticSqrObstacle',
        entry_point='envs:FetchPickStaticSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynSqrObstacle-v1',
        entry_point='envs:FetchPickDynSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLabyrinthEnv-v1',
        entry_point='envs:FetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v1',
        entry_point='envs:FetchPickDynObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v2',
        entry_point='envs:FetchPickDynObstaclesEnv2',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLiftedObstaclesEnv-v1',
        entry_point='envs:FetchPickDynLiftedObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesMaxEnv-v1',
        entry_point='envs:FetchPickDynObstaclesMaxEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLabyrinth-v1',
        entry_point='envs:FetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynLabyrinth-v1',
        entry_point='envs:FrankaFetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynSqrObstacle-v1',
        entry_point='envs:FrankaFetchPickDynSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynDoorObstaclesEnv-v1',
        entry_point='envs:FetchPickDynDoorObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
