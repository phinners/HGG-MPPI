from typing import List

from env_ext.fetch import MPCControlGoalEnv
from policies.mppi_policy import MPPIPolicy
from policies.policy import Policy
from policies.rl_policy import RLPolicy


class MPPIRLPolicy(Policy):
    Vector = Policy.Vector
    InfoVector = Policy.InfoVector

    def __init__(self, args):
        self.rl_policy = RLPolicy(args)
        self.mppi_policy = MPPIPolicy(args)

    def set_envs(self, envs: List[MPCControlGoalEnv]):
        super().set_envs(envs)
        self.rl_policy.set_envs(envs)
        self.mppi_policy.set_envs(envs)

    def initial_info(self, obs: Vector) -> InfoVector:
        pass

    def reset(self):
        self.rl_policy.reset()
        self.mppi_policy.reset()

    def predict(self, obs: Vector) -> (Vector, InfoVector):
        [rl_action], _ = self.rl_policy.predict(obs)

        sub_goal = self.envs[0].subgoal(rl_action)
        # self.mppi_policy.reset()
        [action], _ = self.mppi_policy.predict_with_goal(obs, sub_goal)
        action[3] = rl_action[3]

        return [action], _
