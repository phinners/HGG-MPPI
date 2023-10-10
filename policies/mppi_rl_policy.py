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

        for env in envs:
            env.disable_action_limit()

    def initial_info(self, obs: Vector) -> InfoVector:
        pass

    def reset(self):
        self.rl_policy.reset()
        self.mppi_policy.reset()

    def predict(self, obs: Vector) -> (Vector, InfoVector):
        [rl_action], _ = self.rl_policy.predict(obs)

        desired_goal = obs[0]["desired_goal"]
        sub_goal = self.envs[0].subgoal(rl_action)
        # self.mppi_policy.reset()
        [action], _ = self.mppi_policy.predict_with_goal(obs, sub_goal)
        action[3] = rl_action[3]

        if (linalg.norm(desired_goal - current_pos) > 0.11):
            for i in range(self.mppi_policy.T):
                [rl_action], _ = self.rl_policy.predict(obs_sim)
                sub_goal_sim = self.envs[0].subgoal_sim(rl_action, obs_sim[0]['observation'][0:3])
                obs_pos = np.array(
                    self.envs[0].env.get_obstacles(time=self.envs[0].time + (i + 2) * self.mppi_policy.Δt)).flatten()
                diff = obs_sim[0]['observation'][3:6] - obs_sim[0]['observation'][0:3]
                obs_sim[0]['observation'][0:3] = sub_goal_sim.copy()
                obs_sim[0]['observation'][3:6] = sub_goal_sim.copy() + diff
                obs_sim[0]['observation'][25:25 + len(obs_pos)] = obs_pos
                self.mppi_policy.trajectory.append(sub_goal_sim)
            [action], _ = self.mppi_policy.predict_with_goal(obs, sub_goal)
        else:
            for i in range(self.mppi_policy.T):
                self.mppi_policy.trajectory.append(desired_goal)
            [action], _ = self.mppi_policy.predict_with_goal(obs, desired_goal)
        action[3] = -0.8  # rl_action[3]
        return [action], _
