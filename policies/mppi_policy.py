from typing import List

import numpy as np
import scipy.signal
import torch

import mppi
from env_ext.fetch import MPCControlGoalEnv
from policies.policy import Policy


# TODO get this working for better performance than scipy
def savgol_filter_2d(tensor, window_length, polyorder, axis=0, deriv=0, delta=1.0):
    pass


class MPPIPolicy(Policy):
    Vector = Policy.Vector
    InfoVector = Policy.InfoVector

    def __init__(self, args):
        (
            self.K,
            self.T,
            self.Δt,
            self.α,
            self.F,
            self.q,
            self.ϕ,
            self.Σ,
            self.λ,
            self.convert_to_target,
            self.dtype,
            self.device
        ) = mppi.get_mppi_parameters(args)
        noise_mean_distribution = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.noise_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=noise_mean_distribution, covariance_matrix=self.Σ)
        self.γ = self.λ * (1 - self.α)
        self.Σ_inv = torch.inverse(self.Σ)
        self.u = torch.zeros((self.T, 3), dtype=self.dtype, device=self.device)
        self.u_init = torch.zeros(3, dtype=self.dtype, device=self.device)

        self.prev_u = []

    def initial_info(self, obs: Vector) -> InfoVector:
        pass

    def reset(self):
        self.u = torch.zeros((self.T, 3), dtype=self.dtype, device=self.device)

    def predict(self, obs: Vector) -> (Vector, InfoVector):
        return self.predict_with_goal(obs, obs[0]['desired_goal'])

    def set_envs(self, envs: List[MPCControlGoalEnv]):
        super().set_envs(envs)
        for env in envs:
            env.disable_action_limit()

    def predict_with_goal(self, obs: Vector, goal) -> (Vector, InfoVector):
        x_init, obstacle_positions = self.parse_observation(obs[0], goal)
        goal = torch.tensor(goal, device=self.device)

        # shift the control inputs
        self.u = torch.roll(self.u, -1, dims=0)
        self.u[-1] = self.u_init

        # if the goal is inside the obstacle, we don't update the control sequence
        # and instead immediately return the next action
        if self.is_goal_reachable(goal, obstacle_positions[0]):
            self.update_control(x_init, goal, obstacle_positions)

        target = self.convert_to_target(x_init, self.u[0])
        action = np.array(
            [(target[0] - x_init[0]).cpu(), (target[1] - x_init[1]).cpu(), (target[2] - x_init[2]).cpu(), -0.8])

        # for hyperparameter tuning: check the angle between action and (goal - x_init)
        v1 = (target[0:3] - x_init[0:3]).cpu()
        v2 = (goal - x_init[0:3]).cpu()
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        if angle_rad > np.pi * 1 / 2 or angle_rad < -np.pi * 1 / 2:
            return [action], [{'direction': 'opposite'}]
        return [action], [{'direction': 'forward'}]

    def parse_observation(self, obs, goal):
        ob = obs['observation']
        pos = ob[0:3]
        if self.LastPosition[0] == 0:
            vel = np.zeros(3, )  # ob[20:23]
        else:
            vel = (pos - self.LastPosition) / self.Δt
        self.LastPosition = pos
        x_init = torch.tensor([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=self.dtype, device=self.device)
        parameters = self.envs[0].extract_parameters_3d(self.T, self.Δt, goal)
        obstacle_positions = parameters[:, 6:]
        return x_init, obstacle_positions

    def update_control(self, state, goal, obstacle_positions):
        x = state.view(1, -1).repeat(self.K, 1)
        ε = self.noise_distribution.rsample((self.K, self.T))
        v = torch.clone(ε)
        mask = torch.arange(self.K) < int((1 - self.α) * self.K)
        v[mask] += self.u
        S = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        for i in range(self.T):
            x = self.F(x, v[:, i])
            interim_goal = (goal - state[0:3]) * i + goal
            # print("Interim Goal " + str(i) + ": " + str(interim_goal))
            trajectory_rollouts[:, :, i + 1] = x
            # S += self.q(x, goal, obstacle_positions[i])
            S += self.q(x, interim_goal, obstacle_positions[i])
            S += v[:, i] @ self.Σ_inv @ self.u[i]
        S += self.ϕ(x, goal)

        β = torch.min(S)
        ω = torch.exp((β - S) / self.λ)
        η = torch.sum(ω)
        ω /= η
        δu = torch.sum(ω.view(-1, 1, 1) * ε, dim=0)
        self.u += δu
        self.u = torch.tensor(
            scipy.signal.savgol_filter(self.u, window_length=self.T // 2 * 2 - 1, polyorder=1, axis=0),
            dtype=self.dtype, device=self.device
        )
        # self.u += scipy.signal.savgol_filter(δu, window_length=self.T // 2 * 2 - 1, polyorder=1, axis=0)

    def is_goal_reachable(self, goal, obstacles):
        num_obstacles = len(obstacles) // 6
        for i in range(num_obstacles):
            obstacle_start = i * 6
            obstacle_end = obstacle_start + 6
            obstacle = obstacles[obstacle_start:obstacle_end]
            obstacle_position = obstacle[0:3]
            obstacle_dimensions = obstacle[3:6]
            endeffector_dimensions = np.array([0.04, 0.04, 0.03])

            obstacle_min = torch.tensor(obstacle_position - (obstacle_dimensions + endeffector_dimensions),
                                        dtype=self.dtype, device=self.device)
            obstacle_max = torch.tensor(obstacle_position + (obstacle_dimensions + endeffector_dimensions),
                                        dtype=self.dtype, device=self.device)

            if torch.all(torch.logical_and(goal >= obstacle_min, goal <= obstacle_max)):
                return False

        return True
