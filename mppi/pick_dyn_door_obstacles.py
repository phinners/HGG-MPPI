import ikpy.chain
import numpy as np
import torch
from ikpy.utils import geometry
from scipy.spatial.transform import Rotation

collision = torch.zeros([500, ], dtype=torch.bool)


def getCollisions():
    return collision


def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 40  # 1.62e1
        args.σ = 0.2  # 08  # 0.25  # 4.0505  # 10.52e1
        args.χ = 2.00e-2
        args.ω1 = 15.37
        args.ω2 = 9.16e3
        args.ω_Φ = 5.41

    K = 500
    T = 10
    Δt = 0.01

    dtype = torch.double
    device = 'cpu'  # 'cuda'

    α = args.α
    λ = args.λ
    Σ = args.σ * torch.tensor([
        [1, args.χ, args.χ],
        [args.χ, 1, args.χ],
        [args.χ, args.χ, 1]
    ], dtype=dtype, device=device)

    def dynamics(x, u):
        u[:, 2] = 0  # No Z-Movement
        x[:, 5] = 0  # No Z-Movement
        pt1_vel = u * (1 - torch.exp(torch.tensor(-0.01 / 0.1)))
        new_vel = x[:, 3:6] + u  # * Δt  # + pt1_vel
        new_vel = torch.clamp(new_vel, min=-1.7, max=1.7)
        # evtl noch die halbe zusätzliche neue Velocity dazuaddieren
        new_pos = x[:, 0:3] + new_vel * Δt  # + 0.5 * u * Δt * Δt  # new_vel * Δt  # + (pt1_vel / 2) * Δt
        # new_pos[:, 0] = torch.clamp(new_pos[:, 0], 0.8, 1.5)
        # new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0.0, 0.6)

        return torch.cat((new_pos, new_vel), dim=1)

    def joint_collision_calculation(state, obstacles):
        my_chain = ikpy.chain.Chain.from_urdf_file(urdf_file="franka_panda_description/robots/panda_arm.urdf",
                                                   base_elements=["panda_link0"])
        cost = torch.zeros([500, ], dtype=torch.float)
        # 'robot0_joint1': -2.24,
        # 'robot0_joint2': -0.038,
        # 'robot0_joint3': 2.55,
        # 'robot0_joint4': -2.68,
        # 'robot0_joint5': 0.0,
        # 'robot0_joint6': 0.984,
        # 'robot0_joint7': 0.0327,
        nodes = []
        target_orientation = [0.0, 0.0, 1.0]
        initial_joint_pos = [0.0, -2.24, -0.038, 2.55, -2.68, 0.0, 0.984, 0.0327, 0.0]
        for target_position in state[:, 0:3]:
            inverse = my_chain.inverse_kinematics(target_position.numpy(), target_orientation, orientation_mode="X",
                                                  initial_position=initial_joint_pos)
            real_frame = my_chain.forward_kinematics(inverse, full_kinematics=True)
            # Get the nodes and the orientation from the transformation matrix
            for (index, link) in enumerate(my_chain.links):
                (node, orientation) = geometry.from_transformation_matrix(real_frame[index])

                # Add node corresponding to the link
                # nodes.append(node)
            pass
        return cost

    def state_cost(x, goal, obstacles):
        global collision

        cost = 1000 * torch.norm((x[:, 0:3] - goal), dim=1) ** 2
        # cost -= args.ω1 * torch.norm(x[:, 3:6], dim=1) ** 2

        # Calculate Transformed State to get the correct result
        r = Rotation.from_quat([np.roll(obstacles[3:7], -1)])
        x_translated = x[:, 0:3] - obstacles[0:3]

        state_transformed_obs1 = torch.matmul(torch.tensor(np.squeeze(r.inv().as_matrix())),
                                              x_translated.transpose(0, 1)).transpose(0, 1)

        dist1 = torch.abs(state_transformed_obs1)
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist1,
                                                        torch.tensor(obstacles[7:10], device=device)
                                                        + torch.tensor(
                                                            [0.055, 0.055, 0.03])),
                                               dim=1))

        # Obstacle 2
        dist2 = torch.abs(x[:, 0:3] - torch.tensor(obstacles[10:13], device=device))
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist2,
                                                        torch.tensor(obstacles[17:20], device=device)
                                                        + torch.tensor(
                                                            [0.055, 0.055, 0.03])),
                                               dim=1))

        # Obstacle 3
        dist3 = torch.abs(x[:, 0:3] - torch.tensor(obstacles[20:23], device=device))
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist3,
                                                        torch.tensor(obstacles[27:30], device=device)
                                                        + torch.tensor(
                                                            [0.055, 0.055, 0.03])),
                                               dim=1))

        if torch.any(collision):
            # print("Trajectorie with collision detected!")
            pass

        if torch.all(collision):
            print("All Trajectorie with collision detected!")
            pass

        table_collision = torch.le(x[:, 2], 0.3)

        # cost += joint_collision_calculation(x, obstacles)
        cost += 100 * table_collision
        cost += args.ω2 * collision
        return cost

    def terminal_cost(x, goal):
        global collision
        cost = 10 * torch.norm((x[:, 0:3] - goal), dim=1) ** 2
        cost += args.ω_Φ * torch.norm(x[:, 3:6], dim=1) ** 2
        collision = torch.zeros([500, ], dtype=torch.bool)
        return cost

    def convert_to_target(x, u):
        new_vel = x[3:6] + u / (1 - torch.exp(torch.tensor(-0.01 / 0.100)))  # 0.175
        new_vel[2] = 0  # Block Z-Movement
        target_pos = x[0:3] + new_vel * Δt
        return x[0:3] + new_vel * Δt  # + u * Δt  # x[3:6] * Δt

    return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
