import os

import pytorch_kinematics
import torch

collision = torch.zeros([500, ], dtype=torch.bool)


def getCollisions():
    return collision


def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 40  # 1.62e1
        args.σ = 0.1  # 08  # 0.25  # 4.0505  # 10.52e1
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
        [1, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 1, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 1, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 1]
    ], dtype=dtype, device=device)

    # Ensure we get the path separator correct on windows
    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'panda_arm.urdf')

    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    dtype = torch.double
    chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                            root_link_name="panda_link0")
    chain = chain.to(dtype=dtype, device=device)

    # Translational offset of Robot into World Coordinates
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44],
                                  device=device)

    def dynamics(x, u):

        new_vel = x[:, 7:14] + u

        new_vel[:, 0] = torch.clamp(new_vel[:, 0], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        new_vel[:, 1] = torch.clamp(new_vel[:, 1], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        new_vel[:, 2] = torch.clamp(new_vel[:, 2], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        new_vel[:, 3] = torch.clamp(new_vel[:, 3], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        new_vel[:, 4] = torch.clamp(new_vel[:, 4], min=-2.6100, max=2.6100)  # Limit Joint Velocities
        new_vel[:, 5] = torch.clamp(new_vel[:, 5], min=-2.6100, max=2.6100)  # Limit Joint Velocities
        new_vel[:, 6] = torch.clamp(new_vel[:, 6], min=-2.6100, max=2.6100)  # Limit Joint Velocities

        new_pos = x[:, 0:7] + new_vel * Δt

        new_pos[:, 0] = torch.clamp(new_pos[:, 0], min=-2.8973, max=2.8973)  # Limit Joint Positions
        new_pos[:, 1] = torch.clamp(new_pos[:, 1], min=-1.7628, max=1.7628)  # Limit Joint Positions
        new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=-2.8973, max=2.8973)  # Limit Joint Positions
        new_pos[:, 3] = torch.clamp(new_pos[:, 3], min=-3.0718, max=-0.0698)  # Limit Joint Positions
        new_pos[:, 4] = torch.clamp(new_pos[:, 4], min=-2.8973, max=2.8973)  # Limit Joint Positions
        new_pos[:, 5] = torch.clamp(new_pos[:, 5], min=--0.0175, max=3.7525)  # Limit Joint Positions
        new_pos[:, 6] = torch.clamp(new_pos[:, 6], min=-2.8973, max=2.8973)  # Limit Joint Positions

        return torch.cat((new_pos, new_vel), dim=1)

    def joint_collision_calculation(link_pos, obstacle_pos, obstacle_dim):
        dist_joint = torch.abs(link_pos - obstacle_pos)

        collision = torch.all(torch.le(dist_joint, obstacle_dim + torch.tensor([0.055, 0.055, 0.03])),
                              dim=1)
        return collision

    def state_cost(x, goal, obstacles):
        global collision

        joint_values = x[:, 0:7]
        ret = chain.forward_kinematics(joint_values, end_only=False)
        link_positions = []
        link_rotations = []

        link0_matrix = ret['panda_link0'].get_matrix()
        link_positions.append(link0_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link0_matrix[:, :3, :3]))

        link1_matrix = ret['panda_link1'].get_matrix()
        link_positions.append(link1_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link1_matrix[:, :3, :3]))

        link2_matrix = ret['panda_link2'].get_matrix()
        link_positions.append(link2_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link2_matrix[:, :3, :3]))

        link3_matrix = ret['panda_link3'].get_matrix()
        link_positions.append(link3_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link3_matrix[:, :3, :3]))

        link4_matrix = ret['panda_link4'].get_matrix()
        link_positions.append(link4_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link4_matrix[:, :3, :3]))

        link5_matrix = ret['panda_link5'].get_matrix()
        link_positions.append(link5_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link5_matrix[:, :3, :3]))

        link6_matrix = ret['panda_link6'].get_matrix()
        link_positions.append(link6_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link6_matrix[:, :3, :3]))

        link7_matrix = ret['panda_link7'].get_matrix()
        link_positions.append(link7_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link7_matrix[:, :3, :3]))

        link8_matrix = ret['panda_link8'].get_matrix()  # Equals to panda0_gripper Bode in Mujoco File
        link_positions.append(link8_matrix[:, :3, 3] + robot_base_pos)
        link_rotations.append(pytorch_kinematics.matrix_to_quaternion(link8_matrix[:, :3, :3]))

        goal_dist = torch.norm((link_positions[-1] - goal), dim=1)
        cost = 1000 * goal_dist ** 2
        # cost -= args.ω1 * torch.norm(x[:, 3:6], dim=1) ** 2

        # Obstacle
        dist = torch.abs(link_positions[-1] - torch.tensor(obstacles[0:3], device=device))
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist,
                                                        torch.tensor(obstacles[3:6], device=device)
                                                        + torch.tensor(
                                                            [0.055, 0.055, 0.03])),
                                               dim=1))

        # Obstacle2
        for link_pos in link_positions:
            joint_collision = joint_collision_calculation(link_pos, torch.tensor([1.1, 0.90, 0.9], device=device),
                                                          torch.tensor([0.03, 0.03, 0.06], device=device))
            collision = torch.logical_or(collision,
                                         joint_collision)

        if torch.any(collision):
            # print("Trajectorie with collision detected!")
            pass

        if torch.all(collision):
            print("All Trajectorie with collision detected!")
            pass

        table_collision = torch.le(link_positions[-1][:, 2], 0.4)

        # cost += joint_collision_calculation(x, obstacles)
        cost += 100 * table_collision
        cost += args.ω2 * collision
        return cost

    def terminal_cost(x, goal):
        global collision
        joint_values = x[:, 0:7]
        ret = chain.forward_kinematics(joint_values, end_only=True)

        eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
        cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
        # cost += args.ω_Φ * torch.norm(x[:, 3:6], dim=1) ** 2
        collision = torch.zeros([500, ], dtype=torch.bool)
        return cost

    def convert_to_target(x, u):
        joint_pos = x[0:7]
        joint_vel = x[7:14]
        new_vel = joint_vel + u / (1 - torch.exp(torch.tensor(-0.01 / 0.100)))  # 0.175

        joint_pos = joint_pos + new_vel * Δt  # Calculate new Target Joint Positions
        # Calculate World Coordinate Target Position
        ret = chain.forward_kinematics(joint_pos, end_only=True)
        eef_matrix = ret.get_matrix()
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # Calculate World Coordinate Target
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])

        return torch.concatenate((eef_pos, eef_rot), dim=1)

    return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
