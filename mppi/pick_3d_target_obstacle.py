import math
import os

import pytorch_kinematics
import torch

collision = torch.zeros([500, ], dtype=torch.bool)


def getCollisions():
    return collision


def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 60  # 40  # 1.62e1
        args.σ = 0.201  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
        args.χ = 0.0  # 2.00e-2
        args.ω1 = 9.16e3
        args.ω2 = 9.16e3
        args.ω_Φ = 5.41

    K = 500
    T = 10
    Δt = 0.01
    T_system = 0.01

    dtype = torch.double
    device = 'cpu'  # 'cuda'

    α = args.α
    λ = args.λ
    Σ = args.σ * torch.tensor([
        [1.5, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1.0, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
    ], dtype=dtype, device=device)

    # Ensure we get the path separator correct on windows
    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')

    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    dtype_kinematics = torch.double
    chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                            root_link_name="panda_link0")
    chain = chain.to(dtype=dtype_kinematics, device=device)

    # Translational offset of Robot into World Coordinates
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44],
                                  device=device, dtype=dtype_kinematics)

    link_dimensions = {
        'panda_link0': torch.tensor([0.0, 0.0, 0.333], dtype=dtype_kinematics),
        # 'panda_link1': torch.tensor([0.0, 0.0, 0.000], dtype=dtype_kinematics),
        # Delete from Calculation for Computational Speed
        'panda_link2': torch.tensor([0.0, -0.316, 0.0], dtype=dtype_kinematics),
        'panda_link3': torch.tensor([0.0825, 0.0, 0.0], dtype=dtype_kinematics),
        'panda_link4': torch.tensor([-0.0825, 0.384, 0.0], dtype=dtype_kinematics),
        # 'panda_link5': torch.tensor([0.0, 0.0, 0.0], dtype=dtype_kinematics),
        # Delete from Calculation for Computational Speed
        'panda_link6': torch.tensor([0.088, 0.0, 0.0], dtype=dtype_kinematics),
        'panda_link7': torch.tensor([0.0, 0.0, 0.245], dtype=dtype_kinematics)
        # 'panda_link8': torch.tensor([0.0, 0.0, 0.0], dtype=dtype_kinematics)
        # Delete from Calculation for Computational Speed
    }

    def calculate_link_verticies(links):
        link_verticies = {}
        for link_key in links:
            link = links[link_key]
            length = torch.norm(link)  # Calculate Length of Link in all Dimensions
            points_distance = 0.03
            points_count = math.ceil(length / points_distance)
            points = torch.zeros((points_count, 3))
            for i in range(points_count):
                points[i, :] = torch.tensor([(link[0] / points_count) * i,
                                             (link[1] / points_count) * i,
                                             (link[2] / points_count) * i], dtype=torch.float64)
            link_verticies[link_key] = points
        return link_verticies

    link_verticies = calculate_link_verticies(link_dimensions)

    def dynamics(x, u):

        new_vel = x[:, 7:14] + u

        # new_vel[:, 0] = torch.clamp(new_vel[:, 0], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        # new_vel[:, 1] = torch.clamp(new_vel[:, 1], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        # new_vel[:, 2] = torch.clamp(new_vel[:, 2], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        # new_vel[:, 3] = torch.clamp(new_vel[:, 3], min=-2.1750, max=2.1750)  # Limit Joint Velocities
        # new_vel[:, 4] = torch.clamp(new_vel[:, 4], min=-2.6100, max=2.6100)  # Limit Joint Velocities
        # new_vel[:, 5] = torch.clamp(new_vel[:, 5], min=-2.6100, max=2.6100)  # Limit Joint Velocities
        # new_vel[:, 6] = torch.clamp(new_vel[:, 6], min=-2.6100, max=2.6100)  # Limit Joint Velocities

        new_pos = x[:, 0:7] + new_vel * Δt

        # new_pos[:, 0] = torch.clamp(new_pos[:, 0], min=-2.8973, max=2.8973)  # Limit Joint Positions
        # new_pos[:, 1] = torch.clamp(new_pos[:, 1], min=-1.7628, max=1.7628)  # Limit Joint Positions
        # new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=-2.8973, max=2.8973)  # Limit Joint Positions
        # new_pos[:, 3] = torch.clamp(new_pos[:, 3], min=-3.0718, max=-0.0698)  # Limit Joint Positions
        # new_pos[:, 4] = torch.clamp(new_pos[:, 4], min=-2.8973, max=2.8973)  # Limit Joint Positions
        # new_pos[:, 5] = torch.clamp(new_pos[:, 5], min=-0.0175, max=3.7525)  # Limit Joint Positions
        # new_pos[:, 6] = torch.clamp(new_pos[:, 6], min=-2.8973, max=2.8973)  # Limit Joint Positions

        return torch.cat((new_pos, new_vel), dim=1)

    def joint_collision_calculation(link_transform, link_verticies, obstacle_pos, obstacle_dim):

        transformed_link0 = link_transform.transform_points(link_verticies.to(torch.float64))
        transformed_link0 += robot_base_pos  # Translate Link into World Coordinates

        closest_point = torch.clamp(transformed_link0, min=obstacle_pos - obstacle_dim, max=obstacle_pos + obstacle_dim)
        dist = torch.norm(transformed_link0 - closest_point, dim=2)

        collision_link_segments = torch.le(dist, torch.tensor(
            [0.12]))  # 0.1 Freespace From Obstacle to Center of Link --> Link Dimensions included here!

        collision = torch.any(collision_link_segments, dim=1)

        if torch.any(collision):
            # print("Link Trajectorie with collision detected!")
            pass

        if torch.all(collision):
            # print("All Link Trajectorie with collision detected!")
            pass

        return collision

    def state_cost(x, goal, obstacles):
        global collision

        joint_values = x[:, 0:7]
        ret = chain.forward_kinematics(joint_values, end_only=False)

        link8_matrix = ret['panda_link8'].get_matrix()  # Equals to panda0_gripper Bode in Mujoco File
        link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # Equals to panda0_gripper Bode in Mujoco File

        goal_dist = torch.norm((link8_pos - goal), dim=1)
        cost = 1000 * goal_dist ** 2
        # cost -= args.ω1 * torch.norm(x[:, 3:6], dim=1) ** 2

        # Obstacles
        for link_key in link_verticies:
            link_vertics = link_verticies[link_key]
            link_transform = ret[link_key]

            link_collisions = joint_collision_calculation(link_transform, link_vertics,
                                                          torch.tensor(obstacles[0:3], device=device),
                                                          torch.tensor(obstacles[3:6], device=device))

            collision = torch.logical_or(collision,
                                         link_collisions)

        if torch.any(collision):
            # print("Trajectorie with collision detected!")
            pass

        if torch.all(collision):
            # print("All Trajectorie with collision detected!")
            pass

        table_collision = torch.le(link8_pos[:, 2], 0.45)

        # cost += joint_collision_calculation(x, obstacles)
        cost += args.ω1 * table_collision
        cost += args.ω2 * collision
        collision = torch.zeros([500, ], dtype=torch.bool)
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
        new_vel = joint_vel + u / (
                1 - torch.exp(torch.tensor(-Δt / T_system)) * (
                1 + (Δt / T_system)))  # / (1 - torch.exp(torch.tensor(-0.01 / 0.150)))  # 0.175

        new_joint_pos = joint_pos + new_vel * Δt  # Calculate new Target Joint Positions

        # Calculate World Coordinate Target Position
        ret = chain.forward_kinematics(new_joint_pos, end_only=True)
        eef_matrix = ret.get_matrix()
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # Calculate World Coordinate Target
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])

        return torch.concatenate((eef_pos, eef_rot), dim=1)
        # return u

    return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
