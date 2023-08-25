import torch

collision = torch.zeros([500, ], dtype=torch.bool)


def getCollisions():
    return collision


def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 5.94e-1
        args.λ = 1.62e1
        args.σ = 10.52e1
        args.χ = 2.00e-2
        args.ω1 = 15.37
        args.ω2 = 9.16e3
        args.ω_Φ = 5.41

    K = 500
    T = 15
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
        new_pos = x[:, 0:3] + x[:, 3:6] * Δt + 0.5 * u * Δt * Δt
        # new_pos[:, 0] = torch.clamp(new_pos[:, 0], 0.8, 1.5)
        # new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0.0, 0.6)
        new_vel = x[:, 3:6] + u * Δt
        return torch.cat((new_pos, new_vel), dim=1)

    def state_cost(x, goal, obstacles):
        global collision
        cost = 1000 * torch.norm((x[:, 0:3] - goal), dim=1) ** 2
        # cost -= args.ω1 * torch.norm(x[:, 3:6], dim=1) ** 2

        lowerborder = torch.tensor([1.05, 0.4, 0.0])  # X,Y,Z
        higherborder = torch.tensor([1.55, 1.10, 0.5])  # X,Y,Z

        border_collision = torch.logical_or(torch.any(torch.le(x[:, 0:3], lowerborder), dim=1),
                                            torch.any(torch.ge(x[:, 0:3], higherborder), dim=1))

        if torch.any(border_collision):
            # print("Trajectorie with border collisions detected!")
            pass

        dist1 = torch.abs(x[:, 0:3] - torch.tensor(obstacles[0:3], device=device))
        dist2 = torch.abs(x[:, 0:3] - torch.tensor(obstacles[6:9], device=device))
        dist3 = torch.abs(x[:, 0:3] - torch.tensor(obstacles[12:15], device=device))
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist1,
                                                        torch.tensor(obstacles[3:6], device=device) + torch.tensor(
                                                            [0.035, 0.035, 0.03])),
                                               dim=1))
        collision = torch.logical_or(collision,
                                     torch.all(
                                         torch.le(dist2, torch.tensor(obstacles[9:12], device=device) + torch.tensor(
                                             [0.035, 0.035, 0.03])),
                                         dim=1))
        collision = torch.logical_or(collision,
                                     torch.all(torch.le(dist3,
                                                        torch.tensor(obstacles[15:18], device=device) + torch.tensor(
                                                            [0.035, 0.035, 0.00])),
                                               dim=1))
        collision = torch.logical_or(collision,
                                     border_collision)

        if torch.any(collision):
            # print("Trajectorie with collision detected!")
            pass

        cost += args.ω2 * 2000 * collision

        return cost

    def terminal_cost(x, goal):
        global collision
        cost = 10 * torch.norm((x[:, 0:3] - goal), dim=1) ** 2
        cost += args.ω_Φ * torch.norm(x[:, 3:6], dim=1) ** 2
        collision = torch.zeros([500, ], dtype=torch.bool)
        return cost

    def convert_to_target(x, u):
        return x[0:3] + x[3:6] * Δt + 0.5 * u * Δt + Δt

    return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
