# Collision Free Robot Navigation in Dynamic Environments with Reinforcement Learning and Model Predictive Path Integral Control

## Description

This repository contains the implementation of the MPPI-HGG algorithm for trajectory planning with Reinforcement
Learning (RL) and Model Predictive Path Integral Control (MPPI).
The algorithm aims to improve obstacle avoidance in non-differentiable systems.
The RL agent is trained using Hindsight Goal Generation (HGG) in different environments.
To improve the safety of the RL agent, the MPPI controller is used to generate a safe trajectory to the sub-goal
recommended by the RL agent.

## Requirements

The code is tested on a Omen by HP c0169ng Laptop running Ubuntu 22.04.1 LTS.
To install all requirements, create a new conda environment using the `requirements.txt` file.

```bash
conda create --name <env> --file requirements.txt python=3.7
```

## Usage

All the run commands are preloaded as run configurations when importing the project into PyCharm. Therefore it is
recommended to use PyCharm as IDE and run the Configurations from there.

1. Train the RL agent on the desired environment.
   There are four environments
   available: `FrankaFetchPickDynObstaclesEnv-v1`, `FrankaFetchPickDynLiftedObstaclesEnv-v1`, `FrankaFetchPickDynDoorObstaclesEnv-v2`,
   and `FrankaFetchPick3DTargetObstacle-v1`.
   ```bash
   python train2.py --alg ddpg2 --epochs 20 --env=FrankaFetchPickDynObstaclesEnv-v1 --reward_min -10 --goal mpc
   ```

2. Run the MPPI-HGG algorithm with the trained RL agent.
   ```bash
   python play.py --env FrankaFetchPickDynObstaclesEnv-v1 --play_path log/ddpg2-FrankaFetchPickDynObstaclesEnv-v1-hgg/ --play_epoch 19 --goal mpc --play_policy MPPIRLPolicy --timesteps 1000 --env_n_substeps 5
   ```
   For the different environments, replace both `env` and `play_path` with the desired environment and the path to the
   trained RL agent.
   To compare the performance of MPPI-HGG to pure MPPI, MPC, HGG, or exHGG-MPC, replace the `play_policy` argument
   with `MPPIPolicy`, `MPCPolicy`, `RLPolicy` or `MPCRLMPolicy` respectively.

To run the MPC or exHGG-MPC algorithm, first generate a ForcesPro solver by setting up ForcesPro, following the
instructions in the [ForcesPro documentation](https://forces.embotech.com/Documentation/installation/python.html), and
run the following command:

```bash
cd mpc/
python pick_dyn_sqr_obstacles.py --mpc_gen t
```

For the other environments, replace `pick_dyn_sqr_obstacles.py` with `pick_dyn_obstacles.py`
or `pick_dyn_lifted_obstacles.py`.

## Future Work

- [ ] Add support CUDA for parallel MPPI trajectory rollouts.
