import optuna

from common import get_args
from env_ext import make_env
from envs import register_custom_envs
from policies import MPPIPolicy


def main():
    register_custom_envs()
    args = get_args()
    env = make_env(args)

    def objective(trial):
        args.α = trial.suggest_float('α', 0.0, 1.0)
        # args.α = 0.597
        args.λ = trial.suggest_float('λ', 10, 20)
        # args.λ = 0.429
        args.σ = trial.suggest_float('σ', 35, 50)
        # args.σ = 45.2
        args.χ = trial.suggest_float('χ', 0.0, 0.3)
        # args.χ = 0.02
        args.ω1 = trial.suggest_float('ω_1', 0, 10)
        # args.ω1 = 0.0
        args.ω2 = trial.suggest_float('ω_2', 0, 10000)
        # args.ω2 = 0.0
        # args.ω_Φ = trial.suggest_float('ω_Φ', 0, 10)
        args.ω_Φ = 5.41
        mppi = MPPIPolicy(args)
        mppi.set_envs([env])

        total_elapsed_time = 0
        collisions = 0
        for i in range(5):
            env.np_random.seed(3019 + i)
            mppi.reset()
            ob = env.reset()

            for timestep in range(args.timesteps):
                actions, info = mppi.predict([ob])
                ob, _, _, env_info = env.step(actions[0])
                total_elapsed_time += 1
                if info[0]['direction'] == 'opposite':
                    total_elapsed_time += 1
                if ob['collision_check']:
                    total_elapsed_time += 1000
                    collisions += 1
                    break
                if env_info['Success']:
                    total_elapsed_time -= 50
                    break
                if total_elapsed_time > 6000:
                    return total_elapsed_time
        return total_elapsed_time
        # return collisions

    if args.tune_mppi < 100:
        raise ValueError("Number of trials should be at least 100")
    study = optuna.create_study(storage="sqlite:///mppi_tuning_results/db.mppi-hyperparameter-tuning",
                                study_name='lifted_obstacles-with_vel_reward', load_if_exists=True)
    # study.enqueue_trial({
    #     'α': 0.6154264807467676,
    #     'λ': 5.283829203406557,
    #     'σ_xx': 19.54981155054756,
    #     'σ_xy': 1.4272042343871534e-05,
    #     'ω2': 1.0313018951699435,
    #     'ω5': 2.40629982969234
    # })
    study.optimize(objective, n_trials=args.tune_mppi)


if __name__ == "__main__":
    main()
