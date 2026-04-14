import os
from tqdm import tqdm
from joblib import Parallel, delayed, dump

from src import SACLearner, ForagingEnv, evaluate
import argparse


def run_replicate(
    include_time=False,
    eval_period=100,
    terminal_time=100000,
    stochastic=False,
    verbose=False,
    **kwargs,
):
    env = ForagingEnv(stochastic=stochastic)
    learner = SACLearner(include_time=include_time)

    t_list = []
    pregret_list = []

    progress_bar = tqdm(range(0, terminal_time))
    for t in progress_bar:
        # get current state
        state = env.get_state()

        # evaluate
        if (t + 1) % eval_period == 0:
            pregret = evaluate(learner, env, state, t)
            pregret_list.append(pregret)
            t_list.append(t + 1)
            if verbose:
                print(f"[t = {t}] pregret = {pregret:.3f}")

        # Select action
        action = learner.select_action(state, t)

        # Step in environment
        next_state, reward = env.step(action)

        # Store transition
        learner.replay_buffer.push(state, action, t, reward, next_state)

        # Update SAC
        learner.update()

    return pregret_list, t_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stochastic", action="store_true", help="Enable stochastic environment"
    )
    parser.add_argument(
        "--include_time", action="store_true", help="Enable time-aware learning"
    )
    parser.add_argument(
        "--eval_period", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--terminal_time", type=int, default=100000, help="Terminal time"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--num_reps", type=int, default=10, help="Number of replicates")

    args = parser.parse_args()
    args_dict = vars(args)

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    output = Parallel(n_jobs=args.num_reps)(
        delayed(run_replicate)(**args_dict) for _ in range(args.num_reps)
    )

    pregrets = [p for p, _ in output]
    times = output[0][1]

    # Store all relevant args in metadata for reproducibility
    results = {"pregrets": pregrets, "times": times, "metadata": args_dict}

    # Use the regressor name in the filename
    dump(
        results, f"data/results_sac_{'time' if args.include_time else 'notime'}.joblib"
    )
    print(
        f"Successfully saved results to data/results_sac_{'time' if args.include_time else 'notime'}.joblib"
    )


if __name__ == "__main__":
    main()
