import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, dump

from src import FQILearner, ForagingEnv, evaluate
import argparse


def update_buffer(buffer, s, a, r, t, s_next):
    buffer["s"].append(s)
    buffer["a"].append(a)
    buffer["r"].append(r)
    buffer["t"].append(t)
    buffer["s_next"].append(s_next)


def run_replicate(
    include_time=False,
    stochastic=False,
    gamma=0.99,
    state_dim=7,
    action_dim=3,
    warmup_period=200,
    eval_period=100,
    update_period=100,
    terminal_time=5000,
    epsilon=0.5,
    verbose=False,
    **kwargs,
):
    env = ForagingEnv(stochastic=stochastic)
    learner = FQILearner(
        include_time=include_time,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        epsilon=epsilon,
    )

    buffer = {"s": [], "a": [], "t": [], "r": [], "s_next": []}

    # collect data during the warm-up period
    for t in range(warmup_period):
        # get current state
        state = env.get_state()

        # select action randomly
        action = np.random.choice(action_dim)

        # Step in environment
        next_state, reward = env.step(action)

        # store data
        update_buffer(buffer, state, action, reward, t, next_state)

    # warm start
    learner.update(buffer)

    # inference + online updates
    t_list = []
    pregret_list = []
    for t in tqdm(range(warmup_period, terminal_time)):
        # get current state
        state = env.get_state()

        # evaluate
        if t % eval_period == 0:
            pregret = evaluate(learner, env, state, t)  # TODO: Check
            pregret_list.append(pregret)
            t_list.append(t)
            if verbose:
                print(f"[t = {t}] pregret = {pregret:.3f}")

        # Select action
        action = learner.select_action(state, t)

        # Step in environment
        next_state, reward = env.step(action)

        # Store transition and update the buffer
        update_buffer(buffer, state, action, reward, t, next_state)

        # update the learner
        if (t % 1) % update_period == 0:
            learner.update(buffer)

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
    parser.add_argument("--terminal_time", type=int, default=5000, help="Terminal time")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon", type=float, default=0.5, help="Exploration parameter"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--num_reps", type=int, default=10, help="Number of replicates")

    args = parser.parse_args()
    args_dict = vars(args)

    run_replicate(**args_dict)

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
    dump(results, f"data/results_fqi_{args.regressor}.joblib")
    print(f"Successfully saved results to data/results_fqi_{args.regressor}.joblib")


if __name__ == "__main__":
    main()
