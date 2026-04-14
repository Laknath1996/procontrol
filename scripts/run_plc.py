import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, dump

from src import PLCLearner, ForagingEnv, evaluate
import argparse


def form_buffer(data, gamma=0.99):
    buffer = {"s": [], "a": [], "t": [], "r": [], "s_next": [], "g": []}

    # add the last interaction
    s, a, t, r, s_next = data[-1]
    buffer["s"].append(s)
    buffer["a"].append(a)
    buffer["r"].append(r)
    buffer["t"].append(t)
    buffer["s_next"].append(s_next)
    buffer["g"].append(0)

    # add the other interactions
    g_t = 0
    for i in range(len(data) - 2, -1, -1):
        s, a, t, r, s_next = data[i]
        r_next = data[i + 1][3]
        g_t = r_next + gamma * g_t

        buffer["s"].append(s)
        buffer["a"].append(a)
        buffer["r"].append(r)
        buffer["t"].append(t)
        buffer["s_next"].append(s_next)
        buffer["g"].append(g_t)

    return buffer


def run_replicate(
    regressor="nn",
    warmup_period=200,
    horizon=5,
    eval_period=100,
    terminal_time=2000,
    stochastic=False,
    state_dim=7,
    action_dim=3,
    gamma=0.9,
    verbose=False,
    **kwargs,
):
    env = ForagingEnv(stochastic=stochastic)
    learner = PLCLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        gamma=gamma,
        regressor=regressor,
    )

    # record the interactions
    data = []

    # collect data during the warm-up period
    for t in range(warmup_period):
        # get current state
        state = env.get_state()

        # select action randomly
        action = np.random.choice(action_dim)

        # Step in environment
        next_state, reward = env.step(action)

        # store data
        data.append((state, action, t, reward, next_state))

    # form the initial buffer
    buffer = form_buffer(data, gamma=gamma)

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
            # print(f"Time {t+1}, Pregret: {pregret:.3f}")
            t_list.append(t)
            if verbose:
                print(f"[t = {t}] pregret = {pregret:.3f}")

        # Select action
        action = learner.select_action(state, t, env=env)

        # Step in environment
        next_state, reward = env.step(action)

        # Store transition and update the buffer
        data.append((state, action, t, reward, next_state))
        buffer = form_buffer(data, gamma=gamma)

        # update the learner
        learner.update(buffer)

    return pregret_list, t_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stochastic", action="store_true", help="Enable stochastic environment"
    )
    parser.add_argument(
        "--regressor",
        type=str,
        default="nn",
        choices=["nn", "rf"],
        help="Type of regressor",
    )
    parser.add_argument(
        "--eval_period", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument("--terminal_time", type=int, default=2000, help="Terminal time")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
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
    dump(results, f"data/results_plc_{args.regressor}.joblib")
    print(f"Successfully saved results to data/results_plc_{args.regressor}.joblib")


if __name__ == "__main__":
    main()
