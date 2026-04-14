import numpy as np


def get_normalized_future_rewards(rewards, gamma):
    """
    Calculate normalized discounted future rewards for each timestep.
    Computes the discounted sum of future rewards from time t+1 onward,
    normalized by the sum of discount factors.
    """
    H = len(rewards)
    discounts = gamma ** np.arange(H)

    g_list = []
    for t in range(0, H - 1):
        g_t = np.dot(rewards[t + 1 :], discounts[: H - t - 1])
        g_t /= discounts[: H - t - 1].sum()
        g_list.append(g_t)

    return np.array(g_list)


def evaluate(learner, env, state, time, eval_period=100, gamma=0.5):
    """
    Compute the normalized prospective regret i.e. the mean difference between the
    future rewards of the agent and the optimal policy of the foraging environment
    """
    _, pred_rewards = learner.run_current_policy(env, state, time, eval_period)
    _, optimal_rewards = env.run_optimal_policy(state, time, eval_period)

    pred_prewards = get_normalized_future_rewards(pred_rewards[1:], gamma)
    optimal_prewards = get_normalized_future_rewards(optimal_rewards[1:], gamma)
    pregret = np.mean(optimal_prewards - pred_prewards)
    return pregret
