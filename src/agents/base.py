from abc import ABC, abstractmethod


class BaseLearner(ABC):
    def __init__(self):
        """Base class for Learners/Agents"""
        pass

    @abstractmethod
    def select_action(self, *args, **kwargs):
        """Must be implemented by subclasses. Should return an action."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Must be implemented by subclasses. Handles the learning logic."""
        pass

    def run_current_policy(self, env, state, time, horizon):
        """
        Rolls out the current policy for a fixed horizon.
        Common to all learners.
        Used in evaluation.
        """
        curr_state = state
        curr_time = time
        trajectory = [curr_state]
        rewards = [env._get_current_reward(curr_state, curr_time)]

        for _ in range(horizon):
            action = self.select_action(curr_state, curr_time, env=env, eval_mode=True)
            curr_state = env.get_next_state(state=curr_state, action=action)
            curr_time += 1
            trajectory.append(curr_state)
            rewards.append(env._get_current_reward(curr_state, curr_time))

        return trajectory, rewards
