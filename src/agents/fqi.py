import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseLearner


class FQILearner(BaseLearner):
    def __init__(
        self,
        include_time=True,
        gamma=0.9,
        epsilon=0.5,
        state_dim=7,
        action_dim=3,
        t_dim=50,
    ):
        super().__init__()
        self.include_time = include_time
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_dim = t_dim

        # Q-function
        self.Q = None

    def reset(self):
        self.Q = None

    def _encode(self, s, a, t):
        s_enc = [s]
        a_enc = [a]
        if self.include_time:
            freqs = (2 * np.pi) / np.arange(2, self.t_dim + 1, 2)  # shape: (tdim//2,)
            angles = np.outer(t, freqs)
            sin_emb = np.sin(angles)
            cos_emb = np.cos(angles)
            t_enc = np.concatenate([sin_emb, cos_emb], axis=-1).squeeze()
            return np.concatenate([s_enc, a_enc, t_enc])
        else:
            return np.concatenate([s_enc, a_enc])

    def update(self, buffer, num_iterations=20, n_trees=100):
        # Assuming your buffer keys map to numpy arrays or lists
        s = np.array(buffer["s"]).reshape(-1, 1)
        a = np.array(buffer["a"]).reshape(-1, 1)
        t = np.array(buffer["t"]).reshape(-1, 1)
        r = np.array(buffer["r"])
        s_next = np.array(buffer["s_next"]).reshape(-1, 1)

        # # Shift action from [-1,0,1] to for indexing if necessary
        # a_shifted = a + 1

        # Prepare current state-action-time features
        x_sat = np.hstack([s, a, t])
        X = np.array([self._encode(x[0], x[1], x[2]) for x in x_sat])

        # Prepare next state features for all possible actions (to compute max Q)
        # We use t + 1 as the time for the next state
        t_next = t + 1
        s_next_all = np.repeat(s_next, self.action_dim, axis=0)
        t_next_all = np.repeat(t_next, self.action_dim, axis=0)
        a_next_all = np.tile(np.arange(self.action_dim), len(s_next)).reshape(-1, 1)

        x_next_all = np.hstack([s_next_all, a_next_all, t_next_all])
        X_next = np.array([self._encode(x[0], x[1], x[2]) for x in x_next_all])

        for _ in range(num_iterations):
            if self.Q is None:
                y = r
            else:
                # Predict Q-values for all next actions and take the max
                q_next_all = self.Q.predict(X_next).reshape(
                    len(s_next), self.action_dim
                )
                q_next_max = q_next_all.max(axis=1)
                y = r + self.gamma * q_next_max

            # Fit the regressor (Random Forest)
            self.Q = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
            self.Q.fit(X, y)

    def select_action(self, state, time, eval_mode=False, **kwargs):
        # select an action using episilon-greedy policy
        if not eval_mode and np.random.rand() < max(
            0.01, self.epsilon * (0.999 ** (time))
        ):
            # explore
            return np.random.choice(self.action_dim)
        else:
            # exploit
            return np.argmax(
                [
                    self.Q.predict([self._encode(state, a_, time)])
                    for a_ in np.arange(self.action_dim)
                ]
            )
