import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseLearner


# -------------------------
# Regressor (MLP)
# -------------------------
class MLPRegressor(nn.Module):
    def __init__(self, state_dim, t_dim, mode="reward", hidden_dim=128, batch_size=32):
        super().__init__()
        self.state_dim = state_dim
        self.t_dim = t_dim
        self.mode = mode

        # Calculate input size based on toggle
        self.input_dim = state_dim + t_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def _embed(self, states, times):
        # embed the states
        embed_states = F.one_hot(states, num_classes=self.state_dim).float()

        # embed the times
        freqs = (2 * np.pi) / (torch.arange(2, self.t_dim + 1, 2))
        freqs = freqs.unsqueeze(0)
        sin = torch.sin(freqs * times)
        cos = torch.cos(freqs * times)
        embed_times = torch.cat([sin, cos], dim=-1)

        return torch.cat([embed_states, embed_times], dim=-1)

    def _prepare_data(self, buffer):
        # get the states, times, and targets
        states = buffer["s"]
        times = buffer["t"]
        targets = buffer["r"] if self.mode == "reward" else buffer["g"]

        # convert to tensors
        states = torch.tensor(states, dtype=torch.long)
        times = torch.tensor(times, dtype=torch.float).unsqueeze(1)
        targets = torch.tensor(targets, dtype=torch.float).unsqueeze(1)

        # form the inputs
        inputs = self._embed(states, times)

        # form the dataloader
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=self.batch_size)
        return loader

    def train(self, buffer, epochs=5, batch_size=32, **kwargs):
        self.model.train()  # Set to training mode

        loader = self._prepare_data(buffer)

        for epoch in range(epochs):
            for inputs, targets in loader:
                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

    def __call__(self, state, time):
        """
        Inference logic.
        """
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            states = torch.tensor([state], dtype=torch.long)
            times = torch.tensor([time], dtype=torch.float).unsqueeze(1)

            # form the inputs
            inputs = self._embed(states, times)

            return self.model(inputs)


# -------------------------
# Regressor (RF)
# -------------------------
class RFRegressor:
    def __init__(self, state_dim, t_dim, mode="reward", n_estimators=100):
        self.state_dim = state_dim
        self.t_dim = t_dim
        self.mode = mode

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="exact",
            n_jobs=1,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=n_estimators,
            eval_metric="rmse",
        )

    def _embed(self, states, times):
        # embed the states
        embed_states = F.one_hot(states, num_classes=self.state_dim).float()

        # embed the times
        freqs = (2 * np.pi) / (torch.arange(2, self.t_dim + 1, 2))
        freqs = freqs.unsqueeze(0)
        sin = torch.sin(freqs * times)
        cos = torch.cos(freqs * times)
        embed_times = torch.cat([sin, cos], dim=-1)

        return torch.cat([embed_states, embed_times], dim=-1)

    def _prepare_data(self, buffer):
        # get the states, times, and targets
        states = buffer["s"]
        times = buffer["t"]
        targets = buffer["r"] if self.mode == "reward" else buffer["g"]

        # convert to tensors
        states = torch.tensor(states, dtype=torch.long)
        times = torch.tensor(times, dtype=torch.float).unsqueeze(1)

        inputs = self._embed(states, times)

        return inputs.numpy(), np.array(targets)

    def train(self, buffer):
        inputs, targets = self._prepare_data(buffer)
        self.model.fit(inputs, targets)

    def __call__(self, state, time):
        states = torch.tensor([state], dtype=torch.long)
        times = torch.tensor([time], dtype=torch.float).unsqueeze(1)
        inputs = self._embed(states, times).numpy()
        return self.model.predict(inputs)[0]


# -------------------------
# PL+C Agent
# -------------------------
class PLCLearner(BaseLearner):
    def __init__(
        self, state_dim, action_dim, t_dim=50, horizon=6, gamma=0.99, regressor="nn"
    ):
        self.horizon = horizon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_dim = t_dim

        if regressor == "nn":
            # Predicts instantaneous rewards: r = M_R(s, a)
            self.reward_model = MLPRegressor(state_dim, t_dim, mode="reward")

            # Predicts cumulative discounted returns: G = M_G(s, a)
            self.return_model = MLPRegressor(state_dim, t_dim, mode="return")
        else:
            self.reward_model = RFRegressor(state_dim, t_dim, mode="reward")
            self.return_model = RFRegressor(state_dim, t_dim, mode="return")

    def update(self, buffer):
        self.reward_model.train(buffer)
        self.return_model.train(buffer)

    def select_action(self, state, time, env=None, **kwargs):
        if not env:
            raise ValueError("env needs to be passed")

        # 1. Generate all possible action sequences of length H
        action_sequences = list(
            itertools.product(range(self.action_dim), repeat=self.horizon)
        )
        num_sequences = len(action_sequences)

        # Pre-allocate scores for each sequence
        sequence_scores = np.zeros(num_sequences)

        # Iterate over the sequences
        for idx, seq in enumerate(action_sequences):
            q_value = 0
            curr_state = state

            # Walk down the path defined by the sequence
            for h, action in enumerate(seq):
                # A. Get next state
                next_state = env.get_next_state(state=curr_state, action=action)

                # B. Prepare the inputs for the Reward Model
                s_t = next_state
                t_t = time + h + 1

                # C. Accumulate discounted predicted reward
                imm_reward = self.reward_model(s_t, t_t)
                q_value += (self.gamma**h) * imm_reward

                # Move to next state in the sequence
                curr_state = next_state

            # 2. Tail Correction (The "Stitch")
            # After the 5 steps, use M_G to evaluate the value of the final state
            final_s_t = curr_state
            final_t_t = t_t

            tail_val = self.return_model(final_s_t, final_t_t)
            q_value += (self.gamma**self.horizon) * tail_val

            sequence_scores[idx] = q_value

        # 3. Find the best sequence and return its FIRST action
        best_sequence_idx = np.argmax(sequence_scores)
        best_sequence = action_sequences[best_sequence_idx]

        return best_sequence[0]
