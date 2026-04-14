import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any


class ForagingEnv:
    """
    Defines the foraging environment
    """

    def __init__(
        self,
        initial_reward: int = 10,
        decay_rate: float = 0.6,
        window_size: int = 10,
        session_length: int = 500000,
        stochastic: bool = False,
        noise: float = 0.1,
    ):
        self.initial_reward = initial_reward
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.period = 2 * window_size
        self.session_length = session_length
        self.stochastic = stochastic

        self.patch_A = 1
        self.patch_B = 5

        self.rewards_in_session = self._compute_session_rewards()

        self.action_space = [-1, 0, 1]  # left, stay, right
        self.state_space = [0, 1, 2, 3, 4, 5, 6]
        self.transition_tensor = self.generate_transition_tensor(noise=noise)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.state = 1  # Start at state 1
        return self.state

    def generate_transition_tensor(self, noise=0.1):
        num_actions = len(self.action_space)
        num_states = len(self.state_space)

        # Initialize tensor with zeros: (Actions, States, Next_States)
        tensor = np.zeros((num_actions, num_states, num_states))

        for a_idx, action in enumerate(self.action_space):
            for s in range(num_states):
                # 1. Determine the "intended" outcome
                intended_s_prime = np.clip(s + action, 0, num_states - 1)

                # 2. Assign probabilities
                # (1 - noise) chance to land in the intended state
                tensor[a_idx, s, intended_s_prime] += 1.0 - noise

                # noise chance to stay in the current state (the "slip")
                tensor[a_idx, s, s] += noise

        return tensor

    def get_next_state(self, action, state=None):
        state = self.state if state is None else state
        probs = self.transition_tensor[action, state]

        if self.stochastic:
            # Sample based on the probability distribution
            return np.random.choice(len(probs), p=probs)
        else:
            # Return the most likely state (deterministic)
            return np.argmax(probs)

    def step(self, action):
        reward = self._get_current_reward(self.state, self.current_step)
        next_state = self.get_next_state(action)
        self.state = next_state
        self.current_step += 1
        return next_state, reward

    def get_state(self):
        return self.state

    def get_time(self):
        return self.current_step

    def run_optimal_policy(self, state, time, horizon):
        curr_time = time
        curr_state = state
        trajectory = [curr_state]
        rewards = [self._get_current_reward(curr_state, curr_time)]
        if horizon:
            for _ in range(horizon):
                target = self._get_current_target(curr_time)
                if curr_state > target:
                    curr_state -= 1
                elif curr_state < target:
                    curr_state += 1
                curr_time += 1
                trajectory.append(curr_state)
                rewards.append(self._get_current_reward(curr_state, curr_time))
        return trajectory, rewards

    def _compute_session_rewards(self):
        rewards_per_window = self.initial_reward * self.decay_rate ** np.arange(
            self.window_size
        )
        rewards_per_window = rewards_per_window.tolist()
        num_windows, remainder = divmod(self.session_length, self.window_size)
        return rewards_per_window * num_windows + rewards_per_window[:remainder]

    def _get_current_reward(self, state, time):
        phase = time % self.period
        if state == self.patch_A and phase < self.window_size:
            return self.rewards_in_session[time]
        elif state == self.patch_B and phase >= self.window_size:
            return self.rewards_in_session[time]
        else:
            return 0.0

    def _get_current_target(self, time):
        phase = time % self.period
        q1 = self.period // 4
        q3 = self.period // 4 * 3
        if phase <= q1:
            return self.patch_A
        elif phase > q3:
            return self.patch_A
        else:
            return self.patch_B


class ForagingGymEnv(gym.Env, ForagingEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        initial_reward: int = 10,
        decay_rate: float = 0.6,
        window_size: int = 10,
        session_length: int = 500000,
        stochastic: bool = False,
        noise: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Original environment parameters
        self.initial_reward = initial_reward
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.period = 2 * window_size
        self.session_length = session_length
        self.stochastic = stochastic

        self.patch_A = 1
        self.patch_B = 5

        self.rewards_in_session = self._compute_session_rewards()

        # Original action and state spaces (internal)
        self.action_space_internal = [-1, 0, 1]  # left, stay, right
        self.state_space_internal = [0, 1, 2, 3, 4, 5, 6]
        self.transition_tensor = self.generate_transition_tensor(noise=noise)

        # Gym-compatible spaces
        self.action_space = spaces.Discrete(3)  # 0: left, 1: stay, 2: right
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Discrete(7),
                "time": spaces.Box(
                    low=0, high=session_length, shape=(), dtype=np.int32
                ),
            }
        )

        # Rendering setup
        self.render_mode = render_mode
        self.window_width = 800
        self.window_height = 200
        self.cell_width = self.window_width // 7
        self.window = None
        self.clock = None

        self.tree_img = None
        self.squirrel_img = None

        # State variables
        self.current_step = 0
        self.state = 1

        # Cumulative reward
        self.cum_reward = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.state = 1  # Start at state 1

        observation = {"state": self.state, "time": self.current_step}
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Map gym action (0, 1, 2) to internal action (-1, 0, 1)
        # internal_action = self.action_space_internal[action]

        # Get reward for current state before transition (as per original)
        reward = self._get_current_reward(self.state, self.current_step)

        # Transition to next state
        next_state = self.get_next_state(action)
        self.state = next_state
        self.current_step += 1

        # Check termination
        terminated = False  # No natural termination in this environment
        truncated = self.current_step >= self.session_length

        observation = {"state": self.state, "time": self.current_step}
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    # ========== Helper Methods for Gym ==========

    def _get_info(self) -> Dict[str, Any]:
        phase = self.current_step % self.period
        active_patch = "A" if phase < self.window_size else "B"
        optimal_target = self._get_current_target(self.current_step)

        # Determine optimal action
        if self.state > optimal_target:
            optimal_action = 0  # left
        elif self.state < optimal_target:
            optimal_action = 2  # right
        else:
            optimal_action = 1  # stay

        return {
            "current_phase": phase,
            "active_patch": active_patch,
            "optimal_action": optimal_action,
            "optimal_target": optimal_target,
        }

    def _render_frame(self):
        # 1. Initialize Pygame and Assets if not already done
        if self.tree_img is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
                pygame.display.set_caption("Foraging Environment")
                self.clock = pygame.time.Clock()

            # Load assets for BOTH human and rgb_array
            try:
                # Use absolute paths or ensure relative paths are correct for your PhD workspace
                raw_tree = pygame.image.load("../assets/tree.png")
                raw_squirrel = pygame.image.load("../assets/squirrel.png")
                self.tree_img = pygame.transform.scale(raw_tree, (60, 60))
                self.squirrel_img = pygame.transform.scale(raw_squirrel, (50, 50))
            except Exception as e:
                print(f"Fallback to primitive shapes: {e}")
                self.tree_img = pygame.Surface((60, 60), pygame.SRCALPHA)
                self._draw_tree(self.tree_img, 30, 20, True)
                self.squirrel_img = pygame.Surface((50, 50), pygame.SRCALPHA)
                self._draw_squirrel(self.squirrel_img, 25, 25)

        # Create surface for rendering
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))  # White background

        # Determine which patch is active
        phase = self.current_step % self.period
        patch_A_active = phase < self.window_size
        patch_B_active = phase >= self.window_size

        # Draw grid cells
        lw = 3
        margin = 30
        grid_lower = margin
        grid_upper = self.window_height - margin
        grid_height = grid_upper - grid_lower

        # Draw horizonal grid lines
        pygame.draw.line(
            canvas,
            (0, 0, 0),
            (0, grid_lower - 1),
            (self.window_width, grid_lower - 1),
            lw,
        )
        pygame.draw.line(
            canvas, (0, 0, 0), (0, grid_upper), (self.window_width, grid_upper), lw
        )

        for i in range(7):
            x = i * self.cell_width

            # Color the patches
            if i == self.patch_A:
                if patch_A_active:
                    color = (100, 200, 100)  # Bright green (active)
                else:
                    color = (180, 220, 180)  # Light green (inactive)
                pygame.draw.rect(
                    canvas, color, (x, grid_lower, self.cell_width, grid_height)
                )
                # Draw tree icon for patch A
                tree_rect = self.tree_img.get_rect(
                    center=(x + self.cell_width // 2, self.window_height // 2 - 20)
                )
                canvas.blit(self.tree_img, tree_rect)
                # self._draw_tree(canvas, x + self.cell_width // 2, self.window_height // 2 - 20, patch_A_active)
            elif i == self.patch_B:
                if patch_B_active:
                    color = (100, 200, 100)  # Bright blue (active)
                else:
                    color = (180, 220, 180)  # Light blue (inactive)
                pygame.draw.rect(
                    canvas, color, (x, grid_lower, self.cell_width, grid_height)
                )
                # Draw tree icon for patch B
                tree_rect = self.tree_img.get_rect(
                    center=(x + self.cell_width // 2, self.window_height // 2 - 20)
                )
                canvas.blit(self.tree_img, tree_rect)
            else:
                color = (255, 255, 255)  # Light gray
                pygame.draw.rect(
                    canvas, color, (x, grid_lower, self.cell_width, grid_height)
                )

            # Draw grid lines
            pygame.draw.line(canvas, (0, 0, 0), (x, grid_lower), (x, grid_upper), lw)

        x = (i + 1) * self.cell_width
        pygame.draw.line(canvas, (0, 0, 0), (x, grid_lower), (x, grid_upper), lw)

        # Draw squirrel at current position
        squirrel_x = self.state * self.cell_width + self.cell_width // 2
        squirrel_y = self.window_height // 2 + 40
        squirrel_rect = self.squirrel_img.get_rect(center=(squirrel_x, squirrel_y))
        canvas.blit(self.squirrel_img, squirrel_rect)

        # Draw text info
        if pygame.font.get_init():
            font = pygame.font.Font(None, 24)

            # # Phase info
            # phase_text = "Patch A Active" if patch_A_active else "Patch B Active"

            # # Current reward
            current_reward = self._get_current_reward(self.state, self.current_step)
            self.cum_reward += current_reward

            text = font.render(
                f"State: {self.state} | Time: {self.current_step} | Cumulative Rewards: {self.cum_reward:.2f}",
                True,
                (222, 45, 38),
            )
            canvas.blit(text, (1, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_tree(self, surface, x, y, active):
        # Draw a simple tree
        trunk_color = (101, 67, 33)  # Brown
        if active:
            foliage_color = (34, 139, 34)  # Forest green
        else:
            foliage_color = (144, 238, 144)  # Light green

        # Trunk
        pygame.draw.rect(surface, trunk_color, (x - 5, y + 10, 10, 20))
        # Foliage (circle)
        pygame.draw.circle(surface, foliage_color, (x, y), 15)

    def _draw_squirrel(self, surface, x, y):
        # Draw a simple squirrel (circle with tail)
        body_color = (139, 90, 43)  # Saddle brown

        # Body
        pygame.draw.circle(surface, body_color, (x, y), 12)
        # Tail (arc)
        pygame.draw.circle(surface, body_color, (x - 8, y - 8), 8)
        # Eye
        pygame.draw.circle(surface, (0, 0, 0), (x + 3, y - 2), 2)
