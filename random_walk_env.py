"""
Random Walk environment for RL homework.

States:
    0 ............ n_states-1
Start:
    middle state
Terminal states:
    0 and n_states-1

Actions:
    0 -> left
    1 -> right

Rewards:
    entering left terminal  -> left_terminal_reward
    entering right terminal -> right_terminal_reward
    otherwise               -> 0
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class RandomWalkConfig:
    n_states: int = 7
    start_state: Optional[int] = None
    left_terminal_reward: float = -1.0
    right_terminal_reward: float = 1.0
    max_steps: int = 100


class RandomWalkEnv:
    def __init__(self, config: Optional[RandomWalkConfig] = None):
        self.config = config or RandomWalkConfig()
        assert self.config.n_states >= 3, "Need at least 3 states"

        self.left_terminal = 0
        self.right_terminal = self.config.n_states - 1
        self.non_terminal_states = list(range(1, self.config.n_states - 1))

        self.start_state = (
            self.config.start_state
            if self.config.start_state is not None
            else self.config.n_states // 2
        )

        assert self.start_state in self.non_terminal_states
        self.state = self.start_state
        self.steps = 0

    def reset(self) -> int:
        self.state = self.start_state
        self.steps = 0
        return self.state

    def is_terminal(self, state: int) -> bool:
        return state in (self.left_terminal, self.right_terminal)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        if self.is_terminal(self.state):
            raise RuntimeError("Episode already ended. Call reset().")

        if action not in (0, 1):
            raise ValueError("Action must be 0 (left) or 1 (right).")

        self.steps += 1
        next_state = self.state - 1 if action == 0 else self.state + 1

        reward = 0.0
        done = False

        if next_state <= self.left_terminal:
            next_state = self.left_terminal
            reward = self.config.left_terminal_reward
            done = True
        elif next_state >= self.right_terminal:
            next_state = self.right_terminal
            reward = self.config.right_terminal_reward
            done = True

        if self.steps >= self.config.max_steps:
            done = True

        self.state = next_state
        return next_state, reward, done, {}

    @property
    def n_states(self) -> int:
        return self.config.n_states

    @property
    def n_actions(self) -> int:
        return 2


def random_policy(state: int) -> int:
    return np.random.choice([0, 1])


def equiprobable_policy_probs(state: int) -> np.ndarray:
    return np.array([0.5, 0.5], dtype=np.float64)


def true_values_symmetric(env: RandomWalkEnv) -> np.ndarray:
    """
    Closed-form values only for the classic symmetric random-walk setting:
    - equiprobable left/right policy
    - gamma = 1
    - zero reward except terminal rewards -1 and +1

    For states 1..n-2, the expected values are linearly spaced between -1 and +1.
    """
    n = env.n_states
    vals = np.zeros(n, dtype=np.float64)
    interior = np.linspace(
        env.config.left_terminal_reward,
        env.config.right_terminal_reward,
        num=n
    )
    vals[:] = interior
    return vals


def one_hot_features(state: int, n_states: int) -> np.ndarray:
    x = np.zeros(n_states, dtype=np.float64)
    x[state] = 1.0
    return x


def polynomial_features(state: int, n_states: int, degree: int = 2) -> np.ndarray:
    """
    Normalize state to [-1, 1] and return [1, x, x^2, ...].
    """
    x = -1.0 + 2.0 * state / (n_states - 1)
    feats = [1.0]
    cur = 1.0
    for _ in range(1, degree + 1):
        cur *= x
        feats.append(cur)
    return np.array(feats, dtype=np.float64)