import numpy as np


class GPSEnvironment:
    def __init__(self, task, states, actions):
        self.task = task
        self.states = states
        self.actions = actions
        self.current_step = 0
        self.target = self._get_target(task)

    def _get_target(self, task):
        targets = {
            "reach_top_left": np.array([-0.15, 0.15, 0.01]),
        }
        return targets.get(task, np.zeros(3))

    def reset(self):
        self.current_step = 0
        return self.states[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.states)
        next_state = self.states[self.current_step] if not done else self.states[-1]
        reward = self._compute_reward(next_state)
        return next_state, reward, done, {}

    def _compute_reward(self, state):
        return -np.linalg.norm(state - self.target)


def make(task, states, actions):
    if task not in ["trajenv-v0"]:
        raise ValueError(f"Task '{task}' not found")
    return GPSEnvironment(task, states, actions)
