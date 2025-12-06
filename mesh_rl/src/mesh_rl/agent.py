import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(float)

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        q_vals = [self.Q[(state, a)] for a in valid_actions]
        # If multiple actions have same max Q, argmax takes the first one.
        # This is fine for this simple env.
        best_idx = int(np.argmax(q_vals))
        return valid_actions[best_idx]

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        sa = (state, action)
        q_sa = self.Q[sa]

        if done or not next_valid_actions:
            target = reward
        else:
            next_qs = [self.Q[(next_state, a)] for a in next_valid_actions]
            target = reward + self.gamma * max(next_qs)

        self.Q[sa] = q_sa + self.alpha * (target - q_sa)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
