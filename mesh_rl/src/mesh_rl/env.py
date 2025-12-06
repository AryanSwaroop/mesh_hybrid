import numpy as np

GRAPH = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: []
}

BASE_LATENCY = {
    (0, 1): 2,
    (1, 3): 3,
    (0, 2): 3,
    (2, 3): 2,
}

def edge_key(u, v):
    return tuple(sorted((u, v)))

class SimpleMeshEnv:
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        self.src = 0
        self.dst = 3
        self.reset()

    def reset(self):
        self.current = self.src
        self.steps = 0
        return self.current

    def valid_actions(self):
        return GRAPH[self.current]

    def _sample_latency(self, u, v):
        base = BASE_LATENCY[edge_key(u, v)]
        extra = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        return base + extra

    def step(self, action_node):
        if action_node not in GRAPH[self.current]:
            return self.current, -10.0, True, {"reason": "invalid"}

        u = self.current
        v = action_node
        latency = self._sample_latency(u, v)

        self.current = v
        self.steps += 1

        reward = -latency
        done = False

        if self.current == self.dst:
            reward += 10.0
            done = True
        elif self.steps >= self.max_steps:
            reward -= 5.0
            done = True

        return self.current, reward, done, {"latency": latency}
