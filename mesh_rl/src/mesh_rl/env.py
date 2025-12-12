import numpy as np
import random
from collections import defaultdict

class SimpleMeshEnv:
    def __init__(self, num_nodes=None, max_steps=None, seed=None, topology="random", random_dst=False):
        self.topology = topology
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.random_dst = random_dst

        if self.topology == "classic":
            self.num_nodes = 4
            self.src = 0
            self.dst = 3
            self.graph = {
                0: [1, 2],
                1: [0, 3],
                2: [0, 3],
                3: [1, 2] 
            }
            self.base_latency = {}
            self._set_classic_latency(0, 1, 2)
            self._set_classic_latency(1, 3, 3) 
            self._set_classic_latency(0, 2, 3)
            self._set_classic_latency(2, 3, 2) 
            
        else:
            self.num_nodes = num_nodes if num_nodes is not None else 50
            self.src = 0
            self.dst = self.num_nodes - 1
            self._generate_graph()

        self.max_steps = max_steps if max_steps is not None else self.num_nodes * 2
        # We don't call reset here to avoid side effects if someone just wants the graph info
        # But for compat we initialize current
        self.current = self.src
        self.steps = 0
        
    def _set_classic_latency(self, u, v, lat):
        self.base_latency[tuple(sorted((u, v)))] = lat

    def _generate_graph(self):
        self.graph = defaultdict(list)
        self.base_latency = {}
        
        nodes = list(range(self.num_nodes))
        intermediate_nodes = nodes[1:-1]
        self.rng.shuffle(intermediate_nodes)
        
        path = [0] + intermediate_nodes + [self.num_nodes - 1]
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self._add_edge(u, v)
        
        # Ensure fully connected-ish for random pairs
        num_extra_edges = self.num_nodes * 3 
        for _ in range(num_extra_edges):
            u = self.rng.choice(nodes)
            v = self.rng.choice(nodes)
            if u != v and v not in self.graph[u]:
                self._add_edge(u, v)

    def _add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
        latency = self.rng.randint(1, 5)
        self.base_latency[tuple(sorted((u, v)))] = latency

    def reset(self, new_dst=None):
        """
        Resets the environment.
        """
        if self.random_dst:
            self.dst = self.rng.randint(0, self.num_nodes - 1)
            self.src = self.rng.randint(0, self.num_nodes - 1)
            while self.src == self.dst:
                 self.src = self.rng.randint(0, self.num_nodes - 1)
        
        if new_dst is not None:
             self.dst = new_dst

        self.current = self.src
        self.steps = 0
        return (self.current, self.dst)

    def valid_actions(self):
        return list(self.graph[self.current])

    def _sample_latency(self, u, v):
        key = tuple(sorted((u, v)))
        base = self.base_latency.get(key, 5)
        extra = self.np_rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        return base + extra

    def step(self, action_node):
        if action_node not in self.graph[self.current]:
            return (self.current, self.dst), -10.0, True, {"reason": "invalid"}

        u = self.current
        v = action_node
        latency = self._sample_latency(u, v)

        self.current = v
        self.steps += 1

        reward = -latency
        done = False

        if self.current == self.dst:
            reward += 100.0
            done = True
        elif self.steps >= self.max_steps:
            reward -= 20.0
            done = True

        return (self.current, self.dst), reward, done, {"latency": latency}
