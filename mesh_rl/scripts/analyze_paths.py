#!/usr/bin/env python3
import sys
import os
import argparse
import heapq
from collections import deque

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.env import SimpleMeshEnv

def bfs_shortest_path(graph, start, goal):
    queue = deque([[start]])
    visited = {start}
    
    if start == goal:
        return [start]
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                if neighbor == goal:
                    return new_path
                queue.append(new_path)
    return None

def analyze_paths(seed=42, nodes=50, pairs=10):
    print(f"Analyzing shortest paths for {nodes}-node mesh (Seed: {seed})")
    env = SimpleMeshEnv(num_nodes=nodes, seed=seed)
    graph = env.graph
    
    # Select random pairs
    # Ensure they are connected (the graph generation ensures main component, but let's be safe)
    # Actually env ensures connectivity 0->N-1. But others might be connected too.
    
    test_pairs = []
    
    # Always include Src->Dst
    test_pairs.append((env.src, env.dst))
    
    # Add random pairs
    import random
    rng = random.Random(seed)
    
    used_pairs = set([(env.src, env.dst)])
    
    while len(test_pairs) < pairs:
        u = rng.randint(0, nodes - 1)
        v = rng.randint(0, nodes - 1)
        if u != v and (u, v) not in used_pairs:
            test_pairs.append((u, v))
            used_pairs.add((u, v))
            
    print(f"{'Source':<10} | {'Target':<10} | {'Hops':<5} | {'Path'}")
    print("-" * 60)
    
    results = []
    
    for u, v in test_pairs:
        path = bfs_shortest_path(graph, u, v)
        if path:
            print(f"{u:<10} | {v:<10} | {len(path)-1:<5} | {path}")
            results.append({
                "source": u,
                "target": v,
                "hops": len(path)-1,
                "path": path
            })
        else:
            print(f"{u:<10} | {v:<10} | {'-':<5} | No Path")
            
    # Save to output
    import json
    with open('outputs/shortest_paths_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved to outputs/shortest_paths_analysis.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    
    analyze_paths(seed=args.seed, nodes=args.nodes, pairs=args.pairs)
