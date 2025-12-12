#!/usr/bin/env python3
import sys
import os
import argparse
import pickle
import numpy as np

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.env import SimpleMeshEnv

def evaluate(agent_path, episodes=100, nodes=50):
    print(f"Loading agent from {agent_path}...")
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Disable exploration for evaluation
    agent.epsilon = 0.0
    
    success_count = 0
    total_rewards = []
    
    print(f"Evaluating over {episodes} episodes (Nodes: {nodes})...")
    
    env = SimpleMeshEnv(num_nodes=nodes)
    
    for i in range(episodes):
        # Use different seeds for evaluation (or random if seed=None)
        # env.seed = i # If we want reproducible random set
        # Actually SimpleMeshEnv init sets seed. To re-seed we need to recreate or add seed method.
        # But SimpleMeshEnv init calls _generate_graph.
        # If we want a dynamic graph for each episode, we need to re-init env or re-generate graph.
        # The current env keeps the SAME graph for all resets unless re-init.
        # To evaluate on generalized graphs, we should re-generate graph each time?
        # OR evaluate on the SAME graph (seeded) to see if it learned THAT graph.
        # The user said "train model over it" (synthetic data).
        # The synthetic data was 4000 episodes on ONE graph (seed 42) or random?
        # `generate_data.py` sets `seed=args.seed` (42). So `env` is static.
        # So the agent learned ONE specific 50-node topology.
        # So we should evaluate on THAT topology.
        
        # But wait, `generate_synthetic_data` initializes env ONCE.
        # So all 4000 episodes are on the SAME 50-node graph.
        # So the agent is overfitted to that graph.
        # That is fine for "routing in a specific network".
        
        # So for evaluation, we must use the SAME seed (42) to test if it learned THAT network.
        # If we use a different seed, the graph topology changes, and the agent (which maps (state, action) -> Q)
        # will fail because state 0->1 might be valid in Graph A but not Graph B.
        # IDs are node indices. In a random graph, node 1 might be a neighbor of 0 in one, but not another.
        # So this Q-learning agent is topology-specific.
        
        # So we must use seed 42.
        pass
        
    # Re-init env with the training seed
    env = SimpleMeshEnv(num_nodes=nodes, seed=42)
    
    for i in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            
        total_rewards.append(ep_reward)
        if env.current == env.dst:
            success_count += 1
            
    avg_reward = np.mean(total_rewards)
    accuracy = (success_count / episodes) * 100
    
    print(f"Evaluation Complete.")
    print(f"Success Rate: {accuracy:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=50)
    args = parser.parse_args()
    
    evaluate(args.agent, args.episodes, args.nodes)
