#!/usr/bin/env python3
import sys
import os
import argparse
import pickle

# Ensure the package is in the path if running from source without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl import train, demo
from mesh_rl.env import SimpleMeshEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run demo of the Mesh RL agent.")
    parser.add_argument("--agent", type=str, default=None, help="Path to a trained agent pickle file. If not provided, trains online.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes.")
    
    args = parser.parse_args()

    if args.agent:
        print(f"Loading agent from {args.agent}...")
        with open(args.agent, 'rb') as f:
            agent = pickle.load(f)
            
        print(f"Running demo on environment with {args.nodes} nodes (Seed: {args.seed})...")
        env = SimpleMeshEnv(num_nodes=args.nodes, seed=args.seed)
        
        # We need to manually inject the env into the demo or just pass the agent.
        # The original demo() function creates its own env. We should modify demo() to accept an env or create a new one.
        # For now, let's just inline the demo logic or modify demo.py?
        # Let's modify demo.py to accept an env! 
        # But wait, I can just monkeypatch or just copy the logic.
        # Actually, let's modify demo.py first if I can.
        # But I am in replace_file_content for run_demo.py.
        # I'll just use the demo function but I need to make sure demo() uses the correct env.
        # The current demo() creates `env = SimpleMeshEnv()`.
        # I should update demo.py to accept an optional `env`.
        
        # Let's assume I will update demo.py in the next step.
        demo(agent, env=env)
        
    else:
        print("Training agent for demo (Online)...")
        # Train for fewer episodes for a quick demo, or full if needed.
        # Using full 2000 to ensure it works well.
        env, agent, history, _ = train(num_episodes=2000)
        
        print("\nTraining complete. Running demo...")
        demo(agent, env=env)
