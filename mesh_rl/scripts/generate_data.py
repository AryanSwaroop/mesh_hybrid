#!/usr/bin/env python3
import sys
import os
import argparse

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.data_gen import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data for Mesh RL.")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes to generate.")
    parser.add_argument("--output", type=str, default="data/synthetic_data.json", help="Output JSON file path.")
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes in the mesh.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    
    args = parser.parse_args()
    
    generate_synthetic_data(num_episodes=args.num_episodes, output_file=args.output, num_nodes=args.nodes, seed=args.seed)

if __name__ == "__main__":
    main()
