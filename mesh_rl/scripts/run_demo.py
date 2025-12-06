#!/usr/bin/env python3
import sys
import os

# Ensure the package is in the path if running from source without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl import train, demo

if __name__ == "__main__":
    print("Training agent for demo...")
    # Train for fewer episodes for a quick demo, or full if needed.
    # Using full 2000 to ensure it works well.
    env, agent, history, _ = train(num_episodes=2000)
    
    print("\nTraining complete. Running demo...")
    demo(agent)
