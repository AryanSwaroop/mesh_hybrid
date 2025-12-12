import json
import numpy as np
import pickle
import os
from mesh_rl.agent import QLearningAgent

def train_offline(data_file='outputs/multi_goal_data.json', epochs=10, output_agent='outputs/multi_goal_agent.pkl'):
    """
    Trains a Goal-Conditioned Q-Learning agent.
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        return None

    print(f"Loading data from {data_file}...")
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    transitions = data["transitions"]
    metadata = data.get("metadata", {})
    print(f"Loaded {len(transitions)} transitions. Metadata: {metadata}")

    # Initialize agent
    # Agent now maps ( (current, dst), action ) -> Q
    agent = QLearningAgent(alpha=0.1, gamma=0.95)
    
    print(f"Starting Multi-Goal training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        np.random.shuffle(transitions)
        
        for t in transitions:
            # JSON loads tuples as lists, convert back to tuple for dict key
            state = tuple(t["state"]) 
            action = t["action"]
            reward = t["reward"]
            next_state = tuple(t["next_state"])
            next_valid_actions = t["next_valid_actions"]
            done = t["done"]
            
            agent.update(state, action, reward, next_state, next_valid_actions, done)
            
        print(f"Epoch {epoch} complete.")

    # Save the agent
    with open(output_agent, 'wb') as f:
        pickle.dump(agent, f)
        
    print(f"Training complete. Goal-Conditioned Agent saved to {output_agent}")
    return agent

# CLI wrapper inside logic file? Better to update scripts/train_offline.py
