import json
import os
import random
from mesh_rl.env import SimpleMeshEnv

def generate_synthetic_data(num_episodes=4000, output_file='outputs/multi_goal_data.json', num_nodes=50, seed=42):
    """
    Generates synthetic data with random start/end points (Goal-Conditioned).
    """
    # Enable random destination for diverse training data
    env = SimpleMeshEnv(num_nodes=num_nodes, seed=seed, random_dst=True)
    
    data = {
        "metadata": {
            "num_nodes": num_nodes,
            "seed": seed,
            "num_episodes": num_episodes,
            "type": "goal_conditioned"
        },
        "transitions": []
    }

    print(f"Generating Multi-Goal data with {num_nodes} nodes for {num_episodes} episodes...")

    for ep in range(num_episodes):
        state = env.reset() # Returns (current, dst)
        done = False
        
        while not done:
            valid_actions = env.valid_actions()
            action = random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions() if not done else []
            
            # State is now a tuple (u, dst). We need to serialize it for JSON.
            # We'll store it as a list [u, dst]
            transition = {
                "state": state,         # (u, dst)
                "action": int(action),
                "reward": float(reward),
                "next_state": next_state, # (v, dst)
                "next_valid_actions": [int(a) for a in next_valid],
                "done": bool(done)
            }
            data["transitions"].append(transition)
            
            state = next_state

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Data generation complete. Saved {len(data['transitions'])} transitions to {output_file}")
    return data
