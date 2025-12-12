import json
import random
import os

def fetch_and_merge(synthetic_file='outputs/synthetic_data_4k.json', output_file='outputs/merged_data.json'):
    # 1. Load Synthetic Data
    if os.path.exists(synthetic_file):
        with open(synthetic_file, 'r') as f:
            synthetic_data = json.load(f)
            transitions = synthetic_data.get("transitions", [])
            print(f"Loaded {len(transitions)} transitions from synthetic data.")
    else:
        print(f"Warning: {synthetic_file} not found. Starting empty.")
        transitions = []

    # 2. Simulate "External" Real-World Data
    # Simulating data with higher latency variance and occasional packet loss (reward = -100)
    print("Fetching/Simulating external real-world data...")
    external_transitions = []
    num_external = 1000 # Add 1000 "real" samples
    
    # We'll map "real" data to our 50-node topology (or a subset of it)
    # Using a subset of nodes 0-10 to represent a core network
    for _ in range(num_external):
        u = random.randint(0, 10)
        v = random.randint(0, 10)
        if u == v: continue
        
        # Real world latency has spikes
        latency = random.choice([2, 3, 4, 10, 50]) # 50 is a spike
        reward = -latency
        
        # Occasional failure
        if random.random() < 0.05:
            reward = -100 # Packet loss / timeout
            
        transition = {
            "state": u,
            "action": v, # In our simplified env, action is target node
            "reward": reward,
            "next_state": v, # Assumes successful move mostly
            "next_valid_actions": [], # Unknown or simplified
            "done": False,
            "source": "external_real_world"
        }
        external_transitions.append(transition)
        
    print(f"Generated {len(external_transitions)} external transitions.")
    
    # 3. Merge
    all_transitions = transitions + external_transitions
    random.shuffle(all_transitions)
    
    merged_data = {
        "metadata": {
            "description": "Merged synthetic (4k) and simulated external (1k) data.",
            "total_transitions": len(all_transitions)
        },
        "transitions": all_transitions
    }
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
        
    print(f"Merged data saved to {output_file} with {len(all_transitions)} entries.")

if __name__ == "__main__":
    fetch_and_merge()
