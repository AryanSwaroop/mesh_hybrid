#!/usr/bin/env python3
import sys
import os
import argparse
import pickle
import random
import numpy as np

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.env import SimpleMeshEnv

def generate_traces(agent_path, output_file, count=10, nodes=50, seed=42):
    print(f"Loading agent from {agent_path}...")
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    agent.epsilon = 0.0
    env = SimpleMeshEnv(num_nodes=nodes, seed=seed)
    
    # Generate random pairs
    pairs = []
    rng = random.Random(seed)
    
    # Always include 0->49
    pairs.append((0, 49))
    
    while len(pairs) < count:
        u = rng.randint(0, nodes-1)
        v = rng.randint(0, nodes-1)
        if u != v and (u, v) not in pairs:
            pairs.append((u, v))
            
    all_output = []
    
    for i, (src, dst) in enumerate(pairs):
        lines = []
        def log(msg):
            lines.append(msg)
            
        log(f"\n{'='*40}")
        log(f"TRACE {i+1}/{count}: Node {src} -> Node {dst}")
        log(f"{'='*40}")
        
        env.src = src
        env.dst = dst
        state = env.reset()
        done = False
        step = 0
        path = [state]
        
        log(f"Start: {state}")
        
        while not done:
            step += 1
            valid_actions = env.valid_actions()
            
            # log(f"  [Step {step}] Node {state}")
            # log(f"  Neighbors: {valid_actions}")
            
            best_q = -float('inf')
            # choices_log = []
            
            for action in valid_actions:
                q_val = agent.Q[(state, action)]
                # choices_log.append(f"{action}(Q={q_val:.2f})")
                if q_val > best_q:
                    best_q = q_val
            
            # log(f"  Options: {', '.join(choices_log)}")
            
            chosen_action = agent.choose_action(state, valid_actions)
            # log(f"  >> Chose: {chosen_action}")
            
            next_state, reward, done, info = env.step(chosen_action)
            path.append(next_state)
            
            state = next_state
            if step > nodes * 2:
                log("  STOP: Max steps.")
                break
        
        status = "SUCCESS" if env.current == env.dst else "FAILURE"
        log(f"Result: {status}")
        log(f"Path: {path}")
        
        all_output.append("\n".join(lines))
        print(f"Generated trace for {src}->{dst}: {status}")

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n\n".join(all_output))
        
    print(f"\nSaved {count} traces to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="outputs/trained_agent_improved.pkl")
    parser.add_argument("--output", type=str, default="outputs/path_trace.txt")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    generate_traces(args.agent, args.output, args.count)
