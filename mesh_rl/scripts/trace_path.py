#!/usr/bin/env python3
import sys
import os
import argparse
import pickle
import numpy as np

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.env import SimpleMeshEnv

def trace_path(agent_path, start_node, target_node, nodes=50, seed=42, output_file=None):
    print(f"Loading agent from {agent_path}...")
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    agent.epsilon = 0.0
    
    # Init env
    # Note: We don't set random_dst=True here because we want to manually set src/dst for tracing
    env = SimpleMeshEnv(num_nodes=nodes, seed=seed, random_dst=False)
    env.src = start_node
    env.dst = target_node
    
    # Reset manually to inject start/target
    env.current = env.src
    # Important: In multi-goal, state is (current, dst)
    state = (env.current, env.dst)
    
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
    
    log(f"--- Tracing Path (Multi-Goal): Node {start_node} -> Node {target_node} ---")
    
    done = False
    step = 0
    path = [env.current]
    
    while not done:
        step += 1
        valid_actions = env.valid_actions()
        
        log(f"\n[Step {step}] At Node {state[0]} (Goal: {state[1]})")
        
        best_q = -float('inf')
        
        # log("  Model Decision Logic:")
        for action in valid_actions:
            # Query Q with (state, action) where state include dst
            q_val = agent.Q[(state, action)]
            # log(f"    - Go {action}: Q={q_val:.4f}")
            if q_val > best_q:
                best_q = q_val
                
        chosen_action = agent.choose_action(state, valid_actions)
        log(f"  >> Agent Chose: {chosen_action} (Max Q: {best_q:.4f})")
        
        next_state, reward, done, info = env.step(chosen_action)
        # next_state is (v, dst)
        
        path.append(next_state[0])
        log(f"  -> Moved to {next_state[0]}")
        
        if done:
            if next_state[0] == target_node:
                log(f"\nSUCCESS: Reached Target {target_node}")
            else:
                 log(f"\nFAILURE: Ended at {next_state[0]}")
            break
            
        state = next_state
        if step > nodes * 3:
            log("\nSTOPPING: Max steps.")
            break
            
    log(f"Full Path: {path}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("\n".join(log_lines))
        print(f"\nTrace saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="outputs/multi_goal_agent.pkl")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--target", type=int, default=49)
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    trace_path(args.agent, args.start, args.target, args.nodes, output_file=args.output)
