#!/usr/bin/env python3
import sys
import os
import json
import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl import train, SimpleMeshEnv

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def save_training_logs_and_plots():
    print("Running training and capturing output...")
    
    # Capture stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        env, agent, rewards_history, success_history = train(num_episodes=2000)
    output_log = f.getvalue()
    
    # Save log
    log_path = os.path.join(OUTPUT_DIR, 'training_output.txt')
    with open(log_path, 'w') as log_file:
        log_file.write(output_log)
    print(f"Saved training log to {log_path}")
    
    # Plot Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    reward_plot_path = os.path.join(OUTPUT_DIR, 'reward_curve.png')
    plt.savefig(reward_plot_path)
    plt.close()
    print(f"Saved reward plot to {reward_plot_path}")

    # Plot Success Rate (Moving Average)
    window_size = 50
    success_rate_ma = np.convolve(success_history, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(success_rate_ma)
    plt.title(f'Success Rate (Moving Average window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.ylim(0, 1.1)
    success_plot_path = os.path.join(OUTPUT_DIR, 'success_rate.png')
    plt.savefig(success_plot_path)
    plt.close()
    print(f"Saved success rate plot to {success_plot_path}")

    return agent

def run_demo_evaluation(agent):
    print("Running demo evaluation...")
    env = SimpleMeshEnv()
    
    path_counts = {"upper": 0, "lower": 0, "other": 0}
    success_count = 0
    num_eval_episodes = 100
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        path = [state]
        
        while not done:
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, _, done, _ = env.step(action)
            state = next_state
            path.append(state)
            
        if env.current == env.dst:
            success_count += 1
            
        # Analyze path
        # 0 -> 1 -> 3 is "upper" (if we consider 1 as upper, latency 0-1 is 2)
        # 0 -> 2 -> 3 is "lower" (if we consider 2 as lower, latency 0-2 is 3)
        if path == [0, 1, 3]:
            path_counts["upper"] += 1
        elif path == [0, 2, 3]:
            path_counts["lower"] += 1
        else:
            path_counts["other"] += 1

    agent.epsilon = original_epsilon
    
    results = [
        "Demo Evaluation Results",
        "=======================",
        f"Total Episodes: {num_eval_episodes}",
        f"Success Rate: {success_count/num_eval_episodes:.2%}",
        "",
        "Path Usage:",
        f"  Upper Path (0->1->3): {path_counts['upper']}",
        f"  Lower Path (0->2->3): {path_counts['lower']}",
        f"  Other/Invalid: {path_counts['other']}"
    ]
    
    results_str = "\n".join(results)
    demo_path = os.path.join(OUTPUT_DIR, 'demo_results.txt')
    with open(demo_path, 'w') as f:
        f.write(results_str)
    print(f"Saved demo results to {demo_path}")

def save_q_table(agent):
    # Convert keys to string representation for JSON
    q_dict = {str(k): v for k, v in agent.Q.items()}
    q_path = os.path.join(OUTPUT_DIR, 'q_table.json')
    with open(q_path, 'w') as f:
        json.dump(q_dict, f, indent=2)
    print(f"Saved Q-table to {q_path}")

def save_topology_image():
    # Simple visualization of the graph
    # Nodes: 0, 1, 2, 3
    # Pos: 0=(0,1), 1=(1,2), 2=(1,0), 3=(2,1)
    
    pos = {
        0: (0, 1),
        1: (1, 2),
        2: (1, 0),
        3: (2, 1)
    }
    
    edges = [
        (0, 1), (0, 2),
        (1, 3), (2, 3),
        (1, 0), (2, 0), # bi-directional in graph dict, but lines are same
        (3, 1), (3, 2)
    ]
    
    plt.figure(figsize=(6, 4))
    
    # Draw nodes
    for node, (x, y) in pos.items():
        plt.plot(x, y, 'o', markersize=20, color='lightblue', zorder=2)
        plt.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        
    # Draw edges
    seen_edges = set()
    for u, v in edges:
        if (u, v) not in seen_edges and (v, u) not in seen_edges:
            x_values = [pos[u][0], pos[v][0]]
            y_values = [pos[u][1], pos[v][1]]
            plt.plot(x_values, y_values, 'k-', lw=2, zorder=1)
            seen_edges.add((u, v))
            
    plt.title("Mesh Topology")
    plt.axis('off')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    
    topo_path = os.path.join(OUTPUT_DIR, 'mesh_topology.png')
    plt.savefig(topo_path)
    plt.close()
    print(f"Saved topology image to {topo_path}")

if __name__ == "__main__":
    ensure_output_dir()
    agent = save_training_logs_and_plots()
    run_demo_evaluation(agent)
    save_q_table(agent)
    save_topology_image()
    print("\nAll artifacts generated successfully.")
