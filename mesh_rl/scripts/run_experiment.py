#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.env import SimpleMeshEnv
from mesh_rl.agent import QLearningAgent

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def run_experiment(topology='classic', num_nodes=None, episodes=500, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger()
    
    logger.info(f"Starting experiment with topology={topology}, episodes={episodes}")
    
    # Initialize Environment and Agent
    env = SimpleMeshEnv(topology=topology, num_nodes=num_nodes)
    agent = QLearningAgent()
    
    rewards_history = []
    success_history = []
    
    # Enhanced path stats: path_str -> list of rewards
    path_data = defaultdict(list)
    
    # Training Loop
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        path = [state]
        
        while not done:
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions)
            
            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions() if not done else []
            
            agent.update(state, action, reward, next_state, next_valid, done)
            
            total_reward += reward
            state = next_state
            path.append(state)
        
        # Logging
        rewards_history.append(total_reward)
        is_success = (env.current == env.dst)
        success_history.append(is_success)
        
        if is_success:
            path_str = "->".join(map(str, path))
            # Calculate latency from reward (Reward = -Latency + 100)
            latency = 100.0 - total_reward
            path_data[path_str].append({
                "reward": total_reward,
                "latency": latency,
                "hops": len(path) - 1
            })
            
        if ep % 50 == 0:
            avg_r = np.mean(rewards_history[-50:])
            logger.info(f"Episode {ep}: Avg Reward={avg_r:.2f}, Epsilon={agent.epsilon:.2f}, Success Rate={np.mean(success_history[-50:]):.2f}")

    # Generate Outputs
    logger.info("Generating outputs...")
    
    # 1. Reward Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, alpha=0.3, label='Episode Reward')
    window = 20
    if len(rewards_history) >= window:
        rolling_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), rolling_avg, color='red', label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'reward_curve.png'))
    plt.close()
    
    # 2. Detailed Path Stats
    detailed_stats = {}
    for path_str, records in path_data.items():
        count = len(records)
        avg_reward = np.mean([r['reward'] for r in records])
        avg_latency = np.mean([r['latency'] for r in records])
        avg_hops = np.mean([r['hops'] for r in records])
        
        detailed_stats[path_str] = {
            "count": count,
            "avg_reward": round(avg_reward, 2),
            "avg_latency": round(avg_latency, 2),
            "avg_hops": round(avg_hops, 2),
            "percentage": round((count / len(success_history)) * 100, 2) # % of total episodes (approx) or total successes? Let's say total successes
        }
        
    # Calculate percentages of successes
    total_successes = sum(success_history)
    for k in detailed_stats:
        detailed_stats[k]["percentage"] = round((detailed_stats[k]["count"] / total_successes) * 100, 2)

    with open(os.path.join(output_dir, 'path_stats.json'), 'w') as f:
        json.dump(detailed_stats, f, indent=2)
        
    # 3. Final Q-Table Export
    q_data = []
    for (state, action), value in agent.Q.items():
        q_data.append({"state": state, "action": action, "q_value": value})
    
    q_data.sort(key=lambda x: (x['state'], x['action']))
    
    with open(os.path.join(output_dir, 'q_table.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["state", "action", "q_value"])
        writer.writeheader()
        writer.writerows(q_data)

    logger.info(f"Experiment complete. Artifacts saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="classic", choices=["classic", "random"])
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    
    run_experiment(topology=args.topology, episodes=args.episodes)
