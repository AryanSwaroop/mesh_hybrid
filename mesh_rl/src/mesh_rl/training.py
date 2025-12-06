import numpy as np
from mesh_rl.env import SimpleMeshEnv
from mesh_rl.agent import QLearningAgent

def train(num_episodes=2000):
    env = SimpleMeshEnv()
    agent = QLearningAgent()

    rewards_history = []
    success_history = []
    success_count = 0

    print(f"Starting training for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions() if not done else []

            agent.update(state, action, reward, next_state, next_valid, done)

            total_reward += reward
            state = next_state

            if done and env.current == env.dst:
                success_count += 1
        
        # Check if success for this episode (simple check: if we reached dst)
        # However, the loop updates success_count inside. 
        # Let's just check if the last state was dst.
        is_success = (env.current == env.dst)
        success_history.append(is_success)

        rewards_history.append(total_reward)

        if ep % 200 == 0:
            avg_r = np.mean(rewards_history[-200:])
            current_success_rate = np.mean(success_history[-200:])
            print(f"Episode {ep:4d} | avg_reward={avg_r:6.2f} | "
                  f"success_rate={current_success_rate:5.2f} | epsilon={agent.epsilon:.3f}")

    return env, agent, rewards_history, success_history
