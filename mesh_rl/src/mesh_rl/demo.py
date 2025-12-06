import time
from mesh_rl.env import SimpleMeshEnv

def demo(agent, delay=0.5):
    """
    Runs a demo episode with the trained agent.
    """
    env = SimpleMeshEnv()
    state = env.reset()
    done = False
    
    print("\n--- Starting Demo Run ---")
    print(f"Start Node: {state}")
    
    path = [state]
    total_reward = 0
    
    # Temporarily set epsilon to 0 for greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    while not done:
        valid_actions = env.valid_actions()
        action = agent.choose_action(state, valid_actions)
        
        print(f"Agent chooses to go to node {action}...")
        time.sleep(delay)
        
        next_state, reward, done, info = env.step(action)
        
        latency = info.get('latency', 0)
        print(f" -> Moved to {next_state} (Latency: {latency}, Reward: {reward})")
        
        state = next_state
        total_reward += reward
        path.append(state)
        
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    print("\n--- Demo Finished ---")
    print(f"Path taken: {path}")
    print(f"Total Reward: {total_reward}")
    if env.current == env.dst:
        print("Result: SUCCESS (Reached Destination)")
    else:
        print("Result: FAILURE (Did not reach Destination)")
