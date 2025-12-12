# Mesh RL - Reinforcement Learning for Network Routing

A Reinforcement Learning framework using **Goal-Conditioned Q-Learning** to optimize routing in mesh networks. The agent learns to route packets between any two nodes in a dynamic network topology.

## Table of Contents

- [How the Model Works](#how-the-model-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How to Test](#how-to-test)
- [How to Get Shortest Path](#how-to-get-shortest-path)
- [Sample Training Data](#sample-training-data)
- [Sample Results](#sample-results)
- [Project Structure](#project-structure)
- [Scripts Reference](#scripts-reference)
- [Troubleshooting](#troubleshooting)

---

## How the Model Works

### Architecture Overview

The system implements **Goal-Conditioned Q-Learning** for network routing:

#### 1. **Environment (`SimpleMeshEnv`)**
- **Graph Topology**: Generates a mesh network with N nodes (default: 50)
- **Edge Properties**: Each edge has variable latency (base 1-5 + stochastic noise 0-2)
- **Goal-Conditioned State**: State = `(current_node, target_node)` tuple
- **Reward Function**:
  - `-latency` per step (penalizes longer paths)
  - `+100` on reaching destination (encourages success)
  - `-20` if max steps exceeded

#### 2. **Agent (`QLearningAgent`)**
- **Q-Table**: Maps `(state, action)` → Q-value (expected future reward)
- **Epsilon-Greedy Policy**: 
  - Explores randomly with probability ε
  - Exploits best Q-value action otherwise
- **Q-Learning Update Rule**:
  ```
  Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
  ```
- **Hyperparameters**:
  - `alpha` (learning rate): 0.1
  - `gamma` (discount factor): 0.95
  - `epsilon`: Starts at 1.0, decays to 0.05

#### 3. **Training Process**
- **Offline Training**: Uses pre-generated synthetic data
- **Goal-Conditioned Learning**: Agent learns universal policy for any source-destination pair
- **State Space**: `(current_node, target_node)` tuples
- **Action Space**: Valid neighbor nodes at current position

### Graph Generation

The environment generates a mesh network with:

1. **Backbone Path**: Ensures connectivity from node 0 to node N-1
2. **Extra Edges**: Adds `num_nodes * 3` random edges for redundancy
   - Creates moderately dense graph (~15% density)
   - Average degree: ~7 neighbors per node
3. **Latency Model**: 
   - Base latency: Random integer 1-5
   - Stochastic component: 0, 1, or 2 (with probabilities 0.6, 0.3, 0.1)
4. **Reproducibility**: Uses seed (default: 42) for consistent topology

### Why Goal-Conditioned RL?

Traditional RL learns a single policy for one fixed goal. **Goal-Conditioned RL** learns a **universal policy**:

- State includes both current position AND target: `(current, target)`
- Agent learns: *"From node X, to reach node Y, go to node Z"*
- Enables routing between arbitrary node pairs **without retraining**
- One trained agent can handle any source-destination combination

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. **Navigate to the project directory**:
   ```bash
   cd mesh_rl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Dependencies:
   - `numpy` - Numerical computations
   - `matplotlib` - Plotting (optional, for visualizations)

3. **Verify installation**:
   ```bash
   python -c "import numpy; print('Installation successful!')"
   ```

---

## Quick Start

### Complete Workflow

#### Step 1: Generate Training Data

Generate synthetic training data with random source-destination pairs:

```bash
python scripts/generate_data.py --num-episodes 4000 --nodes 50 --output outputs/multi_goal_data.json
```

**What this does:**
- Creates a 50-node mesh network (seed=42)
- Generates 4000 episodes with random start/end points
- Saves transitions: `(state, action, reward, next_state, done)`
- Output: JSON file with ~1.1M transitions

#### Step 2: Train the Agent

Train the Q-Learning agent on the generated data:

```bash
python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20 --output outputs/multi_goal_agent.pkl
```

**What this does:**
- Loads transitions from JSON
- Trains Q-Learning agent for 20 epochs
- Updates Q-values using Bellman equation
- Saves trained agent as pickle file

#### Step 3: Get Shortest Path

Trace a path from any source to any target:

```bash
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6 --output outputs/trace_49_to_6.txt
```

**Output:**
```
--- Tracing Path (Multi-Goal): Node 49 -> Node 6 ---

[Step 1] At Node 49 (Goal: 6)
  >> Agent Chose: 8 (Max Q: 43.5203)
  -> Moved to 8

[Step 2] At Node 8 (Goal: 6)
  >> Agent Chose: 6 (Max Q: 62.5269)
  -> Moved to 6

SUCCESS: Reached Target 6
Full Path: [49, 8, 6]
```

---

## How to Test

### 1. Verify Agent Training

Check if the agent was properly trained and verify graph structure:

```bash
python scripts/verify_graph_and_agent.py
```

**Output includes:**
- Graph statistics (density: ~14.86%, connectivity)
- Q-table size and non-zero entries (~17,836 entries)
- Path analysis for test cases
- Agent performance metrics
- Detailed report saved to `outputs/verification_report.json`

**Sample Output:**
```
======================================================================
GRAPH AND AGENT VERIFICATION
======================================================================

1. GRAPH ANALYSIS
----------------------------------------------------------------------
Nodes: 50
Edges: 182
Max possible edges: 1225
Graph density: 0.1486 (14.86%)
Average degree: 7.28
Fully connected: True
Reachable nodes from 0: 50/50

2. PATH ANALYSIS: 49 -> 6
----------------------------------------------------------------------
Direct connection (49 -> 6): False
Shortest path: [49, 8, 6]
Shortest path length: 2 hops

Q-values at start state (49, 6):
  Action 8: Q = 43.5203 <-- CHOSEN
  Action 4: Q = 35.3910
  Action 33: Q = 21.0739
  Action 41: Q = 13.7238
  Action 35: Q = 0.0000

3. AGENT TRAINING VERIFICATION
----------------------------------------------------------------------
Q-table size: 17836 entries
Non-zero Q-values: 16274
Epsilon: 0.049911691230058335
Learning rate (alpha): 0.1
Discount factor (gamma): 0.95

4. MULTIPLE PATH SAMPLES
----------------------------------------------------------------------
Start  | Target | Shortest | Agent    | Match 
--------------------------------------------------
49     | 6      | 2        | 2        | YES   
0      | 49     | 2        | 2        | YES   
10     | 20     | 2        | 2        | YES   
5      | 45     | 3        | 3        | YES   
15     | 35     | 2        | 2        | YES   

ASSESSMENT:
AGENT STATUS: Properly trained!
   - Q-table has 17836 entries
   - 16274 non-zero Q-values
   - Agent correctly chose highest Q-value action (8 with Q=43.52)
   - Agent found optimal paths in all test cases
```

### 2. Evaluate Agent Performance

Test the agent on multiple random paths:

```bash
python scripts/evaluate_agent.py --agent outputs/multi_goal_agent.pkl --episodes 100 --nodes 50
```

**Output:**
- Success rate percentage
- Average reward per episode

**Sample Output:**
```
Evaluation Complete.
Success Rate: 98.00%
Average Reward: 89.45
```

### 3. Analyze Shortest Paths

Compare agent paths with optimal BFS shortest paths:

```bash
python scripts/analyze_paths.py --nodes 50 --seed 42 --pairs 10
```

**Output:** Table showing source, target, hops, and path for each pair. Saved to `outputs/shortest_paths_analysis.json`

**Sample Output:**
```
Source     | Target     | Hops   | Path
--------------------------------------------------
0          | 49         | 2      | [0, 33, 49]
40         | 7          | 2      | [40, 9, 7]
1          | 47         | 2      | [1, 21, 47]
```

### 4. Run Interactive Demo

Watch the agent route in real-time:

**With pre-trained agent:**
```bash
python scripts/run_demo.py --agent outputs/multi_goal_agent.pkl --nodes 50
```

**Without pre-trained agent** (trains online first):
```bash
python scripts/run_demo.py
```

**Online Training** (simple 4-node topology):
```bash
python scripts/run_training.py
```

---

## How to Get Shortest Path

### Method 1: Trace Specific Path (Recommended)

Trace how the agent routes from source to target with detailed Q-value information:

```bash
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start <SOURCE> --target <TARGET> [--output <FILE>]
```

**Example:**
```bash
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6 --output outputs/trace_49_to_6.txt
```

**Output format:**
```
--- Tracing Path (Multi-Goal): Node 49 -> Node 6 ---

[Step 1] At Node 49 (Goal: 6)
  >> Agent Chose: 8 (Max Q: 43.5203)
  -> Moved to 8

[Step 2] At Node 8 (Goal: 6)
  >> Agent Chose: 6 (Max Q: 62.5269)
  -> Moved to 6

SUCCESS: Reached Target 6
Full Path: [49, 8, 6]
```

**Analysis:**
- Path length: 2 hops (optimal)
- Node 8 is a common neighbor of nodes 49 and 6
- Agent correctly chose highest Q-value action at each step

### Method 2: Analyze Multiple Paths

Get shortest paths for multiple node pairs using BFS:

```bash
python scripts/analyze_paths.py --nodes 50 --seed 42 --pairs 10
```

**Output:** Table showing source, target, hops, and path for each pair. Results saved to `outputs/shortest_paths_analysis.json`

### Method 3: Online Training (Simple 4-Node)

For the classic 4-node topology, train and test online:

```bash
python scripts/run_training.py
```

This trains an agent online and shows the routing path in a demo.

---

## Sample Training Data

### Data Format

The training data is stored as JSON with the following structure:

```json
{
  "metadata": {
    "num_nodes": 50,
    "seed": 42,
    "num_episodes": 4000,
    "type": "goal_conditioned"
  },
  "transitions": [
    {
      "state": [42, 47],
      "action": 45,
      "reward": -5.0,
      "next_state": [45, 47],
      "next_valid_actions": [48, 1, 20, 19, 4, 42, 27, 2],
      "done": false
    },
    {
      "state": [45, 47],
      "action": 1,
      "reward": -1.0,
      "next_state": [1, 47],
      "next_valid_actions": [45, 20, 7, 26, 14, 21],
      "done": false
    },
    {
      "state": [1, 47],
      "action": 47,
      "reward": 95.0,
      "next_state": [47, 47],
      "next_valid_actions": [],
      "done": true
    }
  ]
}
```

### Sample Training Data Statistics

- **File**: `outputs/multi_goal_data.json`
- **Episodes**: 4000
- **Nodes**: 50
- **Total Transitions**: ~1,100,000
- **State Format**: `[current_node, target_node]`
- **Reward Range**: 
  - Per step: -5 to -1 (latency penalty)
  - Success: +100 (reaching destination)
  - Failure: -100 (timeout/invalid action)

### Generate Your Own Data

```bash
# Generate 1000 episodes for 50-node network
python scripts/generate_data.py --num-episodes 1000 --nodes 50 --seed 42 --output outputs/my_data.json

# Generate 2000 episodes for 100-node network
python scripts/generate_data.py --num-episodes 2000 --nodes 100 --seed 42 --output outputs/large_network_data.json
```

---

## Sample Results

### Example 1: Path Tracing (49 → 6)

**Command:**
```bash
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6
```

**Output:**
```
--- Tracing Path (Multi-Goal): Node 49 -> Node 6 ---

[Step 1] At Node 49 (Goal: 6)
  >> Agent Chose: 8 (Max Q: 43.5203)
  -> Moved to 8

[Step 2] At Node 8 (Goal: 6)
  >> Agent Chose: 6 (Max Q: 62.5269)
  -> Moved to 6

SUCCESS: Reached Target 6
Full Path: [49, 8, 6]
```

**Analysis:**
- Path length: 2 hops (optimal)
- Node 8 is a common neighbor of nodes 49 and 6
- Agent correctly chose highest Q-value action at each step

### Example 2: Verification Results

**Command:**
```bash
python scripts/verify_graph_and_agent.py
```

**Sample Output:**
```
======================================================================
GRAPH AND AGENT VERIFICATION
======================================================================

1. GRAPH ANALYSIS
----------------------------------------------------------------------
Nodes: 50
Edges: 182
Max possible edges: 1225
Graph density: 0.1486 (14.86%)
Average degree: 7.28
Fully connected: True
Reachable nodes from 0: 50/50

2. PATH ANALYSIS: 49 -> 6
----------------------------------------------------------------------
Direct connection (49 -> 6): False
Shortest path: [49, 8, 6]
Shortest path length: 2 hops

Q-values at start state (49, 6):
  Action 8: Q = 43.5203 <-- CHOSEN
  Action 4: Q = 35.3910
  Action 33: Q = 21.0739
  Action 41: Q = 13.7238
  Action 35: Q = 0.0000

3. AGENT TRAINING VERIFICATION
----------------------------------------------------------------------
Q-table size: 17836 entries
Non-zero Q-values: 16274
Epsilon: 0.049911691230058335
Learning rate (alpha): 0.1
Discount factor (gamma): 0.95

4. MULTIPLE PATH SAMPLES
----------------------------------------------------------------------
Start  | Target | Shortest | Agent    | Match 
--------------------------------------------------
49     | 6      | 2        | 2        | YES   
0      | 49     | 2        | 2        | YES   
10     | 20     | 2        | 2        | YES   
5      | 45     | 3        | 3        | YES   
15     | 35     | 2        | 2        | YES   

ASSESSMENT:
AGENT STATUS: Properly trained!
   - Q-table has 17836 entries
   - 16274 non-zero Q-values
   - Agent correctly chose highest Q-value action (8 with Q=43.52)
   - Agent found optimal paths in all test cases
```

### Example 3: Training Progress

**Online Training Output:**
```
Starting training for 2000 episodes...
Episode  200 | avg_reward= 45.23 | success_rate= 0.85 | epsilon=0.810
Episode  400 | avg_reward= 78.56 | success_rate= 0.92 | epsilon=0.656
Episode  600 | avg_reward= 85.34 | success_rate= 0.95 | epsilon=0.531
Episode  800 | avg_reward= 88.12 | success_rate= 0.97 | epsilon=0.430
Episode 1000 | avg_reward= 90.45 | success_rate= 0.98 | epsilon=0.348
Episode 1200 | avg_reward= 91.23 | success_rate= 0.99 | epsilon=0.282
Episode 1400 | avg_reward= 92.01 | success_rate= 0.99 | epsilon=0.228
Episode 1600 | avg_reward= 92.45 | success_rate= 1.00 | epsilon=0.185
Episode 1800 | avg_reward= 92.67 | success_rate= 1.00 | epsilon=0.150
Episode 2000 | avg_reward= 92.89 | success_rate= 1.00 | epsilon=0.121
```

### Example 4: Q-Value Analysis

At state `(49, 6)`, the agent's Q-values for valid actions:

| Action | Q-Value | Decision |
|--------|---------|----------|
| 8      | 43.52   | ✓ Chosen (highest) |
| 4      | 35.39   | - |
| 33     | 21.07   | - |
| 41     | 13.72   | - |
| 35     | 0.00    | - |

The agent correctly selects action 8, which leads to the optimal 2-hop path.

---

## Project Structure

```
mesh_rl/
├── src/
│   └── mesh_rl/
│       ├── __init__.py          # Package exports
│       ├── env.py                # SimpleMeshEnv - network environment
│       ├── agent.py              # QLearningAgent - Q-learning implementation
│       ├── training.py           # Online training loop
│       ├── train_offline.py      # Offline training from data
│       ├── data_gen.py           # Synthetic data generation
│       └── demo.py               # Demo/visualization
├── scripts/
│   ├── generate_data.py         # Generate training data
│   ├── train_offline.py         # Train agent offline
│   ├── trace_path.py            # Trace specific path
│   ├── run_training.py           # Online training
│   ├── run_demo.py               # Interactive demo
│   ├── evaluate_agent.py         # Evaluate performance
│   ├── analyze_paths.py          # Shortest path analysis
│   └── verify_graph_and_agent.py # Verification tool
├── outputs/                      # Generated files
│   ├── multi_goal_data.json      # Training data
│   ├── multi_goal_agent.pkl      # Trained agent
│   ├── trace_*.txt               # Path traces
│   ├── shortest_paths_analysis.json
│   └── verification_report.json
├── requirements.txt              # Dependencies
├── pyproject.toml               # Package config
├── README.md                     # This file
└── ANALYSIS_RESULTS.md           # Detailed analysis document
```

---

## Scripts Reference

### Essential Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_data.py` | Generate training data | `python scripts/generate_data.py --num-episodes 4000 --nodes 50` |
| `train_offline.py` | Train agent from data | `python scripts/train_offline.py --data <file> --epochs 20` |
| `trace_path.py` | Trace specific path | `python scripts/trace_path.py --agent <file> --start X --target Y` |
| `verify_graph_and_agent.py` | Verify training | `python scripts/verify_graph_and_agent.py` |

### Additional Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_training.py` | Online training | `python scripts/run_training.py` |
| `run_demo.py` | Interactive demo | `python scripts/run_demo.py --agent <file>` |
| `evaluate_agent.py` | Evaluate performance | `python scripts/evaluate_agent.py --agent <file> --episodes 100` |
| `analyze_paths.py` | Shortest path analysis | `python scripts/analyze_paths.py --nodes 50 --pairs 10` |

### Script Arguments

#### `generate_data.py`
```bash
--num-episodes N    # Number of episodes (default: 1000)
--nodes N           # Number of nodes (default: 50)
--seed N            # Random seed (default: 42)
--output FILE       # Output file path (default: outputs/multi_goal_data.json)
```

#### `train_offline.py`
```bash
--data FILE         # Training data JSON file (default: outputs/multi_goal_data.json)
--epochs N          # Training epochs (default: 10)
--output FILE       # Output agent pickle file (default: outputs/multi_goal_agent.pkl)
```

#### `trace_path.py`
```bash
--agent FILE        # Trained agent pickle file (default: outputs/multi_goal_agent.pkl)
--start N           # Source node (required)
--target N          # Target node (default: 49)
--nodes N           # Number of nodes (default: 50)
--seed N            # Random seed (default: 42)
--output FILE       # Output trace file (optional)
```

#### `verify_graph_and_agent.py`
```bash
# No arguments - uses defaults (50 nodes, seed 42, agent: outputs/multi_goal_agent.pkl)
```

#### `evaluate_agent.py`
```bash
--agent FILE        # Trained agent (required)
--episodes N        # Evaluation episodes (default: 100)
--nodes N           # Number of nodes (default: 50)
```

#### `analyze_paths.py`
```bash
--nodes N           # Number of nodes (default: 50)
--seed N            # Random seed (default: 42)
--pairs N           # Number of test pairs (default: 10)
```

#### `run_demo.py`
```bash
--agent FILE        # Pre-trained agent (optional, trains online if not provided)
--nodes N           # Number of nodes (default: 50)
--seed N            # Random seed (optional)
```

---

## Troubleshooting

### Common Issues

1. **"Agent file not found"**
   - Solution: Train the agent first using `train_offline.py`
   ```bash
   python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20
   ```

2. **"Data file not found"**
   - Solution: Generate data first using `generate_data.py`
   ```bash
   python scripts/generate_data.py --num-episodes 4000 --nodes 50
   ```

3. **"Module not found"**
   - Solution: Ensure you're in the `mesh_rl` directory and dependencies are installed
   ```bash
   cd mesh_rl
   pip install -r requirements.txt
   ```

4. **Paths seem too short**
   - This is normal! The graph is moderately dense (14.86%), so many pairs are 2-3 hops away
   - See `ANALYSIS_RESULTS.md` for detailed explanation
   - To create longer paths, reduce `num_extra_edges` in `src/mesh_rl/env.py` (line 58)

5. **Low success rate**
   - Ensure agent was trained with sufficient epochs (recommended: 20+)
   - Check that training data was generated with same seed as evaluation
   - Verify graph topology matches training topology

6. **Import errors**
   - Make sure you're running scripts from the `mesh_rl` directory
   - Check that `src/mesh_rl/` contains all required modules

---

## Quick Command Reference

### Complete Workflow (First Time)
```bash
# 1. Generate data
python scripts/generate_data.py --num-episodes 4000 --nodes 50

# 2. Train agent
python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20

# 3. Get shortest path
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6

# 4. Verify everything works
python scripts/verify_graph_and_agent.py
```

### Common Tasks
```bash
# Get path from node X to node Y
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start X --target Y

# Evaluate agent performance
python scripts/evaluate_agent.py --agent outputs/multi_goal_agent.pkl --episodes 100

# Compare with optimal paths
python scripts/analyze_paths.py --nodes 50 --pairs 10

# Run interactive demo
python scripts/run_demo.py --agent outputs/multi_goal_agent.pkl
```

---

## License

MIT License

## Contributing

Contributions welcome! Please ensure code follows existing style and includes tests.
