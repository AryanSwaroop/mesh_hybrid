# Mesh RL Project

A Reinforcement Learning framework for optimizing routing in mesh networks. This project uses Q-Learning to find optimal paths in dynamic network topologies, handling latency, packet loss, and robust routing.

## Key Features

*   **Scalable Simulation**: Supports simulating mesh networks with 50+ nodes.
*   **Goal-Conditioned RL**: Agents can route between *any* two arbitrary nodes (Multi-Goal Routing).
*   **Trace Analysis**: Tools to visualize step-by-step agent decision making (Q-values, neighbors).
*   **Realistic Latency**: Simulates variable latency and packet drops.
*   **Reproducibility**: Seeded environments for consistent testing and evaluation.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `numpy`, `matplotlib`)*

## Quick Start

### 1. Train the Multi-Goal Agent
Generate data and train an agent that can route from any node to any node.

```bash
# 1. Generate synthetic training data (4000 episodes)
python scripts/generate_data.py --num-episodes 4000 --nodes 50 --output outputs/multi_goal_data.json

# 2. Train the agent
python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20 --output outputs/multi_goal_agent.pkl
```

### 2. Verify Routing
Trace how the agent routes a packet from Node 50 to Node 20.

```bash
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 50 --target 20
```

## Detailed Usage

### Classic 4-Node Experiment
Run a controlled experiment on a classic designated topology to visualize learning curves.

```bash
python scripts/run_experiment.py --topology classic --episodes 500
```
*Outputs: `outputs/reward_curve.png`, `outputs/path_stats.json`*

### Analysis Tools

*   **Trace Path**: Debug agent decisions.
    ```bash
    python scripts/trace_path.py --start [SRC] --target [DST] --output [FILE]
    ```
*   **Batch Traces**: Generate multiple random traces for verification.
    ```bash
    python scripts/generate_multiple_traces.py --count 10 --output outputs/path_trace.txt
    ```
*   **Shortest Path Analysis**: Compare agent paths vs BFS optimal hops.
    ```bash
    python scripts/analyze_paths.py --nodes 50 --pairs 15
    ```

## Project Structure

*   `src/mesh_rl/`: Core package (Env, Agent, Data Gen).
*   `scripts/`: Executable scripts for training, demo, and analysis.
*   `outputs/`: Artifacts (models, logs, plots, JSON data).

## Recent Findings

*   **Multi-Goal Success**: The agent successfully learned to generalize routing across the 50-node mesh, achieving 100% success rates on valid random pairs.
*   **Scalability**: The Q-Learning approach scaled effectively to 50 nodes with adequate training data (5000+ transitions).
