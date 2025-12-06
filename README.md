# Mesh RL

A simple Reinforcement Learning project using Q-learning for routing in a small 4-node mesh network.

## Project Structure

- `src/mesh_rl`: Main package source code.
  - `env.py`: Defines the `SimpleMeshEnv` and graph topology.
  - `agent.py`: Implements the `QLearningAgent`.
  - `training.py`: Training loop logic.
  - `demo.py`: Visualization/Demo logic.
- `scripts/`: CLI scripts to run training and demos.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### Run Training
To train the agent and see the progress:
```bash
python scripts/run_training.py
# OR if installed
python -m mesh_rl.training
```

### Run Demo
To train an agent and then watch it perform a routing task:
```bash
python scripts/run_demo.py
```
