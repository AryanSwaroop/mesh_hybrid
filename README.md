# Mesh RL Project

A Reinforcement Learning framework using **Goal-Conditioned Q-Learning** to optimize routing in mesh networks.

## Quick Start

Navigate to the `mesh_rl` directory for full documentation:

```bash
cd mesh_rl
```

See `mesh_rl/README.md` for:
- Complete installation instructions
- How the model works
- How to test and evaluate
- How to get shortest paths
- Sample training data and results
- Full scripts reference

## Main Workflow

1. **Generate training data:**
   ```bash
   cd mesh_rl
   python scripts/generate_data.py --num-episodes 4000 --nodes 50
   ```

2. **Train the agent:**
   ```bash
   python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20
   ```

3. **Test the agent:**
   ```bash
   python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6
   ```

For detailed documentation, see [`mesh_rl/README.md`](mesh_rl/README.md).
