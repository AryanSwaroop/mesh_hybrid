# Quick Reference Guide

## Most Common Commands

### 1. Full Workflow (First Time)

```bash
# Step 1: Generate training data
python scripts/generate_data.py --num-episodes 4000 --nodes 50 --output outputs/multi_goal_data.json

# Step 2: Train agent
python scripts/train_offline.py --data outputs/multi_goal_data.json --epochs 20 --output outputs/multi_goal_agent.pkl

# Step 3: Test agent
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start 49 --target 6
```

### 2. Get Shortest Path

```bash
# Trace path from node X to node Y
python scripts/trace_path.py --agent outputs/multi_goal_agent.pkl --start X --target Y --output outputs/trace_X_to_Y.txt
```

### 3. Verify Everything Works

```bash
# Comprehensive verification
python scripts/verify_graph_and_agent.py
```

### 4. Evaluate Performance

```bash
# Test on 100 random paths
python scripts/evaluate_agent.py --agent outputs/multi_goal_agent.pkl --episodes 100
```

### 5. Compare with Optimal Paths

```bash
# Get shortest paths for 10 random pairs
python scripts/analyze_paths.py --nodes 50 --pairs 10
```

## File Locations

- **Trained Agent**: `outputs/multi_goal_agent.pkl`
- **Training Data**: `outputs/multi_goal_data.json`
- **Path Traces**: `outputs/trace_*.txt`
- **Analysis**: `outputs/shortest_paths_analysis.json`
- **Verification**: `outputs/verification_report.json`

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Agent file not found | Run `train_offline.py` first |
| Data file not found | Run `generate_data.py` first |
| Path seems too short | Normal! Graph is dense (see README) |
| Import errors | Run from `mesh_rl/` directory |

