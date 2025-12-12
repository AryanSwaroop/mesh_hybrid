#!/usr/bin/env python3
import sys
import os
import argparse

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl.train_offline import train_offline

def main():
    parser = argparse.ArgumentParser(description="Train agent offline using synthetic data.")
    parser.add_argument("--data", type=str, default="data/synthetic_data.json", help="Path to training data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--output", type=str, default="trained_agent.pkl", help="Path to save trained agent.")
    
    args = parser.parse_args()
    
    train_offline(data_file=args.data, epochs=args.epochs, output_agent=args.output)

if __name__ == "__main__":
    main()
