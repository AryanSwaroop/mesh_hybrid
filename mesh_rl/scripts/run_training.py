#!/usr/bin/env python3
import sys
import os

# Ensure the package is in the path if running from source without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mesh_rl import train

if __name__ == "__main__":
    print("Running training script...")
    train()
