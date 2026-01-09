#!/usr/bin/env python3
"""
Minimal test script to verify checkpoint loading logic works
"""
import sys
import torch
from pathlib import Path

# Test if we can load a checkpoint
checkpoint_path = "./models/Mutagenicity_1_GIN.pth"

print(f"Testing checkpoint loading from: {checkpoint_path}")
print(f"File exists: {Path(checkpoint_path).exists()}")

if Path(checkpoint_path).exists():
    try:
        # Try loading like the snippet does
        print("\n1. Trying bare torch.load...")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Loaded successfully!")
        print(f"Type: {type(state_dict)}")
        if isinstance(state_dict, dict):
            print(f"Keys: {list(state_dict.keys())[:5]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
else:
    print("Checkpoint file not found!")
