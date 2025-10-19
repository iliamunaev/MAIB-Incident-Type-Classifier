"""
Utility functions for MAIB Incident Type Classifier.
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> str:
  """Get available device (cuda or cpu)."""
  return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> Dict[str, Any]:
  """Get device information."""
  device_info = {
    "device": get_device(),
    "cuda_available": torch.cuda.is_available()
  }

  if torch.cuda.is_available():
    device_info.update({
      "device_name": torch.cuda.get_device_name(0),
      "capability": torch.cuda.get_device_capability(),
      "memory_allocated": torch.cuda.memory_allocated(),
      "memory_reserved": torch.cuda.memory_reserved()
    })

  return device_info


def print_device_info() -> None:
  """Print device information."""
  info = get_device_info()
  print(f"Device: {info['device']}")

  if info['cuda_available']:
    print(f"CUDA Device: {info['device_name']}")
    print(f"Capability: {info['capability']}")
    print(f"Memory Allocated: {info['memory_allocated'] / 1024**3:.2f} GB")
    print(f"Memory Reserved: {info['memory_reserved'] / 1024**3:.2f} GB")
  else:
    print("Running on CPU")


def ensure_dir(path: str) -> None:
  """Ensure directory exists."""
  os.makedirs(path, exist_ok=True)


def get_class_names_from_labels(labels: List[str]) -> List[str]:
  """Extract and sort unique class names from labels."""
  return sorted(set(labels))


def create_label_mappings(class_names: List[str]) -> Dict[str, Dict[int, str]]:
  """Create label mappings from class names."""
  id2label = {i: name for i, name in enumerate(class_names)}
  label2id = {name: i for i, name in id2label.items()}

  return {
    "id2label": id2label,
    "label2id": label2id
  }
