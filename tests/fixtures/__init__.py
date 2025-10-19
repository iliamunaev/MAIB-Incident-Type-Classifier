"""
Test fixtures for MAIB Incident Type Classifier.
"""

import json
from pathlib import Path

# Sample test data
SAMPLE_DATA = [
    {"text": "Crew member fell overboard during rough weather", "label": "Accident to person(s)"},
    {"text": "Vessel collided with another ship in fog", "label": "Collision"},
    {"text": "Engine room fire caused by fuel leak", "label": "Fire / Explosion"},
    {"text": "Ship ran aground on sandbar", "label": "Grounding / Stranding"},
    {"text": "Hull breach due to collision", "label": "Hull Failure"},
]

# Test configuration
TEST_CONFIG = {
    "model": {
        "name": "microsoft/deberta-v3-base",
        "max_length": 128,
        "num_labels": 5,
        "gradient_checkpointing": False
    },
    "training": {
        "output_dir": "test_outputs",
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 1,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "logging_steps": 10,
        "report_to": "none",
        "fp16": False,
        "bf16": False,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "eval_accumulation_steps": 1,
        "torch_compile": False
    },
    "data": {
        "train_split": 0.6,
        "validation_split": 0.2,
        "test_split": 0.2,
        "random_seed": 42,
        "num_proc": 1
    },
    "evaluation": {
        "metrics": ["accuracy", "macro_f1", "weighted_f1"],
        "eval_split": "validation"
    },
    "inference": {
        "top_k": 3,
        "max_length": 128,
        "device": "cpu"
    },
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "test_logs/test.log"
    }
}

def create_test_data_file(file_path: str) -> None:
    """Create a test data file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item) + '\n')

def create_test_config_file(file_path: str) -> None:
    """Create a test configuration file."""
    import yaml

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        yaml.dump(TEST_CONFIG, f, default_flow_style=False, indent=2)
