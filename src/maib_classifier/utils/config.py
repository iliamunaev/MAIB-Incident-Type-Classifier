"""
Configuration management for MAIB Incident Type Classifier.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
  """Model configuration parameters."""
  name: str = "microsoft/deberta-v3-base"
  max_length: int = 256
  num_labels: int = 11
  gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
  """Training configuration parameters."""
  output_dir: str = "outputs"
  learning_rate: float = 2e-5
  per_device_train_batch_size: int = 32
  per_device_eval_batch_size: int = 64
  num_train_epochs: int = 3
  eval_strategy: str = "epoch"
  save_strategy: str = "epoch"
  load_best_model_at_end: bool = True
  metric_for_best_model: str = "macro_f1"
  logging_steps: int = 50
  report_to: str = "none"

  # Optimization
  fp16: bool = True
  bf16: bool = False
  optim: str = "adamw_torch"
  lr_scheduler_type: str = "linear"
  warmup_ratio: float = 0.1
  weight_decay: float = 0.01

  # Data Loading
  dataloader_pin_memory: bool = True
  dataloader_num_workers: int = 2
  eval_accumulation_steps: int = 4
  torch_compile: bool = False


@dataclass
class DataConfig:
  """Data configuration parameters."""
  train_split: float = 0.8
  validation_split: float = 0.1
  test_split: float = 0.1
  random_seed: int = 42
  num_proc: Optional[int] = None


@dataclass
class EvaluationConfig:
  """Evaluation configuration parameters."""
  metrics: list = field(default_factory=lambda: ["accuracy", "macro_f1", "weighted_f1"])
  eval_split: str = "validation"


@dataclass
class InferenceConfig:
  """Inference configuration parameters."""
  top_k: int = 3
  max_length: int = 256
  device: str = "auto"


@dataclass
class LoggingConfig:
  """Logging configuration parameters."""
  level: str = "INFO"
  format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: str = "logs/training.log"


@dataclass
class Config:
  """Main configuration class."""
  model: ModelConfig = field(default_factory=ModelConfig)
  training: TrainingConfig = field(default_factory=TrainingConfig)
  data: DataConfig = field(default_factory=DataConfig)
  evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
  inference: InferenceConfig = field(default_factory=InferenceConfig)
  logging: LoggingConfig = field(default_factory=LoggingConfig)

  @classmethod
  def from_yaml(cls, config_path: str) -> "Config":
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
      raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
      config_dict = yaml.safe_load(f)

    return cls.from_dict(config_dict)

  @classmethod
  def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
    """Create configuration from dictionary."""
    config = cls()

    if 'model' in config_dict:
      config.model = ModelConfig(**config_dict['model'])

    if 'training' in config_dict:
      config.training = TrainingConfig(**config_dict['training'])

    if 'data' in config_dict:
      config.data = DataConfig(**config_dict['data'])

    if 'evaluation' in config_dict:
      config.evaluation = EvaluationConfig(**config_dict['evaluation'])

    if 'inference' in config_dict:
      config.inference = InferenceConfig(**config_dict['inference'])

    if 'logging' in config_dict:
      config.logging = LoggingConfig(**config_dict['logging'])

    return config

  def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
    return {
      'model': self.model.__dict__,
      'training': self.training.__dict__,
      'data': self.data.__dict__,
      'evaluation': self.evaluation.__dict__,
      'inference': self.inference.__dict__,
      'logging': self.logging.__dict__
    }

  def save_yaml(self, config_path: str) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
      yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

  def setup_directories(self) -> None:
    """Create necessary directories."""
    Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
    Path(self.logging.file).parent.mkdir(parents=True, exist_ok=True)

  def setup_data_config(self) -> None:
    """Setup data configuration with system defaults."""
    if self.data.num_proc is None:
      self.data.num_proc = max(1, (os.cpu_count() or 2) // 2)

  def setup_inference_config(self) -> None:
    """Setup inference configuration with device detection."""
    if self.inference.device == "auto":
      import torch
      self.inference.device = "cuda" if torch.cuda.is_available() else "cpu"
