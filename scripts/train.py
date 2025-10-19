#!/usr/bin/env python3
"""
Main training script for MAIB Incident Type Classifier.

This script provides a complete training pipeline for the MAIB incident classification model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maib_classifier.utils.config import Config
from maib_classifier.utils.logger import setup_logger
from maib_classifier.utils.utils import set_seed, print_device_info
from maib_classifier.data.processor import DataProcessor
from maib_classifier.models.trainer import ModelTrainer
from maib_classifier.models.evaluator import ModelEvaluator


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Train MAIB Incident Type Classifier",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )


  parser.add_argument(
    "--config",
    type=str,
    default="configs/config.yaml",
    help="Path to configuration file"
  )

  parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="Output directory for model and results"
  )

  parser.add_argument(
    "--model_name",
    type=str,
    default="microsoft/deberta-v3-base",
    help="Model name to use for training"
  )

  parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs"
  )

  parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Training batch size"
  )

  parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="Learning rate"
  )

  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
  )

  parser.add_argument(
    "--eval_only",
    action="store_true",
    help="Only evaluate existing model (skip training)"
  )

  parser.add_argument(
    "--model_path",
    type=str,
    help="Path to existing model for evaluation only"
  )

  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose logging"
  )

  return parser.parse_args()


def setup_config(args) -> Config:
  """Setup configuration from args and file."""
  # Load base config
  if os.path.exists(args.config):
    config = Config.from_yaml(args.config)
  else:
    config = Config()

  # Override with command line arguments
  config.training.output_dir = args.output_dir
  config.model.name = args.model_name
  config.training.num_train_epochs = args.epochs
  config.training.per_device_train_batch_size = args.batch_size
  config.training.learning_rate = args.learning_rate
  config.data.random_seed = args.seed

  # Setup configuration
  config.setup_directories()
  config.setup_data_config()
  config.setup_inference_config()

  return config


def train_model(config: Config) -> dict:
  """Train the model."""
  logger = setup_logger(
    level="DEBUG" if config.logging.level == "DEBUG" else "INFO",
    log_file=config.logging.file
  )

  logger.info("Starting MAIB Incident Type Classifier training")
  logger.info(f"Output directory: {config.training.output_dir}")
  logger.info(f"Model: {config.model.name}")

  # Print device information
  print_device_info()

  # Set random seed
  set_seed(config.data.random_seed)

  # Initialize data processor
  data_processor = DataProcessor(config)

  # Process data
  logger.info("Processing data...")
  dataset, metadata = data_processor.process_data()

  # Initialize model trainer
  model_trainer = ModelTrainer(config)

  # Train and evaluate
  logger.info("Starting training...")
  results = model_trainer.train_and_evaluate(dataset, metadata)

  logger.info("Training completed successfully!")
  return results


def evaluate_model(config: Config, model_path: str) -> dict:
  """Evaluate existing model."""
  logger = setup_logger(
    level="DEBUG" if config.logging.level == "DEBUG" else "INFO",
    log_file=config.logging.file
  )

  logger.info("Starting model evaluation")
  logger.info(f"Model path: {model_path}")

  # Set random seed
  set_seed(config.data.random_seed)

  # Process data
  data_processor = DataProcessor(config)
  dataset, metadata = data_processor.process_data()

  # Initialize trainer and load model
  model_trainer = ModelTrainer(config)
  model_trainer.setup_model(
    tokenizer=metadata["tokenizer"],
    num_labels=metadata["num_classes"],
    id2label=metadata["label_mappings"]["id2label"],
    label2id=metadata["label_mappings"]["label2id"]
  )

  # Load existing model
  model_trainer.model = model_trainer.model.from_pretrained(model_path)
  model_trainer.model.to(model_trainer.config.inference.device)

  # Setup trainer for evaluation
  model_trainer.setup_trainer(dataset, eval_split=config.evaluation.eval_split)

  # Evaluate
  eval_results = model_trainer.evaluate()

  logger.info("Evaluation completed successfully!")
  return eval_results


def main():
  """Main function."""
  args = parse_args()

  # Setup configuration
  config = setup_config(args)

  # Setup logging
  log_level = "DEBUG" if args.verbose else config.logging.level
  logger = setup_logger(
    level=log_level,
    log_file=config.logging.file
  )

  try:
    if args.eval_only:
      if not args.model_path:
        raise ValueError("--model_path is required for --eval_only")

      # Evaluation only
      results = evaluate_model(config, args.model_path)
      logger.info("Evaluation results:")
      for metric, value in results.items():
        if metric.startswith("eval_"):
          clean_metric = metric.replace("eval_", "")
          logger.info(f"  {clean_metric}: {value:.4f}")

    else:
      # Full training pipeline
      results = train_model(config)

      # Print final results
      logger.info("Final training results:")
      logger.info(f"  Training loss: {results['training']['train_loss']:.4f}")
      logger.info(f"  Training runtime: {results['training']['train_runtime']:.2f}s")

      logger.info("Final evaluation results:")
      for metric, value in results['evaluation'].items():
        if metric.startswith("eval_"):
          clean_metric = metric.replace("eval_", "")
          logger.info(f"  {clean_metric}: {value:.4f}")

      logger.info(f"Model saved to: {results['model_path']}")

  except Exception as e:
    logger.error(f"Training failed: {e}")
    raise


if __name__ == "__main__":
  main()
