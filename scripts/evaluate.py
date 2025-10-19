#!/usr/bin/env python3
"""
Evaluation script for MAIB Incident Type Classifier.

This script provides comprehensive evaluation of the trained model including
confusion matrix, per-class metrics, and visualizations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from maib_classifier.utils.config import Config
from maib_classifier.utils.logger import setup_logger
from maib_classifier.data.processor import DataProcessor
from maib_classifier.models.evaluator import ModelEvaluator
from maib_classifier.inference.predictor import MAIBPredictor


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Evaluate MAIB Incident Type Classifier",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to trained model"
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
    default="evaluation_outputs",
    help="Output directory for evaluation results"
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
  if Path(args.config).exists():
    config = Config.from_yaml(args.config)
  else:
    config = Config()

  # Setup configuration
  config.setup_directories()
  config.setup_data_config()
  config.setup_inference_config()

  return config


def evaluate_model(config: Config, model_path: str, output_dir: str):
  """Evaluate the model comprehensively."""
  logger = setup_logger(
    level="DEBUG" if config.logging.level == "DEBUG" else "INFO",
    log_file=os.path.join(output_dir, "evaluation.log")
  )

  logger.info("Starting comprehensive model evaluation")
  logger.info(f"Model path: {model_path}")
  logger.info(f"Output directory: {output_dir}")

  # Ensure output directory exists
  os.makedirs(output_dir, exist_ok=True)

  # Process test data
  logger.info("Processing test data...")
  data_processor = DataProcessor(config)
  dataset, metadata = data_processor.process_data()

  # Get test dataset
  test_dataset = dataset["test"]

  # Initialize predictor
  predictor = MAIBPredictor(config, model_path)
  predictor.load_model(model_path)

  # Get predictions
  logger.info("Generating predictions...")
  texts = []
  true_labels = []

  # Extract texts and labels from test dataset
  for example in test_dataset:
    # Reconstruct text from tokenized input (approximate)
    text = predictor.tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    texts.append(text)
    true_labels.append(int(example["labels"]))

  # Get predictions
  predictions = predictor.predict_batch(texts)
  # Convert label names to indices using label mappings
  label2id = metadata["label_mappings"]["label2id"]
  predicted_labels = [label2id[pred[0][0]] for pred in predictions]  # Get top prediction

  # Initialize evaluator
  evaluator = ModelEvaluator(metadata["label_mappings"]["id2label"])

  # Run comprehensive evaluation
  logger.info("Running comprehensive evaluation...")
  results = evaluator.evaluate_model(true_labels, predicted_labels, output_dir)

  # Print summary
  logger.info("Evaluation completed successfully!")
  logger.info(f"Results saved to: {output_dir}")

  # Print key metrics
  report = results["classification_report"]
  logger.info("Key Metrics:")
  logger.info(f"  Overall Accuracy: {report['accuracy']:.4f}")
  logger.info(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
  logger.info(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

  return results


def main():
  """Main function."""
  args = parse_args()

  # Setup configuration
  config = setup_config(args)

  # Setup logging
  log_level = "DEBUG" if args.verbose else "INFO"
  logger = setup_logger(level=log_level)

  try:
    # Run evaluation
    results = evaluate_model(config, args.model_path, args.output_dir)

    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Check the following files:")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_f1.png")
    print(f"  - classification_report.txt")
    print(f"  - evaluation.log")

  except Exception as e:
    logger.error(f"Evaluation failed: {e}")
    raise


if __name__ == "__main__":
  main()
