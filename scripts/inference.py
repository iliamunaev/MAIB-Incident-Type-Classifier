#!/usr/bin/env python3
"""
Inference script for MAIB Incident Type Classifier.

This script provides inference capabilities for the trained MAIB incident classification model.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from maib_classifier.utils.config import Config
from maib_classifier.utils.logger import setup_logger
from maib_classifier.inference.predictor import MAIBInferencePipeline


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="MAIB Incident Type Classifier Inference",
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
    "--text",
    type=str,
    help="Single text to classify"
  )

  parser.add_argument(
    "--file",
    type=str,
    help="File containing texts to classify (one per line)"
  )

  parser.add_argument(
    "--output",
    type=str,
    help="Output file for predictions"
  )

  parser.add_argument(
    "--top_k",
    type=int,
    default=3,
    help="Number of top predictions to return"
  )

  parser.add_argument(
    "--interactive",
    action="store_true",
    help="Interactive prediction mode"
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

  # Override with command line arguments
  config.inference.top_k = args.top_k

  # Setup configuration
  config.setup_inference_config()

  return config


def predict_single_text(pipeline: MAIBInferencePipeline, text: str, top_k: int):
  """Predict for a single text."""
  print(f"Input: {text}\n")

  predictions = pipeline.predict(text, top_k=top_k)

  print("Predictions:")
  for label, confidence in predictions:
    print(f"  {label:25s} {confidence:5.2f}%")


def predict_from_file(pipeline: MAIBInferencePipeline, file_path: str, output_path: Optional[str], top_k: int):
  """Predict from file."""
  import json

  # Read texts from file
  with open(file_path, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

  print(f"Loaded {len(texts)} texts from {file_path}")

  # Predict
  predictions = pipeline.predict_batch(texts, top_k=top_k)

  # Format results
  results = []
  for i, (text, preds) in enumerate(zip(texts, predictions)):
    result = {
      "index": i,
      "text": text,
      "predictions": preds,
      "top_prediction": preds[0] if preds else ("Unknown", 0.0)
    }
    results.append(result)

    # Print first few results
    if i < 5:
      print(f"\nText {i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
      print("Predictions:")
      for label, confidence in preds:
        print(f"  {label:25s} {confidence:5.2f}%")

  if len(texts) > 5:
    print(f"\n... and {len(texts) - 5} more predictions")

  # Save results if output path provided
  if output_path:
    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def main():
  """Main function."""
  args = parse_args()

  # Setup configuration
  config = setup_config(args)

  # Setup logging
  log_level = "DEBUG" if args.verbose else "INFO"
  logger = setup_logger(level=log_level)

  try:
    # Initialize pipeline
    pipeline = MAIBInferencePipeline(config)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    pipeline.load_model(args.model_path)

    # Print model info
    model_info = pipeline.predictor.get_model_info()
    logger.info(f"Model loaded successfully:")
    logger.info(f"  Device: {model_info['device']}")
    logger.info(f"  Classes: {model_info['num_labels']}")
    logger.info(f"  Class names: {model_info['class_names']}")

    # Run inference based on mode
    if args.interactive:
      pipeline.interactive_predict()

    elif args.text:
      predict_single_text(pipeline, args.text, args.top_k)

    elif args.file:
      predict_from_file(pipeline, args.file, args.output, args.top_k)

    else:
      print("Please specify --text, --file, or --interactive mode")
      return 1

  except Exception as e:
    logger.error(f"Inference failed: {e}")
    raise

  return 0


if __name__ == "__main__":
  sys.exit(main())
