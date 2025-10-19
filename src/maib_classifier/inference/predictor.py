"""
Inference and prediction for MAIB Incident Type Classifier.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from maib_classifier.utils.config import Config
from maib_classifier.utils.logger import get_logger

logger = get_logger(__name__)


class MAIBPredictor:
  """MAIB incident type predictor."""

  def __init__(self, config: Config, model_path: Optional[str] = None):
    """
    Initialize predictor.

    Args:
      config: Configuration object
      model_path: Path to trained model (optional)
    """
    self.config = config
    self.model_path = model_path
    self.tokenizer = None
    self.model = None
    self.device = None
    self.id2label = None
    self.num_labels = None
    self._is_loaded = False

  def load_model(self, model_path: Optional[str] = None) -> None:
    """
    Load the trained model and tokenizer.

    Args:
      model_path: Path to trained model (optional)
    """
    if model_path is None:
      model_path = self.model_path

    if model_path is None:
      raise ValueError("Model path must be provided either in constructor or load_model()")

    logger.info(f"Loading model from {model_path}")

    # Setup device
    self.device = self.config.inference.device
    if self.device == "auto":
      self.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {self.device}")

    # Load tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    self.model.to(self.device)
    self.model.eval()

    # Get label mappings
    self.id2label = getattr(self.model.config, "id2label", None)
    if not self.id2label:
      # Fallback if config missing maps
      self.num_labels = self.model.config.num_labels
      self.id2label = {i: str(i) for i in range(self.num_labels)}
    else:
      self.num_labels = len(self.id2label)

    self._is_loaded = True
    logger.info(f"Model loaded successfully with {self.num_labels} classes")

  def predict_single(
    self,
    text: str,
    top_k: Optional[int] = None,
    max_length: Optional[int] = None
  ) -> List[Tuple[str, float]]:
    """
    Predict incident type for a single text.

    Args:
      text: Input text
      top_k: Number of top predictions to return
      max_length: Maximum sequence length

    Returns:
      List of (label, confidence) tuples
    """
    if not self._is_loaded:
      raise ValueError("Model not loaded. Call load_model() first.")

    top_k = top_k or self.config.inference.top_k
    max_length = max_length or self.config.inference.max_length

    # Ensure top_k is valid
    top_k = max(1, min(top_k, self.num_labels))

    # Tokenize input
    inputs = self.tokenizer(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_length
    ).to(self.device)

    # Predict
    with torch.inference_mode():
      with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]

    # Get top predictions
    top_probs, top_indices = torch.topk(probs, k=top_k)

    # Format results
    results = [
      (self.id2label[int(idx)], float(prob) * 100.0)
      for prob, idx in zip(top_probs, top_indices)
    ]

    return results

  def predict_batch(
    self,
    texts: List[str],
    top_k: Optional[int] = None,
    max_length: Optional[int] = None
  ) -> List[List[Tuple[str, float]]]:
    """
    Predict incident types for a batch of texts.

    Args:
      texts: List of input texts
      top_k: Number of top predictions to return
      max_length: Maximum sequence length

    Returns:
      List of prediction results for each text
    """
    if not self._is_loaded:
      raise ValueError("Model not loaded. Call load_model() first.")

    top_k = top_k or self.config.inference.top_k
    max_length = max_length or self.config.inference.max_length

    # Ensure top_k is valid
    top_k = max(1, min(top_k, self.num_labels))

    # Tokenize inputs
    inputs = self.tokenizer(
      texts,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_length
    ).to(self.device)

    # Predict
    with torch.inference_mode():
      with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    # Get top predictions for each text
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    # Format results
    results = []
    for probs_row, indices_row in zip(top_probs, top_indices):
      text_results = [
        (self.id2label[int(idx)], float(prob) * 100.0)
        for prob, idx in zip(probs_row, indices_row)
      ]
      results.append(text_results)

    return results

  def predict_from_file(
    self,
    file_path: str,
    output_path: Optional[str] = None,
    top_k: Optional[int] = None
  ) -> List[Dict[str, Any]]:
    """
    Predict incident types from a file containing texts.

    Args:
      file_path: Path to input file
      output_path: Optional path to save results
      top_k: Number of top predictions to return

    Returns:
      List of prediction results
    """
    logger.info(f"Predicting from file: {file_path}")

    # Read texts from file
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line:
          texts.append(line)

    logger.info(f"Loaded {len(texts)} texts from file")

    # Predict
    predictions = self.predict_batch(texts, top_k=top_k)

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

    # Save results if output path provided
    if output_path:
      self._save_predictions(results, output_path)

    return results

  def _save_predictions(self, results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save predictions to file.

    Args:
      results: Prediction results
      output_path: Output file path
    """
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Predictions saved to {output_path}")

  def get_model_info(self) -> Dict[str, Any]:
    """
    Get model information.

    Returns:
      Dictionary with model information
    """
    if not self._is_loaded:
      raise ValueError("Model not loaded. Call load_model() first.")

    return {
      "model_path": self.model_path,
      "device": self.device,
      "num_labels": self.num_labels,
      "class_names": list(self.id2label.values()),
      "id2label": self.id2label,
      "tokenizer_max_length": self.tokenizer.model_max_length
    }

  def print_prediction(self, text: str, top_k: Optional[int] = None) -> None:
    """
    Print formatted prediction results.

    Args:
      text: Input text
      top_k: Number of top predictions to show
    """
    predictions = self.predict_single(text, top_k=top_k)

    print(f"Input: {text}\n")
    for label, confidence in predictions:
      print(f"{label:25s} {confidence:5.2f}%")


class MAIBInferencePipeline:
  """Complete inference pipeline for MAIB incident classification."""

  def __init__(self, config: Config):
    """
    Initialize inference pipeline.

    Args:
      config: Configuration object
    """
    self.config = config
    self.predictor = MAIBPredictor(config)

  def load_model(self, model_path: str) -> None:
    """
    Load the trained model.

    Args:
      model_path: Path to trained model
    """
    self.predictor.load_model(model_path)

  def predict(self, text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Predict incident type for text.

    Args:
      text: Input text
      top_k: Number of top predictions

    Returns:
      List of (label, confidence) tuples
    """
    return self.predictor.predict_single(text, top_k=top_k)

  def predict_batch(self, texts: List[str], top_k: Optional[int] = None) -> List[List[Tuple[str, float]]]:
    """
    Predict incident types for multiple texts.

    Args:
      texts: List of input texts
      top_k: Number of top predictions

    Returns:
      List of prediction results
    """
    return self.predictor.predict_batch(texts, top_k=top_k)

  def interactive_predict(self) -> None:
    """Interactive prediction mode."""
    print("MAIB Incident Type Classifier - Interactive Mode")
    print("Type 'quit' to exit\n")

    while True:
      try:
        text = input("Enter incident description: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
          break

        if not text:
          continue

        self.predictor.print_prediction(text)
        print()

      except KeyboardInterrupt:
        break
      except Exception as e:
        print(f"Error: {e}")
        print()

    print("Goodbye!")
