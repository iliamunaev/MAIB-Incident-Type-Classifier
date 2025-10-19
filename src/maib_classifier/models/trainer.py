"""
Model training and evaluation for MAIB Incident Type Classifier.
"""

import os
import numpy as np
import torch
import evaluate
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
  AutoModelForSequenceClassification,
  DataCollatorWithPadding,
  TrainingArguments,
  Trainer,
  set_seed
)
from datasets import DatasetDict
from maib_classifier.utils.config import Config
from maib_classifier.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
  """Model trainer for MAIB incident classification."""

  def __init__(self, config: Config):
    """
    Initialize model trainer.

    Args:
      config: Configuration object
    """
    self.config = config
    self.model = None
    self.tokenizer = None
    self.data_collator = None
    self.trainer = None
    self.metrics = {}

  def setup_model(self, tokenizer: Any, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]) -> None:
    """
    Setup the model for training.

    Args:
      tokenizer: Tokenizer instance
      num_labels: Number of classes
      id2label: ID to label mapping
      label2id: Label to ID mapping
    """
    logger.info(f"Setting up model: {self.config.model.name}")

    self.tokenizer = tokenizer

    # Load model
    self.model = AutoModelForSequenceClassification.from_pretrained(
      self.config.model.name,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
    )

    # Enable gradient checkpointing if specified
    if self.config.model.gradient_checkpointing and torch.cuda.is_available():
      self.model.gradient_checkpointing_enable()
      logger.info("Gradient checkpointing enabled")

    # Setup data collator
    self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    logger.info("Model setup completed successfully")

  def setup_metrics(self) -> None:
    """Setup evaluation metrics."""
    logger.info("Setting up evaluation metrics...")

    # Load metrics using the correct method for evaluate library
    try:
      # Try the newer evaluate library API
      self.metrics = {
        "accuracy": evaluate.load("accuracy"),
        "macro_f1": evaluate.load("f1"),
        "weighted_f1": evaluate.load("f1")
      }
    except (AttributeError, Exception) as e:
      # Fallback: use sklearn metrics directly
      logger.warning(f"Failed to load evaluate metrics: {e}. Using sklearn fallback.")
      from sklearn.metrics import accuracy_score, f1_score

      self.metrics = {
        "accuracy": None,  # Will use sklearn accuracy_score
        "macro_f1": None,  # Will use sklearn f1_score with average='macro'
        "weighted_f1": None  # Will use sklearn f1_score with average='weighted'
      }

    logger.info(f"Metrics loaded: {list(self.metrics.keys())}")

  def compute_metrics(self, eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
      eval_pred: Evaluation predictions

    Returns:
      Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    results = {}

    # Check if we have evaluate metrics or need to use sklearn fallback
    if self.metrics["accuracy"] is not None:
      # Use evaluate library metrics
      results["accuracy"] = self.metrics["accuracy"].compute(
        predictions=predictions,
        references=labels
      )["accuracy"]

      results["macro_f1"] = self.metrics["macro_f1"].compute(
        predictions=predictions,
        references=labels,
        average="macro"
      )["f1"]

      results["weighted_f1"] = self.metrics["weighted_f1"].compute(
        predictions=predictions,
        references=labels,
        average="weighted"
      )["f1"]
    else:
      # Use sklearn metrics directly
      from sklearn.metrics import accuracy_score, f1_score

      results["accuracy"] = accuracy_score(labels, predictions)
      results["macro_f1"] = f1_score(labels, predictions, average="macro")
      results["weighted_f1"] = f1_score(labels, predictions, average="weighted")

    return results

  def setup_trainer(self, dataset: DatasetDict, eval_split: str = "validation") -> None:
    """
    Setup the Hugging Face Trainer.

    Args:
      dataset: Processed dataset
      eval_split: Evaluation split name
    """
    logger.info("Setting up trainer...")

    # Setup metrics
    self.setup_metrics()

    # Training arguments
    training_args = TrainingArguments(
      output_dir=self.config.training.output_dir,
      learning_rate=self.config.training.learning_rate,
      per_device_train_batch_size=self.config.training.per_device_train_batch_size,
      per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
      num_train_epochs=self.config.training.num_train_epochs,
      eval_strategy=self.config.training.eval_strategy,
      save_strategy=self.config.training.save_strategy,
      load_best_model_at_end=self.config.training.load_best_model_at_end,
      metric_for_best_model=self.config.training.metric_for_best_model,
      logging_steps=self.config.training.logging_steps,
      report_to=self.config.training.report_to,

      # Optimization
      fp16=self.config.training.fp16,
      bf16=self.config.training.bf16,
      optim=self.config.training.optim,
      lr_scheduler_type=self.config.training.lr_scheduler_type,
      warmup_ratio=self.config.training.warmup_ratio,
      weight_decay=self.config.training.weight_decay,

      # Data loading
      dataloader_pin_memory=self.config.training.dataloader_pin_memory,
      dataloader_num_workers=self.config.training.dataloader_num_workers,
      eval_accumulation_steps=self.config.training.eval_accumulation_steps,
      torch_compile=self.config.training.torch_compile,
    )

    # Create trainer
    self.trainer = Trainer(
      model=self.model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset[eval_split],
      processing_class=self.tokenizer,
      data_collator=self.data_collator,
      compute_metrics=self.compute_metrics,
    )

    logger.info("Trainer setup completed successfully")

  def train(self) -> Dict[str, Any]:
    """
    Train the model.

    Returns:
      Training output dictionary
    """
    if self.trainer is None:
      raise ValueError("Trainer not initialized. Call setup_trainer() first.")

    logger.info("Starting model training...")

    # Set seed for reproducibility
    set_seed(self.config.data.random_seed)

    # Train the model
    train_output = self.trainer.train()

    logger.info("Training completed successfully")
    logger.info(f"Training loss: {train_output.training_loss:.4f}")
    logger.info(f"Training runtime: {train_output.metrics['train_runtime']:.2f} seconds")

    return train_output.metrics

  def evaluate(self) -> Dict[str, float]:
    """
    Evaluate the model.

    Returns:
      Evaluation metrics dictionary
    """
    if self.trainer is None:
      raise ValueError("Trainer not initialized. Call setup_trainer() first.")

    logger.info("Evaluating model...")

    metrics = self.trainer.evaluate()

    logger.info("Evaluation completed:")
    for metric, value in metrics.items():
      if metric.startswith("eval_"):
        clean_metric = metric.replace("eval_", "")
        logger.info(f"  {clean_metric}: {value:.4f}")

    return metrics

  def save_model(self, save_path: str) -> None:
    """
    Save the trained model and tokenizer.

    Args:
      save_path: Path to save the model
    """
    if self.trainer is None:
      raise ValueError("Trainer not initialized.")

    logger.info(f"Saving model to {save_path}")

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save model and tokenizer
    self.trainer.save_model(save_path)
    self.tokenizer.save_pretrained(save_path)

    logger.info("Model saved successfully")

  def train_and_evaluate(self, dataset: DatasetDict, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline.

    Args:
      dataset: Processed dataset
      metadata: Dataset metadata

    Returns:
      Dictionary with training and evaluation results
    """
    logger.info("Starting training and evaluation pipeline...")

    # Setup model
    self.setup_model(
      tokenizer=metadata["tokenizer"],
      num_labels=metadata["num_classes"],
      id2label=metadata["label_mappings"]["id2label"],
      label2id=metadata["label_mappings"]["label2id"]
    )

    # Setup trainer
    self.setup_trainer(dataset, eval_split=self.config.evaluation.eval_split)

    # Train model
    train_results = self.train()

    # Evaluate model
    eval_results = self.evaluate()

    # Save model
    model_save_path = os.path.join(self.config.training.output_dir, "best_model")
    self.save_model(model_save_path)

    # Combine results
    results = {
      "training": train_results,
      "evaluation": eval_results,
      "model_path": model_save_path,
      "metadata": metadata
    }

    logger.info("Training and evaluation pipeline completed successfully")
    return results
