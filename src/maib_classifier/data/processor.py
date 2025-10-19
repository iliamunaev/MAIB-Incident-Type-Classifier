"""
Data loading and preprocessing for MAIB Incident Type Classifier.
"""

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from .utils.config import Config
from .utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
  """Data processor for MAIB incident reports."""

  def __init__(self, config: Config):
    """
    Initialize data processor.

    Args:
      config: Configuration object
    """
    self.config = config
    self.tokenizer = None
    self.class_names = None
    self.label_mappings = None

  def load_dataset(self) -> DatasetDict:
    """
    Load dataset from Hugging Face.

    Returns:
      DatasetDict with train/validation/test splits
    """
    logger.info("Loading dataset from Hugging Face: baker-street/maib-incident-reports-5K")

    # Load the dataset from Hugging Face
    ds = load_dataset("baker-street/maib-incident-reports-5K", split="train")

    # Split into train/test (80/20)
    ds = ds.train_test_split(
      test_size=self.config.data.test_split,
      seed=self.config.data.random_seed
    )

    # Split test into validation/test (50/50 of remaining 20%)
    tmp = ds["test"].train_test_split(
      test_size=0.5,
      seed=self.config.data.random_seed
    )

    # Recombine into final structure
    ds = {
      "train": ds["train"],
      "validation": tmp["train"],
      "test": tmp["test"]
    }

    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Train: {len(ds['train'])} samples")
    logger.info(f"  Validation: {len(ds['validation'])} samples")
    logger.info(f"  Test: {len(ds['test'])} samples")

    return ds

  def prepare_labels(self, dataset: DatasetDict) -> DatasetDict:
    """
    Prepare labels by converting to ClassLabel format.

    Args:
      dataset: DatasetDict with text and label columns

    Returns:
      DatasetDict with prepared labels
    """
    logger.info("Preparing labels...")

    # Extract unique labels and sort them
    labels_sorted = sorted(set(dataset["train"]["label"]))
    self.class_names = labels_sorted

    logger.info(f"Found {len(labels_sorted)} unique classes:")
    for i, label in enumerate(labels_sorted):
      logger.info(f"  {i}: {label}")

    # Convert label column to ClassLabel
    for split in ["train", "validation", "test"]:
      dataset[split] = dataset[split].cast_column("label", ClassLabel(names=labels_sorted))

    # Create labels column for Trainer compatibility
    def to_labels(batch):
      return {"labels": batch["label"]}

    for split in dataset:
      dataset[split] = dataset[split].map(
        to_labels,
        batched=True,
        num_proc=self.config.data.num_proc
      )

    # Create label mappings
    self.label_mappings = self._create_label_mappings(labels_sorted)

    return dataset

  def setup_tokenizer(self) -> None:
    """Setup tokenizer for the model."""
    logger.info(f"Setting up tokenizer: {self.config.model.name}")

    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
    self.tokenizer.model_max_length = self.config.model.max_length

    logger.info(f"Tokenizer max length: {self.tokenizer.model_max_length}")

  def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
    """
    Tokenize the dataset.

    Args:
      dataset: DatasetDict with text column

    Returns:
      DatasetDict with tokenized inputs
    """
    if self.tokenizer is None:
      raise ValueError("Tokenizer not initialized. Call setup_tokenizer() first.")

    logger.info("Tokenizing dataset...")

    def tokenize(batch):
      return self.tokenizer(
        batch["text"],
        truncation=True,
        max_length=self.tokenizer.model_max_length,
      )

    # Tokenize each split
    for split in dataset:
      cols_to_remove = [c for c in ("text",) if c in dataset[split].column_names]
      dataset[split] = dataset[split].map(
        tokenize,
        batched=True,
        remove_columns=cols_to_remove,
        desc=f"Tokenizing {split}",
        num_proc=self.config.data.num_proc
      )

    logger.info("Dataset tokenized successfully")
    return dataset

  def prepare_for_training(self, dataset: DatasetDict) -> DatasetDict:
    """
    Prepare dataset for training by setting torch format.

    Args:
      dataset: Tokenized DatasetDict

    Returns:
      DatasetDict ready for training
    """
    logger.info("Preparing dataset for training...")

    # Choose columns for PyTorch tensors
    torch_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
      torch_cols.append("token_type_ids")

    # Set format for each split
    for split in dataset:
      dataset[split].set_format(type="torch", columns=torch_cols)

    logger.info(f"Dataset prepared with columns: {torch_cols}")
    return dataset

  def process_data(self) -> Tuple[DatasetDict, Dict[str, any]]:
    """
    Complete data processing pipeline.

    Returns:
      Tuple of (processed_dataset, metadata)
    """
    logger.info("Starting data processing pipeline...")

    # Load dataset
    dataset = self.load_dataset()

    # Prepare labels
    dataset = self.prepare_labels(dataset)

    # Setup tokenizer
    self.setup_tokenizer()

    # Tokenize dataset
    dataset = self.tokenize_dataset(dataset)

    # Prepare for training
    dataset = self.prepare_for_training(dataset)

    # Create metadata
    metadata = {
      "class_names": self.class_names,
      "label_mappings": self.label_mappings,
      "num_classes": len(self.class_names),
      "tokenizer": self.tokenizer
    }

    logger.info("Data processing pipeline completed successfully")
    return dataset, metadata

  def _create_label_mappings(self, class_names: List[str]) -> Dict[str, Dict[int, str]]:
    """Create label mappings from class names."""
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}

    return {
      "id2label": id2label,
      "label2id": label2id
    }
