"""
Model evaluation and visualization utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional, Tuple
from maib_classifier.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
  """Model evaluator with visualization capabilities."""

  def __init__(self, id2label: Dict[int, str]):
    """
    Initialize model evaluator.

    Args:
      id2label: ID to label mapping
    """
    self.id2label = id2label
    self.num_classes = len(id2label)

  def generate_confusion_matrix(
    self,
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 8)
  ) -> np.ndarray:
    """
    Generate and optionally save confusion matrix.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      save_path: Optional path to save the plot
      normalize: Whether to normalize the matrix
      figsize: Figure size

    Returns:
      Confusion matrix array
    """
    logger.info("Generating confusion matrix...")

    # Compute confusion matrix
    cm = confusion_matrix(
      y_true, y_pred,
      labels=list(range(self.num_classes)),
      normalize="true" if normalize else None
    )

    # Create plot
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Set ticks and labels
    plt.xticks(
      ticks=np.arange(self.num_classes),
      labels=[self.id2label[i] for i in range(self.num_classes)],
      rotation=90
    )
    plt.yticks(
      ticks=np.arange(self.num_classes),
      labels=[self.id2label[i] for i in range(self.num_classes)]
    )

    # Add colorbar
    plt.colorbar()

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(self.num_classes):
      for j in range(self.num_classes):
        plt.text(
          j, i, f"{cm[i, j]:.2f}",
          ha="center", va="center",
          color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      plt.savefig(save_path, dpi=150, bbox_inches="tight")
      logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    return cm

  def generate_per_class_f1(
    self,
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
  ) -> List[float]:
    """
    Generate per-class F1 score visualization.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      save_path: Optional path to save the plot
      figsize: Figure size

    Returns:
      List of per-class F1 scores
    """
    logger.info("Generating per-class F1 scores...")

    # Generate classification report
    report = classification_report(
      y_true, y_pred,
      labels=list(range(self.num_classes)),
      target_names=[self.id2label[i] for i in range(self.num_classes)],
      output_dict=True,
      zero_division=0
    )

    # Extract F1 scores
    per_class_f1 = [report[self.id2label[i]]["f1-score"] for i in range(self.num_classes)]

    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(self.num_classes), per_class_f1, color="skyblue", alpha=0.7)

    # Add value labels on bars
    for bar, f1 in zip(bars, per_class_f1):
      plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{f1:.3f}",
        ha="center", va="bottom"
      )

    plt.xticks(
      range(self.num_classes),
      [self.id2label[i] for i in range(self.num_classes)],
      rotation=90
    )
    plt.ylabel("F1-score")
    plt.title("Per-class F1 Scores")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save if path provided
    if save_path:
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      plt.savefig(save_path, dpi=150, bbox_inches="tight")
      logger.info(f"Per-class F1 plot saved to {save_path}")

    plt.show()
    return per_class_f1

  def generate_classification_report(
    self,
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None
  ) -> Dict[str, Any]:
    """
    Generate detailed classification report.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      save_path: Optional path to save the report

    Returns:
      Classification report dictionary
    """
    logger.info("Generating classification report...")

    report = classification_report(
      y_true, y_pred,
      labels=list(range(self.num_classes)),
      target_names=[self.id2label[i] for i in range(self.num_classes)],
      output_dict=True,
      zero_division=0
    )

    # Save if path provided
    if save_path:
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      with open(save_path, 'w') as f:
        f.write(classification_report(
          y_true, y_pred,
          labels=list(range(self.num_classes)),
          target_names=[self.id2label[i] for i in range(self.num_classes)],
          zero_division=0
        ))
      logger.info(f"Classification report saved to {save_path}")

    return report

  def evaluate_model(
    self,
    y_true: List[int],
    y_pred: List[int],
    output_dir: str = "outputs"
  ) -> Dict[str, Any]:
    """
    Complete model evaluation with visualizations.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      output_dir: Output directory for saving plots

    Returns:
      Dictionary with evaluation results
    """
    logger.info("Starting comprehensive model evaluation...")

    # Generate confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm = self.generate_confusion_matrix(y_true, y_pred, save_path=cm_path)

    # Generate per-class F1 scores
    f1_path = os.path.join(output_dir, "per_class_f1.png")
    per_class_f1 = self.generate_per_class_f1(y_true, y_pred, save_path=f1_path)

    # Generate classification report
    report_path = os.path.join(output_dir, "classification_report.txt")
    report = self.generate_classification_report(y_true, y_pred, save_path=report_path)

    # Compile results
    results = {
      "confusion_matrix": cm,
      "per_class_f1": per_class_f1,
      "classification_report": report,
      "plots_saved": {
        "confusion_matrix": cm_path,
        "per_class_f1": f1_path,
        "classification_report": report_path
      }
    }

    logger.info("Model evaluation completed successfully")
    return results
