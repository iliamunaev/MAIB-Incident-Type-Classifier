# MAIB Incident Type Classifier - API Documentation

## Overview

This document provides detailed API documentation for the MAIB Incident Type Classifier system.

## Core Modules

### Configuration Management

#### `maib_classifier.utils.config.Config`

Main configuration class for the system.

```python
from maib_classifier.utils.config import Config

# Load from YAML file
config = Config.from_yaml("configs/config.yaml")

# Create from dictionary
config = Config.from_dict(config_dict)

# Save configuration
config.save_yaml("output_config.yaml")

# Setup directories and device detection
config.setup_directories()
config.setup_data_config()
config.setup_inference_config()
```

**Configuration Classes:**

- `ModelConfig`: Model parameters (name, max_length, num_labels, etc.)
- `TrainingConfig`: Training parameters (learning_rate, batch_size, epochs, etc.)
- `DataConfig`: Data processing parameters (splits, seed, num_proc, etc.)
- `EvaluationConfig`: Evaluation parameters (metrics, eval_split, etc.)
- `InferenceConfig`: Inference parameters (top_k, device, etc.)
- `LoggingConfig`: Logging parameters (level, format, file, etc.)

### Data Processing

#### `maib_classifier.data.processor.DataProcessor`

Handles data loading, preprocessing, and tokenization.

```python
from maib_classifier.data.processor import DataProcessor

processor = DataProcessor(config)

# Complete data processing pipeline
dataset, metadata = processor.process_data("data/maib-data.jsonl")

# Individual steps
dataset = processor.load_dataset("data/maib-data.jsonl")
dataset = processor.prepare_labels(dataset)
processor.setup_tokenizer()
dataset = processor.tokenize_dataset(dataset)
dataset = processor.prepare_for_training(dataset)
```

**Methods:**

- `load_dataset(data_path)`: Load JSONL dataset and create splits
- `prepare_labels(dataset)`: Convert labels to ClassLabel format
- `setup_tokenizer()`: Initialize tokenizer
- `tokenize_dataset(dataset)`: Tokenize text data
- `prepare_for_training(dataset)`: Set PyTorch format
- `process_data(data_path)`: Complete processing pipeline

### Model Training

#### `maib_classifier.models.trainer.ModelTrainer`

Handles model training and evaluation.

```python
from maib_classifier.models.trainer import ModelTrainer

trainer = ModelTrainer(config)

# Setup model
trainer.setup_model(tokenizer, num_labels, id2label, label2id)

# Setup trainer
trainer.setup_trainer(dataset, eval_split="validation")

# Train model
train_results = trainer.train()

# Evaluate model
eval_results = trainer.evaluate()

# Save model
trainer.save_model("outputs/best_model")

# Complete pipeline
results = trainer.train_and_evaluate(dataset, metadata)
```

**Methods:**

- `setup_model(tokenizer, num_labels, id2label, label2id)`: Initialize model
- `setup_metrics()`: Setup evaluation metrics
- `compute_metrics(eval_pred)`: Compute metrics from predictions
- `setup_trainer(dataset, eval_split)`: Setup Hugging Face Trainer
- `train()`: Train the model
- `evaluate()`: Evaluate the model
- `save_model(save_path)`: Save model and tokenizer
- `train_and_evaluate(dataset, metadata)`: Complete training pipeline

### Model Evaluation

#### `maib_classifier.models.evaluator.ModelEvaluator`

Provides comprehensive model evaluation with visualizations.

```python
from maib_classifier.models.evaluator import ModelEvaluator

evaluator = ModelEvaluator(id2label)

# Generate confusion matrix
cm = evaluator.generate_confusion_matrix(
    y_true, y_pred,
    save_path="outputs/confusion_matrix.png"
)

# Generate per-class F1 scores
f1_scores = evaluator.generate_per_class_f1(
    y_true, y_pred,
    save_path="outputs/per_class_f1.png"
)

# Generate classification report
report = evaluator.generate_classification_report(
    y_true, y_pred,
    save_path="outputs/classification_report.txt"
)

# Complete evaluation
results = evaluator.evaluate_model(y_true, y_pred, "outputs")
```

**Methods:**

- `generate_confusion_matrix(y_true, y_pred, save_path, normalize, figsize)`: Create confusion matrix
- `generate_per_class_f1(y_true, y_pred, save_path, figsize)`: Create F1 score visualization
- `generate_classification_report(y_true, y_pred, save_path)`: Generate detailed report
- `evaluate_model(y_true, y_pred, output_dir)`: Complete evaluation pipeline

### Inference and Prediction

#### `maib_classifier.inference.predictor.MAIBPredictor`

Handles model inference and predictions.

```python
from maib_classifier.inference.predictor import MAIBPredictor

predictor = MAIBPredictor(config, model_path="outputs/best_model")

# Load model
predictor.load_model("outputs/best_model")

# Single prediction
predictions = predictor.predict_single("Incident description text")

# Batch prediction
predictions = predictor.predict_batch(["text1", "text2", "text3"])

# Predict from file
results = predictor.predict_from_file("input.txt", "output.json")

# Get model information
info = predictor.get_model_info()

# Print formatted prediction
predictor.print_prediction("Incident description text")
```

**Methods:**

- `load_model(model_path)`: Load trained model and tokenizer
- `predict_single(text, top_k, max_length)`: Predict for single text
- `predict_batch(texts, top_k, max_length)`: Predict for multiple texts
- `predict_from_file(file_path, output_path, top_k)`: Predict from file
- `get_model_info()`: Get model information
- `print_prediction(text, top_k)`: Print formatted results

#### `maib_classifier.inference.predictor.MAIBInferencePipeline`

High-level inference pipeline.

```python
from maib_classifier.inference.predictor import MAIBInferencePipeline

pipeline = MAIBInferencePipeline(config)

# Load model
pipeline.load_model("outputs/best_model")

# Predict
predictions = pipeline.predict("Incident description text")

# Batch predict
predictions = pipeline.predict_batch(["text1", "text2"])

# Interactive mode
pipeline.predictor.interactive_predict()
```

### Utilities

#### `maib_classifier.utils.logger`

Logging utilities.

```python
from maib_classifier.utils.logger import setup_logger, get_logger

# Setup logger
logger = setup_logger(
    name="maib_classifier",
    level="INFO",
    log_file="logs/training.log"
)

# Get existing logger
logger = get_logger("maib_classifier")
```

#### `maib_classifier.utils.utils`

General utility functions.

```python
from maib_classifier.utils.utils import (
    set_seed, get_device, get_device_info,
    print_device_info, ensure_dir,
    get_class_names_from_labels, create_label_mappings
)

# Set random seed
set_seed(42)

# Get device information
device = get_device()
info = get_device_info()
print_device_info()

# Directory operations
ensure_dir("outputs")

# Label operations
class_names = get_class_names_from_labels(labels)
mappings = create_label_mappings(class_names)
```

## Command Line Interface

### Training Script

```bash
python scripts/train.py [OPTIONS]

Options:
  --data_path PATH        Path to JSONL data file (required)
  --config PATH           Path to configuration file
  --output_dir PATH       Output directory for model and results
  --model_name STR        Model name to use for training
  --epochs INT            Number of training epochs
  --batch_size INT        Training batch size
  --learning_rate FLOAT   Learning rate
  --seed INT              Random seed for reproducibility
  --eval_only             Only evaluate existing model
  --model_path PATH       Path to existing model for evaluation
  --verbose               Enable verbose logging
```

### Inference Script

```bash
python scripts/inference.py [OPTIONS]

Options:
  --model_path PATH       Path to trained model (required)
  --config PATH           Path to configuration file
  --text STR              Single text to classify
  --file PATH             File containing texts to classify
  --output PATH           Output file for predictions
  --top_k INT             Number of top predictions to return
  --interactive           Interactive prediction mode
  --verbose               Enable verbose logging
```

### Evaluation Script

```bash
python scripts/evaluate.py [OPTIONS]

Options:
  --model_path PATH       Path to trained model (required)
  --data_path PATH        Path to test data JSONL file (required)
  --config PATH           Path to configuration file
  --output_dir PATH       Output directory for evaluation results
  --verbose               Enable verbose logging
```

## Configuration Reference

### Model Configuration

```yaml
model:
  name: "microsoft/deberta-v3-base"  # Model name
  max_length: 256                     # Maximum sequence length
  num_labels: 11                      # Number of classes
  gradient_checkpointing: true       # Enable gradient checkpointing
```

### Training Configuration

```yaml
training:
  output_dir: "outputs"              # Output directory
  learning_rate: 2e-5                # Learning rate
  per_device_train_batch_size: 32     # Training batch size
  per_device_eval_batch_size: 64      # Evaluation batch size
  num_train_epochs: 3                # Number of epochs
  eval_strategy: "epoch"             # Evaluation strategy
  save_strategy: "epoch"              # Save strategy
  load_best_model_at_end: true       # Load best model
  metric_for_best_model: "macro_f1"  # Best model metric
  logging_steps: 50                   # Logging frequency
  report_to: "none"                   # Reporting destination

  # Optimization
  fp16: true                          # Mixed precision training
  bf16: false                         # BF16 precision
  optim: "adamw_torch"                # Optimizer
  lr_scheduler_type: "linear"        # Learning rate scheduler
  warmup_ratio: 0.1                   # Warmup ratio
  weight_decay: 0.01                  # Weight decay

  # Data Loading
  dataloader_pin_memory: true         # Pin memory
  dataloader_num_workers: 2           # Number of workers
  eval_accumulation_steps: 4          # Evaluation accumulation
  torch_compile: false                # Torch compilation
```

### Data Configuration

```yaml
data:
  train_split: 0.8                    # Training split ratio
  validation_split: 0.1               # Validation split ratio
  test_split: 0.1                     # Test split ratio
  random_seed: 42                     # Random seed
  num_proc: null                      # Number of processes (auto-detected)
```

### Evaluation Configuration

```yaml
evaluation:
  metrics: ["accuracy", "macro_f1", "weighted_f1"]  # Metrics to compute
  eval_split: "validation"            # Evaluation split
```

### Inference Configuration

```yaml
inference:
  top_k: 3                            # Number of top predictions
  max_length: 256                     # Maximum sequence length
  device: "auto"                      # Device (auto, cuda, cpu)
```

### Logging Configuration

```yaml
logging:
  level: "INFO"                       # Log level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Format
  file: "logs/training.log"           # Log file
```

## Error Handling

The system includes comprehensive error handling:

- **Configuration errors**: Invalid YAML, missing files, invalid parameters
- **Data errors**: Missing files, invalid formats, processing failures
- **Model errors**: Loading failures, device issues, memory problems
- **Training errors**: CUDA issues, convergence problems, resource limits
- **Inference errors**: Model not loaded, invalid inputs, device mismatches

## Performance Considerations

### Memory Usage

- **Training**: ~8-12GB GPU memory (Tesla T4)
- **Inference**: ~2-4GB GPU memory
- **CPU**: ~4-8GB RAM

### Speed Optimization

- Use mixed precision training (`fp16: true`)
- Enable gradient checkpointing for memory efficiency
- Adjust batch size based on available memory
- Use multiple workers for data loading
- Consider model compilation for inference speedup

### Scaling

- For larger datasets: Increase batch size, use gradient accumulation
- For faster training: Use multiple GPUs with DataParallel
- For production: Use model quantization and optimization
