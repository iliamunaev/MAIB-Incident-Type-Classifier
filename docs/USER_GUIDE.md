# MAIB Incident Type Classifier - User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Inference](#inference)
6. [Evaluation](#evaluation)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Install the package
pip install -e .

# 2. Ready to use - data automatically downloaded from Hugging Face
# No data preparation required

# 3. Train the model
python scripts/train.py

# 4. Run inference
python scripts/inference.py --model_path outputs/best_model --interactive
```

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 5GB+ free space
- **GPU**: CUDA-compatible GPU (recommended for training)

### Installation Methods

#### Method 1: Direct Installation

```bash
git clone <repository-url>
cd MAIB-Incident-Type-Classifier
pip install -e .
```

#### Method 2: Development Installation

```bash
git clone <repository-url>
cd MAIB-Incident-Type-Classifier
pip install -e ".[dev,jupyter]"
```

#### Method 3: Docker Installation

```bash
git clone <repository-url>
cd MAIB-Incident-Type-Classifier
docker build -t maib-classifier .
```

### Verify Installation

```bash
python -c "import maib_classifier; print('Installation successful!')"
```

## Data Source

### Hugging Face Dataset

The system automatically loads data from the `baker-street/maib-incident-reports-5K` dataset on Hugging Face. This dataset contains MAIB incident reports in the following format:

```json
{"text": "Incident description text", "label": "Accident to person(s)"}
{"text": "Another incident description", "label": "Collision"}
{"text": "Third incident description", "label": "Fire / Explosion"}
```

### Dataset Fields

- **text**: The incident description (string)
- **label**: The incident type (string, matches predefined classes)

### Supported Classes

The system supports 11 incident types:

1. Accident to person(s)
2. Capsizing / Listing
3. Collision
4. Contact
5. Damage / Loss Of Equipment
6. Fire / Explosion
7. Flooding / Foundering
8. Grounding / Stranding
9. Hull Failure
10. Loss Of Control
11. Non-accidental Event

### Data Quality Guidelines

- **Text length**: 50-500 characters recommended
- **Language**: English only
- **Format**: Plain text, avoid special formatting
- **Balance**: Aim for balanced class distribution
- **Quality**: Ensure accurate labels

### Dataset Information

The `baker-street/maib-incident-reports-5K` dataset includes:

- **Size**: ~5,000 incident reports
- **Source**: Marine Accident Investigation Branch (MAIB)
- **Format**: Automatically processed by the system
- **Caching**: Downloaded once and cached locally by Hugging Face
- **Updates**: Automatically gets latest version when available

## Training

### Basic Training

```bash
python scripts/train.py
```

### Advanced Training Options

```bash
python scripts/train.py \
  --output_dir my_outputs \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --model_name microsoft/deberta-v3-large \
  --seed 123
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `outputs` | Output directory |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--model_name` | `microsoft/deberta-v3-base` | Model to use |
| `--seed` | `42` | Random seed |

### Training Output

Training creates the following outputs:

```
outputs/
├── best_model/           # Best model checkpoint
│   ├── config.json      # Model configuration
│   ├── pytorch_model.bin # Model weights
│   ├── tokenizer.json   # Tokenizer
│   └── tokenizer_config.json
├── training_args.bin    # Training arguments
├── trainer_state.json   # Training state
└── logs/                # Training logs
```

### Monitoring Training

Training progress is logged to:
- Console output
- `logs/training.log` file
- TensorBoard (if enabled)

Key metrics to monitor:
- Training loss (should decrease)
- Validation accuracy (should increase)
- Validation F1-score (should increase)

## Inference

### Interactive Mode

```bash
python scripts/inference.py --model_path outputs/best_model --interactive
```

### Single Text Prediction

```bash
python scripts/inference.py \
  --model_path outputs/best_model \
  --text "Crew member injured during maintenance work"
```

### Batch Prediction from File

```bash
python scripts/inference.py \
  --model_path outputs/best_model \
  --file input_texts.txt \
  --output predictions.json \
  --top_k 5
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Required | Path to trained model |
| `--text` | - | Single text to classify |
| `--file` | - | File with texts to classify |
| `--output` | - | Output file for predictions |
| `--top_k` | `3` | Number of top predictions |
| `--interactive` | False | Interactive mode |

### Output Format

Predictions are returned as:

```json
[
  {
    "index": 0,
    "text": "Input text",
    "predictions": [
      ["Accident to person(s)", 85.2],
      ["Collision", 8.1],
      ["Fire / Explosion", 3.2]
    ],
    "top_prediction": ["Accident to person(s)", 85.2]
  }
]
```

## Evaluation

### Basic Evaluation

```bash
python scripts/evaluate.py \
  --model_path outputs/best_model
```

### Comprehensive Evaluation

```bash
python scripts/evaluate.py \
  --model_path outputs/best_model \
  --output_dir evaluation_results
```

### Evaluation Outputs

Evaluation creates:

```
evaluation_results/
├── confusion_matrix.png      # Confusion matrix visualization
├── per_class_f1.png         # Per-class F1 scores
├── classification_report.txt # Detailed metrics
└── evaluation.log           # Evaluation logs
```

### Key Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1-score across all classes
- **Weighted F1**: F1-score weighted by class frequency
- **Per-class F1**: F1-score for each individual class

## Configuration

### Configuration File

Create or modify `configs/config.yaml`:

```yaml
# Model Configuration
model:
  name: "microsoft/deberta-v3-base"
  max_length: 256
  num_labels: 11

# Training Configuration
training:
  learning_rate: 2e-5
  per_device_train_batch_size: 32
  num_train_epochs: 3
  output_dir: "outputs"

# Data Configuration
data:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  random_seed: 42
```

### Environment Variables

Set environment variables for configuration:

```bash
export MAIB_MODEL_NAME="microsoft/deberta-v3-base"
export MAIB_LEARNING_RATE="2e-5"
export MAIB_BATCH_SIZE="32"
export MAIB_OUTPUT_DIR="outputs"
```

### Programmatic Configuration

```python
from maib_classifier.utils.config import Config

# Create custom configuration
config = Config()
config.model.name = "microsoft/deberta-v3-large"
config.training.learning_rate = 1e-5
config.training.num_train_epochs = 5

# Save configuration
config.save_yaml("custom_config.yaml")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch_size 16`
- Enable gradient checkpointing in config
- Use mixed precision: `fp16: true`
- Close other GPU applications

#### 2. Data Loading Errors

**Error**: `FileNotFoundError: Data file not found`

**Solutions**:
- Check internet connection for Hugging Face dataset download
- Verify Hugging Face authentication if required
- Check available disk space for dataset caching

#### 3. Model Loading Errors

**Error**: `OSError: Model not found`

**Solutions**:
- Verify model path: `--model_path outputs/best_model`
- Check if training completed successfully
- Ensure model files exist

#### 4. Poor Performance

**Symptoms**: Low accuracy, high loss

**Solutions**:
- Increase training epochs
- Adjust learning rate
- Check data quality
- Verify label consistency
- Increase model size

### Debug Mode

Enable verbose logging:

```bash
python scripts/train.py --verbose
```

### Log Analysis

Check training logs:

```bash
tail -f logs/training.log
```

## Best Practices

### Data Preparation

1. **Quality over Quantity**: Focus on high-quality, accurately labeled data
2. **Balance Classes**: Ensure reasonable class distribution
3. **Text Length**: Keep descriptions concise but informative
4. **Consistency**: Use consistent terminology and formatting

### Training

1. **Start Small**: Begin with smaller models and datasets
2. **Monitor Progress**: Watch training metrics closely
3. **Save Checkpoints**: Regular model saving for recovery
4. **Validation**: Use validation set for model selection

### Model Selection

1. **DeBERTa-v3-base**: Good balance of performance and speed
2. **DeBERTa-v3-large**: Higher accuracy, slower training
3. **Custom Models**: Consider domain-specific models

### Production Deployment

1. **Model Optimization**: Use quantization for faster inference
2. **Batch Processing**: Process multiple texts together
3. **Error Handling**: Implement robust error handling
4. **Monitoring**: Track model performance over time

### Performance Optimization

1. **GPU Usage**: Use CUDA when available
2. **Batch Size**: Optimize for your hardware
3. **Mixed Precision**: Enable FP16 for faster training
4. **Data Loading**: Use multiple workers for data loading

### Maintenance

1. **Regular Retraining**: Retrain with new data periodically
2. **Model Versioning**: Keep track of model versions
3. **Performance Monitoring**: Monitor accuracy over time
4. **Data Updates**: Keep training data current
