# MAIB Incident Type Classifier - Development Guide

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Code Standards](#code-standards)
4. [Testing](#testing)
5. [Contributing](#contributing)
6. [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-compatible GPU (recommended)
- Docker (optional)

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MAIB-Incident-Type-Classifier/new
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,jupyter]"
   ```

4. **Setup pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Tools

- **Code Formatting**: Black
- **Linting**: Flake8
- **Type Checking**: MyPy
- **Testing**: Pytest
- **Documentation**: Sphinx (optional)

## Project Structure

```
new/
├── src/maib_classifier/          # Main package
│   ├── __init__.py
│   ├── data/                     # Data processing module
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── models/                   # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── inference/                 # Inference and prediction
│   │   ├── __init__.py
│   │   └── predictor.py
│   └── utils/                    # Utilities and configuration
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── utils.py
├── scripts/                      # Command-line scripts
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── configs/                      # Configuration files
│   └── config.yaml
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   ├── test_inference/
│   └── test_utils/
├── docs/                         # Documentation
│   ├── API.md
│   └── USER_GUIDE.md
├── outputs/                      # Training outputs
├── logs/                         # Log files
├── data/                         # Data directory
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Modern Python packaging
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose
├── Makefile                      # Build automation
└── README.md                     # Main documentation
```

### Module Responsibilities

#### `data/` Module
- Data loading and preprocessing
- Dataset splitting and formatting
- Tokenization and encoding
- Data validation

#### `models/` Module
- Model training and evaluation
- Metrics computation
- Visualization generation
- Model saving and loading

#### `inference/` Module
- Model inference and prediction
- Batch processing
- Interactive prediction
- Output formatting

#### `utils/` Module
- Configuration management
- Logging setup
- Utility functions
- Device management

## Code Standards

### Python Style Guide

Follow PEP 8 with these modifications:

- **Line length**: 100 characters
- **Indentation**: 2 spaces
- **Imports**: Grouped and sorted
- **Docstrings**: Google style

### Code Formatting

```bash
# Format code
make format

# Check formatting
black --check src/ scripts/
```

### Linting

```bash
# Run linting
make lint

# Check specific files
flake8 src/maib_classifier/data/processor.py
```

### Type Hints

Use type hints for all functions:

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    data_path: str,
    config: Config,
    output_dir: Optional[str] = None
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Process data with type hints."""
    pass
```

### Documentation

#### Function Docstrings

```python
def train_model(
    dataset: DatasetDict,
    config: Config,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Train the MAIB incident classification model.

    Args:
        dataset: Processed dataset with train/validation/test splits
        config: Configuration object with training parameters
        output_dir: Directory to save model outputs

    Returns:
        Dictionary containing training results and metrics

    Raises:
        ValueError: If dataset is invalid
        RuntimeError: If training fails

    Example:
        >>> config = Config()
        >>> dataset = load_dataset("data.jsonl")
        >>> results = train_model(dataset, config)
        >>> print(results["accuracy"])
    """
```

#### Class Docstrings

```python
class DataProcessor:
    """
    Data processor for MAIB incident reports.

    This class handles the complete data processing pipeline including
    loading, preprocessing, tokenization, and formatting for training.

    Attributes:
        config: Configuration object
        tokenizer: Tokenizer instance
        class_names: List of class names

    Example:
        >>> processor = DataProcessor(config)
        >>> dataset, metadata = processor.process_data("data.jsonl")
    """
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── test_data/
│   ├── __init__.py
│   └── test_processor.py
├── test_models/
│   ├── __init__.py
│   ├── test_trainer.py
│   └── test_evaluator.py
├── test_inference/
│   ├── __init__.py
│   └── test_predictor.py
├── test_utils/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_logger.py
│   └── test_utils.py
└── fixtures/
    ├── sample_data.jsonl
    └── test_config.yaml
```

### Writing Tests

#### Unit Tests

```python
import pytest
from maib_classifier.data.processor import DataProcessor
from maib_classifier.utils.config import Config

class TestDataProcessor:
    """Test cases for DataProcessor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def processor(self, config):
        """Create DataProcessor instance."""
        return DataProcessor(config)

    def test_load_dataset(self, processor):
        """Test dataset loading."""
        # Test implementation
        pass

    def test_prepare_labels(self, processor):
        """Test label preparation."""
        # Test implementation
        pass
```

#### Integration Tests

```python
@pytest.mark.integration
def test_full_training_pipeline():
    """Test complete training pipeline."""
    # Test implementation
    pass

@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing large datasets."""
    # Test implementation
    pass
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_data/test_processor.py

# Run with coverage
pytest --cov=maib_classifier tests/

# Run integration tests
pytest -m integration

# Run fast tests only
pytest -m "not slow"
```

### Test Data

Create test fixtures in `tests/fixtures/`:

```python
# tests/fixtures/sample_data.jsonl
{"text": "Crew member fell overboard", "label": "Accident to person(s)"}
{"text": "Vessel collided with another ship", "label": "Collision"}
{"text": "Engine room fire", "label": "Fire / Explosion"}
```

## Contributing

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make changes**:
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Run tests and checks**:
   ```bash
   make test
   make lint
   make format
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add new feature"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/new-feature
   ```

### Pull Request Guidelines

#### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Integration tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

#### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: At least one reviewer required
3. **Testing**: Manual testing on different environments
4. **Documentation**: Update relevant documentation

### Code Review Guidelines

#### For Authors

- Write clear, self-documenting code
- Add comprehensive tests
- Update documentation
- Write descriptive commit messages
- Keep PRs focused and small

#### For Reviewers

- Check code quality and style
- Verify tests are adequate
- Test functionality manually
- Ensure documentation is updated
- Provide constructive feedback

## Release Process

### Versioning

Follow Semantic Versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Steps

1. **Update version**:
   ```bash
   # Update version in setup.py and pyproject.toml
   # Update CHANGELOG.md
   ```

2. **Create release branch**:
   ```bash
   git checkout -b release/v1.0.0
   ```

3. **Run full test suite**:
   ```bash
   make test
   make lint
   make format
   ```

4. **Build package**:
   ```bash
   python -m build
   ```

5. **Create release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

6. **Publish to PyPI**:
   ```bash
   twine upload dist/*
   ```

### Changelog

Maintain `CHANGELOG.md` with:

```markdown
# Changelog

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Training pipeline
- Inference capabilities
- Evaluation tools

### Changed
- None

### Fixed
- None

### Removed
- None
```

### Docker Releases

```bash
# Build Docker image
docker build -t maib-classifier:v1.0.0 .

# Push to registry
docker push registry/maib-classifier:v1.0.0
```

## Development Tools

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm

- Configure Python interpreter to virtual environment
- Enable Black formatter
- Configure pytest as test runner
- Enable MyPy type checking

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Continuous Integration

#### GitHub Actions

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ --cov=maib_classifier

    - name: Run linting
      run: |
        flake8 src/ scripts/
        mypy src/
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile training
profiler = cProfile.Profile()
profiler.enable()

# Run training
train_model(dataset, config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_dataset(dataset):
    """Process large dataset with memory profiling."""
    pass
```

### GPU Optimization

```python
# Enable mixed precision
config.training.fp16 = True

# Enable gradient checkpointing
config.model.gradient_checkpointing = True

# Optimize data loading
config.training.dataloader_num_workers = 4
config.training.dataloader_pin_memory = True
```
