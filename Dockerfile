# MAIB Incident Type Classifier - Docker Configuration

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p outputs logs data

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MAIB_OUTPUT_DIR=/app/outputs
ENV MAIB_LOG_FILE=/app/logs/training.log

# Default command
CMD ["python", "scripts/train.py", "--help"]
