# MAIB Incident Type Classifier

A machine learning project for automatically classifying maritime incident types from MAIB (Marine Accident Investigation Branch) safety reports.

## Overview

This project processes maritime incident data and trains a transformer-based classifier to automatically categorize incident types based on report descriptions.

## Project Structure

- `preprocess.py` - Data preprocessing and cleaning pipeline
- `train.py` - Model training using DistilBERT
- `explore.py` - Data exploration and analysis
- `data/` - Contains raw MAIB data samples and processed datasets samples

## Current Status

**Work in Progress** - This is a raw, unfinished project

The project currently includes:
- Data preprocessing pipeline for MAIB occurrence reports
- Basic transformer model training setup
- Initial data exploration scripts

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Datasets

## Usage

1. Place raw MAIB data in `data/raw_data/`
2. Run preprocessing: `python preprocess.py`
3. Train model: `python train.py`

## License

See LICENSE file for details.
