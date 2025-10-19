"""
Setup script for MAIB Incident Type Classifier.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
  with open(requirements_path, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
  name="maib-incident-classifier",
  version="1.0.0",
  author="MAIB Incident Type Classifier Team",
  author_email="team@maib-classifier.com",
  description="Marine Accident Investigation Branch Incident Type Classification System",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/maib-team/incident-classifier",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  classifiers=[
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
  ],
  python_requires=">=3.8",
  install_requires=requirements,
  extras_require={
    "dev": [
      "pytest>=7.0.0",
      "pytest-cov>=4.0.0",
      "black>=22.0.0",
      "flake8>=5.0.0",
      "mypy>=1.0.0",
    ],
    "jupyter": [
      "jupyter>=1.0.0",
      "ipykernel>=6.0.0",
    ],
  },
  entry_points={
    "console_scripts": [
      "maib-train=maib_classifier.scripts.train:main",
      "maib-inference=maib_classifier.scripts.inference:main",
      "maib-evaluate=maib_classifier.scripts.evaluate:main",
    ],
  },
  include_package_data=True,
  zip_safe=False,
)
