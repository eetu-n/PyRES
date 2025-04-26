#!/bin/bash

ENV_NAME="pyres-env"

# Check if conda is available
if command -v conda &> /dev/null
then
    echo "Conda detected."
    echo "Creating conda environment..."
    conda env create -f environment.yml
    echo "Done. To activate, run: conda activate $ENV_NAME"
    exit 0
fi

# Otherwise, fallback to python venv
echo "Conda not found. Proceeding with python -m venv..."

# Create virtual environment
python3 -m venv $ENV_NAME

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

echo "Done. To activate the environment, run: source $ENV_NAME/bin/activate"
