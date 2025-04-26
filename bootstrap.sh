#!/bin/bash

ENV_NAME="pyres-env"

# Check if conda is available
if command -v conda &> /dev/null
then
    echo "Conda detected."
    read -p "Would you like to create the environment using conda and environment.yml? (y/n): " use_conda

    if [[ "$use_conda" == "y" || "$use_conda" == "Y" ]]; then
        echo "Creating conda environment..."
        conda env create -f environment.yml
        echo "Done. To activate, run: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Otherwise, fallback to python venv
echo "Conda not used. Proceeding with python -m venv..."

# Check if python is available
if ! command -v python &> /dev/null
then
    echo "Python not found. Please install Python 3.10 first."
    exit 1
fi

# Create virtual environment
python -m venv $ENV_NAME

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

echo "Done. To activate the environment, run: source $ENV_NAME/bin/activate"
