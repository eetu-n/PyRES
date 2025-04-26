#!/bin/bash
set -e

ENV_NAME="pyres-env"

# Check if conda is available
if command -v conda &> /dev/null
then
    echo "🔍 Conda detected. Creating environment with conda..."

    # Check if environment already exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo "⚠️ Conda environment '$ENV_NAME' already exists. Skipping creation."
    else
        conda create -n $ENV_NAME python=3.10 -y
    fi

    echo "✅ Environment created. Activate with: conda activate $ENV_NAME"
else
    echo "🔍 Conda not found. Falling back to Python venv..."

    # Create local venv
    python3 -m venv $ENV_NAME

    # Activate venv
    source $ENV_NAME/bin/activate

    # Upgrade pip tools
    pip install --upgrade pip setuptools wheel

    # Install dependencies
    pip install -r requirements.txt

    echo "✅ Environment setup complete. Activate with: source $ENV_NAME/bin/activate"
fi
