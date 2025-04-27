#!/bin/bash

ENV_NAME="pyres-env"

# Detect OS
OS_TYPE="$(uname)"

# Check if conda is available
if command -v conda &> /dev/null
then
    echo "Conda detected."
    echo "Creating conda environment..."
    conda env create -f environment.yml
    exit 0
fi

# Otherwise, fallback to python venv
echo "Conda not found. Proceeding with python -m venv..."

# Create virtual environment
python3 -m venv $ENV_NAME

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
python -m pip install -r requirements.txt

# Install libsndfile depending on OS
if [ "$OS_TYPE" == "Darwin" ]; then
    echo "macOS detected."
    if ! brew list libsndfile &> /dev/null; then
        echo "Installing libsndfile with brew..."
        brew install libsndfile
    else
        echo "libsndfile already installed via brew."
    fi
    # Add DYLD_LIBRARY_PATH setup into the activate script
    echo "Configuring environment to find libsndfile..."
    echo "export DYLD_LIBRARY_PATH=\$(brew --prefix libsndfile)/lib:\$DYLD_LIBRARY_PATH" >> $ENV_NAME/bin/activate
elif [ "$OS_TYPE" == "Linux" ]; then
    echo "Linux detected."
    echo "Installing libsndfile with apt..."
    sudo apt-get update
    sudo apt-get install -y libsndfile1
else
    echo "Unsupported OS: $OS_TYPE. Please install libsndfile manually."
fi

echo "Done. To activate the environment, run: source $ENV_NAME/bin/activate"