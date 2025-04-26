#!/bin/bash

echo "Creating virtual environment 'pyres-env'..."
python3 -m venv pyres-env

echo "Activating virtual environment..."
source pyres-env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing PyRES and dependencies..."
pip install .

echo "Installation complete."
echo "To activate the environment later, run:"
echo "source pyres-env/bin/activate"
