@echo off
set ENV_NAME=pyres-env

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel%==0 (
    echo Conda detected.
    echo Creating conda environment...
    conda env create -f environment.yml
    echo Done. To activate, run: conda activate %ENV_NAME%
    exit /b 0
)

:: Otherwise fallback to python venv
echo Conda not found. Proceeding with python -m venv...

:: Create virtual environment
python3 -m venv %ENV_NAME%

:: Activate environment
call %ENV_NAME%\Scripts\activate

:: Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

:: Install dependencies
pip install -r requirements.txt

echo Done. To activate the environment, run: call %ENV_NAME%\Scripts\activate
