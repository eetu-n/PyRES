@echo off
set ENV_NAME=pyres-env

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel%==0 (
    echo Conda detected.
    set /p use_conda=Would you like to create the environment using conda and environment.yml? (y/n):

    if /I "%use_conda%"=="y" (
        echo Creating conda environment...
        conda env create -f environment.yml
        echo Done. To activate, run: conda activate %ENV_NAME%
        exit /b 0
    )
)

:: Otherwise fallback to python venv
echo Conda not used. Proceeding with python -m venv...

where python >nul 2>nul
if %errorlevel% NEQ 0 (
    echo Python not found. Please install Python 3.10 first.
    exit /b 1
)

:: Create virtual environment
python -m venv %ENV_NAME%

:: Activate environment
call %ENV_NAME%\Scripts\activate

:: Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

:: Install dependencies
pip install -r requirements.txt

echo Done. To activate the environment, run: call %ENV_NAME%\Scripts\activate
