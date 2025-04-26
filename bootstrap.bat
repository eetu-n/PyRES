@echo off
setlocal enabledelayedexpansion

set ENV_NAME=pyres-env

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel%==0 (
    echo 🔍 Conda detected. Creating environment with conda...

    REM Check if environment already exists
    conda env list | findstr /C:"%ENV_NAME%" >nul
    if %errorlevel%==0 (
        echo ⚠️ Conda environment '%ENV_NAME%' already exists. Skipping creation.
    ) else (
        conda create -n %ENV_NAME% python=3.10 -y
    )

    echo ✅ Environment created. Activate with: conda activate %ENV_NAME%
    echo Once activated, install dependencies manually with: pip install -r requirements.txt
) else (
    echo 🔍 Conda not found. Falling back to python venv...

    REM Create venv
    python3 -m venv %ENV_NAME%

    REM Activate venv
    call %ENV_NAME%\Scripts\activate.bat

    REM Upgrade pip tools
    python -m pip install --upgrade pip setuptools wheel

    REM Install dependencies
    pip install -r requirements.txt

    echo ✅ Environment setup complete. Activate with: call %ENV_NAME%\Scripts\activate.bat
)

endlocal
