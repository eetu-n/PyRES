@echo off

echo Creating virtual environment 'pyres-env'...
python -m venv pyres-env

echo Activating virtual environment...
call pyres-env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo Installing PyRES and dependencies...
pip install .

echo Installation complete.
echo To activate the environment later, run:
echo call pyres-env\Scripts\activate.bat
