
@echo off

:: Create a virtual environment
python -m venv venv || (
    echo Failed to create venv
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate || (
    echo Failed to activate venv
    exit /b 1
)

:: Install requirements
pip install -r requirements.txt || (
    echo Failed to install requirements
    exit /b 1
)

echo Installation completed successfully.
