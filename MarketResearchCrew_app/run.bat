
@echo off

:: Activate the virtual environment
call venv\Scripts\activate || (
    echo Failed to activate venv
    exit /b 1
)

:: Run the Streamlit app
streamlit run app.py --server.headless true
