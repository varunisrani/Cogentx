@echo off
call venv\Scripts\activate
streamlit run app.py --server.headless true
