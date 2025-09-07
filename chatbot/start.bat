@echo off
echo Starting MineSafe Chatbot Server...
echo.

REM Check if .env file exists
if not exist .env (
    echo Creating .env file from template...
    copy env.template .env
    echo.
    echo Please edit .env file and add your OpenAI API key before running again.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Start the application
echo.
echo Starting Flask application...
echo Access the chatbot at: http://localhost:5000/chatbot
echo.
python app.py

pause
