@echo off
echo Starting Cattle vs Buffalo Classifier Web Application...
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Check if Flask is installed
echo Checking dependencies...
python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo Installing Flask dependencies...
    pip install Flask Werkzeug
    if %errorlevel% neq 0 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the Flask application
echo.
echo Starting Flask application...
echo Web server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause