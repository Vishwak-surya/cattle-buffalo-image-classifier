Write-Host "Starting Cattle vs Buffalo Classifier Web Application..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Check if Flask is installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Flask not found"
    }
}
catch {
    Write-Host "Installing Flask dependencies..." -ForegroundColor Cyan
    pip install Flask Werkzeug
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Start the Flask application
Write-Host ""
Write-Host "Starting Flask application..." -ForegroundColor Green
Write-Host "Web server will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py

Read-Host "Press Enter to exit"