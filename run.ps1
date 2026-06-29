# Activate virtual environment
if (Test-Path ".build2r\Scripts\activate.ps1") {
    . ".build2r\Scripts\activate.ps1"
} else {
    Write-Host "Warning: Virtual environment not found." -ForegroundColor Yellow
}

# Force Python to use UTF-8 natively
$env:PYTHONUTF8 = "1"

# Run the engine
python main.py
