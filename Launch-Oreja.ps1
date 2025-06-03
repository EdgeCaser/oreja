# Launch-Oreja.ps1 - Single-Click Oreja Launcher
# Comprehensive launcher for backend + frontend components

Write-Host "🎙️ Oreja Complete Launcher" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

try {
    # Check if virtual environment exists
    $VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
    
    if (Test-Path $VenvPython) {
        Write-Host "✅ Using virtual environment Python..." -ForegroundColor Green
        & $VenvPython "launch_oreja_analytics.py"
    } else {
        Write-Host "⚠️ Virtual environment not found, using system Python..." -ForegroundColor Yellow
        python "launch_oreja_analytics.py"
    }
} catch {
    Write-Host "❌ Error starting launcher: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   • Make sure Python is installed and in your PATH" -ForegroundColor Gray
    Write-Host "   • Check that the virtual environment is set up correctly" -ForegroundColor Gray
    Write-Host "   • Verify dependencies are installed: pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
} 