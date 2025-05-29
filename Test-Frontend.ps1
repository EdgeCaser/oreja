# Simple test script for the C# frontend
Write-Host "Testing Oreja Frontend..." -ForegroundColor Green

$ExePath = ".\publish\Oreja.exe"

if (!(Test-Path $ExePath)) {
    Write-Host "Error: Oreja.exe not found at $ExePath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting Oreja application..." -ForegroundColor Yellow

try {
    # Start the application and capture any immediate errors
    $process = Start-Process $ExePath -WorkingDirectory . -PassThru -ErrorAction Stop
    
    Write-Host "Application started with PID: $($process.Id)" -ForegroundColor Green
    
    # Wait a moment to see if it crashes immediately
    Start-Sleep -Seconds 3
    
    if ($process.HasExited) {
        Write-Host "Application exited with code: $($process.ExitCode)" -ForegroundColor Red
    } else {
        Write-Host "Application is running normally" -ForegroundColor Green
    }
    
} catch {
    Write-Host "Failed to start application: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press Enter to exit..."
Read-Host 