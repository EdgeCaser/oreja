# Quick Start Guide - Oreja

## 🚀 How to Run Oreja

### Option 1: Double-Click Launch (Recommended)
1. **Double-click** `Start-Oreja.bat` 
   - This will automatically start both the Python backend and C# frontend
   - A console window will open showing the startup process
   - The Oreja desktop application will launch automatically

### Option 2: PowerShell Launch
1. Right-click `Start-Oreja.ps1` → "Run with PowerShell"

### Option 3: Direct Executable
1. First manually start the backend:
   - Open PowerShell in the `backend` folder
   - Run: `.\..\.venv\Scripts\Activate.ps1`
   - Run: `uvicorn server:app --host 127.0.0.1 --port 8000`
2. Then run: `publish\Oreja.exe`

## 📁 Files Included

- `Start-Oreja.bat` - **Main launcher** (double-click this!)
- `Start-Oreja.ps1` - PowerShell launcher script
- `publish\Oreja.exe` - Self-contained C# application (168MB)
- `backend\` - Python server files
- `venv\` - Python virtual environment

## ✅ What Should Happen

1. **Backend Console**: A PowerShell window opens running the Python server
2. **Frontend Window**: The Oreja desktop application opens
3. **Ready to Use**: Select audio sources and click "Start" to begin transcription

## 🛑 To Stop Oreja

1. Close the Oreja desktop application window
2. Close the Python backend console window

## 🔧 Troubleshooting

- **Backend fails to start**: Check that the virtual environment is set up correctly
- **Frontend fails to launch**: Ensure you've run `dotnet publish` to create the executable
- **Can't find models**: First run may take time to download Hugging Face models (several GB)

## 📋 Requirements Met

- ✅ Single-click execution
- ✅ Self-contained executable (no .NET runtime needed)
- ✅ Automatic backend startup
- ✅ Error checking and user feedback 