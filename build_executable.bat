@echo off
setlocal EnableDelayedExpansion

echo.
echo ==========================================
echo    OREJA EXECUTABLE BUILDER
echo ==========================================
echo.
echo This script will create the Oreja.exe file needed for live transcription.
echo This may take 5-10 minutes on first run.
echo.

:: Check if .NET SDK is installed
echo [1/4] Checking for .NET SDK...
dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: .NET SDK not found!
    echo.
    echo Please install .NET 8 SDK first:
    echo 1. Go to: https://dotnet.microsoft.com/download/dotnet/8.0
    echo 2. Download and install ".NET 8.0 SDK" (not Runtime^)
    echo 3. Restart your computer
    echo 4. Run this script again
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('dotnet --version') do set DOTNET_VERSION=%%i
echo ✅ Found .NET SDK version: %DOTNET_VERSION%
echo.

:: Check if we're in the right directory
echo [2/4] Checking project files...
if not exist "Oreja.csproj" (
    echo.
    echo ❌ ERROR: Cannot find Oreja.csproj file!
    echo.
    echo Make sure you're running this script from the oreja project folder.
    echo The folder should contain: Oreja.csproj, README.markdown, etc.
    echo.
    pause
    exit /b 1
)
echo ✅ Project files found
echo.

:: Clean previous build if it exists
echo [3/4] Preparing build environment...
if exist "publish-standalone" (
    echo Removing previous build...
    rmdir /s /q "publish-standalone" 2>nul
    if exist "publish-standalone" (
        echo ⚠️  Warning: Could not remove old build folder completely
        echo This might cause issues. Try deleting 'publish-standalone' folder manually.
        echo.
    )
)
echo ✅ Build environment ready
echo.

:: Build the executable
echo [4/4] Building Oreja executable...
echo This will take a few minutes - please be patient!
echo.

dotnet publish -c Release -r win-x64 --self-contained true -o publish-standalone --verbosity quiet

if %errorlevel% neq 0 (
    echo.
    echo ❌ BUILD FAILED!
    echo.
    echo Common solutions:
    echo 1. Make sure you have a stable internet connection
    echo 2. Try running as Administrator
    echo 3. Check if antivirus is blocking the process
    echo 4. Delete publish-standalone folder and try again
    echo.
    pause
    exit /b 1
)

:: Check if executable was created
if not exist "publish-standalone\Oreja.exe" (
    echo.
    echo ❌ BUILD COMPLETED but Oreja.exe not found!
    echo.
    echo Please check the publish-standalone folder manually.
    echo.
    pause
    exit /b 1
)

:: Success!
echo.
echo ==========================================
echo ✅ SUCCESS! Oreja.exe has been created!
echo ==========================================
echo.
echo 📁 Location: %CD%\publish-standalone\Oreja.exe
echo 📏 File size: 
for %%A in ("publish-standalone\Oreja.exe") do echo    %%~zA bytes (~149 MB)
echo.

:: Test the executable
echo 🧪 Testing executable...
"publish-standalone\Oreja.exe" --help >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Executable test passed!
) else (
    echo ⚠️  Executable created but may have issues
    echo    Try running it manually to check
)
echo.

:: Offer to create desktop shortcut
set /p CREATE_SHORTCUT="Would you like to create a desktop shortcut? (Y/N): "
if /i "%CREATE_SHORTCUT%"=="Y" (
    echo.
    echo 🔗 Creating desktop shortcut...
    
    :: Get desktop path
    for /f "usebackq tokens=3*" %%A in (`reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop 2^>nul`) do set DESKTOP=%%A %%B
    
    if defined DESKTOP (
        :: Create shortcut using PowerShell
        powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\Oreja Live Transcription.lnk'); $Shortcut.TargetPath = '%CD%\publish-standalone\Oreja.exe'; $Shortcut.WorkingDirectory = '%CD%\publish-standalone'; $Shortcut.Description = 'Oreja Live Transcription - Real-time conference call transcription'; $Shortcut.Save()" 2>nul
        
        if exist "%DESKTOP%\Oreja Live Transcription.lnk" (
            echo ✅ Desktop shortcut created: "Oreja Live Transcription"
        ) else (
            echo ⚠️  Could not create desktop shortcut automatically
            echo    You can create one manually by right-clicking Oreja.exe
        )
    ) else (
        echo ⚠️  Could not find desktop folder
        echo    You can create a shortcut manually by right-clicking Oreja.exe
    )
    echo.
)

:: Final instructions
echo ==========================================
echo 🎯 WHAT'S NEXT:
echo ==========================================
echo.
echo 1. Double-click Oreja.exe to start live transcription
echo 2. Or use the desktop shortcut if you created one
echo 3. Select your microphone and system audio sources
echo 4. Click "Start" to begin transcription!
echo.
echo 📚 For advanced features (conversation analysis, privacy mode):
echo    See the "Full Development Setup" section in README.markdown
echo.
echo 🆘 If you have problems:
echo    Check the troubleshooting section in README.markdown
echo.

:: Offer to launch the application
set /p LAUNCH_NOW="Would you like to launch Oreja now? (Y/N): "
if /i "%LAUNCH_NOW%"=="Y" (
    echo.
    echo 🚀 Launching Oreja...
    start "" "publish-standalone\Oreja.exe"
    echo.
    echo Oreja should be starting now!
    echo If it doesn't appear, check for error messages above.
)

echo.
echo ==========================================
echo BUILD COMPLETE - You can close this window
echo ==========================================
echo.
pause 