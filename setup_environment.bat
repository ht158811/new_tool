@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" -SetupOnly
if errorlevel 1 (
    echo.
    echo Environment setup failed. Review the error above.
    pause
    exit /b 1
)
echo.
echo Environment setup completed. Double-click run.bat to start.
pause
endlocal
