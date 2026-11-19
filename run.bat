@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
if errorlevel 1 (
    echo.
    echo Dataset Manager failed to start. Review the error above.
    pause
    exit /b 1
)
endlocal
