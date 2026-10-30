@echo off
setlocal enabledelayedexpansion

:: CRITICAL: Set working directory to the folder containing this batch file
cd /d "%~dp0"

echo ========================================================
echo   Auto-Commit and Push to GitHub (yashsidana/Retail_demand)
echo ========================================================
echo Current folder: %CD%
echo.

set "TOKEN=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"
set "REPO_URL=https://x-access-token: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@github.com/yashsidana/Retail_demand.git"

:: Method 1: Try Python API Script (Zero git requirement)
echo [1/3] Checking for Python to push via GitHub REST API...
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Python found! Running push_to_github.py...
    python "%~dp0push_to_github.py"
    if !errorlevel% equ 0 (
        echo.
        echo ========================================================
        echo SUCCESS! Everything committed to GitHub!
        echo URL: https://github.com/yashsidana/Retail_demand
        echo ========================================================
        pause
        exit /b 0
    )
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    echo Python (py) found! Running push_to_github.py...
    py "%~dp0push_to_github.py"
    if !errorlevel% equ 0 (
        echo.
        echo ========================================================
        echo SUCCESS! Everything committed to GitHub!
        echo URL: https://github.com/yashsidana/Retail_demand
        echo ========================================================
        pause
        exit /b 0
    )
)

:: Method 2: Try Git
echo.
echo [2/3] Trying Git CLI...
set "GIT_CMD=git"
where git >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\Git\cmd\git.exe" set "GIT_CMD=C:\Program Files\Git\cmd\git.exe"
    if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Git\cmd\git.exe" set "GIT_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Git\cmd\git.exe"
)

"%GIT_CMD%" --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Git found: "%GIT_CMD%"
    if not exist ".git" "%GIT_CMD%" init
    "%GIT_CMD%" config user.name "Yash Sidana"
    "%GIT_CMD%" config user.email "yashsidana@users.noreply.github.com"
    "%GIT_CMD%" remote remove origin >nul 2>&1
    "%GIT_CMD%" remote add origin %REPO_URL%
    "%GIT_CMD%" add .
    "%GIT_CMD%" commit -m "feat: complete retail demand forecasting & inventory optimization project"
    "%GIT_CMD%" branch -M main
    "%GIT_CMD%" push -u origin main --force
    if !errorlevel! equ 0 (
        echo.
        echo ========================================================
        echo SUCCESS! Pushed via Git to GitHub!
        echo URL: https://github.com/yashsidana/Retail_demand
        echo ========================================================
        pause
        exit /b 0
    )
)

:: Method 3: Try PowerShell Invoke-RestMethod via system powershell.exe
echo.
echo [3/3] Trying Windows PowerShell API Upload...
set "PS=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%PS%" (
    echo System PowerShell found! Running API upload...
    "%PS%" -ExecutionPolicy Bypass -File "%~dp0push_via_api.ps1"
    if !errorlevel! equ 0 (
        echo.
        echo ========================================================
        echo SUCCESS! Pushed via PowerShell API to GitHub!
        echo URL: https://github.com/yashsidana/Retail_demand
        echo ========================================================
        pause
        exit /b 0
    )
)

echo.
echo Failed to push automatically.
pause
