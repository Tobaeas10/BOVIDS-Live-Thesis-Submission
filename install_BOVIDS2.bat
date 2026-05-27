@echo off
title BOVIDS2 Installer
echo ============================================
echo     BOVIDS2 One-Click Installer
echo ============================================
echo.

:: -------------------------------
:: Step 0: Download & install Python
:: -------------------------------
echo [INFO] Installing Python 3 (latest release)...
curl -L -o python_installer.exe https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe
start /wait "" python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del python_installer.exe
echo [DONE] Python installed and added to PATH.

:: Refresh PATH for this session
set "PATH=C:\Program Files\Python313\;C:\Program Files\Python313\Scripts\;%PATH%"

:: -------------------------------
:: Step 1: Install VC++ Redistributable
:: -------------------------------
echo.
echo [INFO] Installing Microsoft Visual C++ 2015-2019 Redistributable (x64)...
curl -L -o vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
start /wait "" vc_redist.x64.exe /install /quiet /norestart
del vc_redist.x64.exe
echo [DONE] VC++ Redistributable installed.

:: -------------------------------
:: Step 2: Install Python dependencies
:: -------------------------------
echo.
echo [INFO] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r server\bovids_v2\docs\requirements_yolov8.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements. Aborting.
    pause
    exit /b 1
)

:: -------------------------------
:: Step 3: Install Ultralytics
:: -------------------------------
echo.
echo [INFO] Installing Ultralytics...
python -m pip install ultralytics
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Ultralytics. Aborting.
    pause
    exit /b 1
)

:: -------------------------------
:: Step 4: Install Google Chrome (latest stable)
:: -------------------------------
echo.
echo [INFO] Installing Google Chrome...
curl -L -o chrome_installer.exe https://dl.google.com/chrome/install/latest/chrome_installer.exe
start /wait "" chrome_installer.exe /silent /install
del chrome_installer.exe
echo [DONE] Google Chrome installed.

echo.
echo ============================================
echo   Python + VC++ + dependencies + Chrome successfully installed!
echo ============================================
pause
