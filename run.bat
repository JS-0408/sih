@echo off
TITLE Yaazhi GeoAlign OS - Geospatial Registration Platform
COLOR 0A
CLS

echo ======================================================================
echo    YAAZHI GEOALIGN OS -- GEOSPATIAL IMAGE REGISTRATION PLATFORM
echo ======================================================================
echo.

:: 1. Check Python installation
py --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not found on PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [1/3] Python environment verified.

:: 2. Install / verify dependencies
echo [2/3] Verifying required Python packages...
py -m pip install -r requirements.txt flask --quiet >nul 2>&1

:: 3. Launch Web Server and open browser
echo [3/3] Starting Web Server and opening browser dashboard...
echo.
echo ======================================================================
echo  Dashboard URL: http://127.0.0.1:5000
echo ======================================================================
echo.

:: Open browser after 2 seconds delay
start "" http://127.0.0.1:5000

:: Start server
py server.py

pause
