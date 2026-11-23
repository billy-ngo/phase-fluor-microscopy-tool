@echo off
REM Launch the Microscopy Timelapse Processor.
REM Double-click this file to open the app, or drop an .nd2 file on it.

setlocal DisableDelayedExpansion
set "HERE=%~dp0"
set "PYTHONW=%HERE%.venv\Scripts\pythonw.exe"
set "PYTHON=%HERE%.venv\Scripts\python.exe"
set "APP=%HERE%microscopy_timelapse.py"
set "LOG=%HERE%launch-log.txt"

if not exist "%PYTHONW%" goto missing_venv
if not exist "%APP%" goto missing_app

REM Preflight the imports with the console python first. pythonw.exe has no
REM console, so without this a missing dependency would make the app appear
REM to do nothing at all when double-clicked.
"%PYTHON%" -c "import PySide6, nd2, numpy, cv2, imageio, imageio_ffmpeg, PIL" 2>"%LOG%"
if errorlevel 1 goto missing_deps

REM Errors after this point are appended to launch-log.txt by the app itself.
start "" "%PYTHONW%" "%APP%" %*
endlocal
exit /b 0

:missing_deps
echo.
echo === The app can't start: a required package is missing. ===
echo.
type "%LOG%"
echo.
echo Fix this by re-running Setup.bat (double-click it) to finish installing
echo the dependencies.
echo.
pause
exit /b 1

:missing_venv
echo.
echo ERROR: The Python environment hasn't been set up yet.
echo.
echo Please run Setup.bat first (double-click it). That creates the .venv
echo folder and installs the dependencies (about 5 minutes the first time).
echo.
pause
exit /b 1

:missing_app
echo.
echo ERROR: App file not found at:
echo   %APP%
echo.
echo This file should sit next to Launch.bat. Make sure you extracted the
echo entire folder.
echo.
pause
exit /b 1
