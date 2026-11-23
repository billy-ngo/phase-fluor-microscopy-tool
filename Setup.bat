@echo off
REM One-time setup for the Microscopy Timelapse Processor.
REM Creates a Python virtual environment in .venv and installs dependencies.

setlocal DisableDelayedExpansion
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "REQS=%HERE%requirements.txt"

echo.
echo === Microscopy Timelapse Processor - Setup ===
echo.

REM 1. Find a Python the packages support. The py launcher's own default is
REM    the newest Python installed, which can be too new for ready-made
REM    packages, so ask for specific versions first.
set "PYCMD="
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.13 3.12 3.11 3.14 3.10) do (
        if not defined PYCMD (
            py -%%V -c "import sys" >nul 2>&1
            if not errorlevel 1 set "PYCMD=py -%%V"
        )
    )
)
REM "call" is needed in case python is a batch file (pyenv-win).
if not defined PYCMD (
    call python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo ERROR: Python was not found.
    echo.
    echo Install Python 3.12 from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH" on the first installer screen.
    echo Then run Setup.bat again.
    echo.
    pause
    exit /b 1
)

echo Using Python: %PYCMD%
call %PYCMD% --version

REM 2. Require Python 3.10 or later (PySide6 6.x needs it).
call %PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or later is required.
    pause
    exit /b 1
)

REM 3. An environment whose Python has gone (the folder was copied from
REM    another computer, or Python was uninstalled) cannot be repaired.
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo The existing .venv no longer works - rebuilding it.
        rmdir /s /q "%VENV%"
    )
)

REM 4. Create the virtual environment if it doesn't already exist.
if exist "%VENV%\Scripts\python.exe" (
    echo Reusing existing .venv at "%VENV%".
) else (
    echo Creating virtual environment at "%VENV%"...
    call %PYCMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM 5. Upgrade pip and install dependencies. PySide6 and ffmpeg are large
REM    (about 500 MB to download), so this can take a few minutes.
echo.
echo Upgrading pip...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
echo.
echo Installing dependencies (this can take a few minutes the first time)...
"%VENV%\Scripts\python.exe" -m pip install -r "%REQS%"
if errorlevel 1 (
    echo.
    echo ERROR: Dependency install failed. See messages above.
    pause
    exit /b 1
)

echo.
echo === Setup complete! ===
echo Double-click Launch.bat to start the app.
echo.
pause
exit /b 0
