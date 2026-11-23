#!/bin/bash
# One-time setup for the Microscopy Timelapse Processor on macOS.
# Creates a Python virtual environment in .venv and installs dependencies.
#
# Usage:
#   1. Open Terminal (Cmd+Space, type "Terminal", press Enter).
#   2. In Terminal, type:  cd
#      (with a trailing space) then drag this folder from Finder into the
#      Terminal window and press Enter.
#   3. Run:  bash setup_macos.sh

cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"

echo
echo "=== Microscopy Timelapse Processor - macOS Setup ==="
echo

# Remove macOS quarantine attribute from any files that got tagged when the
# folder was downloaded. Without this, Gatekeeper blocks the scripts with a
# "developer cannot be verified" dialog on first launch.
xattr -dr com.apple.quarantine . 2>/dev/null || true

# 1. Find a Python 3.10+ interpreter. macOS system python3 is often too old,
#    so prefer specific versioned binaries first.
PYCMD=""
for cmd in python3.13 python3.12 python3.11 python3.14 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            PYCMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYCMD" ]; then
    echo "ERROR: Python 3.10 or later was not found on PATH."
    echo
    echo "Install Python with one of:"
    echo "  https://www.python.org/downloads/macos/   (official installer)"
    echo "  brew install python@3.12                  (Homebrew)"
    echo
    echo "Then run this script again."
    exit 1
fi

echo "Using Python: $(command -v "$PYCMD")"
"$PYCMD" --version

# 2. An environment whose Python has gone (the folder was copied from another
#    computer, or Python was removed) cannot be repaired: rebuild it.
if [ -e "$HERE/.venv" ] && ! "$HERE/.venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    echo "The existing .venv no longer works - rebuilding it."
    rm -rf "$HERE/.venv"
fi

# 3. Create the virtual environment if it doesn't already exist.
if [ -f "$HERE/.venv/bin/python" ]; then
    echo "Reusing existing virtual environment at $HERE/.venv"
else
    echo "Creating virtual environment at $HERE/.venv ..."
    if ! "$PYCMD" -m venv "$HERE/.venv"; then
        echo "ERROR: Failed to create the virtual environment."
        exit 1
    fi
fi

# 4. Upgrade pip and install dependencies. PySide6 + bundled ffmpeg are
#    a few hundred MB total; this can take a few minutes the first time.
echo
echo "Upgrading pip..."
"$HERE/.venv/bin/python" -m pip install --upgrade pip

echo
echo "Installing dependencies (this can take a few minutes the first time)..."
if ! "$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements.txt"; then
    echo "ERROR: Dependency install failed. See messages above."
    exit 1
fi

# 5. Make Launch.command double-clickable from Finder (downloads don't always
#    preserve the executable bit, so we set it after install).
chmod +x "$HERE/Launch.command" 2>/dev/null || true
chmod +x "$HERE/setup_macos.sh" 2>/dev/null || true

echo
echo "=== Setup complete! ==="
echo "Double-click Launch.command in Finder to start the app."
echo
