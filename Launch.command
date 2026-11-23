#!/bin/bash
# Launch the Microscopy Timelapse Processor on macOS.
# Double-click this file in Finder after running setup_macos.sh.

cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
PYTHON="$HERE/.venv/bin/python"
APP="$HERE/microscopy_timelapse.py"
LOG="$HERE/launch-log.txt"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: The Python environment hasn't been set up yet."
    echo
    echo "Open Terminal in this folder and run:"
    echo "    bash setup_macos.sh"
    echo
    read -p "Press Enter to close..."
    exit 1
fi

if [ ! -f "$APP" ]; then
    echo "ERROR: App file not found at:"
    echo "    $APP"
    echo
    echo "Make sure microscopy_timelapse.py is in the same folder as this script."
    read -p "Press Enter to close..."
    exit 1
fi

# Launch detached so this Terminal window can be closed while the GUI runs.
# Output goes to a log file rather than /dev/null: if the app fails to start we
# need to show the reason, otherwise a crash looks exactly like a good launch.
nohup "$PYTHON" "$APP" "$@" >"$LOG" 2>&1 &
APP_PID=$!
disown

# Give it a moment, then confirm it actually survived startup.
sleep 4

if kill -0 "$APP_PID" 2>/dev/null; then
    echo "Microscopy Timelapse Processor launched."
    echo "You can close this Terminal window."
    sleep 2
    exit 0
fi

echo
echo "=== The app failed to start. ==="
echo
if [ -s "$LOG" ]; then
    echo "Error details:"
    echo
    tail -n 25 "$LOG" | sed 's/^/    /'
    echo
fi
echo "Most common cause: the dependency install didn't finish. Try re-running:"
echo "    bash setup_macos.sh"
echo
echo "The full log is saved at:"
echo "    $LOG"
echo
read -p "Press Enter to close..."
exit 1
