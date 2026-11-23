# Microscopy Timelapse Processor

A desktop app for Windows and macOS that reads Nikon `.nd2` timelapse files
with phase, GFP and mCherry channels and exports videos, TIFF frames and figure
panels, with adjustable display ranges, a scale bar and a timer.

## Requirements

- Windows 10 or 11, or macOS 11 or later
- Python 3.10 to 3.14
- About 1 GB of disk space for the Python environment
- Memory of about three times the size of the `.nd2` file

## Install

Download this repository (**Code → Download ZIP** on GitHub, then extract it),
or clone it:

```bash
git clone https://github.com/billy-ngo/phase-fluor-microscopy-tool
```

### Windows

1. Install Python 3.12 from <https://www.python.org/downloads/> and tick
   **Add python.exe to PATH** on the first installer screen.
2. Double-click `Setup.bat` in the downloaded folder. It creates a Python
   environment in `.venv` and installs the packages in `requirements.txt`
   (3–5 minutes).

### macOS

1. Install Python 3.10 or later from <https://www.python.org/downloads/macos/>
   or with `brew install python@3.12`.
2. Open Terminal, type `cd ` followed by a space, drag the downloaded folder
   onto the Terminal window and press Enter.
3. Run `bash setup_macos.sh` (3–5 minutes).

## Running

- **Windows:** double-click `Launch.bat`, or drop an `.nd2` file on it.
- **macOS:** double-click `Launch.command`. If macOS blocks it as being from an
  unidentified developer, right-click it and choose *Open*; on macOS 15 and
  later allow it under *System Settings → Privacy & Security → Open Anyway*.
- **From a terminal:** `.venv\Scripts\python.exe microscopy_timelapse.py [file.nd2]`
  on Windows, `.venv/bin/python microscopy_timelapse.py [file.nd2]` on macOS.

If the app does not appear, the error is in `launch-log.txt` in this folder.

## Workflow

1. **Open a file** with `Open .nd2…`, or drag one onto the window. Channels are
   matched by name (for example "ph3", "GFP", "mCherry"); a missing channel is
   shown black. A Z-stack is reduced to its maximum projection.
2. **Background ROI:** drag a box over an empty area. Its mean is subtracted
   from the GFP and mCherry channels in every frame. Phase is not changed.
3. **Display range:** for each channel, drag the two handles on the histogram
   for minimum and maximum and the dot on the curve for gamma (above 1 brightens
   the midtones), or type the minimum and maximum. `Auto` restores the automatic
   range.
4. **Crop region** (optional): drag a box, or type X, Y, W and H. The crop can
   be free, a fixed aspect ratio or a fixed size.
5. **Preview export** shows the frame exactly as the video will contain it.
6. **Export** (see below).

## Exports

| Button | Output |
| --- | --- |
| `Process and Export` | One MP4 of the 2×2 grid (phase, GFP, mCherry, merge) for the chosen frame range, with scale bar, timer and channel labels as set on the Export tab. Channels that are unticked are black. |
| `Export Example Set…` | Into a folder you choose: one folder of TIFF frames per channel and per merge, a `SCALE_…` frame with the scale bar, two 4-panel videos (with and without mCherry), the custom strip if ticked, and `example_settings.json`. Always contains every channel. |
| `Export Strip Only…` | The custom strip (chosen panels side by side, one TIFF per frame, optional video) and `strip_settings.json`. |
| `Timeline Export…` | A figure with one row per channel and one column per chosen frame, or each chosen frame as its own PNG or TIFF. |
| `Save Metadata…` | Every setting as a JSON file. `Load Settings…` restores such a file, and the settings files written by the exports, reopening the source file if needed. |

- Exported TIFF and PNG images are 8-bit RGB with the display range applied,
  not raw camera values.
- All exports use the same background subtraction and display range as the
  preview.
- Videos are H.264. The three 4:4:4 qualities (the default) keep full colour
  resolution and play in VLC, QuickTime, Premiere and ImageJ, but not in
  Windows' own players or PowerPoint; choose `Compatible` for those.
- `Size` enlarges a video by a whole factor with sharp pixels. TIFF frames are
  never enlarged.
- Files of the same name in the chosen folder are replaced.

## Other controls

| Control | Use |
| --- | --- |
| Time slider, `Play`, `P` | Step through or play the export frame range. |
| Position | Switch XY position in a multi-position file. |
| `Show` buttons, `Ctrl`+`1`–`5` | View all four panels or one panel large. Preview only. |
| Start / End frame | The frame range that is exported. |
| LUT mode | *Stack auto*: one range per channel for all frames. *Smart per-frame auto*: an automatic range for every frame. *Manual per-frame*: set the range frame by frame with `Next Frame`. |
| Merge mode | Additive, mCherry over GFP, or GFP over mCherry. |
| Rotate | Turn the image by 90°, 180° or 270°. The boxes turn with it. |
| `Append .nd2…` | Join a continuation file after the loaded one (in memory only). Image size, positions and pixel size must match. The field can be aligned at the join by whole pixels, and the timer keeps the real time gap. |
| Timer | Frame number or elapsed time on every frame, from a fixed interval or the microscope's own acquisition times. |
| Scale bar | Length, corner and offsets. An export stops with a message if the bar is longer than the image it is drawn on. |
| Layout, `Ctrl`+`\` | Wide monitor, laptop (side panel) or image focus; fold the side panel. |
| Zoom and pan | Mouse wheel or pinch zooms; `Space`+drag or middle-drag pans; double-click fits. |

On macOS the `Ctrl` shortcuts use `⌘`.

## Measurement notes

- The scale bar is `length ÷ pixel size` rounded to whole pixels, using the
  pixel size stored in the `.nd2` file. A file without a calibrated pixel size
  gets no scale bar.
- Background subtraction is the mean of the background box in each frame,
  subtracted in 32-bit float before the display range is applied.
- Appended files are shifted by whole pixels only; no pixel is interpolated.

## Tests

`test_core.py` checks the image processing on synthetic data and needs no
`.nd2` file:

```bash
.venv/bin/python test_core.py
```

(`.venv\Scripts\python.exe test_core.py` on Windows.)

## Files

| File | Purpose |
| --- | --- |
| `microscopy_timelapse.py` | The application |
| `requirements.txt` | Python packages it needs |
| `Setup.bat`, `Launch.bat` | Windows: one-time install, and start |
| `setup_macos.sh`, `Launch.command` | macOS: one-time install, and start |
| `test_core.py` | Tests |

`.venv/` (the Python environment) and `launch-log.txt` are created in this
folder by the scripts.

## Authors

Written by [billy-ngo](https://github.com/billy-ngo) and
[chipashvili](https://github.com/chipashvili).

## Troubleshooting

- **Windows: "Python was not found"** — install Python from python.org with
  *Add python.exe to PATH* ticked, then run `Setup.bat` again.
- **"A required package is missing"** — run the setup script again.
- **The app worked before and no longer starts** — the folder was moved to
  another computer, or Python was updated. Run the setup script again; it
  rebuilds `.venv`.
- **macOS: "permission denied" on `Launch.command`** — run
  `bash setup_macos.sh` again; it makes the launcher executable.
- **A video does not play** — see the note on 4:4:4 videos under *Exports*.
