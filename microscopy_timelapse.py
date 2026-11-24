"""Microscopy Timelapse Processor.

Reads a multi-position Nikon .nd2 timelapse with Phase / GFP / mCherry channels
and lets the user interactively tune the look of the exported MP4:

  * Navigate timepoints (slider) and stage positions (dropdown).
  * Toggle channel visibility (changes the export, not just the preview).
  * Set a background ROI for per-frame fluorescence background subtraction.
  * Define a crop region (the export is cropped, but the crop is non-destructive
    — the underlying ND2 data is never modified, so you can adjust it any time).
  * Adjust per-channel LUTs graphically: drag min/max handles on the channel
    histogram; drag the dot on the curve for gamma (above 1 brightens midtones).
  * Preview the exact 2x2 grid frame the export will produce, with a scale bar
    sized from the .nd2's spatial calibration.
  * Export videos, per-channel TIFF frames, strips and timeline figures.

Run:    python microscopy_timelapse.py [file.nd2]
        python microscopy_timelapse.py --version
"""
from __future__ import annotations

# The one place the version is set. It is shown in the window title, printed
# by --version and written to every settings file. CHANGELOG.md lists what
# changed in each version.
__version__ = "1.0"
APP_NAME = "Microscopy Timelapse Processor"

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Started by double-click on Windows (pythonw.exe) there is no console, so an
# error at start-up or inside a button's action would vanish. Send it to
# launch-log.txt next to this file instead.
if sys.stderr is None or sys.stdout is None:
    try:
        _log = open(Path(__file__).resolve().with_name("launch-log.txt"), "a",
                    encoding="utf-8", buffering=1)
        sys.stderr = sys.stderr or _log
        sys.stdout = sys.stdout or _log
    except OSError:
        pass

import cv2
import imageio.v2 as imageio
import nd2
import numpy as np
from PySide6.QtCore import (
    QEvent, QLocale, QObject, QPoint, QPointF, QRect, QRectF, QSettings, QSize, Qt, QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Modifier keys as the keyboard labels them (Qt maps Ctrl shortcuts to ⌘ on macOS).
_CMD_KEY = "⌘" if sys.platform == "darwin" else "Ctrl+"
_ALT_KEY = "⌥" if sys.platform == "darwin" else "Alt"

# Channel name lookup. .nd2 files store human-typed channel names; tolerate
# variants like "ph3", "Phase Contrast", "mCherry-Texas Red", etc.
_PHASE_TOKENS = ("ph", "phase", "bf", "brightfield", "trans", "dic")
_GFP_TOKENS = ("gfp", "egfp", "fitc")
_MCHERRY_TOKENS = ("mcherry", "mch", "rfp", "tritc", "txred", "txr", "cy3")

# Display colors / accent tints per channel.
_ACCENT_PHASE = (200, 200, 210)
_ACCENT_GFP = (95, 200, 95)
_ACCENT_MCHERRY = (230, 95, 95)


# ===========================================================================
# Unicode-aware text rendering via Pillow (OpenCV putText is ASCII-only)
# ===========================================================================

_PIL_FONT_CACHE: dict = {}


# Font files for the text drawn on exports, tried in order.
_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
    # Windows (which need not be installed on C:)
    str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"),
    str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"),
    # Linux: Debian/Ubuntu, Fedora, Arch, openSUSE
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
]


def _get_pil_font(size: int):
    """Return a cached Pillow font at `size` pixels."""
    from PIL import ImageFont
    if size not in _PIL_FONT_CACHE:
        font = None
        for p in _FONT_CANDIDATES:
            if Path(p).exists():
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if font is None:
            # No known font file: Pillow's built-in font, at the size asked
            # for where Pillow can scale it (10.1 and later).
            try:
                font = ImageFont.load_default(size)
            except TypeError:
                font = ImageFont.load_default()
        _PIL_FONT_CACHE[size] = font
    return _PIL_FONT_CACHE[size]


def _micrometre(font_size: int) -> str:
    """ "µm", or "um" when the font in use has no µ (Pillow's built-in font)."""
    font = _get_pil_font(font_size)
    try:
        missing = bytes(font.getmask("\uffff"))
        return "µm" if bytes(font.getmask("µ")) != missing else "um"
    except Exception:
        return "µm"


def _draw_text_at(draw, x: int, y: int, text: str, font, fill) -> None:
    """Draw `text` so that its ink starts exactly at (x, y).

    Pillow places the ink a few pixels below and right of the drawing origin
    (more for larger fonts), whereas _text_size_pil() measures the ink alone.
    """
    bb = draw.textbbox((0, 0), text, font=font)
    draw.text((x - bb[0], y - bb[1]), text, fill=fill, font=font)


def _text_size_pil(text: str, font_size: int) -> tuple:
    """Return (width, height) in pixels for `text` at the given PIL font size."""
    from PIL import Image, ImageDraw
    font = _get_pil_font(font_size)
    tmp = Image.new("RGB", (1, 1))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _put_text_pil(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int,
    color: tuple = (255, 255, 255),
    bg_alpha: float = 0.0,
) -> None:
    """Draw Unicode text at pixel position (x, y) top-left, in-place on frame."""
    from PIL import Image, ImageDraw
    font = _get_pil_font(font_size)
    tw, th = _text_size_pil(text, font_size)
    img = Image.fromarray(frame)
    if bg_alpha > 0.0:
        pad = 3
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img.width, x + tw + pad); y2 = min(img.height, y + th + pad)
        if x2 > x1 and y2 > y1:
            overlay = Image.new("RGB", img.size, (0, 0, 0))
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).rectangle((x1, y1, x2, y2), fill=int(bg_alpha * 255))
            img = Image.composite(overlay, img, mask)
    _draw_text_at(ImageDraw.Draw(img), x, y, text, font, color)
    frame[:] = np.asarray(img)


# ===========================================================================
# Theming / skinning engine
# ===========================================================================

@dataclass
class ThemeColors:
    """Custom-paint colors for QPainter-drawn widgets (HistogramCanvas, ROIView).

    Qt QSS only covers standard widgets; anything painted manually in
    paintEvent must read colors from this struct instead of hardcoding them.
    """
    hist_bg:        tuple  # HistogramCanvas background fill
    hist_axis:      tuple  # baseline axis line
    hist_handles:   tuple  # vmin/vmax vertical lines + triangle markers
    hist_curve:     tuple  # LUT transfer curve (RGBA)
    hist_dot_border: tuple # gamma midpoint dot outline
    hist_dot_fill:  tuple  # gamma midpoint dot interior
    hist_labels:    tuple  # axis tick-label text
    groupbox_border: str   # CSS color string for HistogramLUTWidget border
    roi_bg:         tuple  # ROIView background brush
    roi_bg_pen:     tuple  # BG ROI rectangle pen
    roi_crop_pen:   tuple  # Crop ROI rectangle pen


@dataclass
class Theme:
    name: str
    display: str
    qss: str
    colors: ThemeColors


# ---- theme definitions -------------------------------------------------------

_QSS_CLASSIC = """
QWidget          { background-color: #2A2A2A; color: #00DD00;
                   font-family: "Courier New", monospace; font-size: 11px; }
QMainWindow      { background-color: #2A2A2A; }
QPushButton      { background-color: #3A3A3A; color: #00FF00;
                   border: 2px solid;
                   border-top-color: #666666; border-left-color: #666666;
                   border-bottom-color: #111111; border-right-color: #111111;
                   padding: 4px 10px; min-height: 20px; }
QPushButton:pressed { border-top-color: #111111; border-left-color: #111111;
                      border-bottom-color: #666666; border-right-color: #666666;
                      background-color: #252525; }
QPushButton:hover    { background-color: #444444; }
QPushButton:disabled { color: #336633; border-color: #333333;
                       background-color: #252525; }
QGroupBox        { border: 1px solid #007700; border-radius: 0px;
                   margin-top: 8px; background-color: #252525; }
QGroupBox::title { subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: #00CC00; }
QComboBox        { background-color: #333333; border: 1px solid #555555;
                   color: #00DD00; padding: 2px 6px; }
QComboBox QAbstractItemView { background-color: #2A2A2A; border: 1px solid #555555;
                               color: #00DD00;
                               selection-background-color: #003300; }
QSpinBox, QDoubleSpinBox { background-color: #1A1A1A; border: 1px solid #555555;
                           color: #00DD00; padding: 2px; }
QSlider::groove:horizontal { background-color: #1A1A1A; height: 4px;
                              border: 1px solid #444444; }
QSlider::handle:horizontal { background-color: #00AA00; width: 12px;
                              margin: -4px 0px; border: 1px solid #008800; }
QLabel           { color: #00AA00; background-color: transparent; }
QProgressBar     { background-color: #1A1A1A; border: 1px solid #444444;
                   color: #00DD00; text-align: center; }
QProgressBar::chunk { background-color: #009900; }
QStatusBar       { background-color: #222222; color: #009900; }
QCheckBox        { color: #00CC00; background-color: transparent; }
QCheckBox::indicator { width: 13px; height: 13px;
                       background-color: #1A1A1A; border: 1px solid #555555; }
QCheckBox::indicator:checked { background-color: #00AA00; border-color: #009900; }
QLineEdit        { background-color: #1A1A1A; border: 1px solid #555555;
                   color: #00DD00; padding: 2px; }
QScrollBar:vertical   { background: #1A1A1A; width: 10px; }
QScrollBar::handle:vertical { background: #444444; min-height: 20px; }
"""

_QSS_Y2K = """
QWidget          { background-color: #C8CDD4; color: #001540;
                   font-family: Arial, sans-serif; font-size: 11px; }
QMainWindow      { background-color: #B8BDC4; }
QPushButton      { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                       stop:0 #EAEEF2, stop:0.5 #B8BEC6, stop:1 #A0A8B2);
                   color: #001040; border: 1px solid #7A8490;
                   border-radius: 6px; padding: 4px 12px; min-height: 20px; }
QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                          stop:0 #8090A0, stop:1 #C0C8D0); }
QPushButton:hover    { border-color: #0060FF; color: #0040CC; }
QPushButton:disabled { color: #8090A0; background: #C0C8D0; }
QGroupBox        { border: 1px solid #5070A0; border-radius: 4px;
                   margin-top: 8px; background-color: rgba(255,255,255,60); }
QGroupBox::title { subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: #0050BB; }
QComboBox        { background-color: #DDE2E8; border: 1px solid #7A8494;
                   color: #001040; padding: 2px 6px; border-radius: 3px; }
QComboBox QAbstractItemView { background-color: #DDE2E8; border: 1px solid #7A8494;
                               color: #001040;
                               selection-background-color: #0060FF;
                               selection-color: #FFFFFF; }
QSpinBox, QDoubleSpinBox { background-color: #E8ECF2; border: 1px solid #7A8494;
                           color: #001040; padding: 2px; }
QSlider::groove:horizontal { background-color: #8090A0; height: 3px;
                              border-radius: 2px; }
QSlider::handle:horizontal { background-color: #0070FF; width: 14px;
                              margin: -5px 0px; border-radius: 7px;
                              border: 1px solid #0050CC; }
QLabel           { color: #002080; background-color: transparent; }
QProgressBar     { background-color: #B0B8C4; border: 1px solid #7A8494;
                   border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #0060FF, stop:1 #00C0FF); border-radius: 3px; }
QStatusBar       { background-color: #A8B0BA; color: #002080; }
QCheckBox        { color: #001A60; background-color: transparent; }
QCheckBox::indicator { width: 13px; height: 13px;
                       background-color: #E0E6EE; border: 1px solid #7A8494;
                       border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #0060FF; border-color: #0040CC; }
QLineEdit        { background-color: #E8ECF2; border: 1px solid #7A8494;
                   color: #001040; padding: 2px; }
QScrollBar:vertical   { background: #B8BEC6; width: 10px; }
QScrollBar::handle:vertical { background: #8090A0; min-height: 20px;
                               border-radius: 5px; }
"""

_QSS_HIFI = """
QWidget          { background-color: #1C1C1C; color: #C8A848;
                   font-family: Helvetica, Arial, sans-serif; font-size: 11px; }
QMainWindow      { background-color: #141414; }
QPushButton      { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                       stop:0 #585858, stop:0.45 #383838, stop:1 #282828);
                   color: #C8A848; border: 1px solid #555555;
                   border-radius: 3px; padding: 5px 12px; min-height: 20px; }
QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                          stop:0 #181818, stop:1 #383838); }
QPushButton:hover    { border-color: #C8A848; }
QPushButton:disabled { color: #605030; border-color: #383830;
                       background: #202018; }
QGroupBox        { border: 1px solid #4A4030; border-radius: 3px;
                   margin-top: 8px; background-color: #181818; }
QGroupBox::title { subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: #B09040; }
QComboBox        { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                       stop:0 #383838, stop:1 #262626);
                   border: 1px solid #4A4030; color: #C8A848; padding: 2px 6px; }
QComboBox QAbstractItemView { background-color: #262626; border: 1px solid #4A4030;
                               color: #C8A848;
                               selection-background-color: #4A3820; }
QSpinBox, QDoubleSpinBox { background-color: #181818; border: 1px solid #4A4030;
                           color: #B09040; padding: 2px; }
QSlider::groove:horizontal { background-color: #282010; height: 6px;
                              border: 1px solid #4A3820; border-radius: 3px; }
QSlider::handle:horizontal { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                  stop:0 #C0A060, stop:1 #806040);
                              width: 16px; margin: -5px 0px;
                              border-radius: 8px; border: 1px solid #604020; }
QLabel           { color: #9A8040; background-color: transparent; }
QProgressBar     { background-color: #181818; border: 1px solid #4A4030;
                   border-radius: 3px; text-align: center; color: #C8A848; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #803800, stop:1 #D07800); }
QStatusBar       { background-color: #101010; color: #9A8040; }
QCheckBox        { color: #B09040; background-color: transparent; }
QCheckBox::indicator { width: 13px; height: 13px;
                       background-color: #181818; border: 1px solid #4A4030;
                       border-radius: 2px; }
QCheckBox::indicator:checked { background-color: #A07830; border-color: #805820; }
QLineEdit        { background-color: #181818; border: 1px solid #4A4030;
                   color: #C8A848; padding: 2px; }
QScrollBar:vertical   { background: #181818; width: 10px; }
QScrollBar::handle:vertical { background: #4A4030; min-height: 20px; }
"""

_QSS_OFFICE2K = """
/* ---- Office 2000 --------------------------------------------------------
   Battleship Gray (#D4D0C8) panels, white content areas, navy selection.
   No gradients. No rounded corners. Strict 1-px bevel borders.
   Buttons are flat (invisible border) until hovered.
   ----------------------------------------------------------------------- */
QWidget          { background-color: #D4D0C8; color: #000000;
                   font-family: Tahoma, "MS Sans Serif", Arial, sans-serif;
                   font-size: 11px; }
QMainWindow      { background-color: #D4D0C8; }

/* ---- Buttons: completely flat until hovered ---- */
QPushButton      { background-color: #D4D0C8; color: #000000;
                   border: 1px solid transparent;
                   border-radius: 0px; padding: 3px 10px; min-height: 20px; }
QPushButton:hover    { border-top-color: #FFFFFF;   border-left-color: #FFFFFF;
                       border-bottom-color: #808080; border-right-color: #808080; }
QPushButton:pressed  { border-top-color: #808080; border-left-color: #808080;
                       border-bottom-color: #FFFFFF; border-right-color: #FFFFFF;
                       background-color: #C8C4BC; }
QPushButton:disabled { color: #808080; border-color: transparent; }
QPushButton:default  { border-top-color: #FFFFFF;   border-left-color: #FFFFFF;
                       border-bottom-color: #808080; border-right-color: #808080; }

/* ---- Tool buttons (mode buttons etc.) same flat logic ---- */
QToolButton      { background-color: #D4D0C8; color: #000000;
                   border: 1px solid transparent; border-radius: 0px;
                   padding: 3px 8px; }
QToolButton:hover    { border-top-color: #FFFFFF;   border-left-color: #FFFFFF;
                       border-bottom-color: #808080; border-right-color: #808080; }
QToolButton:pressed  { border-top-color: #808080; border-left-color: #808080;
                       border-bottom-color: #FFFFFF; border-right-color: #FFFFFF; }
QToolButton:checked  { border-top-color: #808080; border-left-color: #808080;
                       border-bottom-color: #FFFFFF; border-right-color: #FFFFFF;
                       background-color: #C0BDB4; }

/* ---- GroupBoxes: thin 1-px gray border, square corners ---- */
QGroupBox        { border: 1px solid #808080; border-radius: 0px;
                   margin-top: 8px; background-color: #D4D0C8; }
QGroupBox::title { subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: #000000; font-weight: bold; }

/* ---- Dropdowns: flat gray, navy highlight ---- */
QComboBox        { background-color: #D4D0C8; color: #000000;
                   border: 1px solid #808080; border-radius: 0px;
                   padding: 2px 6px; }
QComboBox::drop-down { border-left: 1px solid #808080; width: 16px; }
QComboBox::down-arrow { width: 0; height: 0;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-top: 5px solid #000000; }
QComboBox QAbstractItemView { background-color: #D4D0C8; color: #000000;
                               border: 1px solid #808080; outline: none;
                               selection-background-color: #000080;
                               selection-color: #FFFFFF; }

/* ---- Input fields: white background, sunken bevel ---- */
QSpinBox, QDoubleSpinBox { background-color: #FFFFFF; color: #000000;
                           border-top-color: #808080; border-left-color: #808080;
                           border-bottom-color: #DFDFDF; border-right-color: #DFDFDF;
                           border-style: solid; border-width: 1px;
                           padding: 1px 2px; border-radius: 0px; }
QLineEdit        { background-color: #FFFFFF; color: #000000;
                   border-top-color: #808080; border-left-color: #808080;
                   border-bottom-color: #DFDFDF; border-right-color: #DFDFDF;
                   border-style: solid; border-width: 1px;
                   padding: 1px 2px; border-radius: 0px; }

/* ---- Slider: Win98-style flat track + raised thumb ---- */
QSlider::groove:horizontal { background-color: #C0C0C0; height: 4px;
                              border-top-color: #808080; border-left-color: #808080;
                              border-bottom-color: #FFFFFF; border-right-color: #FFFFFF;
                              border-style: solid; border-width: 1px; }
QSlider::handle:horizontal { background-color: #D4D0C8; width: 12px;
                              margin: -5px 0px;
                              border-top-color: #FFFFFF; border-left-color: #FFFFFF;
                              border-bottom-color: #808080; border-right-color: #808080;
                              border-style: solid; border-width: 1px; }

/* ---- Labels ---- */
QLabel           { color: #000000; background-color: transparent; }

/* ---- Progress bar: navy fill ---- */
QProgressBar     { background-color: #FFFFFF; color: #000000;
                   border: 1px solid #808080; border-radius: 0px;
                   text-align: center; }
QProgressBar::chunk { background-color: #000080; }

/* ---- Status bar: recessed look ---- */
QStatusBar       { background-color: #D4D0C8; color: #000000;
                   border-top: 1px solid #808080; }

/* ---- Checkboxes: white box, sunken ---- */
QCheckBox        { color: #000000; background-color: transparent; spacing: 4px; }
QCheckBox::indicator { width: 13px; height: 13px; background-color: #FFFFFF;
                       border-top-color: #808080; border-left-color: #808080;
                       border-bottom-color: #DFDFDF; border-right-color: #DFDFDF;
                       border-style: solid; border-width: 1px; }
QCheckBox::indicator:checked { background-color: #FFFFFF;
                                image: none; }

/* ---- Scrollbars: classic raised thumb ---- */
QScrollBar:vertical   { background: #D4D0C8; width: 16px; border: none; }
QScrollBar::handle:vertical { background-color: #D4D0C8; min-height: 20px;
                               border-top-color: #FFFFFF; border-left-color: #FFFFFF;
                               border-bottom-color: #808080; border-right-color: #808080;
                               border-style: solid; border-width: 1px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: #D4D0C8; height: 16px; border: none; }
QScrollBar::handle:horizontal { background-color: #D4D0C8; min-width: 20px;
                                 border-top-color: #FFFFFF; border-left-color: #FFFFFF;
                                 border-bottom-color: #808080; border-right-color: #808080;
                                 border-style: solid; border-width: 1px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

/* ---- Scroll area ---- */
QScrollArea      { border: 1px solid #808080; }

/* ---- Dialog / message box ---- */
QDialog          { background-color: #D4D0C8; }
QMessageBox      { background-color: #D4D0C8; }
"""

_QSS_GRUNGE = """
QWidget          { background-color: #1A1B14; color: #9EA890;
                   font-family: "Arial Black", Impact, sans-serif; font-size: 11px; }
QMainWindow      { background-color: #141510; }
QPushButton      { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                       stop:0 #484838, stop:1 #282818);
                   color: #C0C870; border: 2px solid #6A5A38;
                   border-radius: 0px; padding: 4px 10px; min-height: 20px; }
QPushButton:pressed  { background: #181808; border-color: #8A7A4A; }
QPushButton:hover    { border-color: #9A8A50; color: #E0E860; }
QPushButton:disabled { color: #585840; border-color: #3A3820;
                       background: #141410; }
QGroupBox        { border: 2px solid #504830; border-radius: 0px;
                   margin-top: 8px; background-color: #141410; }
QGroupBox::title { subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: #989068; }
QComboBox        { background-color: #282818; border: 2px solid #504830;
                   color: #9EA878; padding: 2px 6px; border-radius: 0px; }
QComboBox QAbstractItemView { background-color: #1A1A10; border: 2px solid #504830;
                               color: #9EA878;
                               selection-background-color: #3A3820; }
QSpinBox, QDoubleSpinBox { background-color: #181810; border: 2px solid #504830;
                           color: #9EA878; padding: 2px; }
QSlider::groove:horizontal { background-color: #383828; height: 6px;
                              border: 1px solid #504830; }
QSlider::handle:horizontal { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                  stop:0 #888050, stop:1 #484030);
                              width: 14px; margin: -4px 0px;
                              border: 1px solid #6A5A38; border-radius: 0px; }
QLabel           { color: #909080; background-color: transparent; }
QProgressBar     { background-color: #181810; border: 2px solid #504830;
                   border-radius: 0px; text-align: center; color: #9EA878; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #583020, stop:1 #A05828); }
QStatusBar       { background-color: #101008; color: #808070; }
QCheckBox        { color: #909080; background-color: transparent; }
QCheckBox::indicator { width: 13px; height: 13px;
                       background-color: #181810; border: 2px solid #504830; }
QCheckBox::indicator:checked { background-color: #707040; border-color: #504830; }
QLineEdit        { background-color: #181810; border: 2px solid #504830;
                   color: #9EA878; padding: 2px; }
QScrollBar:vertical   { background: #181810; width: 10px; }
QScrollBar::handle:vertical { background: #504830; min-height: 20px; }
"""

# ---- shadcn/ui-inspired themes -----------------------------------------------
# Design tokens taken from shadcn/ui's default "neutral" palette (Tailwind's
# neutral scale): a near-achromatic ramp, hairline 1px borders, generous corner
# radii and muted-foreground secondary text. shadcn itself is React + Tailwind,
# so none of its code is usable here — this is the visual language re-expressed
# in Qt stylesheets. Qt QSS has no transitions or box-shadow, so the motion and
# elevation parts of that design language are deliberately absent.

# No font-family is set on purpose. QSS treats even "sans-serif" as a literal
# family name rather than a CSS generic, so any value Qt cannot match costs an
# alias-resolution pass at startup and logs a warning — and Qt exposes neither
# "Inter" nor "SF Pro Text" nor "Segoe UI" on macOS. Omitting it leaves Qt on
# the platform UI font (SF on macOS, Segoe UI on Windows), which is both the
# native choice and the closest match to shadcn's Inter on each platform.


def _shadcn_qss(
    bg, fg, card, muted, muted_fg, border, primary, primary_fg, ring, destructive
) -> str:
    """Build a shadcn-style stylesheet from a token set (light or dark)."""
    return f"""
QWidget {{ background-color: {bg}; color: {fg}; font-size: 12px; }}
QMainWindow, QDialog {{ background-color: {bg}; }}

/* Buttons: outline/secondary by default, solid for the default action. */
QPushButton {{
    background-color: {card}; color: {fg};
    border: 1px solid {border}; border-radius: 8px;
    padding: 6px 14px; min-height: 18px; font-weight: 500;
}}
QPushButton:hover    {{ background-color: {muted}; }}
QPushButton:pressed  {{ background-color: {border}; }}
QPushButton:disabled {{ color: {muted_fg}; background-color: {bg};
                        border-color: {border}; }}
QPushButton:default  {{ background-color: {primary}; color: {primary_fg};
                        border-color: {primary}; }}
QPushButton:default:hover  {{ background-color: {muted_fg}; border-color: {muted_fg}; }}
/* Opt-in primary action: setProperty("accent", True) on the button. */
QPushButton[accent="true"] {{ background-color: {primary}; color: {primary_fg};
                              border-color: {primary}; }}
QPushButton[accent="true"]:hover    {{ background-color: {muted_fg};
                                       border-color: {muted_fg}; }}
QPushButton[accent="true"]:disabled {{ background-color: {muted};
                                       color: {muted_fg};
                                       border-color: {border}; }}
QToolButton {{ background-color: transparent; border: 1px solid transparent;
               border-radius: 8px; padding: 4px 8px; }}
QToolButton:hover   {{ background-color: {muted}; }}
QToolButton:pressed,
QToolButton:checked {{ background-color: {border}; }}

/* Cards */
QGroupBox {{
    background-color: {card}; border: 1px solid {border};
    border-radius: 10px; margin-top: 10px; padding-top: 10px; font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 4px; color: {muted_fg}; font-size: 11px;
}}

/* Inputs */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {card}; color: {fg};
    border: 1px solid {border}; border-radius: 8px;
    padding: 4px 8px; min-height: 18px; selection-background-color: {muted};
    selection-color: {fg};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border: 1px solid {ring};
}}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QLineEdit:disabled {{ color: {muted_fg}; background-color: {muted}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    width: 0; height: 0; margin-right: 7px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {muted_fg};
}}
QComboBox QAbstractItemView {{
    background-color: {card}; color: {fg};
    border: 1px solid {border}; border-radius: 8px;
    selection-background-color: {muted}; selection-color: {fg}; padding: 4px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: transparent; border: none; width: 14px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0; border-left: 3px solid transparent;
    border-right: 3px solid transparent; border-bottom: 4px solid {muted_fg};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0; border-left: 3px solid transparent;
    border-right: 3px solid transparent; border-top: 4px solid {muted_fg};
}}

QLabel {{ background-color: transparent; color: {fg}; }}

/* Checkbox */
QCheckBox {{ background-color: transparent; spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {border};
    border-radius: 4px; background-color: {card};
}}
QCheckBox::indicator:hover   {{ border-color: {ring}; }}
QCheckBox::indicator:checked {{ background-color: {primary};
                                border-color: {primary}; }}

/* Tabs: muted pill track, raised active tab */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 10px;
                    background-color: {card}; top: -1px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background-color: transparent; color: {muted_fg};
    border: 1px solid transparent; border-radius: 7px;
    padding: 5px 14px; margin: 2px; font-weight: 500;
}}
QTabBar::tab:hover    {{ color: {fg}; background-color: {muted}; }}
QTabBar::tab:selected {{ background-color: {card}; color: {fg};
                         border: 1px solid {border}; }}

/* Table */
QTableWidget, QTableView {{
    background-color: {card}; alternate-background-color: {muted};
    color: {fg}; gridline-color: {border};
    border: 1px solid {border}; border-radius: 10px;
    selection-background-color: {muted}; selection-color: {fg};
}}
QHeaderView::section {{
    background-color: {card}; color: {muted_fg};
    border: none; border-bottom: 1px solid {border};
    padding: 6px 8px; font-weight: 500;
}}
QTableCornerButton::section {{ background-color: {card};
                               border-bottom: 1px solid {border}; }}

/* Progress */
QProgressBar {{
    background-color: {muted}; border: none; border-radius: 5px;
    height: 8px; text-align: center; color: {muted_fg}; font-size: 10px;
}}
QProgressBar::chunk {{ background-color: {primary}; border-radius: 5px; }}

QStatusBar {{ background-color: {bg}; color: {muted_fg};
              border-top: 1px solid {border}; }}
QStatusBar::item {{ border: none; }}

/* Slider */
QSlider::groove:horizontal {{ background: {muted}; height: 5px;
                              border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {card}; border: 1px solid {border};
                              width: 15px; height: 15px; margin: -6px 0;
                              border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ border-color: {ring}; }}
QSlider::sub-page:horizontal {{ background: {primary}; height: 5px;
                                border-radius: 3px; }}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px;
                               min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {muted_fg}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {border}; border-radius: 5px;
                                 min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {muted_fg}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0;
                                              border: none; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QScrollArea {{ background-color: {bg}; border: none; }}
QToolTip {{ background-color: {primary}; color: {primary_fg};
            border: none; border-radius: 6px; padding: 5px 8px; }}
"""


_QSS_SHADCN = _shadcn_qss(
    bg="#FFFFFF", fg="#0A0A0A", card="#FFFFFF", muted="#F5F5F5",
    muted_fg="#737373", border="#E5E5E5", primary="#171717",
    primary_fg="#FAFAFA", ring="#A1A1A1", destructive="#DC2626",
)

_QSS_SHADCN_DARK = _shadcn_qss(
    bg="#0A0A0A", fg="#FAFAFA", card="#171717", muted="#262626",
    muted_fg="#A1A1A1", border="#2A2A2A", primary="#FAFAFA",
    primary_fg="#171717", ring="#737373", destructive="#EF4444",
)


THEMES: dict[str, Theme] = {
    "classic": Theme(
        name="classic", display="Classic Base",
        qss=_QSS_CLASSIC,
        colors=ThemeColors(
            hist_bg=(10, 10, 10),
            hist_axis=(0, 80, 0),
            hist_handles=(0, 255, 0),
            hist_curve=(0, 220, 0, 230),
            hist_dot_border=(0, 80, 0),
            hist_dot_fill=(0, 255, 0),
            hist_labels=(0, 170, 0),
            groupbox_border="#007700",
            roi_bg=(10, 10, 10),
            roi_bg_pen=(0, 255, 0),
            roi_crop_pen=(0, 180, 255),
        ),
    ),
    "y2k": Theme(
        name="y2k", display="Y2K Cyber",
        qss=_QSS_Y2K,
        colors=ThemeColors(
            hist_bg=(20, 28, 44),
            hist_axis=(48, 64, 96),
            hist_handles=(255, 128, 0),
            hist_curve=(0, 192, 255, 230),
            hist_dot_border=(0, 80, 160),
            hist_dot_fill=(192, 232, 255),
            hist_labels=(128, 160, 200),
            groupbox_border="#5070A0",
            roi_bg=(16, 22, 36),
            roi_bg_pen=(255, 128, 0),
            roi_crop_pen=(0, 192, 255),
        ),
    ),
    "hifi": Theme(
        name="hifi", display="Skeuomorphic Hi-Fi",
        qss=_QSS_HIFI,
        colors=ThemeColors(
            hist_bg=(12, 12, 12),
            hist_axis=(58, 48, 32),
            hist_handles=(212, 176, 72),
            hist_curve=(192, 128, 64, 230),
            hist_dot_border=(96, 64, 32),
            hist_dot_fill=(224, 192, 128),
            hist_labels=(128, 112, 64),
            groupbox_border="#4A4030",
            roi_bg=(8, 8, 8),
            roi_bg_pen=(212, 176, 72),
            roi_crop_pen=(192, 64, 0),
        ),
    ),
    "grunge": Theme(
        name="grunge", display="Grunge / Gaming",
        qss=_QSS_GRUNGE,
        colors=ThemeColors(
            hist_bg=(14, 14, 10),
            hist_axis=(58, 58, 42),
            hist_handles=(192, 200, 96),
            hist_curve=(128, 136, 80, 230),
            hist_dot_border=(74, 64, 32),
            hist_dot_fill=(160, 168, 112),
            hist_labels=(96, 104, 80),
            groupbox_border="#504830",
            roi_bg=(10, 10, 6),
            roi_bg_pen=(192, 200, 96),
            roi_crop_pen=(128, 160, 80),
        ),
    ),
    "shadcn": Theme(
        name="shadcn", display="Shadcn Light",
        qss=_QSS_SHADCN,
        colors=ThemeColors(
            hist_bg=(255, 255, 255),
            hist_axis=(229, 229, 229),
            hist_handles=(23, 23, 23),
            hist_curve=(115, 115, 115, 230),
            hist_dot_border=(23, 23, 23),
            hist_dot_fill=(250, 250, 250),
            hist_labels=(115, 115, 115),
            groupbox_border="#E5E5E5",
            roi_bg=(250, 250, 250),
            # ROI pen hues carry meaning (yellow = background, cyan = crop),
            # so they stay saturated even in an otherwise neutral palette.
            roi_bg_pen=(202, 138, 4),
            roi_crop_pen=(8, 145, 178),
        ),
    ),
    "shadcn_dark": Theme(
        name="shadcn_dark", display="Shadcn Dark",
        qss=_QSS_SHADCN_DARK,
        colors=ThemeColors(
            hist_bg=(10, 10, 10),
            hist_axis=(42, 42, 42),
            hist_handles=(250, 250, 250),
            hist_curve=(161, 161, 161, 230),
            hist_dot_border=(250, 250, 250),
            hist_dot_fill=(23, 23, 23),
            hist_labels=(161, 161, 161),
            groupbox_border="#2A2A2A",
            roi_bg=(10, 10, 10),
            roi_bg_pen=(234, 179, 8),
            roi_crop_pen=(34, 211, 238),
        ),
    ),
    "office2k": Theme(
        name="office2k", display="Office 2000",
        qss=_QSS_OFFICE2K,
        colors=ThemeColors(
            # White content area for the histogram canvas
            hist_bg=(255, 255, 255),
            hist_axis=(160, 160, 160),
            hist_handles=(0, 0, 128),       # Deep Navy
            hist_curve=(0, 0, 128, 200),    # Navy, semi-transparent
            hist_dot_border=(0, 0, 80),
            hist_dot_fill=(0, 0, 160),
            hist_labels=(0, 0, 0),
            groupbox_border="#808080",
            # ROI view: battleship gray background
            roi_bg=(212, 208, 200),         # #D4D0C8
            roi_bg_pen=(0, 0, 128),         # Navy BG ROI
            roi_crop_pen=(128, 0, 0),       # Dark red crop ROI
        ),
    ),
}


class ThemeManager(QObject):
    """Application-wide theme state.  Singleton; access via ThemeManager.instance()."""

    theme_changed = Signal(object)   # emits the new Theme dataclass

    _instance: Optional["ThemeManager"] = None

    def __init__(self) -> None:
        super().__init__()
        self._current: Theme = THEMES["y2k"]

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def current(self) -> Theme:
        return self._current

    def apply(self, name: str) -> None:
        theme = THEMES[name]
        self._current = theme
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.qss)
        self.theme_changed.emit(theme)


# ===========================================================================
# Channel LUT model (vmin, vmax, gamma)
# ===========================================================================

@dataclass
class ChannelLUT:
    """Display mapping for one channel.

    Pixel values are first clipped to [vmin, vmax] and linearly normalized to
    [0, 1], then a gamma curve is applied (out = x^(1/gamma): gamma > 1 brightens
    midtones), and finally scaled to uint8 [0, 255].
    """

    vmin: float
    vmax: float
    gamma: float = 1.0

    @staticmethod
    def auto_from(
        stack: np.ndarray, lo_pct: float = 0.1, hi_pct: float = 99.9
    ) -> "ChannelLUT":
        flat = stack.ravel()
        if flat.size > 5_000_000:
            flat = flat[:: flat.size // 5_000_000]
        lo, hi = np.percentile(flat, [lo_pct, hi_pct])
        if hi <= lo:
            hi = lo + 1.0
        return ChannelLUT(float(lo), float(hi), 1.0)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        span = self.vmax - self.vmin
        if span <= 0:
            span = 1.0
        x = (arr - self.vmin) * (1.0 / span)
        np.clip(x, 0.0, 1.0, out=x)
        if abs(self.gamma - 1.0) > 1e-6:
            # gamma > 1 brightens midtones (out = x^(1/gamma)).
            x = np.power(x, 1.0 / max(self.gamma, 1e-3), out=x)
        return (x * 255).astype(np.uint8)


# ===========================================================================
# Data model with per-position channel stacks
# ===========================================================================

@dataclass
class ND2Data:
    """Channel stacks shaped (P, T, Y, X), plus shared metadata. P=1 when the
    file has no position axis."""

    phase: np.ndarray
    gfp: np.ndarray
    mcherry: np.ndarray
    um_per_px: Optional[float]
    source_path: Path
    channel_names: tuple[str, str, str]
    frame_interval_s: Optional[float] = None
    # Absolute acquisition time of every frame, (P, T) seconds on a common
    # clock (Julian day × 86400), or None when the file has no timestamps.
    # Absolute rather than relative so appended files keep their real gap.
    abs_times_s: Optional[np.ndarray] = None
    # Empty for a single file. After append_nd2: one dict per source file
    # (file, path, start, n_frames, shift_xy) in frame order.
    segments: tuple = ()

    @property
    def n_positions(self) -> int: return int(self.phase.shape[0])
    @property
    def n_frames(self) -> int:    return int(self.phase.shape[1])
    @property
    def height(self) -> int:      return int(self.phase.shape[2])
    @property
    def width(self) -> int:       return int(self.phase.shape[3])

    def at(self, p: int, channel: str) -> np.ndarray:
        """Get one position's stack for a named channel as (T, Y, X)."""
        return {"phase": self.phase, "gfp": self.gfp, "mcherry": self.mcherry}[channel][p]


# Whole-image rotation applied on import. Values are degrees clockwise.
ROTATIONS = [("0°", 0), ("90° ↻", 90), ("180°", 180), ("90° ↺", 270)]

# Preview panel choices: (button label, view_combo label, panel key). None = 2×2 grid.
VIEW_CHOICES = [("All 4", "2×2 grid", None), ("Phase", "Phase", "phase"),
                ("GFP", "GFP", "gfp"), ("mCherry", "mCherry", "mcherry"),
                ("Merge", "Merge", "composite")]
PLAY_SPEEDS = (1, 2, 3, 5, 8, 10, 15, 20, 30)   # frames per second


def rot_k(deg_cw: int) -> int:
    """np.rot90 turn count (counter-clockwise) for a clockwise angle."""
    return (-int(deg_cw) // 90) % 4


def rotate_nd2(data: ND2Data, k: int) -> ND2Data:
    """Return a copy of `data` with every frame turned k×90° counter-clockwise.

    Contiguous copies, not rot90 views: cv2 rejects some non-contiguous arrays.
    """
    k %= 4
    if k == 0:
        return data
    from dataclasses import replace
    turn = lambda a: np.ascontiguousarray(np.rot90(a, k, axes=(2, 3)))

    # The alignment shift recorded for an appended file is a displacement in
    # image coordinates, so it turns with the image: one CCW turn maps
    # (dx, dy) to (dy, -dx).
    def turn_shift(v):
        dx, dy = v
        for _ in range(k):
            dx, dy = dy, -dx
        return [int(dx), int(dy)]

    segments = tuple(
        {**s, "shift_xy": None if s.get("shift_xy") is None
         else [turn_shift(v) for v in s["shift_xy"]]}
        for s in data.segments)
    return replace(data, phase=turn(data.phase), gfp=turn(data.gfp),
                   mcherry=turn(data.mcherry), segments=segments)


def rotate_roi(roi, k: int, w: int, h: int):
    """Map an (x, y, w, h) box on a w×h image through k×90° CCW turns, so it
    still frames the same pixels after rotate_nd2(…, k)."""
    if roi is None:
        return None
    x, y, rw, rh = roi
    for _ in range(k % 4):
        # One CCW turn: pixel (x, y) -> (y, W-1-x); the image becomes h×w.
        x, y, rw, rh = y, w - x - rw, rh, rw
        w, h = h, w
    return (int(x), int(y), int(rw), int(rh))


# ---------------------------------------------------------------------------
# Appending a continuation file
# ---------------------------------------------------------------------------

def measure_join_shift(before: np.ndarray, after: np.ndarray
                       ) -> tuple[int, int, float]:
    """Integer (dx, dy) by which `after` is displaced relative to `before`,
    plus the phase-correlation peak (≈1 = confident, < ~0.2 = unreliable).

    Each file of a continued run is drift-corrected on its own, so the stage
    offset between the two survives into the "Aligned" files. Integer only:
    a sub-pixel shift would interpolate, and every other pixel in this app
    stays a raw camera value.
    """
    a = before.astype(np.float32); b = after.astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return int(round(dx)), int(round(dy)), float(resp)


def shift_frames(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move image content by (dx, dy) px on the last two axes, zero-filled.
    Zero is the same fill NIS uses for registration padding in *aligned*
    files, so downstream code already treats it as "no data"."""
    if dx == 0 and dy == 0:
        return arr
    out = np.zeros_like(arr)
    H, W = arr.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = arr[..., ys, xs]
    return out


def _segments_of(d: ND2Data) -> list[dict]:
    if d.segments:
        return [dict(s) for s in d.segments]
    return [{"file": d.source_path.name, "path": str(d.source_path),
             "start": 0, "n_frames": d.n_frames, "shift_xy": None}]


def append_nd2(a: ND2Data, b: ND2Data,
               shifts: Optional[list] = None) -> ND2Data:
    """`a` followed by `b` along T. `shifts` is one (dx, dy) per position:
    the displacement measured for `b`, which is undone so `b` lines up with
    `a`. Shapes, positions and pixel size must already have been checked."""
    from dataclasses import replace
    shifts = shifts or [(0, 0)] * b.n_positions

    def join(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        xb = np.stack([shift_frames(xb[p], -dx, -dy)
                       for p, (dx, dy) in enumerate(shifts)])
        return np.concatenate([xa, xb], axis=1)

    segs = _segments_of(a)
    for s in _segments_of(b):
        s["start"] += a.n_frames
        if s["shift_xy"] is None:
            s["shift_xy"] = [list(v) for v in shifts]
        segs.append(s)
    times = (np.concatenate([a.abs_times_s, b.abs_times_s], axis=1)
             if a.abs_times_s is not None and b.abs_times_s is not None else None)
    return replace(a, phase=join(a.phase, b.phase), gfp=join(a.gfp, b.gfp),
                   mcherry=join(a.mcherry, b.mcherry),
                   abs_times_s=times, segments=tuple(segs))


# ===========================================================================
# ND2 loader
# ===========================================================================

class ND2Loader:
    """Open an .nd2 file, identify the three channels, extract pixel size."""

    @staticmethod
    def load(path: str | Path) -> ND2Data:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with nd2.ND2File(str(path)) as f:
            arr = f.asarray()
            axes = "".join(f.sizes.keys())
            channel_names = ND2Loader._safe_channel_names(f)
            um_per_px = ND2Loader._safe_um_per_px(f)
            frame_interval_s = ND2Loader._safe_frame_interval_s(f)
            abs_times_s = ND2Loader._safe_abs_times_s(
                f, int(f.sizes.get("P", 1)), int(f.sizes.get("T", 1)))

        # Z-stack: take max projection per (T, C, P) so downstream is always 2D.
        if "Z" in axes:
            arr = arr.max(axis=axes.index("Z"))
            axes = axes.replace("Z", "")
        # nd2 leaves out axes of length 1, so a single-channel file has no C.
        if "C" not in axes:
            arr = arr[np.newaxis, ...]; axes = "C" + axes
        if not all(a in axes for a in "CYX"):
            raise ValueError(f"Unexpected ND2 axis layout: {axes}")
        # Inject T and P if missing, then transpose into (P, T, C, Y, X).
        if "T" not in axes:
            arr = arr[np.newaxis, ...]; axes = "T" + axes
        if "P" not in axes:
            arr = arr[np.newaxis, ...]; axes = "P" + axes
        order = [axes.index(a) for a in "PTCYX"]
        arr = np.transpose(arr, order)

        if arr.shape[2] < 1:
            raise ValueError("ND2 file has no image channels.")
        ph_i, gfp_i, mch_i = ND2Loader._match_channels(channel_names, arr.shape[2])

        def _ch(idx: Optional[int], ref_idx: int) -> np.ndarray:
            if idx is not None:
                return arr[:, :, idx]
            return np.zeros_like(arr[:, :, ref_idx])

        def _name(idx: Optional[int], default: str) -> str:
            if idx is not None and idx < len(channel_names):
                return channel_names[idx]
            return default

        ref = ph_i if ph_i is not None else 0
        return ND2Data(
            phase=_ch(ph_i, ref),
            gfp=_ch(gfp_i, ref),
            mcherry=_ch(mch_i, ref),
            um_per_px=um_per_px,
            source_path=path,
            channel_names=(
                _name(ph_i, "Phase"),
                _name(gfp_i, "GFP"),
                _name(mch_i, "mCherry"),
            ),
            frame_interval_s=frame_interval_s,
            abs_times_s=abs_times_s,
        )

    @staticmethod
    def _safe_abs_times_s(f: "nd2.ND2File", n_pos: int, n_frames: int
                          ) -> Optional[np.ndarray]:
        """Per-frame acquisition times, (P, T) seconds on an absolute clock.

        events() gives "Time [s]" relative to the file's own start; frame 0's
        TimeStamp gives that start as a Julian day, which is what lets two
        files be placed on one time axis. Files extracted from a multipoint
        run carry extra events with no T/P index, so those are skipped.
        """
        try:
            ts = f.frame_metadata(0).channels[0].time
            base = float(ts.absoluteJulianDayNumber) * 86400.0 \
                - float(ts.relativeTimeMs) / 1000.0
            out = np.full((n_pos, n_frames), np.nan, dtype=np.float64)
            for e in f.events():
                ti, t = e.get("T Index"), e.get("Time [s]")
                if ti is None or t is None or not (0 <= int(ti) < n_frames):
                    continue
                pi = int(e.get("P Index") or 0)
                if 0 <= pi < n_pos:
                    out[pi, int(ti)] = base + float(t)
            for pi in range(n_pos):
                ok = np.isfinite(out[pi])
                if ok.sum() == 0:
                    return None
                if not ok.all():
                    if ok.sum() < 2:
                        return None
                    out[pi, ~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0],
                                             out[pi, ok])
            return out
        except Exception:
            return None

    @staticmethod
    def _safe_channel_names(f: "nd2.ND2File") -> list[str]:
        try:    return [c.channel.name for c in f.metadata.channels]
        except Exception:  return []

    @staticmethod
    def _safe_um_per_px(f: "nd2.ND2File") -> Optional[float]:
        # An uncalibrated file reports a pixel size of exactly 1.0, which would
        # be drawn as a scale bar of 1 px per µm. Treat it as "no scale".
        try:
            if not all(f.metadata.channels[0].volume.axesCalibrated[:2]):
                return None
        except (AttributeError, IndexError, TypeError):
            pass
        try:
            v = f.voxel_size()
            return float(v.x) if v.x > 0 else None
        except Exception:
            return None

    @staticmethod
    def _safe_frame_interval_s(f: "nd2.ND2File") -> Optional[float]:
        try:
            for loop in f.experiment:
                params = getattr(loop, "parameters", None)
                if params is None:
                    continue
                period_ms = getattr(params, "periodMs", None)
                if period_ms is not None and float(period_ms) > 0:
                    return float(period_ms) / 1000.0
            return None
        except Exception:
            return None

    @staticmethod
    def _match_channels(
        names: list[str], n_channels: int
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Return (phase_idx, gfp_idx, mcherry_idx).

        Named channels are matched by token first.  Any channel not matched
        by name is treated as a spare and assigned to the first role that
        still needs one.  If no spare is available the role gets None and
        a synthetic zero array will be used instead.

        Example: 2-channel file with ['ph3', 'mCherry'] → (0, None, 1).
        """
        lower = [n.lower() for n in names]

        def find(tokens) -> Optional[int]:
            for i, name in enumerate(lower):
                if any(t in name for t in tokens):
                    return i
            return None

        picks = [find(_PHASE_TOKENS), find(_GFP_TOKENS), find(_MCHERRY_TOKENS)]

        # Channels claimed by a name-based match must not be reused as fallbacks.
        named: set[int] = {p for p in picks if p is not None}
        spare = [i for i in range(n_channels) if i not in named]

        used: set[int] = set()
        out: list[Optional[int]] = []
        for picked in picks:
            if picked is not None and picked not in used:
                used.add(picked)
                out.append(picked)
            elif spare:
                idx = spare.pop(0)
                used.add(idx)
                out.append(idx)
            else:
                out.append(None)
        return (out[0], out[1], out[2])


# ===========================================================================
# Processing pipeline
# ===========================================================================

@dataclass
class ChannelVisibility:
    phase: bool = True
    gfp: bool = True
    mcherry: bool = True

    def __iter__(self):
        return iter((self.phase, self.gfp, self.mcherry))


# How the fluorescence channels combine in the merged (composite) panel.
# Additive is the Nikon NIS-Elements look: overlapping GFP + mCherry read as
# yellow. The layered modes paint one channel *over* the other instead, so
# where the top channel is displayed it hides the one beneath (no yellow).
MERGE_MODES: list[tuple[str, str]] = [
    ("Additive (yellow overlap)", "additive"),
    ("mCherry over GFP",          "mch_over_gfp"),
    ("GFP over mCherry",          "gfp_over_mch"),
]

# In a layered merge the top channel becomes fully opaque at this display
# value (0-255, i.e. after its LUT) and fades in linearly below it, so dim
# edges and residual noise don't punch hard-edged holes in the lower layer.
LAYER_OPAQUE_AT = 128.0


def merge_channels(
    phase: np.ndarray, gfp: np.ndarray, mch: np.ndarray, mode: str = "additive",
) -> np.ndarray:
    """Merge LUT-mapped uint8 channels into one (H, W, 3) uint8 RGB panel.

    Phase is always a gray underlay added to every channel. GFP adds to G and
    mCherry to R; in a layered mode the lower fluor channel is attenuated by
    the top channel's opacity before it is added. int16 intermediates let the
    sum exceed 255 before clipping.
    """
    ph16 = phase.astype(np.int16)
    g16 = gfp.astype(np.int16)
    m16 = mch.astype(np.int16)
    if mode == "mch_over_gfp":
        alpha = np.clip(mch.astype(np.float32) / LAYER_OPAQUE_AT, 0.0, 1.0)
        g16 = np.rint(gfp * (1.0 - alpha)).astype(np.int16)
    elif mode == "gfp_over_mch":
        alpha = np.clip(gfp.astype(np.float32) / LAYER_OPAQUE_AT, 0.0, 1.0)
        m16 = np.rint(mch * (1.0 - alpha)).astype(np.int16)
    elif mode != "additive":
        raise ValueError(f"unknown merge mode {mode!r}")
    rgb = np.empty((*phase.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(ph16 + m16, 0, 255)   # R = phase + mCherry
    rgb[..., 1] = np.clip(ph16 + g16, 0, 255)   # G = phase + GFP
    rgb[..., 2] = phase                         # B = phase only
    return rgb


class Pipeline:
    """Build per-timepoint 2x2 grid frames with scale bar.

    Honors per-channel LUTs (gamma included), channel visibility (a channel
    that's hidden contributes black to its panel and to the composite), and an
    optional crop region (applied to each panel before the grid is composed).
    """

    @staticmethod
    def process(
        data: ND2Data,
        position: int,
        bg_roi: Optional[tuple[int, int, int, int]],
        crop_roi: Optional[tuple[int, int, int, int]],
        visibility: ChannelVisibility,
        phase_lut: Optional[ChannelLUT] = None,
        gfp_lut: Optional[ChannelLUT] = None,
        mcherry_lut: Optional[ChannelLUT] = None,
        phase_luts: Optional[list[ChannelLUT]] = None,
        gfp_luts: Optional[list[ChannelLUT]] = None,
        mcherry_luts: Optional[list[ChannelLUT]] = None,
        scale_bar_um: float = 10.0,
        scale_bar_visible: bool = True,
        scale_bar_position: str = "bottom_right",
        scale_bar_x_offset: int = 20,
        scale_bar_y_offset: int = 20,
        scale_bar_text_center: bool = True,
        scale_bar_text_above: bool = True,
        timer_enabled: bool = False,
        timer_format: str = "frame",
        timer_interval_s: float = 60.0,
        timer_t0: int = 0,
        timer_times_s: Optional[np.ndarray] = None,
        timer_position: str = "top_left",
        timer_x_offset: int = 20,
        timer_y_offset: int = 20,
        timer_decimals: int = 0,
        channel_labels: Optional[dict] = None,
        channel_label_position: str = "top_left",
        label_x_offset: int = 8,
        label_y_offset: int = 8,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        progress: Optional[callable] = None,
        merge_mode: str = "additive",         # see MERGE_MODES
    ) -> list[np.ndarray]:
        """Render the timelapse to a list of (2H, 2W, 3) uint8 frames.

        Per-frame LUT lists (`*_luts`) take priority over the single-LUT args
        (`*_lut`). Pass a list of length T to vary the LUT across timepoints
        (Smart or Manual per-frame mode); pass a single LUT (or None for auto)
        for Stack mode.
        """
        T = data.n_frames

        # Slice to the requested frame range FIRST so that background
        # subtraction and LUT auto-normalisation are computed only from
        # the selected (in-focus) frames, not the full stack.
        t_start = max(0, frame_start)
        t_end   = min(T - 1, frame_end if frame_end is not None else T - 1)
        n_export = t_end - t_start + 1

        phase_f = data.at(position, "phase")[t_start:t_end + 1].astype(np.float32)
        gfp_f   = data.at(position, "gfp"  )[t_start:t_end + 1].astype(np.float32)
        mch_f   = data.at(position, "mcherry")[t_start:t_end + 1].astype(np.float32)

        # Per-frame ROI-mean background subtraction on fluorescence only.
        if bg_roi is not None:
            x, y, w, h = bg_roi
            for i in range(n_export):
                gfp_f[i] -= float(gfp_f[i, y:y + h, x:x + w].mean())
                mch_f[i] -= float(mch_f[i, y:y + h, x:x + w].mean())

        # Slice per-frame LUT lists to the selected range.
        phase_luts_s   = phase_luts[t_start:t_end + 1]   if phase_luts   is not None else None
        gfp_luts_s     = gfp_luts[t_start:t_end + 1]     if gfp_luts     is not None else None
        mcherry_luts_s = mcherry_luts[t_start:t_end + 1] if mcherry_luts is not None else None

        phase_u8 = Pipeline._apply_luts(phase_f, phase_lut, phase_luts_s)
        gfp_u8   = Pipeline._apply_luts(gfp_f,   gfp_lut,   gfp_luts_s)
        mch_u8   = Pipeline._apply_luts(mch_f,   mcherry_lut, mcherry_luts_s)

        # Hidden channels contribute black to their panel and to the composite.
        if not visibility.phase:   phase_u8[:] = 0
        if not visibility.gfp:     gfp_u8[:]   = 0
        if not visibility.mcherry: mch_u8[:]   = 0

        um_per_px = data.um_per_px
        frames: list[np.ndarray] = []
        for i in range(n_export):
            t_abs = t_start + i   # absolute 0-based index (used by timer)
            ph, gp, mc = phase_u8[i], gfp_u8[i], mch_u8[i]
            if crop_roi is not None:
                cx, cy, cw, ch = crop_roi
                ph = ph[cy:cy + ch, cx:cx + cw]
                gp = gp[cy:cy + ch, cx:cx + cw]
                mc = mc[cy:cy + ch, cx:cx + cw]
            grid = Pipeline._make_grid(ph, gp, mc, merge_mode)
            if channel_labels:
                Pipeline._draw_channel_labels(
                    grid, channel_labels, channel_label_position,
                    x_offset=label_x_offset, y_offset=label_y_offset,
                )
            if scale_bar_visible and um_per_px:
                Pipeline._draw_scale_bar(
                    grid, um_per_px, scale_bar_um, scale_bar_position,
                    x_offset=scale_bar_x_offset, y_offset=scale_bar_y_offset,
                    text_center=scale_bar_text_center, text_above=scale_bar_text_above,
                )
            if timer_enabled:
                Pipeline._draw_timer(
                    grid, t_abs, timer_format, timer_interval_s, timer_position,
                    x_offset=timer_x_offset, y_offset=timer_y_offset,
                    decimals=timer_decimals, t0=timer_t0, times_s=timer_times_s,
                )
            frames.append(grid)
            if progress is not None:
                progress(i + 1, n_export)
        return frames

    @staticmethod
    def _apply_luts(
        stack: np.ndarray,
        single: Optional[ChannelLUT],
        per_frame: Optional[list[ChannelLUT]],
    ) -> np.ndarray:
        """Return a uint8 (T, Y, X) stack.

        Resolution order: per_frame > single > auto-computed-from-stack.
        """
        T = stack.shape[0]
        if per_frame is not None:
            if len(per_frame) != T:
                raise ValueError(
                    f"per-frame LUT list has {len(per_frame)} entries, expected {T}"
                )
            out = np.empty_like(stack, dtype=np.uint8)
            for t in range(T):
                out[t] = per_frame[t].apply(stack[t])
            return out
        lut = single or ChannelLUT.auto_from(stack)
        return lut.apply(stack)

    @staticmethod
    def _make_grid(
        phase: np.ndarray, gfp: np.ndarray, mch: np.ndarray,
        merge_mode: str = "additive",
    ) -> np.ndarray:
        """Build a (2H, 2W, 3) uint8 panel with Nikon-style pseudocolor.

        Per-panel:
          * Phase   -> gray             (R=G=B=phase)
          * GFP     -> green tint       (R=0, G=gfp, B=0)
          * mCherry -> red tint         (R=mch, G=0, B=0)
          * Composite -> merge_channels(): phase is a gray underlay; GFP and
            mCherry either add (default, overlap reads yellow) or one is
            layered over the other (see MERGE_MODES).
        """
        zeros = np.zeros_like(phase)
        phase_panel = np.stack([phase, phase, phase], axis=-1)
        gfp_panel   = np.stack([zeros, gfp, zeros],   axis=-1)
        mch_panel   = np.stack([mch, zeros, zeros],   axis=-1)
        composite = merge_channels(phase, gfp, mch, merge_mode)

        top    = np.concatenate([phase_panel, gfp_panel], axis=1)
        bottom = np.concatenate([mch_panel,   composite], axis=1)
        return np.concatenate([top, bottom], axis=0)

    @staticmethod
    def _draw_scale_bar(
        frame: np.ndarray,
        um_per_px: float,
        length_um: float,
        position: str = "bottom_right",
        x_offset: int = 20,
        y_offset: int = 20,
        text_center: bool = True,
        text_above: bool = True,
    ) -> None:
        h, w = frame.shape[:2]
        length_px = max(2, int(round(length_um / um_per_px)))
        if length_px > w:
            # The bar does not fit. A shortened bar would still carry the full
            # length in its label, so draw nothing; the export actions check
            # this first and tell the user.
            return
        thickness = max(3, h // 250)
        gap = 4 + thickness // 2     # clear space between the label and the bar
        font_size = max(14, h // 50)
        label = f"{length_um:g} {_micrometre(font_size)}"
        tw, th = _text_size_pil(label, font_size)

        # Horizontal bar extents. The bar is exactly length_px pixels
        # (x1 .. x2-1) because it is traced as the figure's scale reference.
        # Never shorten it to honour the offset: on a small crop that drew a
        # bar shorter than its label. Shrink the margin instead.
        x_off = max(0, min(x_offset, w - length_px))
        if "right" in position:
            x2 = w - x_off
            x1 = x2 - length_px
        else:
            x1 = x_off
            x2 = x1 + length_px

        # Text horizontal: centered over bar, or flush with bar end
        if text_center:
            bar_cx = (x1 + x2) // 2
            tx = max(0, min(w - tw, bar_cx - tw // 2))
        else:
            tx = (x2 - tw) if "right" in position else x1

        # Vertical positions for bar and text
        if "bottom" in position:
            y_bar = h - y_offset
            ty = (y_bar - th - gap) if text_above else (y_bar + thickness + gap)
        else:
            if text_above:
                ty = y_offset
                y_bar = ty + th + gap
            else:
                y_bar = y_offset
                ty = y_bar + thickness + gap

        # A filled rectangle, not an anti-aliased line: cv2.line adds round caps
        # that made the bar ~thickness px longer than length_px.
        y0 = max(0, min(h - thickness, y_bar - thickness // 2))
        cv2.rectangle(frame, (x1, y0), (x2 - 1, y0 + thickness - 1),
                      color=(255, 255, 255), thickness=-1)
        _put_text_pil(frame, label, tx, ty, font_size)

    @staticmethod
    def _format_time(t: int, fmt: str, interval_s: float, decimals: int = 0,
                     t0: int = 0, times_s: Optional[np.ndarray] = None) -> str:
        """Return a timestamp string for frame index `t` (0-based).

        t0: the frame index that represents time zero. Frames before t0 will
        show negative times. For the "frame" format t0 shifts the displayed
        frame number so that frame t0 is labelled "Frame 1".
        times_s: optional per-frame seconds (any origin). When given, elapsed
        time is times_s[t] - times_s[t0] instead of (t - t0) * interval_s —
        needed for appended files, whose join is not one interval long.
        """
        rel = t - t0
        if times_s is not None and 0 <= t < len(times_s) and 0 <= t0 < len(times_s):
            secs = float(times_s[t] - times_s[t0])
        else:
            secs = rel * interval_s
        if fmt == "frame":
            return f"Frame {rel + 1}"
        elif fmt == "seconds":
            return f"t = {secs:.{decimals}f} s"
        elif fmt == "minutes":
            return f"t = {secs / 60.0:.{decimals}f} min"
        else:  # hms
            total_s = int(round(secs))
            sign = "-" if total_s < 0 else ""
            total_s = abs(total_s)
            h_val, rem = divmod(total_s, 3600)
            m_val, s_val = divmod(rem, 60)
            return f"{sign}{h_val}:{m_val:02d}:{s_val:02d}"

    @staticmethod
    def _draw_timer(
        frame: np.ndarray,
        t: int,
        fmt: str,
        interval_s: float,
        position: str = "top_left",
        x_offset: int = 20,
        y_offset: int = 20,
        decimals: int = 0,
        t0: int = 0,
        times_s: Optional[np.ndarray] = None,
    ) -> None:
        """Draw a timestamp on frame `t` (0-based absolute index).

        fmt: "frame" | "seconds" | "minutes" | "hms"
        x_offset / y_offset: pixels from the chosen corner edge.
        t0: frame index treated as time zero.
        """
        label = Pipeline._format_time(t, fmt, interval_s, decimals, t0, times_s)
        fh, fw = frame.shape[:2]
        font_size = max(14, fh // 50)
        tw, th = _text_size_pil(label, font_size)

        tx = (fw - x_offset - tw) if "right" in position else x_offset
        ty = (fh - y_offset - th) if "bottom" in position else y_offset

        _put_text_pil(frame, label, tx, ty, font_size, bg_alpha=0.6)

    @staticmethod
    def _draw_channel_labels(
        grid: np.ndarray,
        labels: dict,
        position: str = "top_left",
        x_offset: int = 8,
        y_offset: int = 8,
    ) -> None:
        """Overlay a text label in a corner of each 2×2 panel."""
        gh, gw = grid.shape[:2]
        ph, pw = gh // 2, gw // 2
        font_size = max(14, ph // 25)
        panel_origins = {
            "phase":     (0,  0),
            "gfp":       (0,  pw),
            "mcherry":   (ph, 0),
            "composite": (ph, pw),
        }
        for key, (oy, ox) in panel_origins.items():
            text = labels.get(key, "")
            if not text:
                continue
            tw, th = _text_size_pil(text, font_size)
            tx = (ox + pw - tw - x_offset) if "right" in position else (ox + x_offset)
            ty = (oy + ph - th - y_offset) if "bottom" in position else (oy + y_offset)
            _put_text_pil(grid, text, tx, ty, font_size, bg_alpha=0.55)


# ===========================================================================
# Video exporter
# ===========================================================================

class VideoExporter:
    """Encode a list of uint8 RGB frames to .mp4 (libx264, 15 fps default)."""

    @staticmethod
    def write(
        frames: list[np.ndarray], out_path: str | Path,
        fps: int = 15,
        crf: int = 1,
        pixelformat: str = "yuv444p",
        progress: Optional[callable] = None,
        upscale: int = 1,
    ) -> Path:
        """`upscale`: whole-number nearest-neighbour enlargement (0 = auto, see
        resolve_upscale). 4:2:0 needs even sizes, so odd frames get a 1 px edge."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise ValueError("No frames to write.")
        # Remove an earlier file of this name first: if the encoder fails
        # without an error, the old video must not pass for the new one.
        if out_path.is_file():
            out_path.unlink()
        f = resolve_upscale(upscale, frames[0].shape[0], frames[0].shape[1])
        if f > 1:
            frames = [np.repeat(np.repeat(fr, f, axis=0), f, axis=1) for fr in frames]
        if pixelformat.startswith("yuv420"):
            h, w = frames[0].shape[:2]
            if h % 2 or w % 2:
                frames = [np.pad(fr, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
                          for fr in frames]
        params = ["-crf", str(crf), "-preset", "slow"]
        if frames[0].shape[0] > 576:
            # ffmpeg converts RGB to YUV with the BT.601 matrix and writes no
            # colour tags, but players read untagged video taller than 576 px
            # as BT.709, which shifts the colours (green about 15 % dimmer).
            # Convert with BT.709 and record it in the file.
            params += ["-vf", "scale=out_color_matrix=bt709", "-colorspace", "bt709",
                       "-color_primaries", "bt709", "-color_trc", "bt709"]
        else:
            params += ["-colorspace", "smpte170m"]     # BT.601, now stated in the file
        writer = imageio.get_writer(
            str(out_path), format="FFMPEG", mode="I", fps=fps,
            codec="libx264", macro_block_size=1, pixelformat=pixelformat,
            output_params=params,
        )
        try:
            for i, frame in enumerate(frames):
                writer.append_data(frame)
                if progress is not None:
                    progress(i + 1, len(frames))
        finally:
            writer.close()
        # ffmpeg can fail without an exception reaching us (the frames go into
        # a pipe, and its exit code is not read), so check what it wrote.
        try:
            import imageio_ffmpeg
            written, _secs = imageio_ffmpeg.count_frames_and_secs(str(out_path))
        except Exception as exc:
            raise IOError(f"The video encoder did not write a readable file:\n{out_path}") from exc
        if written != len(frames):
            raise IOError(f"The video is incomplete: {written} of {len(frames)} frames "
                          f"were written to\n{out_path}")
        return out_path


# Video size choices: (label, factor). 0 = auto. Enlarging uses nearest-
# neighbour (each pixel becomes an N×N block) so small crops stay sharp: a
# 94 px video is otherwise smoothed by every player and by Premiere.
VIDEO_SCALES: list[tuple[str, int]] = [
    ("Native (1×)", 1), ("2× sharp", 2), ("4× sharp", 4), ("8× sharp", 8),
    ("Auto — sharp, ≥ 720 px tall", 0),
]


def resolve_upscale(factor: int, h: int, w: int) -> int:
    """Turn a VIDEO_SCALES factor into a whole number ≥ 1 (0 = auto)."""
    if factor and factor > 0:
        return int(factor)
    f = max(1, math.ceil(720 / max(1, h)))
    return max(1, min(f, 16, 3840 // max(1, w), 2160 // max(1, h) or 1))


def _make_video_scale_combo(default: int = 1) -> QComboBox:
    cb = QComboBox()
    for lbl, f in VIDEO_SCALES:
        cb.addItem(lbl, f)
    cb.setCurrentIndex([f for _l, f in VIDEO_SCALES].index(default))
    cb.setToolTip(
        "Enlarge the video with sharp, blocky pixels (nearest neighbour).\n"
        "Small crops otherwise look soft, because players and Premiere smooth\n"
        "them when they scale them up. TIFF frames are never enlarged.")
    return cb


# Panels a custom strip can contain: (label, key) and the folder-name word.
STRIP_CHOICES: list[tuple[str, Optional[str]]] = [
    ("—", None), ("Phase", "phase"), ("GFP", "gfp"), ("mCherry", "mcherry"),
    ("Merge", "merge"), ("Merge (no mCherry)", "merge_no_mch"),
]
STRIP_NAMES = {"phase": "Phase", "gfp": "GFP", "mcherry": "mCherry",
               "merge": "Merge", "merge_no_mch": "Merge no mCherry"}


def compose_strip(panels: list[np.ndarray], horizontal: bool = True,
                  gap: int = 0, bg: int = 255) -> np.ndarray:
    """Join equal-size RGB panels into one strip with `gap` px of `bg` between."""
    if not panels:
        raise ValueError("No panels to compose.")
    h, w = panels[0].shape[:2]
    n = len(panels)
    if horizontal:
        out = np.full((h, n * w + (n - 1) * gap, 3), bg, np.uint8)
        for i, p in enumerate(panels):
            out[:, i * (w + gap): i * (w + gap) + w] = p
    else:
        out = np.full((n * h + (n - 1) * gap, w, 3), bg, np.uint8)
        for i, p in enumerate(panels):
            out[i * (h + gap): i * (h + gap) + h] = p
    return out


# ===========================================================================
# Timeline figure exporter
# ===========================================================================

def _parse_frame_list(text: str, max_frame: int) -> list[int]:
    """Parse a frame-selection string into sorted 0-based indices.

    Syntax:
      "1, 5, 10"       → frames 1, 5, 10 (1-indexed → 0-based)
      "1-10"           → frames 1 through 10
      "1-55:5"         → every 5th frame from 1 to 55

    A range typed backwards ("10-1") or with an en dash is read as intended.
    """
    result: set[int] = set()
    for token in text.replace(" ", "").replace("–", "-").replace("—", "-").split(","):
        if not token:
            continue
        # Range with optional step: start-end or start-end:step
        if "-" in token or ":" in token:
            try:
                step = 1
                if ":" in token:
                    range_part, step_part = token.rsplit(":", 1)
                    step = max(1, int(step_part))
                else:
                    range_part = token
                start_s, end_s = range_part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    start, end = end, start
                result.update(range(start - 1, end, step))
            except (ValueError, TypeError):
                pass
        else:
            try:
                result.add(int(token) - 1)
            except ValueError:
                pass
    return sorted(i for i in result if 0 <= i < max_frame)


class TimelineExporter:
    """Build a publication-quality timeline figure.

    Layout (all regions optional):
    ┌──────────┬──col0──┬─spacing─┬──col1──┐
    │  (blank) │ t-lbl0 │         │ t-lbl1 │  ← time_label_h rows
    ├──────────┼────────┤         ├────────┤
    │ ch-lbl 0 │ panel  │         │ panel  │  ← panel_h rows
    ├──────────┼────────┤         ├────────┤
    │  spacing │        │         │        │
    ├──────────┼────────┤         ├────────┤
    │ ch-lbl 1 │ panel  │         │ panel  │  ← panel_h rows
    └──────────┴────────┴─────────┴────────┘
    """

    # Pseudo-colors matching the main 2×2 grid
    _CHANNEL_COLOR: dict[str, tuple] = {
        "phase":   (200, 200, 210),
        "gfp":     (60,  210,  60),
        "mcherry": (230,  60,  60),
    }

    @staticmethod
    def _colorize(img_u8: np.ndarray, channel: str, colorize: bool) -> np.ndarray:
        """Convert (H, W) uint8 → (H, W, 3) uint8 RGB panel."""
        zeros = np.zeros_like(img_u8)
        if not colorize or channel == "phase":
            return np.stack([img_u8, img_u8, img_u8], axis=-1)
        if channel == "gfp":
            return np.stack([zeros, img_u8, zeros], axis=-1)
        # mcherry
        return np.stack([img_u8, zeros, zeros], axis=-1)

    @staticmethod
    def _make_composite(ph: np.ndarray, gfp: np.ndarray, mch: np.ndarray,
                        merge_mode: str = "additive") -> np.ndarray:
        """Build the merged RGB panel, identical to the 2×2 grid's composite."""
        return merge_channels(ph, gfp, mch, merge_mode)

    @staticmethod
    def render_panels(
        data: "ND2Data",
        position: int,
        bg_roi: Optional[tuple],
        crop_roi: Optional[tuple],
        frame_indices: list[int],
        channels: list[str],
        luts: dict,
        colorize: bool,
        merge_mode: str = "additive",
        visibility: Optional["ChannelVisibility"] = None,
    ) -> dict[str, list[np.ndarray]]:
        """Return {channel: [RGB_panel, ...]} for each requested frame.

        'composite' is a valid channel key — it is computed from the three
        base channels and does not require a separate LUT entry. Each entry
        of `luts` is one ChannelLUT, or a list with one ChannelLUT per frame
        of the stack. A channel switched off in `visibility` is left out of
        the composite, as in the preview.

        The pixels are those Pipeline.process() produces: background
        subtraction is the box mean per frame with nothing clipped before the
        display range is applied.
        """
        result: dict[str, list[np.ndarray]] = {ch: [] for ch in channels}

        # Always load the three base channels; they are needed for composite
        # even if not explicitly listed.
        base_needed = {"phase", "gfp", "mcherry"}
        raw: dict[str, np.ndarray] = {}
        for ch in base_needed:
            stack = data.at(position, ch).astype(np.float32)
            if bg_roi is not None and ch != "phase":
                rx, ry, rw, rh = bg_roi
                for t in range(stack.shape[0]):
                    patch = stack[t, ry:ry + rh, rx:rx + rw]
                    if patch.size:
                        stack[t] -= float(patch.mean())
            raw[ch] = stack

        def _crop(arr):
            if crop_roi is not None:
                cx, cy, cw, ch_ = crop_roi
                return arr[cy:cy + ch_, cx:cx + cw]
            return arr

        def _lut(ch: str, fi: int) -> "ChannelLUT":
            # One LUT for the whole stack, or a list with one per frame
            # (Smart / Manual per-frame modes).
            lut = luts[ch]
            return lut[fi] if isinstance(lut, (list, tuple)) else lut

        for fi in frame_indices:
            ph_u8  = _crop(_lut("phase", fi).apply(raw["phase"][fi]))
            gfp_u8 = _crop(_lut("gfp", fi).apply(raw["gfp"][fi]))
            mch_u8 = _crop(_lut("mcherry", fi).apply(raw["mcherry"][fi]))

            for ch in channels:
                if ch == "composite":
                    vis = visibility or ChannelVisibility()
                    off = np.zeros_like(ph_u8)
                    panel = TimelineExporter._make_composite(
                        ph_u8 if vis.phase else off, gfp_u8 if vis.gfp else off,
                        mch_u8 if vis.mcherry else off, merge_mode)
                elif ch == "phase":
                    panel = TimelineExporter._colorize(ph_u8, "phase", colorize)
                elif ch == "gfp":
                    panel = TimelineExporter._colorize(gfp_u8, "gfp", colorize)
                else:  # mcherry
                    panel = TimelineExporter._colorize(mch_u8, "mcherry", colorize)
                result[ch].append(panel)

        return result

    @staticmethod
    def compose(
        panels: dict[str, list[np.ndarray]],
        channels: list[str],
        channel_names: dict[str, str],
        time_labels: Optional[list[str]],
        scale_bar_info: Optional[dict],    # {um_per_px, length_um, row, col}
        spacing: int,
        bg_white: bool,
        ch_label_font: int,
        time_label_font: int,
    ) -> np.ndarray:
        """Compose the full figure from pre-rendered RGB panels."""
        from PIL import Image, ImageDraw

        if not channels or not panels[channels[0]]:
            raise ValueError("No panels to compose.")

        n_cols = len(panels[channels[0]])
        n_rows = len(channels)
        ph, pw = panels[channels[0]][0].shape[:2]

        fg_color = (0, 0, 0) if bg_white else (255, 255, 255)
        bg_rgb = (255, 255, 255) if bg_white else (0, 0, 0)

        # Measure label areas
        ch_label_w = 0
        if channel_names:
            for ch in channels:
                name = channel_names.get(ch, ch)
                tw, _ = _text_size_pil(name, ch_label_font)
                ch_label_w = max(ch_label_w, tw + 16)

        time_label_h = 0
        if time_labels:
            _, th = _text_size_pil("Frame 0", time_label_font)
            time_label_h = th + 12

        total_w = ch_label_w + n_cols * pw + max(0, n_cols - 1) * spacing
        total_h = time_label_h + n_rows * ph + max(0, n_rows - 1) * spacing

        img = Image.new("RGB", (total_w, total_h), bg_rgb)
        draw = ImageDraw.Draw(img)
        font_ch  = _get_pil_font(ch_label_font)
        font_t   = _get_pil_font(time_label_font)

        # Draw time labels (centered above each column)
        if time_labels:
            for ci, lbl in enumerate(time_labels):
                col_x = ch_label_w + ci * (pw + spacing)
                tw, th = _text_size_pil(lbl, time_label_font)
                tx = col_x + (pw - tw) // 2
                ty = (time_label_h - th) // 2
                draw.text((tx, ty), lbl, fill=fg_color, font=font_t)

        # Draw channel labels (vertically centred in each row)
        if channel_names:
            for ri, ch in enumerate(channels):
                name = channel_names.get(ch, ch)
                _, th = _text_size_pil(name, ch_label_font)
                row_y = time_label_h + ri * (ph + spacing)
                ty = row_y + (ph - th) // 2
                draw.text((8, ty), name, fill=fg_color, font=font_ch)

        # Paste panels
        for ri, ch in enumerate(channels):
            for ci, panel_arr in enumerate(panels[ch]):
                x0 = ch_label_w + ci * (pw + spacing)
                y0 = time_label_h + ri * (ph + spacing)
                img.paste(Image.fromarray(panel_arr), (x0, y0))

        # Draw scale bar on the specified panel
        if scale_bar_info:
            sb_row = scale_bar_info.get("row", n_rows - 1)
            sb_col = scale_bar_info.get("col", n_cols - 1)
            sb_row = max(0, min(n_rows - 1, sb_row))
            sb_col = max(0, min(n_cols - 1, sb_col))
            px0 = ch_label_w + sb_col * (pw + spacing)
            py0 = time_label_h + sb_row * (ph + spacing)
            panel_np = np.asarray(img)[py0:py0 + ph, px0:px0 + pw].copy()
            um_per_px = scale_bar_info["um_per_px"]
            length_um = scale_bar_info["length_um"]
            # Use 20px offset from panel edges
            Pipeline._draw_scale_bar(
                panel_np, um_per_px, length_um,
                position="bottom_right", x_offset=12, y_offset=12,
                text_center=True, text_above=True,
            )
            img_arr = np.asarray(img).copy()
            img_arr[py0:py0 + ph, px0:px0 + pw] = panel_np
            img = Image.fromarray(img_arr)

        return np.asarray(img).copy()

    @staticmethod
    def save(
        figure: np.ndarray,
        out_path: str | Path,
        fmt: str = "png",
        dpi: int = 300,
    ) -> None:
        """Save the composed figure as PNG or TIFF."""
        from PIL import Image
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(figure)
        if fmt.lower() in ("tiff", "tif"):
            img.save(str(out_path), dpi=(dpi, dpi))
        else:
            img.save(str(out_path), dpi=(dpi, dpi))


# ===========================================================================
# Histogram canvas: custom-painted with draggable vmin/vmax handles
# ===========================================================================

class HistogramCanvas(QWidget):
    """NIS-Elements-style LUT editor.

    Shows the channel histogram with two draggable handles (vmin/vmax) and a
    midpoint dot on the LUT line. Drag a handle horizontally to move it; drag
    the midpoint dot vertically for non-linear (gamma) scaling; drag between
    the handles to shift the whole window without changing its width.
    """

    range_changed = Signal(float, float)  # vmin, vmax in data units
    gamma_changed = Signal(float)         # gamma
    HANDLE_GRAB_PX = 8
    DOT_GRAB_PX = 10
    HEIGHT = 140
    AXIS_PAD = 14    # bottom space reserved for min/max tick labels
    PLOT_TOP = 4     # top inset of the plot area inside the canvas

    def __init__(self, accent: tuple[int, int, int], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._accent = accent
        self._tc: ThemeColors = ThemeManager.instance().current().colors
        self._lo = 0.0
        self._hi = 1.0
        self._vmin = 0.0
        self._vmax = 1.0
        self._gamma = 1.0
        self._counts = np.zeros(256, dtype=np.float64)
        self._dragging: Optional[str] = None
        self._drag_anchor_x: int = 0
        self._drag_anchor_vmin: float = 0.0
        self._drag_anchor_vmax: float = 1.0

    def apply_theme(self, theme: Theme) -> None:
        self._tc = theme.colors
        self.update()

    # ---- public API -------------------------------------------------------

    def set_data(self, counts: np.ndarray, lo: float, hi: float) -> None:
        self._counts = counts.astype(np.float64)
        self._lo, self._hi = float(lo), float(hi)
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0
        self.update()

    def set_range(self, vmin: float, vmax: float, gamma: float) -> None:
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._gamma = float(gamma)
        self.update()

    # ---- painting ---------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        plot_top, plot_bottom = self.PLOT_TOP, h - self.AXIS_PAD
        plot_h = plot_bottom - plot_top

        # Background and axis baseline.
        p.fillRect(self.rect(), QColor(*self._tc.hist_bg))
        p.setPen(QPen(QColor(*self._tc.hist_axis), 1))
        p.drawLine(0, plot_bottom, w, plot_bottom)

        # Histogram bars: log-scaled, mostly opaque so they read clearly even
        # on the dark background.
        if self._counts.size > 0 and self._counts.max() > 0:
            log_counts = np.log1p(self._counts)
            scale = plot_h / float(log_counts.max())
            n = self._counts.size
            p.setPen(Qt.PenStyle.NoPen)
            r, g, b = self._accent
            p.setBrush(QColor(r, g, b, 235))
            for i in range(n):
                x0 = int(i * w / n)
                x1 = int((i + 1) * w / n)
                bar_h = int(log_counts[i] * scale)
                if bar_h > 0:
                    p.drawRect(x0, plot_bottom - bar_h, max(1, x1 - x0), bar_h)

        x_min = self._value_to_x(self._vmin)
        x_max = self._value_to_x(self._vmax)

        # vmin / vmax handle lines extend through the plot area only.
        p.setPen(QPen(QColor(*self._tc.hist_handles), 2))
        p.drawLine(x_min, plot_top, x_min, plot_bottom)
        p.drawLine(x_max, plot_top, x_max, plot_bottom)

        # LUT curve from (vmin, plot_bottom) to (vmax, plot_top), bent by gamma.
        # This is the visualized transfer function applied to the data.
        dot_x: Optional[float] = None
        dot_y: Optional[float] = None
        if x_max > x_min + 1:
            p.setPen(QPen(QColor(*self._tc.hist_curve), 2))
            pts = []
            for i in range(96):
                t = i / 95.0
                out = t ** (1.0 / max(self._gamma, 1e-3))
                cx = x_min + t * (x_max - x_min)
                cy = plot_bottom - out * plot_h
                pts.append((cx, cy))
            for j in range(len(pts) - 1):
                p.drawLine(int(pts[j][0]), int(pts[j][1]),
                           int(pts[j + 1][0]), int(pts[j + 1][1]))
            # Midpoint dot — same point where the curve crosses x = (vmin+vmax)/2.
            mid_out = 0.5 ** (1.0 / max(self._gamma, 1e-3))
            dot_x = (x_min + x_max) / 2.0
            dot_y = plot_bottom - mid_out * plot_h

        # Handle markers (small triangles below the plot area).
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(*self._tc.hist_handles))
        for x in (x_min, x_max):
            p.drawPolygon(QPolygon([
                QPoint(x - 5, plot_bottom + 8),
                QPoint(x + 5, plot_bottom + 8),
                QPoint(x, plot_bottom),
            ]))

        # Midpoint gamma dot — drawn on top of everything so it's grabbable.
        if dot_x is not None:
            p.setPen(QPen(QColor(*self._tc.hist_dot_border), 1))
            p.setBrush(QColor(*self._tc.hist_dot_fill))
            p.drawEllipse(QPointF(dot_x, dot_y), 5.5, 5.5)

        # Axis tick labels at the data extents.
        p.setPen(QPen(QColor(*self._tc.hist_labels), 1))
        font = p.font(); font.setPointSize(7); p.setFont(font)
        p.drawText(2, h - 1, f"{self._lo:.0f}")
        hi_text = f"{self._hi:.0f}"
        text_w = p.fontMetrics().horizontalAdvance(hi_text)
        p.drawText(w - text_w - 2, h - 1, hi_text)
        p.end()

    # ---- geometry helpers --------------------------------------------------

    def _plot_bounds(self) -> tuple[int, int]:
        """Return (plot_top, plot_bottom) y-coords inside this widget."""
        return self.PLOT_TOP, self.height() - self.AXIS_PAD

    def _gamma_dot(self) -> tuple[float, float]:
        """Current (x, y) of the gamma midpoint dot in widget coords."""
        x_min = self._value_to_x(self._vmin)
        x_max = self._value_to_x(self._vmax)
        plot_top, plot_bottom = self._plot_bounds()
        out_mid = 0.5 ** (1.0 / max(self._gamma, 1e-3))
        return ((x_min + x_max) / 2.0, plot_bottom - out_mid * (plot_bottom - plot_top))

    # ---- mouse interaction ------------------------------------------------

    def mousePressEvent(self, event) -> None:
        x = int(event.position().x())
        y = int(event.position().y())
        x_min = self._value_to_x(self._vmin)
        x_max = self._value_to_x(self._vmax)
        dot_x, dot_y = self._gamma_dot()
        # Order of priority:
        #   1. min / max handle (close to the vertical line in X)
        #   2. gamma midpoint dot (close in both X and Y)
        #   3. interior of the window -> drag whole window
        #   4. exterior -> snap nearest handle
        if abs(x - x_min) <= self.HANDLE_GRAB_PX:
            self._dragging = "min"
        elif abs(x - x_max) <= self.HANDLE_GRAB_PX:
            self._dragging = "max"
        elif (abs(x - dot_x) <= self.DOT_GRAB_PX
              and abs(y - dot_y) <= self.DOT_GRAB_PX):
            self._dragging = "gamma"
        elif x_min < x < x_max:
            self._dragging = "both"
            self._drag_anchor_x = x
            self._drag_anchor_vmin = self._vmin
            self._drag_anchor_vmax = self._vmax
        else:
            self._dragging = "min" if abs(x - x_min) < abs(x - x_max) else "max"
            self._set_handle_from_x(x)

    def mouseMoveEvent(self, event) -> None:
        x = int(event.position().x())
        y = int(event.position().y())
        if self._dragging is None:
            x_min = self._value_to_x(self._vmin)
            x_max = self._value_to_x(self._vmax)
            dot_x, dot_y = self._gamma_dot()
            on_handle = (abs(x - x_min) <= self.HANDLE_GRAB_PX
                         or abs(x - x_max) <= self.HANDLE_GRAB_PX)
            on_dot = (abs(x - dot_x) <= self.DOT_GRAB_PX
                      and abs(y - dot_y) <= self.DOT_GRAB_PX)
            if on_handle:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif on_dot:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._dragging == "gamma":
            self._set_gamma_from_y(y)
        elif self._dragging == "both":
            delta_v = self._x_to_value(x) - self._x_to_value(self._drag_anchor_x)
            self._vmin = self._drag_anchor_vmin + delta_v
            self._vmax = self._drag_anchor_vmax + delta_v
            self.range_changed.emit(self._vmin, self._vmax)
        else:
            self._set_handle_from_x(x)
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = None

    def _set_handle_from_x(self, x: int) -> None:
        v = self._x_to_value(x)
        min_span = 1e-3 * (self._hi - self._lo)
        if self._dragging == "min":
            self._vmin = min(v, self._vmax - min_span)
        elif self._dragging == "max":
            self._vmax = max(v, self._vmin + min_span)
        self.range_changed.emit(self._vmin, self._vmax)

    def _set_gamma_from_y(self, y: int) -> None:
        """Drag the midpoint dot up -> brighter midtones (gamma > 1),
        drag it down -> darker midtones (gamma < 1).

        Math: the LUT curve passes through (mid_x, out_mid) where
        out_mid = 0.5 ** (1/gamma). Solving for gamma:
            gamma = log(0.5) / log(out_mid)
        Out_mid is clamped to (0.02, 0.98) to keep gamma in a sane range.
        """
        plot_top, plot_bottom = self._plot_bounds()
        plot_h = plot_bottom - plot_top
        y_clamped = max(plot_top, min(plot_bottom, y))
        out_mid = (plot_bottom - y_clamped) / plot_h
        out_mid = max(0.02, min(0.98, out_mid))
        gamma = math.log(0.5) / math.log(out_mid)
        self._gamma = max(0.1, min(10.0, gamma))
        self.gamma_changed.emit(self._gamma)

    # ---- coordinate maps --------------------------------------------------

    def _value_to_x(self, v: float) -> int:
        t = (v - self._lo) / (self._hi - self._lo)
        return int(np.clip(t, 0.0, 1.0) * self.width())

    def _x_to_value(self, x: float) -> float:
        t = float(np.clip(x / max(1, self.width()), 0.0, 1.0))
        return self._lo + t * (self._hi - self._lo)


# ===========================================================================
# Per-channel LUT widget: histogram + gamma slider + Auto button
# ===========================================================================

class HistogramLUTWidget(QGroupBox):
    """NIS-Elements-style LUT panel for one channel.

    UI:
      * Histogram canvas with draggable vmin/vmax handles and a draggable
        gamma midpoint dot.
      * Editable min and max numeric fields (click and type a value).
      * Read-only gamma readout.
      * Auto button to reset to the robust default.
    """

    changed = Signal()

    def __init__(self, title: str, accent: tuple[int, int, int]):
        super().__init__(title)
        self._accent = accent
        self._auto_lut: Optional[ChannelLUT] = None

        self.canvas = HistogramCanvas(accent)
        self.canvas.range_changed.connect(self._on_canvas_range)
        self.canvas.gamma_changed.connect(self._on_canvas_gamma)

        self.min_field = QLineEdit("0")
        self.max_field = QLineEdit("0")
        # The fields are written and read with a decimal point, so validate
        # them the same way whatever the system's regional format; otherwise
        # "500.0" is read back as 5000 where the decimal sign is a comma.
        c_locale = QLocale.c()
        c_locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
        for fld in (self.min_field, self.max_field):
            validator = QDoubleValidator(fld)
            validator.setLocale(c_locale)
            fld.setValidator(validator)
            fld.setMaximumWidth(90)
            fld.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.min_field.editingFinished.connect(self._on_min_typed)
        self.max_field.editingFinished.connect(self._on_max_typed)

        self.gamma_label = QLabel("γ: 1.00")
        self.gamma_label.setToolTip(
            "Drag the dot on the LUT curve up to brighten midtones, "
            "down to darken them."
        )
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setFixedWidth(60)
        self.auto_btn.clicked.connect(self.reset_to_auto)

        self._apply_groupbox_style(ThemeManager.instance().current().colors.groupbox_border)

        body = QVBoxLayout(self)
        body.setContentsMargins(8, 14, 8, 8)
        body.setSpacing(6)
        body.addWidget(self.canvas)

        row = QHBoxLayout()
        row.addWidget(QLabel("min"))
        row.addWidget(self.min_field)
        row.addStretch(1)
        row.addWidget(self.gamma_label)
        row.addStretch(1)
        row.addWidget(QLabel("max"))
        row.addWidget(self.max_field)
        row.addWidget(self.auto_btn)
        body.addLayout(row)
        self.setEnabled(False)

    # ---- theming ----------------------------------------------------------

    def _apply_groupbox_style(self, border_color: str) -> None:
        rgb = f"rgb({self._accent[0]}, {self._accent[1]}, {self._accent[2]})"
        self.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {rgb}; "
            f"border: 1px solid {border_color}; border-radius: 4px; margin-top: 8px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}"
        )

    def apply_theme(self, theme: Theme) -> None:
        self._apply_groupbox_style(theme.colors.groupbox_border)
        self.canvas.apply_theme(theme)

    # ---- configuration ----------------------------------------------------

    def configure(
        self,
        stack: np.ndarray,
        auto_lut: ChannelLUT,
        data_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """Bind this widget to a (T, Y, X) (or larger) data stack.

        Precomputes a strided-subsample histogram of `stack` so the canvas has
        something to draw, and seeds vmin/vmax/gamma from `auto_lut`.
        """
        if data_range is None:
            lo = float(stack.min())
            hi = float(stack.max())
        else:
            lo, hi = float(data_range[0]), float(data_range[1])
        if hi <= lo:
            hi = lo + 1.0
        flat = stack.ravel()
        if flat.size > 5_000_000:
            flat = flat[:: flat.size // 5_000_000]
        counts, _ = np.histogram(flat, bins=256, range=(lo, hi))
        self.canvas.set_data(counts, lo, hi)

        self._auto_lut = auto_lut
        self.canvas.set_range(auto_lut.vmin, auto_lut.vmax, auto_lut.gamma)
        self._refresh_fields(auto_lut.vmin, auto_lut.vmax, auto_lut.gamma)
        self.setEnabled(True)
        self.changed.emit()

    # ---- public API -------------------------------------------------------

    def lut(self) -> ChannelLUT:
        return ChannelLUT(
            vmin=self.canvas._vmin,
            vmax=self.canvas._vmax,
            gamma=self.canvas._gamma,
        )

    def set_lut(self, lut: ChannelLUT) -> None:
        """Show `lut` in the canvas and the fields, without emitting `changed`."""
        self.canvas.set_range(lut.vmin, lut.vmax, lut.gamma)
        self._refresh_fields(lut.vmin, lut.vmax, lut.gamma)

    def is_hand_set(self) -> bool:
        """True if the range differs from the automatic one last given to
        configure(), i.e. the user has dragged a handle or typed a value."""
        return self._auto_lut is not None and self.lut() != self._auto_lut

    def reset_to_auto(self) -> None:
        if self._auto_lut is None:
            return
        self.canvas.set_range(
            self._auto_lut.vmin, self._auto_lut.vmax, self._auto_lut.gamma
        )
        self._refresh_fields(
            self._auto_lut.vmin, self._auto_lut.vmax, self._auto_lut.gamma
        )
        self.changed.emit()

    # ---- internal slots ---------------------------------------------------

    def _on_canvas_range(self, vmin: float, vmax: float) -> None:
        # Canvas already has the new vmin/vmax internally — just sync labels
        # and re-emit. (We don't call canvas.set_range again or we'd loop.)
        self._refresh_fields(vmin, vmax, self.canvas._gamma)
        self.changed.emit()

    def _on_canvas_gamma(self, gamma: float) -> None:
        self._refresh_fields(self.canvas._vmin, self.canvas._vmax, gamma)
        self.changed.emit()

    def _on_min_typed(self) -> None:
        try:
            v = float(self.min_field.text())
        except ValueError:
            # Show the value still in use rather than text that was not applied.
            self._refresh_fields(self.canvas._vmin, self.canvas._vmax, self.canvas._gamma)
            return
        v = min(v, self.canvas._vmax - 1e-6 * (self.canvas._hi - self.canvas._lo))
        self.canvas.set_range(v, self.canvas._vmax, self.canvas._gamma)
        self.canvas.range_changed.emit(v, self.canvas._vmax)
        self._refresh_fields(v, self.canvas._vmax, self.canvas._gamma)
        self.changed.emit()

    def _on_max_typed(self) -> None:
        try:
            v = float(self.max_field.text())
        except ValueError:
            self._refresh_fields(self.canvas._vmin, self.canvas._vmax, self.canvas._gamma)
            return
        v = max(v, self.canvas._vmin + 1e-6 * (self.canvas._hi - self.canvas._lo))
        self.canvas.set_range(self.canvas._vmin, v, self.canvas._gamma)
        self.canvas.range_changed.emit(self.canvas._vmin, v)
        self._refresh_fields(self.canvas._vmin, v, self.canvas._gamma)
        self.changed.emit()

    def _refresh_fields(self, vmin: float, vmax: float, gamma: float) -> None:
        # blockSignals to avoid recursive editingFinished firing from setText.
        for fld, val in ((self.min_field, vmin), (self.max_field, vmax)):
            fld.blockSignals(True)
            fld.setText(f"{val:.1f}")
            fld.blockSignals(False)
        self.gamma_label.setText(f"γ: {gamma:.2f}")


# ===========================================================================
# ROI / preview canvas
# ===========================================================================

# Interaction modes for the preview canvas.
MODE_BG = "bg"
MODE_CROP = "crop"
MODE_PREVIEW = "preview"

# Window layouts offered in the toolbar's Layout dropdown: (label, key).
LAYOUTS: list[tuple[str, str]] = [
    ("Wide monitor", "wide"),
    ("Laptop (side panel)", "laptop"),
    ("Image focus", "focus"),
]

# LUT modes.
LUT_STACK = "stack"
LUT_SMART = "smart"
LUT_MANUAL = "manual"


class FlowLayout(QLayout):
    """A left-to-right row that wraps onto extra lines when it runs out of width.

    Used for control rows so the same widgets fit a wide monitor (one line) and
    a narrow side panel (several lines). A QLabel is glued to the item after it,
    so "Label: [control]" pairs never split across lines. addStretch() is a
    no-op and addSpacing() adds a fixed gap that is dropped at a line start,
    which makes this a drop-in replacement for a QHBoxLayout row.
    """

    def __init__(self, parent=None, hspacing: int = 6, vspacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h = hspacing
        self._v = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    # QLayout API
    def addItem(self, item) -> None:
        self._items.append(item)

    def addSpacing(self, w: int) -> None:
        self.addItem(QSpacerItem(int(w), 0, QSizePolicy.Policy.Fixed,
                                 QSizePolicy.Policy.Minimum))

    def addStretch(self, *_args) -> None:
        pass

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        chunks = self._chunks()
        if not chunks:
            return QSize(0, 0)
        w = sum(c[1] for c in chunks) + self._h * (len(chunks) - 1)
        h = max(c[2] for c in chunks)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QSize:
        chunks = self._chunks()
        if not chunks:
            return QSize(0, 0)
        m = self.contentsMargins()
        return QSize(max(c[1] for c in chunks) + m.left() + m.right(),
                     max(c[2] for c in chunks) + m.top() + m.bottom())

    # internals
    def _chunks(self) -> list:
        """[(items, width, height, is_spacer)] with each QLabel glued to its successor."""
        vis = [it for it in self._items
               if it.spacerItem() is not None or not it.isEmpty()]
        out, i = [], 0
        while i < len(vis):
            group = [vis[i]]
            wdg = vis[i].widget()
            if isinstance(wdg, QLabel) and i + 1 < len(vis) and vis[i + 1].spacerItem() is None:
                group.append(vis[i + 1]); i += 1
            i += 1
            sizes = [g.sizeHint() for g in group]
            w = sum(sz.width() for sz in sizes) + 4 * (len(group) - 1)
            h = max(sz.height() for sz in sizes)
            out.append((group, w, h, group[0].spacerItem() is not None))
        return out

    def _do_layout(self, rect, test_only: bool) -> int:
        m = self.contentsMargins()
        left, top = rect.x() + m.left(), rect.y() + m.top()
        right = rect.right() - m.right()
        # Break into lines first, so every item can be centred on its line.
        lines, cur, x = [], [], left
        for group, w, h, is_spacer in self._chunks():
            if is_spacer:
                if x > left:
                    x += w
                continue
            if x > left and x + w > right + 1:
                lines.append(cur); cur, x = [], left
            cur.append((group, x, w, h))
            x += w + self._h
        if cur:
            lines.append(cur)
        y = top
        for line in lines:
            line_h = max(h for _g, _x, _w, h in line)
            if not test_only:
                for group, gx, _w, _h in line:
                    cx = gx
                    for it in group:
                        sz = it.sizeHint()
                        it.setGeometry(QRect(QPoint(cx, y + (line_h - sz.height()) // 2), sz))
                        cx += sz.width() + 4
            y += line_h + self._v
        if lines:
            y -= self._v
        return y - rect.y() + m.bottom()


def reflow_row(row: QHBoxLayout) -> FlowLayout:
    """Replace a QHBoxLayout control row, in place, with a wrapping FlowLayout.

    Handles a row nested in a QBoxLayout or one that is a widget's own layout.
    Nested sub-layouts are carried over inside a holder widget.
    """
    flow = FlowLayout(hspacing=max(4, row.spacing()))
    holder_parent = row.parentWidget()
    items = []
    while row.count():
        items.append(row.takeAt(0))
    for it in items:
        if it.widget() is not None:
            flow.addWidget(it.widget())
        elif it.layout() is not None:
            holder = QWidget()
            sub = it.layout()
            holder.setLayout(sub)
            sub.setContentsMargins(0, 0, 0, 0)
            flow.addWidget(holder)
        elif it.spacerItem() is not None:
            sp = it.spacerItem()
            if sp.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed:
                flow.addSpacing(sp.sizeHint().width())
    parent = row.parent()
    if isinstance(parent, QLayout):
        idx = parent.indexOf(row)
        parent.takeAt(idx)
        parent.insertLayout(idx, flow)
    elif holder_parent is not None and holder_parent.layout() is row:
        m = row.contentsMargins()
        QWidget().setLayout(row)          # detach the old layout
        holder_parent.setLayout(flow)
        flow.setContentsMargins(m)
    return flow


class ROIView(QGraphicsView):
    """Displays the 2x2 grid preview, with two overlaid ROIs (background and
    crop) that the user can draw and adjust.

    The "preview" mode hides the ROIs and re-renders the canvas with the crop
    region applied — what you see is exactly the export frame.
    """

    bg_roi_changed = Signal(object)    # (x, y, w, h) or None
    crop_roi_changed = Signal(object)  # (x, y, w, h) or None
    # Live (kind, roi) while a rectangle is being drawn, moved or resized, so
    # numeric fields can follow the mouse. *_roi_changed fires on release.
    roi_dragging = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QBrush(QColor(15, 15, 20)))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setToolTip(
            "Drag on the image to draw · drag inside a box to move it · drag its edge "
            "or corner to resize · arrow keys nudge (Shift = 10 px)\n"
            "Trackpad: two-finger scroll pans, pinch or ⌘/Ctrl+scroll zooms · "
            "Mouse wheel zooms · Space+drag or middle-drag pans · double-click = fit")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drag_kind: Optional[str] = None     # "new" | "move" | "resize"
        self._drag_edge: str = ""                 # e.g. "l", "tr"
        self._drag_origin: Optional[tuple] = None # ROI at press time
        self._space_down: bool = False
        # When set, the scene shows this single panel instead of the 2x2 grid
        # (same image coordinates, so ROI editing is unchanged).
        self._single_panel: Optional[str] = None
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._bg_rect_item: Optional[QGraphicsRectItem] = None
        self._crop_rect_item: Optional[QGraphicsRectItem] = None
        self._bg_roi: Optional[tuple[int, int, int, int]] = None
        self._crop_roi: Optional[tuple[int, int, int, int]] = None
        self._drag_start: Optional[QPointF] = None
        self._pan_start: Optional[QPoint] = None
        self._mode: str = MODE_BG
        self._user_zoom: bool = False   # True once user has scrolled/zoomed
        # Crop shape constraint: (aspect_w, aspect_h) ratio, or None for free.
        self._crop_aspect: Optional[tuple] = None
        # Fixed pixel size constraint: (w, h), or None.
        self._crop_px: Optional[tuple] = None
        # Image (single-channel) dimensions and the grid offset of the Phase
        # quadrant (where ROI rects are drawn) within the scene. Updated by
        # show_grid().
        self._img_w = 1
        self._img_h = 1
        # Size of the whole image. It differs from _img_w/_img_h while
        # "Preview export" shows a crop; boxes are always clamped to this one.
        self._full_w = 1
        self._full_h = 1
        self._roi_before_press: Optional[tuple] = None

    # ---- public API -------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        tc = theme.colors
        self.setBackgroundBrush(QBrush(QColor(*tc.roi_bg)))
        # Refresh ROI rectangles in the scene so they pick up the new pen colors.
        if self._bg_rect_item is not None:
            pen = QPen(QColor(*tc.roi_bg_pen)); pen.setCosmetic(True); pen.setWidth(2)
            self._bg_rect_item.setPen(pen)
        if self._crop_rect_item is not None:
            pen = QPen(QColor(*tc.roi_crop_pen)); pen.setCosmetic(True); pen.setWidth(2)
            self._crop_rect_item.setPen(pen)

    def set_mode(self, mode: str) -> None:
        assert mode in (MODE_BG, MODE_CROP, MODE_PREVIEW)
        self._mode = mode
        self._update_roi_visibility()

    def show_grid(self, grid_rgb: np.ndarray, img_wh: tuple[int, int],
                  full_wh: Optional[tuple[int, int]] = None) -> None:
        """Set/replace the pixmap.

        grid_rgb: (H_g, W_g, 3) uint8 — pre-rendered 2x2 grid (possibly cropped
        in preview mode).
        img_wh:  (img_w, img_h) of a SINGLE channel image as it currently is
        (post-crop in preview mode, pre-crop otherwise). Used to map mouse
        coords back to image coords for ROI drawing.
        full_wh: (width, height) of the whole image, when it differs from
        img_wh (preview of a crop).
        """
        pix = self._pixmap_from_array(grid_rgb)
        old_size = self._pixmap_item.pixmap().size() if self._pixmap_item is not None else None
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self._bg_rect_item = None
        self._crop_rect_item = None
        self._img_w, self._img_h = img_wh
        self._full_w, self._full_h = full_wh or img_wh
        # Recreate visible ROI rectangles if applicable.
        if self._mode != MODE_PREVIEW:
            if self._bg_roi is not None:
                self._bg_rect_item = self._make_roi_item(self._bg_roi, "bg")
            if self._crop_roi is not None:
                self._crop_rect_item = self._make_roi_item(self._crop_roi, "crop")
        # A zoom or pan set on an image of another size (before a crop was
        # previewed or removed) no longer frames anything sensible: re-fit.
        if old_size is not None and old_size != pix.size():
            self._user_zoom = False
        if not self._user_zoom:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_grid(self, grid_rgb: np.ndarray) -> None:
        """Swap in a new pixmap without disturbing the ROI rectangles."""
        if self._pixmap_item is None:
            return
        new_pix = self._pixmap_from_array(grid_rgb)
        if new_pix.size() != self._pixmap_item.pixmap().size():
            # Geometry changed (e.g. enter/exit preview mode crop) — full rebuild.
            self.show_grid(grid_rgb, (self._img_w, self._img_h),
                           (self._full_w, self._full_h))
            return
        self._pixmap_item.setPixmap(new_pix)

    def bg_roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._bg_roi

    def crop_roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._crop_roi

    def clear_bg_roi(self) -> None:
        if self._bg_rect_item is not None:
            self._scene.removeItem(self._bg_rect_item)
            self._bg_rect_item = None
        self._bg_roi = None
        self.bg_roi_changed.emit(None)

    def set_crop_constraint(
        self,
        aspect: Optional[tuple] = None,
        px_size: Optional[tuple] = None,
    ) -> None:
        """Set crop shape constraint.

        aspect: (w_ratio, h_ratio) e.g. (16, 9) — constrains drag to that ratio.
        px_size: (width, height) in image pixels — rectangle always that exact size.
        Pass both None for free-form dragging.
        """
        self._crop_aspect = aspect
        self._crop_px = px_size

    def clear_crop_roi(self) -> None:
        if self._crop_rect_item is not None:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None
        self._crop_roi = None
        self.crop_roi_changed.emit(None)

    def set_crop_roi(self, roi: Optional[tuple]) -> None:
        """Set the crop numerically (clamped to the image) and emit the change."""
        if roi is None:
            self.clear_crop_roi()
            return
        x, y, w, h = (int(v) for v in roi)
        w = max(1, min(w, self._full_w)); h = max(1, min(h, self._full_h))
        x = max(0, min(x, self._full_w - w)); y = max(0, min(y, self._full_h - h))
        roi = (x, y, w, h)
        self._crop_roi = roi
        if self._crop_rect_item is not None:
            self._crop_rect_item.setRect(QRectF(x, y, w, h))
        elif self._pixmap_item is not None and self._mode != MODE_PREVIEW:
            self._crop_rect_item = self._make_roi_item(roi, "crop")
        self.crop_roi_changed.emit(roi)

    def set_bg_roi(self, roi: Optional[tuple]) -> None:
        """Set the background box numerically (clamped) and emit the change."""
        if roi is None:
            self.clear_bg_roi()
            return
        x, y, w, h = (int(v) for v in roi)
        w = max(1, min(w, self._full_w)); h = max(1, min(h, self._full_h))
        x = max(0, min(x, self._full_w - w)); y = max(0, min(y, self._full_h - h))
        roi = (x, y, w, h)
        self._bg_roi = roi
        if self._bg_rect_item is not None:
            self._bg_rect_item.setRect(QRectF(x, y, w, h))
        elif self._pixmap_item is not None and self._mode != MODE_PREVIEW:
            self._bg_rect_item = self._make_roi_item(roi, "bg")
        self.bg_roi_changed.emit(roi)

    def set_single_panel(self, panel: Optional[str]) -> None:
        self._single_panel = panel

    def rotate_rois(self, k: int, w: int, h: int) -> None:
        """Carry both boxes through a k×90° CCW rotation of a w×h image.
        No signals: the caller refreshes the caches and the preview."""
        self._bg_roi = rotate_roi(self._bg_roi, k, w, h)
        self._crop_roi = rotate_roi(self._crop_roi, k, w, h)
        if k % 2:
            self._img_w, self._img_h = self._img_h, self._img_w
            self._full_w, self._full_h = self._full_h, self._full_w

    def clear_image(self) -> None:
        """Remove the picture (no file is loaded). The boxes are kept."""
        self._scene.clear()
        self._pixmap_item = None
        self._bg_rect_item = None
        self._crop_rect_item = None

    def fit(self) -> None:
        self._user_zoom = False
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_crop(self) -> None:
        """Fill the view with the crop box (plus a margin) to fine-tune it."""
        if self._crop_roi is None or self._pixmap_item is None or self._mode == MODE_PREVIEW:
            self.fit()
            return
        x, y, w, h = self._crop_roi
        pad = max(w, h) * 0.6
        self._user_zoom = True
        self.fitInView(QRectF(x - pad, y - pad, w + 2 * pad, h + 2 * pad),
                       Qt.AspectRatioMode.KeepAspectRatio)

    # ---- internals --------------------------------------------------------

    def _update_roi_visibility(self) -> None:
        show = self._mode != MODE_PREVIEW
        if self._bg_rect_item is not None:
            self._bg_rect_item.setVisible(show)
        if self._crop_rect_item is not None:
            self._crop_rect_item.setVisible(show)

    def _make_roi_item(
        self, roi: tuple[int, int, int, int], kind: str
    ) -> QGraphicsRectItem:
        x, y, w, h = roi
        tc = ThemeManager.instance().current().colors
        color = QColor(*tc.roi_bg_pen) if kind == "bg" else QColor(*tc.roi_crop_pen)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2)
        item = self._scene.addRect(QRectF(x, y, w, h), pen)
        item.setZValue(10)
        return item

    @staticmethod
    def _pixmap_from_array(frame: np.ndarray) -> QPixmap:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected (Y, X, 3) uint8 RGB, got {frame.shape}/{frame.dtype}"
            )
        h, w, _ = frame.shape
        qimg = QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _clip_to_image(self, scene_point: QPointF) -> QPointF:
        """Treat the canvas as if only the top-left (Phase) quadrant is the
        editable image region. Drags outside it get clipped."""
        return QPointF(
            float(np.clip(scene_point.x(), 0, self._img_w)),
            float(np.clip(scene_point.y(), 0, self._img_h)),
        )

    # ---- mouse events -----------------------------------------------------

    # -- ROI hit testing / editing --

    def _active_roi(self) -> Optional[tuple]:
        if self._mode == MODE_CROP:
            return self._crop_roi
        if self._mode == MODE_BG:
            return self._bg_roi
        return None

    def _active_item(self):
        return self._crop_rect_item if self._mode == MODE_CROP else self._bg_rect_item

    def _hit(self, sp: QPointF) -> Optional[str]:
        """'move', an edge/corner code ('l','r','t','b','tl','tr','bl','br') or None."""
        roi = self._active_roi()
        if roi is None:
            return None
        x, y, w, h = roi
        tol = 7.0 / max(abs(self.transform().m11()), 1e-6)   # 7 screen px
        px, py = sp.x(), sp.y()
        if not (x - tol <= px <= x + w + tol and y - tol <= py <= y + h + tol):
            return None
        inside = x <= px <= x + w and y <= py <= y + h
        fixed = self._mode == MODE_CROP and self._crop_px is not None
        # A box only a few screen px wide is all edge; let it be grabbed to move.
        if fixed or (inside and (w < 3 * tol or h < 3 * tol)):
            return "move" if inside or fixed else None
        v = "t" if abs(py - y) <= tol else ("b" if abs(py - (y + h)) <= tol else "")
        hz = "l" if abs(px - x) <= tol else ("r" if abs(px - (x + w)) <= tol else "")
        if v or hz:
            return v + hz
        return "move" if inside else None

    _CURSORS = {
        "move": Qt.CursorShape.SizeAllCursor,
        "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
        "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
        "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
    }

    def _update_hover_cursor(self, sp: QPointF) -> None:
        if self._space_down:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._mode == MODE_PREVIEW:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            hit = self._hit(sp)
            self.viewport().setCursor(self._CURSORS.get(hit, Qt.CursorShape.CrossCursor))

    def _rect_to_roi(self, r: QRectF) -> tuple:
        r = r.normalized().intersected(QRectF(0, 0, self._img_w, self._img_h))
        x, y = int(round(r.x())), int(round(r.y()))
        return (max(0, x), max(0, y),
                max(1, int(round(r.width()))), max(1, int(round(r.height()))))

    def _resized_rect(self, cur: QPointF) -> QRectF:
        x, y, w, h = self._drag_origin
        x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
        e = self._drag_edge
        if "l" in e: x0 = min(cur.x(), x1 - 1)
        if "r" in e: x1 = max(cur.x(), x0 + 1)
        if "t" in e: y0 = min(cur.y(), y1 - 1)
        if "b" in e: y1 = max(cur.y(), y0 + 1)
        if self._mode == MODE_CROP and self._crop_aspect is not None:
            aw, ah = self._crop_aspect
            r = aw / ah
            nw, nh = x1 - x0, y1 - y0
            if e in ("l", "r"):
                nh = nw / r
            elif e in ("t", "b"):
                nw = nh * r
            elif nw / r >= nh:
                nh = nw / r
            else:
                nw = nh * r
            if "t" in e: y0 = y1 - nh
            else:        y1 = y0 + nh
            if "l" in e: x0 = x1 - nw
            else:        x1 = x0 + nw
        return QRectF(QPointF(x0, y0), QPointF(x1, y1))

    def _moved_rect(self, cur: QPointF) -> QRectF:
        x, y, w, h = self._drag_origin
        nx = x + (cur.x() - self._drag_start.x())
        ny = y + (cur.y() - self._drag_start.y())
        nx = max(0.0, min(nx, self._img_w - w))
        ny = max(0.0, min(ny, self._img_h - h))
        return QRectF(round(nx), round(ny), w, h)

    def _start_pan(self, event) -> None:
        self._pan_start = event.position().toPoint()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        btn = event.button()
        if btn == Qt.MouseButton.MiddleButton or (
                btn == Qt.MouseButton.LeftButton and self._space_down):
            self._start_pan(event)
            event.accept()
            return
        if btn != Qt.MouseButton.LeftButton or self._pixmap_item is None:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._mode == MODE_PREVIEW:
            self._start_pan(event)         # nothing to edit in preview: drag pans
            event.accept()
            return
        hit = self._hit(scene_pos)
        self._drag_start = self._clip_to_image(scene_pos)
        self._roi_before_press = self._active_roi()   # restored by a double-click
        if hit is not None:
            self._drag_kind = "move" if hit == "move" else "resize"
            self._drag_edge = hit
            self._drag_origin = self._active_roi()
            event.accept()
            return
        # Draw a new rectangle (replacing this mode's existing one).
        self._drag_kind = "new"
        if self._mode == MODE_BG and self._bg_rect_item is not None:
            self._scene.removeItem(self._bg_rect_item)
            self._bg_rect_item = None
        if self._mode == MODE_CROP and self._crop_rect_item is not None:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None
        tc = ThemeManager.instance().current().colors
        _c = QColor(*tc.roi_bg_pen) if self._mode == MODE_BG else QColor(*tc.roi_crop_pen)
        pen = QPen(_c); pen.setCosmetic(True); pen.setWidth(2)
        start_rect = QRectF(self._drag_start, self._drag_start)
        if self._mode == MODE_CROP and self._crop_px is not None:
            # Fixed size: drop a box of exactly that size centred on the click,
            # then let the same drag move it.
            pw, ph = self._crop_px
            cx = max(0.0, min(self._drag_start.x() - pw / 2, self._img_w - pw))
            cy = max(0.0, min(self._drag_start.y() - ph / 2, self._img_h - ph))
            start_rect = QRectF(round(cx), round(cy), pw, ph)
            self._drag_kind = "move"
            self._drag_origin = (int(start_rect.x()), int(start_rect.y()), pw, ph)
        rect = self._scene.addRect(start_rect, pen)
        rect.setZValue(10)
        if self._mode == MODE_BG:   self._bg_rect_item = rect
        else:                       self._crop_rect_item = rect
        event.accept()

    def _constrain_crop_end(self, current: QPointF) -> QPointF:
        """Apply the aspect-ratio constraint to the drag endpoint of a new box."""
        if self._crop_aspect is not None:
            aw, ah = self._crop_aspect
            if aw <= 0 or ah <= 0:
                return current
            r = aw / ah
            dx = current.x() - self._drag_start.x()
            dy = current.y() - self._drag_start.y()
            if abs(dx) >= abs(dy) * r:
                adj_dy = math.copysign(abs(dx) / r, dy if dy != 0 else (1 if dx >= 0 else -1))
                return QPointF(current.x(), self._drag_start.y() + adj_dy)
            else:
                adj_dx = math.copysign(abs(dy) * r, dx if dx != 0 else (1 if dy >= 0 else -1))
                return QPointF(self._drag_start.x() + adj_dx, current.y())
        return current

    def mouseMoveEvent(self, event) -> None:
        if self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._drag_kind is None:
            if self._pixmap_item is not None:
                self._update_hover_cursor(scene_pos)
            super().mouseMoveEvent(event)
            return
        active = self._active_item()
        if active is None:
            return
        current = self._clip_to_image(scene_pos)
        if self._drag_kind == "move":
            rect = self._moved_rect(scene_pos)
        elif self._drag_kind == "resize":
            rect = self._resized_rect(current)
        else:
            if self._mode == MODE_CROP:
                current = self._constrain_crop_end(current)
            rect = QRectF(self._drag_start, current).normalized()
        active.setRect(rect)
        self.roi_dragging.emit("crop" if self._mode == MODE_CROP else "bg",
                               self._rect_to_roi(rect))

    def _commit(self, roi: tuple) -> None:
        if self._mode == MODE_BG:
            self._bg_roi = roi
            self.bg_roi_changed.emit(roi)
        else:
            self._crop_roi = roi
            self.crop_roi_changed.emit(roi)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_start is not None and event.button() in (
                Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._pan_start = None
            self._update_hover_cursor(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_kind is not None:
            active = self._active_item()
            kind = self._drag_kind
            self._drag_kind = None
            self._drag_start = None
            if active is not None:
                r = active.rect()
                if kind == "new" and (r.width() < 2 or r.height() < 2):
                    # A click without a drag: keep the previous box, if any.
                    self._scene.removeItem(active)
                    if self._mode == MODE_BG:
                        self._bg_rect_item = None
                        if self._bg_roi is not None:
                            self._bg_rect_item = self._make_roi_item(self._bg_roi, "bg")
                    else:
                        self._crop_rect_item = None
                        if self._crop_roi is not None:
                            self._crop_rect_item = self._make_roi_item(self._crop_roi, "crop")
                else:
                    self._commit(self._rect_to_roi(r))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        steps = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                 Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        roi = self._active_roi()
        if key in steps and roi is not None and self._drag_kind is None:
            dx, dy = steps[key]
            n = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            x, y, w, h = roi
            if event.modifiers() & Qt.KeyboardModifier.AltModifier and not (
                    self._mode == MODE_CROP and self._crop_px is not None):
                w = max(1, min(w + dx * n, self._img_w - x))     # Alt/Option + arrows resize
                h = max(1, min(h + dy * n, self._img_h - y))
            else:
                x = max(0, min(x + dx * n, self._img_w - w))
                y = max(0, min(y + dy * n, self._img_h - h))
            new = (x, y, w, h)
            item = self._active_item()
            if item is not None:
                item.setRect(QRectF(x, y, w, h))
            self._commit(new)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _zoom_by(self, factor: float) -> None:
        if self._pixmap_item is None or factor <= 0:
            return
        self._user_zoom = True
        self.scale(factor, factor)
        # Never zoom out past "fit": there is nothing to see beyond the image.
        vr = self.viewport().rect()
        sr = self._scene.sceneRect()
        # Only when zooming out: fitInView leaves a small margin, so a small
        # zoom-in step (a slow trackpad pinch) would otherwise be undone.
        if factor < 1.0 and sr.width() > 0 and sr.height() > 0:
            fit = min(vr.width() / sr.width(), vr.height() / sr.height())
            if self.transform().m11() <= fit:
                self.fit()

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            super().wheelEvent(event)
            return
        mods = event.modifiers()
        wants_zoom = bool(mods & (Qt.KeyboardModifier.ControlModifier
                                  | Qt.KeyboardModifier.MetaModifier))
        # Trackpads report scroll phases; a classic mouse wheel does not.
        trackpad = event.phase() != Qt.ScrollPhase.NoScrollPhase
        if trackpad and not wants_zoom:
            d = event.pixelDelta()
            if d.isNull():
                d = event.angleDelta() / 8
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        dy = event.angleDelta().y() or event.pixelDelta().y()
        self._zoom_by(1.0015 ** dy)
        event.accept()

    def viewportEvent(self, event) -> bool:
        # macOS trackpad pinch (and two-finger double-tap = fit).
        if event.type() == QEvent.Type.NativeGesture:
            gt = event.gestureType()
            if gt == Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom_by(1.0 + event.value())
                return True
            if gt == Qt.NativeGestureType.SmartZoomNativeGesture:
                self.fit()
                return True
        return super().viewportEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item is not None:
            # With a fixed-size crop, the first click of the double-click has
            # already dropped a new box at the pointer: put the old one back.
            if self._mode == MODE_CROP and self._crop_px is not None:
                self.set_crop_roi(self._roi_before_press)
            self._user_zoom = False
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap_item is not None and not self._user_zoom:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ===========================================================================
# Metadata export
# ===========================================================================

import json as _json
import datetime as _datetime


class MetadataExporter:
    """Collect and save a JSON file describing all settings used to produce the output."""

    @staticmethod
    def collect(mw: "MainWindow") -> dict:
        d = mw._data
        meta: dict = {
            "software": {"name": APP_NAME, "version": __version__},
            "export_timestamp": _datetime.datetime.now().isoformat(timespec="seconds"),
            "source": {
                "path": d.source_path if d is not None else None,
                "position_index": mw._position,
                "n_frames": d.n_frames if d is not None else None,
                "n_positions": d.n_positions if d is not None else None,
                "um_per_px": d.um_per_px if d is not None else None,
                "frame_interval_s": d.frame_interval_s if d is not None else None,
            },
        }

        # Crop / BG ROI
        crop = mw.roi_view.crop_roi()
        bg   = mw.roi_view.bg_roi()
        # roi_view returns (x, y, w, h) tuples, not QRects.
        def _roi(r):
            if r is None:
                return None
            x, y, w, h = (int(v) for v in r)
            return {"x": x, "y": y, "width": w, "height": h}
        meta["regions"] = {"crop_roi": _roi(crop), "bg_roi": _roi(bg)}

        # LUT settings
        def _lut_dict(lut: ChannelLUT) -> dict:
            return {"vmin": lut.vmin, "vmax": lut.vmax, "gamma": lut.gamma}

        meta["luts"] = {
            "mode":    mw.lut_mode_combo.currentData(),
            "phase":   _lut_dict(mw.phase_lut_widget.lut()),
            "gfp":     _lut_dict(mw.gfp_lut_widget.lut()),
            "mcherry": _lut_dict(mw.mcherry_lut_widget.lut()),
        }
        # In the per-frame modes every frame has its own range; without them
        # the export cannot be reproduced from this file.
        if d is not None and mw._lut_mode != LUT_STACK and mw._per_frame_luts["phase"]:
            mw._export_luts()       # store the current frame's widget values first
            meta["luts"]["per_frame"] = {
                ch: [[lut.vmin, lut.vmax, lut.gamma] for lut in mw._per_frame_luts[ch]]
                for ch in ("phase", "gfp", "mcherry")}

        # Channel visibility
        meta["channel_visibility"] = {
            "phase":   mw.phase_cb.isChecked(),
            "gfp":     mw.gfp_cb.isChecked(),
            "mcherry": mw.mch_cb.isChecked(),
        }
        meta["merge_mode"] = mw.merge_combo.currentData()
        meta["rotation_deg_clockwise"] = getattr(mw, "_rotation_deg", 0)
        if d is not None and d.segments:
            meta["joined_files"] = [dict(sg) for sg in d.segments]
        if hasattr(mw, "timer_clock_combo"):
            meta["timer_clock"] = mw.timer_clock_combo.currentData()

        # Frame range
        meta["frame_range"] = {
            "start": mw.start_frame_spin.value(),
            "end":   mw.end_frame_spin.value(),
        }

        # Scale bar
        meta["scale_bar"] = {
            "enabled":    mw.sb_visible_cb.isChecked(),
            "length_um":  mw.sb_size_spin.value(),
            "position":   mw.sb_pos_combo.currentData(),
            "x_offset":   mw.sb_x_spin.value(),
            "y_offset":   mw.sb_y_spin.value(),
            "text_center": mw.sb_center_cb.isChecked(),
            "text_above":  mw.sb_above_cb.isChecked(),
        }

        # Timer
        meta["timer"] = {
            "enabled":    mw.timer_cb.isChecked(),
            "format":     mw.timer_fmt_combo.currentData(),
            "interval_s": mw.timer_interval_spin.value(),
            "t0_frame":   mw.timer_t0_spin.value(),
            "position":   mw.timer_pos_combo.currentData(),
            "x_offset":   mw.timer_x_spin.value(),
            "y_offset":   mw.timer_y_spin.value(),
            "decimals":   mw.timer_decimals_spin.value(),
        }

        # Channel labels
        meta["channel_labels"] = {
            "enabled":   mw.labels_cb.isChecked(),
            "position":  mw.label_pos_combo.currentData(),
            "x_offset":  mw.label_x_spin.value(),
            "y_offset":  mw.label_y_spin.value(),
            "phase":     mw.label_phase_field.text(),
            "gfp":       mw.label_gfp_field.text(),
            "mcherry":   mw.label_mcherry_field.text(),
            "composite": mw.label_composite_field.text(),
        }

        # Export settings
        crf, pxfmt = mw.quality_combo.currentData()
        meta["export"] = {
            "fps": mw.fps_spin.value(),
            "crf": crf,
            "pixel_format": pxfmt,
            "video_size": mw.vscale_combo.currentText(),
            "theme": mw.theme_combo.currentData(),
        }

        # Custom strip and example-set options (the export actions add what
        # they actually wrote under "strip" / "example_set").
        meta["strip"] = {**mw._strip_settings(),
                         "video": mw.strip_video_cb.isChecked(),
                         "include_in_example_set": mw.strip_in_set_cb.isChecked()}
        meta["example_options"] = {
            "video_fps": mw.ex_fps_spin.value(),
            "video_scale": mw.ex_vscale_combo.currentData(),
            "tiff_dpi": mw.ex_dpi_spin.value(),
            "scale_frame_in_every_folder": mw.ex_scale_each_cb.isChecked(),
        }

        return meta

    @staticmethod
    def save(mw: "MainWindow") -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        out_path, _ = QFileDialog.getSaveFileName(
            mw, "Save Metadata", str(Path.home() / "microscopy_settings.json"),
            "JSON (*.json)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".json"):
            out_path += ".json"
        meta = MetadataExporter.collect(mw)
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                _json.dump(meta, fh, indent=2, default=str)
            QMessageBox.information(mw, "Saved", f"Metadata saved to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(mw, "Save failed", str(exc))

    # ---- loading a settings file back ---------------------------------------

    @staticmethod
    def sources_of(meta: dict) -> list[dict]:
        """The source file(s) a settings file was made from, in frame order:
        dicts with file, path and (for appended files) shift_xy."""
        joined = meta.get("joined_files")
        if joined:
            return [dict(sg) for sg in joined]
        src = (meta.get("source") or {}).get("path")
        if not src:
            return []
        # The path may have been written on another operating system, where
        # Path(...).name would return all of it.
        name = str(src).replace("\\", "/").rsplit("/", 1)[-1]
        return [{"file": name, "path": str(src), "shift_xy": None}]

    @staticmethod
    def apply(mw: "MainWindow", meta: dict) -> list[str]:
        """Put the settings in `meta` back into the widgets of `mw`, whose
        data must already be the files they were saved from. Returns notes
        for anything that could not be restored.

        Order matters: changing the frame range or background box re-runs the
        Stack auto LUT, so the saved display range is applied last.
        """
        notes: list[str] = []
        d = mw._data

        def combo(c, value) -> bool:
            i = c.findData(value)
            if i < 0:
                return False
            c.setCurrentIndex(i)
            return True

        def section(name, fn):
            try:
                fn()
            except Exception as exc:        # a bad section must not block the rest
                notes.append(f"{name}: {exc}")

        # A file written before a setting existed lacks its key. Such a file
        # was made with what is now the default, so apply the default instead
        # of leaving whatever the window happens to show.
        def rotation():
            deg = meta.get("rotation_deg_clockwise") or 0
            if not combo(mw.rotate_combo, int(deg)):
                notes.append(f"rotation {deg}° is not one of the choices")

        def position():
            pi = (meta.get("source") or {}).get("position_index")
            if pi is None:
                return
            if not 0 <= int(pi) < d.n_positions:
                notes.append(f"position {int(pi) + 1} does not exist in this file "
                             f"({d.n_positions} position(s))")
            elif mw.pos_combo.isEnabled():
                mw.pos_combo.setCurrentIndex(int(pi))

        def frames():
            fr = meta.get("frame_range")
            if fr:
                want = (int(fr["start"]), int(fr["end"]))
                mw.start_frame_spin.setValue(want[0])
                mw.end_frame_spin.setValue(want[1])
                got = (mw.start_frame_spin.value(), mw.end_frame_spin.value())
                if got != want:
                    notes.append(f"frame range {want[0]}–{want[1]} does not fit this "
                                 f"file's {d.n_frames} frames — set to {got[0]}–{got[1]}")

        def visibility_merge():
            cv = meta.get("channel_visibility") or {}
            for key, cb in (("phase", mw.phase_cb), ("gfp", mw.gfp_cb),
                            ("mcherry", mw.mch_cb)):
                if key in cv:
                    cb.setChecked(bool(cv[key]))
            merge = (meta.get("merge_mode")
                     or (meta.get("example_set") or {}).get("merge_mode")
                     or (meta.get("strip") or {}).get("merge_mode") or "additive")
            if not combo(mw.merge_combo, merge):
                notes.append(f"merge mode {merge!r} not recognised")

        def regions():
            reg = meta.get("regions") or {}
            for key, setter, label in (("bg_roi", mw.roi_view.set_bg_roi, "background"),
                                       ("crop_roi", mw.roi_view.set_crop_roi, "crop")):
                r = reg.get(key)
                if r is None:
                    if key in reg:
                        setter(None)
                    continue
                x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                if x < 0 or y < 0 or x + w > d.width or y + h > d.height:
                    notes.append(f"{label} box ({x},{y},{w}×{h}) does not fit the "
                                 f"{d.width}×{d.height} image (different file or rotation?) — skipped")
                    continue
                setter((x, y, w, h))

        def luts():
            lt = meta.get("luts") or {}
            mode = lt.get("mode") or LUT_STACK
            channels = (("phase", mw.phase_lut_widget), ("gfp", mw.gfp_lut_widget),
                        ("mcherry", mw.mcherry_lut_widget))
            if mode in (LUT_SMART, LUT_MANUAL):
                per_frame = lt.get("per_frame") or {}
                complete = all(len(per_frame.get(ch) or ()) == d.n_frames
                               for ch, _w in channels)
                if mode == LUT_MANUAL and not complete:
                    notes.append("display ranges were not restored: the file holds no "
                                 f"per-frame ranges for this file's {d.n_frames} frames")
                    return
                combo(mw.lut_mode_combo, mode)     # Smart recomputes its ranges
                if complete:
                    mw._per_frame_luts = {
                        ch: [ChannelLUT(*(float(x) for x in v)) for v in per_frame[ch]]
                        for ch, _w in channels}
                    mw._configure_lut_widgets(initial=True)
                return
            combo(mw.lut_mode_combo, LUT_STACK)
            for key, wdg in channels:
                v = lt.get(key)
                if not v:
                    continue
                wdg.canvas.set_range(v["vmin"], v["vmax"], v["gamma"])
                wdg._refresh_fields(v["vmin"], v["vmax"], v["gamma"])
                wdg.changed.emit()

        def scale_bar():
            sb = meta.get("scale_bar") or {}
            if not sb:
                return
            mw.sb_visible_cb.setChecked(bool(sb["enabled"]))
            mw.sb_size_spin.setValue(float(sb["length_um"]))
            combo(mw.sb_pos_combo, sb["position"])
            mw.sb_x_spin.setValue(int(sb["x_offset"])); mw.sb_y_spin.setValue(int(sb["y_offset"]))
            mw.sb_center_cb.setChecked(bool(sb["text_center"]))
            mw.sb_above_cb.setChecked(bool(sb["text_above"]))

        def timer():
            tm = meta.get("timer") or {}
            if not tm:
                return
            mw.timer_cb.setChecked(bool(tm["enabled"]))
            combo(mw.timer_fmt_combo, tm["format"])
            mw.timer_interval_spin.setValue(float(tm["interval_s"]))
            mw.timer_t0_spin.setValue(int(tm["t0_frame"]))
            if mw.timer_t0_spin.value() != int(tm["t0_frame"]):
                notes.append(f"timer zero frame {tm['t0_frame']} does not exist in this "
                             f"file — set to {mw.timer_t0_spin.value()}")
            combo(mw.timer_pos_combo, tm["position"])
            mw.timer_x_spin.setValue(int(tm["x_offset"])); mw.timer_y_spin.setValue(int(tm["y_offset"]))
            mw.timer_decimals_spin.setValue(int(tm["decimals"]))
            if mw.timer_clock_combo.isEnabled():
                combo(mw.timer_clock_combo, meta.get("timer_clock") or "fixed")

        def labels():
            lb = meta.get("channel_labels") or {}
            if not lb:
                return
            mw.labels_cb.setChecked(bool(lb["enabled"]))
            combo(mw.label_pos_combo, lb["position"])
            mw.label_x_spin.setValue(int(lb["x_offset"])); mw.label_y_spin.setValue(int(lb["y_offset"]))
            for key, fld in (("phase", mw.label_phase_field), ("gfp", mw.label_gfp_field),
                             ("mcherry", mw.label_mcherry_field),
                             ("composite", mw.label_composite_field)):
                if key in lb:
                    fld.setText(lb[key])

        def export():
            ex = meta.get("export") or {}
            if "fps" in ex:
                mw.fps_spin.setValue(int(ex["fps"]))
            for i in range(mw.quality_combo.count()):
                if tuple(mw.quality_combo.itemData(i)) == (ex.get("crf"), ex.get("pixel_format")):
                    mw.quality_combo.setCurrentIndex(i)
                    break
            i = mw.vscale_combo.findText(ex.get("video_size", ""))
            if i >= 0:
                mw.vscale_combo.setCurrentIndex(i)
            # What an example-set export recorded, else the options as saved.
            exs = {**(meta.get("example_options") or {}), **(meta.get("example_set") or {})}
            if "tiff_dpi" in exs:
                mw.ex_dpi_spin.setValue(int(exs["tiff_dpi"]))
            if "video_fps" in exs:
                mw.ex_fps_spin.setValue(int(exs["video_fps"]))
            if "video_scale" in exs:
                combo(mw.ex_vscale_combo, exs["video_scale"])
            if "scale_frame_in_every_folder" in exs:
                mw.ex_scale_each_cb.setChecked(bool(exs["scale_frame_in_every_folder"]))

        def strip():
            st = (meta.get("strip") or (meta.get("example_set") or {}).get("strip") or {})
            if not st:
                return
            keys = list(st.get("panels") or [])
            for i, cb in enumerate(mw.strip_combos):
                cb.setCurrentIndex(max(0, cb.findData(keys[i])) if i < len(keys) else 0)
            if "horizontal" in st:
                combo(mw.strip_dir_combo, st["horizontal"])
            if "gap_value" in st:
                combo(mw.strip_bg_combo, st["gap_value"])
            if "gap_px" in st:
                mw.strip_gap_spin.setValue(int(st["gap_px"]))
            if "tiff_dpi" in st:
                mw.ex_dpi_spin.setValue(int(st["tiff_dpi"]))
            if "video" in st:
                mw.strip_video_cb.setChecked(bool(st["video"]))
            if "include_in_example_set" in st:
                mw.strip_in_set_cb.setChecked(bool(st["include_in_example_set"]))

        for name, fn in (("rotation", rotation), ("position", position),
                         ("frame range", frames), ("channels", visibility_merge),
                         ("boxes", regions), ("scale bar", scale_bar), ("timer", timer),
                         ("labels", labels), ("export", export), ("strip", strip),
                         ("display range", luts)):
            section(name, fn)
        mw._update_preview()
        return notes

    @staticmethod
    def load(mw: "MainWindow") -> None:
        """Pick a settings .json (Save Metadata, or the one written beside an
        example set / strip export) and restore it, reopening its files if
        they are not the ones loaded."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        start = str(mw._data.source_path.parent) if mw._data is not None else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(mw, "Load settings", start, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:   # tolerate a byte-order mark
                meta = _json.load(fh)
            if not isinstance(meta, dict) or "regions" not in meta and "luts" not in meta:
                raise ValueError("this does not look like a settings file from this app")
        except Exception as exc:
            QMessageBox.critical(mw, "Load failed", f"Could not read {Path(path).name}:\n\n{exc}")
            return

        segs = MetadataExporter.sources_of(meta)
        saved = [sg["file"] for sg in segs]
        have = [sg["file"] for sg in _segments_of(mw._data)] if mw._data is not None else []
        if saved and saved != have:
            missing = [sg["path"] for sg in segs if not Path(sg["path"]).exists()]
            box = QMessageBox(mw)
            box.setWindowTitle("Different file")
            box.setTextFormat(Qt.TextFormat.RichText)
            what = " + ".join(f"<b>{n}</b>" for n in saved)
            now = " + ".join(f"<b>{n}</b>" for n in have) if have else "nothing"
            box.setText(f"These settings were saved for {what}; {now} is loaded."
                        + (f"<br><br>Not found on disk: {', '.join(missing)}" if missing else ""))
            open_btn = None
            if not missing:
                open_btn = box.addButton("Open the saved file" + ("s" if len(segs) > 1 else ""),
                                         QMessageBox.ButtonRole.AcceptRole)
            apply_btn = None
            if mw._data is not None:
                apply_btn = box.addButton("Apply to the loaded file",
                                          QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or (clicked is not open_btn and clicked is not apply_btn):
                return
            if clicked is open_btn:
                # Turn the data on load, as it was when the settings were saved.
                before = mw.rotate_combo.currentIndex()
                i = mw.rotate_combo.findData(int(meta.get("rotation_deg_clockwise") or 0))
                if i >= 0:
                    mw.rotate_combo.blockSignals(True)
                    mw.rotate_combo.setCurrentIndex(i)
                    mw.rotate_combo.blockSignals(False)
                if not mw._load_sources(segs):
                    if mw._data is not None:     # the old file is still shown, as it was
                        mw.rotate_combo.blockSignals(True)
                        mw.rotate_combo.setCurrentIndex(before)
                        mw.rotate_combo.blockSignals(False)
                    return
        elif mw._data is None:
            QMessageBox.information(mw, "Nothing to apply to",
                                    "These settings have no source file recorded. Open a "
                                    "file first, then load them.")
            return

        notes = MetadataExporter.apply(mw, meta)
        msg = f"Settings restored from {Path(path).name}."
        if notes:
            msg += "\n\nNot restored:\n• " + "\n• ".join(notes)
        (QMessageBox.warning if notes else QMessageBox.information)(mw, "Settings loaded", msg)
        mw.statusBar().showMessage(f"Loaded settings from {Path(path).name}"
                                   + (f" ({len(notes)} item(s) skipped)" if notes else ""))


# ===========================================================================
# Timeline export dialog
# ===========================================================================

from PySide6.QtWidgets import QDialog, QScrollArea


class TimelineDialog(QDialog):
    """Settings dialog for exporting a publication-quality timeline figure."""

    def __init__(self, main_win: "MainWindow", parent=None) -> None:
        super().__init__(parent)
        self._mw = main_win
        self.setWindowTitle("Timeline Export")
        self.setMinimumWidth(560)
        self._build()

    # ---- build UI ---------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        form = QVBoxLayout(inner)
        form.setSpacing(6)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # --- Frame selection ------------------------------------------------
        fs_group = QGroupBox("Frame Selection")
        fs_lay = QHBoxLayout(fs_group)
        fs_lay.setContentsMargins(8, 6, 8, 6)
        self.frame_select_edit = QLineEdit()
        self.frame_select_edit.setPlaceholderText("e.g. 1, 5, 10-55:5")
        self.frame_select_edit.setToolTip(
            "Comma-separated frame numbers (1-indexed).\n"
            "Ranges: 10-30   Step ranges: 1-55:5 (every 5th frame)"
        )
        fs_lay.addWidget(self.frame_select_edit, 1)
        auto_btn = QPushButton("From crop range")
        auto_btn.setToolTip("Populate from the current export frame range.")
        auto_btn.clicked.connect(self._fill_from_crop_range)
        fs_lay.addWidget(auto_btn)
        form.addWidget(fs_group)

        # --- Channels -------------------------------------------------------
        ch_group = QGroupBox("Channels")
        ch_lay = QHBoxLayout(ch_group)
        ch_lay.setContentsMargins(8, 6, 8, 6)
        self.ch_phase_cb = QCheckBox("Phase"); self.ch_phase_cb.setChecked(True)
        self.ch_gfp_cb   = QCheckBox("GFP");   self.ch_gfp_cb.setChecked(True)
        self.ch_mch_cb   = QCheckBox("mCherry"); self.ch_mch_cb.setChecked(True)
        self.ch_composite_cb = QCheckBox("Composite"); self.ch_composite_cb.setChecked(False)
        self.colorize_cb = QCheckBox("Pseudocolor (GFP=green, mCherry=red)")
        self.colorize_cb.setChecked(True)
        for w in (self.ch_phase_cb, self.ch_gfp_cb, self.ch_mch_cb, self.ch_composite_cb,
                  QLabel("  "), self.colorize_cb):
            ch_lay.addWidget(w)
        ch_lay.addStretch(1)
        form.addWidget(ch_group)

        # --- Channel labels -------------------------------------------------
        cl_group = QGroupBox("Channel Labels  (shown left of images)")
        cl_lay = QGridLayout(cl_group)
        cl_lay.setContentsMargins(8, 6, 8, 6)
        cl_lay.setSpacing(4)
        self.cl_show_cb = QCheckBox("Show")
        self.cl_show_cb.setChecked(True)
        self.cl_phase_field     = QLineEdit("Phase");     self.cl_phase_field.setMaximumWidth(90)
        self.cl_gfp_field       = QLineEdit("GFP");       self.cl_gfp_field.setMaximumWidth(90)
        self.cl_mch_field       = QLineEdit("mCherry");   self.cl_mch_field.setMaximumWidth(90)
        self.cl_composite_field = QLineEdit("Merge"); self.cl_composite_field.setMaximumWidth(90)
        self.cl_font_spin = QSpinBox(); self.cl_font_spin.setRange(8, 72)
        self.cl_font_spin.setValue(20); self.cl_font_spin.setSuffix(" pt")
        cl_lay.addWidget(self.cl_show_cb, 0, 0)
        cl_lay.addWidget(QLabel("Phase:"),     0, 1); cl_lay.addWidget(self.cl_phase_field,     0, 2)
        cl_lay.addWidget(QLabel("GFP:"),       0, 3); cl_lay.addWidget(self.cl_gfp_field,       0, 4)
        cl_lay.addWidget(QLabel("mCherry:"),   0, 5); cl_lay.addWidget(self.cl_mch_field,       0, 6)
        cl_lay.addWidget(QLabel("Composite:"), 1, 1); cl_lay.addWidget(self.cl_composite_field, 1, 2)
        cl_lay.addWidget(QLabel("Font:"), 0, 7); cl_lay.addWidget(self.cl_font_spin, 0, 8)
        form.addWidget(cl_group)

        # --- Time labels ----------------------------------------------------
        tl_group = QGroupBox("Time Labels  (shown above each column)")
        tl_lay = QHBoxLayout(tl_group)
        tl_lay.setContentsMargins(8, 6, 8, 6)
        self.tl_show_cb = QCheckBox("Show")
        self.tl_show_cb.setChecked(True)
        self.tl_fmt_combo = QComboBox()
        for _lbl, _val in [
            ("Frame N", "frame"), ("t = N s", "seconds"),
            ("t = N min", "minutes"), ("H:MM:SS", "hms"),
        ]:
            self.tl_fmt_combo.addItem(_lbl, _val)
        self.tl_font_spin = QSpinBox(); self.tl_font_spin.setRange(8, 72)
        self.tl_font_spin.setValue(18); self.tl_font_spin.setSuffix(" pt")
        tl_lay.addWidget(self.tl_show_cb)
        tl_lay.addWidget(self.tl_fmt_combo)
        tl_lay.addWidget(QLabel("Font:"))
        tl_lay.addWidget(self.tl_font_spin)
        tl_lay.addStretch(1)
        form.addWidget(tl_group)

        # --- Scale bar ------------------------------------------------------
        sb_group2 = QGroupBox("Scale Bar")
        sb_lay2 = QHBoxLayout(sb_group2)
        sb_lay2.setContentsMargins(8, 6, 8, 6)
        self.tl_sb_cb = QCheckBox("Show")
        self.tl_sb_cb.setChecked(True)
        self.tl_sb_len_spin = QDoubleSpinBox()
        self.tl_sb_len_spin.setRange(0.1, 10000); self.tl_sb_len_spin.setValue(10)
        self.tl_sb_len_spin.setSuffix(" µm"); self.tl_sb_len_spin.setDecimals(1)
        self.tl_sb_len_spin.setMinimumWidth(90)
        self.tl_sb_panel_combo = QComboBox()
        for _l, _v in [("Bottom-right panel", "br"), ("Bottom-left panel", "bl"),
                       ("Top-right panel", "tr"), ("Top-left panel", "tl")]:
            self.tl_sb_panel_combo.addItem(_l, _v)
        sb_lay2.addWidget(self.tl_sb_cb)
        sb_lay2.addWidget(self.tl_sb_len_spin)
        sb_lay2.addWidget(QLabel("In:"))
        sb_lay2.addWidget(self.tl_sb_panel_combo)
        sb_lay2.addStretch(1)
        form.addWidget(sb_group2)

        # --- Layout & output ------------------------------------------------
        lo_group = QGroupBox("Layout & Output")
        lo_lay = QGridLayout(lo_group)
        lo_lay.setContentsMargins(8, 6, 8, 6)
        lo_lay.setSpacing(6)
        self.spacing_spin = QSpinBox(); self.spacing_spin.setRange(0, 100)
        self.spacing_spin.setValue(4); self.spacing_spin.setSuffix(" px")
        self.bg_combo = QComboBox()
        self.bg_combo.addItem("Black background", False)
        self.bg_combo.addItem("White background", True)
        self.fmt_combo2 = QComboBox()
        self.fmt_combo2.addItem("PNG (.png)", "png")
        self.fmt_combo2.addItem("TIFF (.tif)", "tif")
        self.fmt_combo2.setCurrentIndex(1)   # TIFF by default
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(600); self.dpi_spin.setSuffix(" DPI")
        lo_lay.addWidget(QLabel("Panel gap:"), 0, 0); lo_lay.addWidget(self.spacing_spin, 0, 1)
        lo_lay.addWidget(QLabel("Background:"), 0, 2); lo_lay.addWidget(self.bg_combo, 0, 3)
        lo_lay.addWidget(QLabel("Format:"), 1, 0);  lo_lay.addWidget(self.fmt_combo2, 1, 1)
        lo_lay.addWidget(QLabel("DPI:"), 1, 2);     lo_lay.addWidget(self.dpi_spin, 1, 3)
        form.addWidget(lo_group)
        form.addStretch(1)

        # --- Buttons --------------------------------------------------------
        btn_row = QHBoxLayout()
        self.export_timeline_btn = QPushButton("Export Timeline Figure…")
        self.export_timeline_btn.setDefault(True)
        self.export_timeline_btn.clicked.connect(self._on_export_timeline)
        self.export_frames_btn = QPushButton("Export Individual Frames…")
        self.export_frames_btn.clicked.connect(self._on_export_frames)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.export_timeline_btn)
        btn_row.addWidget(self.export_frames_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        # Pre-populate interval from main window
        if self._mw._data is not None:
            end = self._mw.end_frame_spin.value()
            start = self._mw.start_frame_spin.value()
            self.frame_select_edit.setText(f"{start}, {end}")

    # ---- helpers ----------------------------------------------------------

    def _fill_from_crop_range(self) -> None:
        if self._mw._data is None:
            return
        s = self._mw.start_frame_spin.value()
        e = self._mw.end_frame_spin.value()
        self.frame_select_edit.setText(f"{s}-{e}")

    def _selected_channels(self) -> list[str]:
        ch = []
        if self.ch_phase_cb.isChecked():     ch.append("phase")
        if self.ch_gfp_cb.isChecked():       ch.append("gfp")
        if self.ch_mch_cb.isChecked():       ch.append("mcherry")
        if self.ch_composite_cb.isChecked(): ch.append("composite")
        return ch

    def _channel_names(self) -> dict[str, str]:
        return {
            "phase":     self.cl_phase_field.text(),
            "gfp":       self.cl_gfp_field.text(),
            "mcherry":   self.cl_mch_field.text(),
            "composite": self.cl_composite_field.text(),
        }

    def _luts(self) -> dict:
        """The display ranges as the video export uses them: one per channel,
        or one per frame in the Smart / Manual per-frame modes (the range of
        the frame on the slider used to be applied to every frame)."""
        e = self._mw._export_luts()
        return {
            "phase":   e["phase_luts"] or e["phase_lut"],
            "gfp":     e["gfp_luts"] or e["gfp_lut"],
            "mcherry": e["mcherry_luts"] or e["mcherry_lut"],
        }

    def _scale_bar_fits(self) -> bool:
        """The scale bar is drawn on one panel. Stop, rather than export
        without it, when it is longer than the panel is wide."""
        d = self._mw._data
        if not (self.tl_sb_cb.isChecked() and d.um_per_px):
            return True
        crop = self._mw.roi_view.crop_roi()
        panel_w = crop[2] if crop else d.width
        bar_px = int(round(self.tl_sb_len_spin.value() / d.um_per_px))
        if bar_px > panel_w:
            QMessageBox.warning(
                self, "Scale bar does not fit",
                f"A {self.tl_sb_len_spin.value():g} µm bar is {bar_px} px, but each "
                f"panel is only {panel_w} px wide ({panel_w * d.um_per_px:.1f} µm). "
                "Choose a shorter bar and try again.")
            return False
        return True

    def _get_frame_indices(self) -> list[int]:
        if self._mw._data is None:
            return []
        return _parse_frame_list(
            self.frame_select_edit.text(), self._mw._data.n_frames
        )

    # ---- export actions ---------------------------------------------------

    def _on_export_timeline(self) -> None:
        mw = self._mw
        if mw._data is None:
            QMessageBox.warning(self, "No data", "Please load an ND2 file first.")
            return
        channels = self._selected_channels()
        if not channels:
            QMessageBox.warning(self, "No channels", "Select at least one channel.")
            return
        frame_indices = self._get_frame_indices()
        if not frame_indices:
            QMessageBox.warning(self, "No frames",
                                "Enter valid frame numbers in the Frame Selection field.")
            return
        if not self._scale_bar_fits():
            return

        fmt = self.fmt_combo2.currentData()
        suffix = ".tif" if fmt == "tif" else ".png"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Timeline Figure",
            str(Path.home() / f"timeline{suffix}"),
            "TIFF (*.tif *.tiff)" if fmt == "tif" else "PNG (*.png)"
        )
        if not out_path:
            return
        # The file type is taken from the name, and not every file dialog
        # adds the extension.
        if Path(out_path).suffix.lower() not in (".tif", ".tiff", ".png"):
            out_path += suffix

        try:
            luts = self._luts()
            panels = TimelineExporter.render_panels(
                mw._data, mw._position,
                mw.roi_view.bg_roi(), mw.roi_view.crop_roi(),
                frame_indices, channels, luts,
                colorize=self.colorize_cb.isChecked(),
                merge_mode=mw.merge_combo.currentData(),
                visibility=ChannelVisibility(
                    phase=mw.phase_cb.isChecked(), gfp=mw.gfp_cb.isChecked(),
                    mcherry=mw.mch_cb.isChecked()),
            )

            # Time labels
            time_labels = None
            if self.tl_show_cb.isChecked():
                interval = mw.timer_interval_spin.value()
                fmt_str = self.tl_fmt_combo.currentData()
                decimals = mw.timer_decimals_spin.value()
                t0 = mw.timer_t0_spin.value() - 1
                times = mw._timer_times()
                time_labels = [
                    Pipeline._format_time(fi, fmt_str, interval, decimals, t0, times)
                    for fi in frame_indices
                ]

            # Scale bar info
            sb_info = None
            if self.tl_sb_cb.isChecked() and mw._data.um_per_px:
                sv = self.tl_sb_panel_combo.currentData()
                sb_row = (len(channels) - 1) if sv.startswith("b") else 0
                sb_col = (len(frame_indices) - 1) if sv.endswith("r") else 0
                sb_info = {
                    "um_per_px": mw._data.um_per_px,
                    "length_um": self.tl_sb_len_spin.value(),
                    "row": sb_row, "col": sb_col,
                }

            figure = TimelineExporter.compose(
                panels, channels,
                channel_names=self._channel_names() if self.cl_show_cb.isChecked() else {},
                time_labels=time_labels,
                scale_bar_info=sb_info,
                spacing=self.spacing_spin.value(),
                bg_white=self.bg_combo.currentData(),
                ch_label_font=self.cl_font_spin.value(),
                time_label_font=self.tl_font_spin.value(),
            )
            TimelineExporter.save(figure, out_path, fmt=fmt, dpi=self.dpi_spin.value())
            QMessageBox.information(self, "Done", f"Timeline saved:\n{out_path}")
        except Exception:
            import traceback
            QMessageBox.critical(self, "Export failed", traceback.format_exc())

    def _on_export_frames(self) -> None:
        """Export each selected frame as an individual image file."""
        mw = self._mw
        if mw._data is None:
            QMessageBox.warning(self, "No data", "Please load an ND2 file first.")
            return
        channels = self._selected_channels()
        if not channels:
            QMessageBox.warning(self, "No channels", "Select at least one channel.")
            return
        frame_indices = self._get_frame_indices()
        if not frame_indices:
            QMessageBox.warning(self, "No frames",
                                "Enter valid frame numbers in the Frame Selection field.")
            return
        if not self._scale_bar_fits():
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path.home())
        )
        if not out_dir:
            return

        fmt = self.fmt_combo2.currentData()
        ext = ".tif" if fmt == "tif" else ".png"
        try:
            luts = self._luts()
            panels = TimelineExporter.render_panels(
                mw._data, mw._position,
                mw.roi_view.bg_roi(), mw.roi_view.crop_roi(),
                frame_indices, channels, luts,
                colorize=self.colorize_cb.isChecked(),
                merge_mode=mw.merge_combo.currentData(),
                visibility=ChannelVisibility(
                    phase=mw.phase_cb.isChecked(), gfp=mw.gfp_cb.isChecked(),
                    mcherry=mw.mch_cb.isChecked()),
            )
            # Includes the position and "_joined", so that frames of another
            # position do not replace these in the same folder.
            stem = mw._export_stem()
            for col_idx, fi in enumerate(frame_indices):
                # Stack the selected channel panels vertically for this frame
                col_panels = [panels[ch][col_idx] for ch in channels]
                col_img = np.concatenate(col_panels, axis=0)

                # Optionally draw a scale bar on the bottom panel
                if self.tl_sb_cb.isChecked() and mw._data.um_per_px:
                    ph = col_panels[0].shape[0]
                    bar_region = col_img[-ph:].copy()
                    Pipeline._draw_scale_bar(
                        bar_region, mw._data.um_per_px, self.tl_sb_len_spin.value(),
                        position="bottom_right", x_offset=12, y_offset=12,
                        text_center=True, text_above=True,
                    )
                    col_img[-ph:] = bar_region

                TimelineExporter.save(
                    col_img,
                    Path(out_dir) / f"{stem}_frame{fi + 1:04d}{ext}",
                    fmt=fmt, dpi=self.dpi_spin.value(),
                )
            QMessageBox.information(
                self, "Done",
                f"Saved {len(frame_indices)} frame(s) to:\n{out_dir}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


# ===========================================================================
# Main window
# ===========================================================================

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        # 1200x920 fits on 13" MacBook screens (1440x900 logical) with the
        # menu bar and Dock visible. Windows users with larger monitors can
        # resize freely.
        self.resize(1200, 920)
        _scr = QApplication.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(1440, int(_av.width() * 0.96)),
                        min(920, int(_av.height() * 0.96)))
        self.setAcceptDrops(True)
        self._data: Optional[ND2Data] = None
        self._position: int = 0
        self._time: int = 0
        # Cached float32 stacks for the *current* position (computed on
        # position change, reused for every preview render).
        self._phase_pos: Optional[np.ndarray] = None
        self._gfp_pos: Optional[np.ndarray] = None
        self._mch_pos: Optional[np.ndarray] = None
        # Cached bg-subtracted float32 stacks (recomputed when BG ROI changes).
        self._gfp_sub: Optional[np.ndarray] = None
        self._mch_sub: Optional[np.ndarray] = None
        # LUT mode + per-frame storage.
        self._lut_mode: str = LUT_STACK
        self._per_frame_luts: dict[str, list[ChannelLUT]] = {
            "phase": [], "gfp": [], "mcherry": []
        }
        # Ranges set in Manual per-frame mode, kept while another mode is shown.
        self._manual_luts: Optional[dict[str, list[ChannelLUT]]] = None
        # Suppress LUT-widget signals during programmatic updates (e.g. loading
        # the per-frame LUT for the current timepoint).
        self._suppress_lut_save: bool = False
        self._rotation_deg = 0   # clockwise; applied to self._data on load
        self._build_ui()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Tab pages — populated as widgets are built below. The QTabWidget
        # itself is added to `layout` after the preview canvas so that the
        # preview stays visible on every tab. The pages are scrollable
        # because the LUT histograms can exceed window height.
        from PySide6.QtWidgets import QScrollArea  # local — small dep surface
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        def _make_page() -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            v = QVBoxLayout(page)
            v.setContentsMargins(6, 6, 6, 6); v.setSpacing(6)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            return page, v, scroll

        display_page, display_layout, display_scroll = _make_page()
        export_page,  export_layout,  export_scroll  = _make_page()
        self.tabs.addTab(display_scroll,  "Display")
        self.tabs.addTab(export_scroll,   "Export")

        # Toolbar
        toolbar = QHBoxLayout()
        self.open_btn = QPushButton("Open .nd2…")
        self.open_btn.clicked.connect(self._on_open)
        toolbar.addWidget(self.open_btn)
        self.append_btn = QPushButton("Append .nd2…")
        self.append_btn.setEnabled(False)
        self.append_btn.setToolTip(
            "Add a continuation file after the last frame of the one loaded\n"
            "(e.g. a run restarted the next day). Checks the size, positions and\n"
            "pixel size match, lines up the field at the join, and keeps the real\n"
            "time gap in the timer. Nothing is written to disk.")
        self.append_btn.clicked.connect(self._on_append)
        toolbar.addWidget(self.append_btn)
        self.process_btn = QPushButton("Process and Export")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._on_process)
        toolbar.addWidget(self.process_btn)
        self.timeline_btn = QPushButton("Timeline Export…")
        self.timeline_btn.setEnabled(False)
        self.timeline_btn.setToolTip(
            "Export selected frames as a publication-quality timeline figure "
            "with channels stacked vertically."
        )
        self.timeline_btn.clicked.connect(self._on_timeline_export)
        toolbar.addWidget(self.timeline_btn)
        self.example_btn = QPushButton("Export Example Set…")
        self.example_btn.setEnabled(False)
        self.example_btn.setProperty("accent", True)
        self.example_btn.setToolTip(
            "One click: per-channel TIFF folders, both composites, scale-bar\n"
            "frames and two 4-panel videos, all from the current LUTs, crop,\n"
            "background ROI, merge mode and export frame range.\n"
            "Options are in the Export tab under 'Example set'."
        )
        self.example_btn.clicked.connect(self._on_export_example_set)
        toolbar.addWidget(self.example_btn)
        self.save_meta_btn = QPushButton("Save Metadata…")
        self.save_meta_btn.setEnabled(False)
        self.save_meta_btn.setToolTip(
            "Save a JSON file recording all current settings (file path, crop/BG ROIs, "
            "LUTs, scale bar, timer, etc.) so you can reproduce the exact output later."
        )
        self.save_meta_btn.clicked.connect(lambda: MetadataExporter.save(self))
        toolbar.addWidget(self.save_meta_btn)
        self.load_meta_btn = QPushButton("Load Settings…")
        self.load_meta_btn.setToolTip(
            "Restore a settings .json (from Save Metadata, or the example_settings.json /\n"
            "strip_settings.json written beside an export): display ranges, background and\n"
            "crop boxes, rotation, frame range, merge mode, scale bar, timer and labels.\n"
            "If the file it was saved from is not the one loaded, it can open it for you,\n"
            "including a joined pair with the same alignment.")
        self.load_meta_btn.clicked.connect(lambda: MetadataExporter.load(self))
        toolbar.addWidget(self.load_meta_btn)
        toolbar.addSpacing(20)
        # FPS + quality live in their own widget: on the toolbar in the wide
        # layout, at the top of the Export tab in the laptop layouts.
        self._w_video = QWidget()
        video_row = QHBoxLayout(self._w_video)
        video_row.setContentsMargins(0, 0, 0, 0); video_row.setSpacing(6)
        self._w_video.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        video_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(5)
        self.fps_spin.setToolTip("Output frame rate for the exported MP4.")
        video_row.addWidget(self.fps_spin)
        video_row.addSpacing(14)
        video_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        for _lbl, _val in [
            ("Best (CRF 0, 4:4:4)",          (0,  "yuv444p")),
            ("Near-lossless (CRF 1, 4:4:4)", (1,  "yuv444p")),
            ("High        (CRF 5, 4:4:4)",   (5,  "yuv444p")),
            ("Compatible  (CRF 10, 4:2:0)",  (10, "yuv420p")),
        ]:
            self.quality_combo.addItem(_lbl, _val)
        self.quality_combo.setToolTip(
            "Best = H.264 CRF 0 4:4:4: as decoded by macOS, pixels are within 1 grey\n"
            "level of the preview. (True RGB-lossless formats were tested: macOS either\n"
            "can't open them or shows wrong colours.) CRF 1 is within ~4 levels.\n"
            "4:4:4 videos do not play in Windows' own players or in PowerPoint; use VLC,\n"
            "QuickTime, Premiere or ImageJ, or choose Compatible.\n"
            "4:2:0 plays everywhere but halves colour resolution — avoid for fine detail.\n"
            "If a video looks soft, the cause is usually enlargement, not compression:\n"
            "use Size to enlarge with sharp pixels."
        )
        video_row.addWidget(self.quality_combo)
        video_row.addSpacing(14)
        video_row.addWidget(QLabel("Size:"))
        self.vscale_combo = _make_video_scale_combo(default=1)
        video_row.addWidget(self.vscale_combo)
        toolbar.addWidget(self._w_video)
        toolbar.addStretch(1)
        toolbar.addSpacing(14)
        toolbar.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        for _lbl, _val in LAYOUTS:
            self.layout_combo.addItem(_lbl, _val)
        self.layout_combo.setToolTip(
            "Wide monitor: controls under the image (the original layout).\n"
            "Laptop: image on the left at full height, controls in a side panel.\n"
            "Image focus: as Laptop with the side panel folded away — drag the\n"
            f"divider at the right edge, or press {_CMD_KEY}\\, to bring it back.")
        toolbar.addWidget(self.layout_combo)
        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        for _tname, _theme in THEMES.items():
            self.theme_combo.addItem(_theme.display, _tname)
        # Connect the signal BEFORE setting the index so the initial change
        # fires _on_theme_changed and actually applies the Y2K stylesheet
        # (otherwise the app keeps whatever theme main() preloaded).
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        _y2k_idx = next(i for i, k in enumerate(THEMES) if k == "y2k")
        self.theme_combo.setCurrentIndex(_y2k_idx)
        # Belt-and-suspenders: if Y2K happened to be index 0 (no change signal),
        # apply it explicitly so we're never out of sync with the combo.
        self._on_theme_changed()
        toolbar.addWidget(self.theme_combo)
        toolbar.setContentsMargins(0, 0, 0, 0)
        self._toolbar = toolbar
        self._video_idx = toolbar.indexOf(self._w_video)
        self._export_layout = export_layout
        self._w_top = QWidget(); self._w_top.setLayout(toolbar)

        self.info_label = QLabel("No file loaded. Click 'Open .nd2…' to begin.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.RichText)

        # Navigation row: time slider + position combo
        nav = QHBoxLayout()
        nav.addWidget(QLabel("Time"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        nav.addWidget(self.time_slider, stretch=1)
        self.time_label = QLabel("- / -")
        self.time_label.setMinimumWidth(70)
        nav.addWidget(self.time_label)
        nav.addSpacing(20)
        nav.addWidget(QLabel("Position"))
        self.pos_combo = QComboBox()
        self.pos_combo.setMinimumWidth(80)
        self.pos_combo.setEnabled(False)
        self.pos_combo.currentIndexChanged.connect(self._on_position_changed)
        nav.addWidget(self.pos_combo)
        nav.setContentsMargins(0, 0, 0, 0)
        self.time_slider.sliderPressed.connect(self._stop_play)   # grabbing it pauses

        # Second line: playback and which panel to show.
        play_row = FlowLayout(hspacing=8)
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setEnabled(False)
        self.play_btn.setMinimumWidth(88)
        self.play_btn.setToolTip(
            "Play the timelapse in the preview (shortcut: P).\n"
            "Plays the export frame range (Start–End frames), so trimming out-of-focus\n"
            "frames there also trims what plays. Drag the slider to pause.")
        self.play_btn.clicked.connect(self._toggle_play)
        play_row.addWidget(self.play_btn)
        play_row.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for _fps in PLAY_SPEEDS:
            self.speed_combo.addItem(f"{_fps} fps", _fps)
        self.speed_combo.setCurrentIndex(PLAY_SPEEDS.index(8))
        self.speed_combo.setToolTip("Playback speed in frames per second. The preview "
                                    "renders at roughly 25-30 fps at most on a 1024 px field.")
        self.speed_combo.currentIndexChanged.connect(self._on_play_speed_changed)
        play_row.addWidget(self.speed_combo)
        self.loop_cb = QCheckBox("Loop")
        self.loop_cb.setChecked(True)
        play_row.addWidget(self.loop_cb)
        play_row.addSpacing(18)
        play_row.addWidget(QLabel("Show:"))
        self.view_group = QButtonGroup(self)
        self.view_buttons: list = []
        for _i, (_btn_lbl, _combo_lbl, _val) in enumerate(VIEW_CHOICES):
            _b = QToolButton(); _b.setText(_btn_lbl); _b.setCheckable(True)
            _b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            _b.setToolTip(("Show all four panels (phase, GFP, mCherry, merge)."
                           if _val is None else f"Show only the {_combo_lbl} panel, large.")
                          + f"\nShortcut: {_CMD_KEY}{_i + 1}.\nPreview only: it does not change what is exported.")
            self.view_group.addButton(_b, _i)
            self.view_buttons.append(_b)
            play_row.addWidget(_b)
        self.view_buttons[0].setChecked(True)
        self.view_group.idClicked.connect(lambda i: self.view_combo.setCurrentIndex(i))
        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._play_tick)

        nav_box = QVBoxLayout(); nav_box.setContentsMargins(0, 0, 0, 0); nav_box.setSpacing(3)
        nav_box.addLayout(nav); nav_box.addLayout(play_row)
        self._w_nav = QWidget(); self._w_nav.setLayout(nav_box)

        # Export frame range + overlay settings row
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Export frames:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(1, 1)
        self.start_frame_spin.setEnabled(False)
        self.start_frame_spin.setToolTip("First frame to include in the export (1-indexed).")
        export_row.addWidget(self.start_frame_spin)
        export_row.addWidget(QLabel("–"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(1, 1)
        self.end_frame_spin.setEnabled(False)
        self.end_frame_spin.setToolTip("Last frame to include in the export (1-indexed).")
        export_row.addWidget(self.end_frame_spin)
        self.export_count_label = QLabel("")
        export_row.addWidget(self.export_count_label)
        export_row.addStretch(1)

        # Scale bar group
        sb_group = QGroupBox("Scale Bar")
        sb_vlay = QVBoxLayout(sb_group)
        sb_vlay.setContentsMargins(8, 6, 8, 6); sb_vlay.setSpacing(3)
        sb_row1 = QHBoxLayout(); sb_row1.setSpacing(6)
        self.sb_visible_cb = QCheckBox("Show")
        self.sb_visible_cb.setChecked(True)
        sb_row1.addWidget(self.sb_visible_cb)
        self.sb_size_spin = QDoubleSpinBox()
        self.sb_size_spin.setRange(0.1, 10000.0); self.sb_size_spin.setValue(2.0)
        self.sb_size_spin.setSuffix(" µm"); self.sb_size_spin.setDecimals(1)
        self.sb_size_spin.setMinimumWidth(90)
        self.sb_size_spin.setToolTip("Scale bar length in micrometres.")
        sb_row1.addWidget(self.sb_size_spin)
        self.sb_pos_combo = QComboBox()
        for _lbl, _val in [
            ("Bottom Right", "bottom_right"), ("Bottom Left", "bottom_left"),
            ("Top Right",    "top_right"),    ("Top Left",   "top_left"),
        ]:
            self.sb_pos_combo.addItem(_lbl, _val)
        sb_row1.addWidget(self.sb_pos_combo)
        sb_row1.addStretch(1)
        sb_row2 = QHBoxLayout(); sb_row2.setSpacing(4)
        sb_row2.addWidget(QLabel("X:"))
        self.sb_x_spin = QSpinBox(); self.sb_x_spin.setRange(0, 2000)
        self.sb_x_spin.setValue(20); self.sb_x_spin.setSuffix(" px")
        self.sb_x_spin.setMaximumWidth(72); self.sb_x_spin.setToolTip("Pixels from horizontal edge.")
        sb_row2.addWidget(self.sb_x_spin)
        sb_row2.addWidget(QLabel("Y:"))
        self.sb_y_spin = QSpinBox(); self.sb_y_spin.setRange(0, 2000)
        self.sb_y_spin.setValue(20); self.sb_y_spin.setSuffix(" px")
        self.sb_y_spin.setMaximumWidth(72); self.sb_y_spin.setToolTip("Pixels from vertical edge.")
        sb_row2.addWidget(self.sb_y_spin)
        self.sb_center_cb = QCheckBox("Center text")
        self.sb_center_cb.setChecked(True)
        self.sb_center_cb.setToolTip("Center the µm label horizontally over the bar.")
        sb_row2.addWidget(self.sb_center_cb)
        self.sb_above_cb = QCheckBox("Text above")
        self.sb_above_cb.setChecked(False)
        self.sb_above_cb.setToolTip("Place the µm label above (vs below) the bar line.")
        sb_row2.addWidget(self.sb_above_cb)
        sb_row2.addStretch(1)
        sb_vlay.addLayout(sb_row1); sb_vlay.addLayout(sb_row2)
        export_layout.addWidget(self._build_example_set_group())
        export_layout.addWidget(self._build_strip_group())
        export_layout.addWidget(sb_group)  # moved to Export tab

        # Timer group
        timer_group = QGroupBox("Timer")
        timer_vlay = QVBoxLayout(timer_group)
        timer_vlay.setContentsMargins(8, 6, 8, 6); timer_vlay.setSpacing(3)
        timer_row1 = QHBoxLayout(); timer_row1.setSpacing(6)
        self.timer_cb = QCheckBox("Show")
        timer_row1.addWidget(self.timer_cb)
        self.timer_fmt_combo = QComboBox()
        for _lbl, _val in [
            ("Frame N",   "frame"),   ("t = N s",    "seconds"),
            ("t = N min", "minutes"), ("H:MM:SS",    "hms"),
        ]:
            self.timer_fmt_combo.addItem(_lbl, _val)
        timer_row1.addWidget(self.timer_fmt_combo)
        timer_row1.addWidget(QLabel("Interval:"))
        self.timer_interval_spin = QDoubleSpinBox()
        self.timer_interval_spin.setDecimals(2)   # before the range: it rounds the limits
        self.timer_interval_spin.setRange(0.01, 86400.0); self.timer_interval_spin.setValue(60.0)
        self.timer_interval_spin.setSuffix(" s/fr")
        self.timer_interval_spin.setMinimumWidth(90); self.timer_interval_spin.setEnabled(False)
        self.timer_interval_spin.setToolTip(
            "Time between frames in seconds. Used by all timer formats except 'Frame N'."
        )
        timer_row1.addWidget(self.timer_interval_spin)
        self.timer_clock_combo = QComboBox()
        self.timer_clock_combo.addItem("Fixed interval", "fixed")
        self.timer_clock_combo.addItem("Real acquisition times", "real")
        self.timer_clock_combo.setEnabled(False)
        self.timer_clock_combo.setToolTip(
            "Fixed interval: frame N shows N × interval — round numbers.\n"
            "Real acquisition times: the timestamp the microscope recorded for\n"
            "each frame (intervals slip when the scope is busy).\n\n"
            "For appended files, Fixed interval still restarts each file at its\n"
            "real start time, so the gap at the join is never hidden.")
        timer_row1.addWidget(self.timer_clock_combo)
        timer_row1.addWidget(QLabel("T0 frame:"))
        self.timer_t0_spin = QSpinBox()
        self.timer_t0_spin.setRange(0, 9999); self.timer_t0_spin.setValue(0)
        self.timer_t0_spin.setMinimumWidth(58)
        self.timer_t0_spin.setToolTip(
            "Frame number (1-based) to treat as time zero.\n"
            "Frames before this will show negative times."
        )
        timer_row1.addWidget(self.timer_t0_spin)
        self.timer_pos_combo = QComboBox()
        for _lbl, _val in [
            ("Top Left",     "top_left"),     ("Top Right",    "top_right"),
            ("Bottom Left",  "bottom_left"),  ("Bottom Right", "bottom_right"),
        ]:
            self.timer_pos_combo.addItem(_lbl, _val)
        timer_row1.addWidget(self.timer_pos_combo)
        timer_row1.addStretch(1)
        timer_row2 = QHBoxLayout(); timer_row2.setSpacing(4)
        timer_row2.addWidget(QLabel("X:"))
        self.timer_x_spin = QSpinBox(); self.timer_x_spin.setRange(0, 2000)
        self.timer_x_spin.setValue(20); self.timer_x_spin.setSuffix(" px")
        self.timer_x_spin.setMaximumWidth(72)
        timer_row2.addWidget(self.timer_x_spin)
        timer_row2.addWidget(QLabel("Y:"))
        self.timer_y_spin = QSpinBox(); self.timer_y_spin.setRange(0, 2000)
        self.timer_y_spin.setValue(20); self.timer_y_spin.setSuffix(" px")
        self.timer_y_spin.setMaximumWidth(72)
        timer_row2.addWidget(self.timer_y_spin)
        timer_row2.addWidget(QLabel("Decimals:"))
        self.timer_decimals_spin = QSpinBox(); self.timer_decimals_spin.setRange(0, 3)
        self.timer_decimals_spin.setValue(0); self.timer_decimals_spin.setMaximumWidth(48)
        self.timer_decimals_spin.setToolTip("Decimal places for seconds/minutes formats")
        timer_row2.addWidget(self.timer_decimals_spin)
        timer_row2.addStretch(1)
        timer_vlay.addLayout(timer_row1); timer_vlay.addLayout(timer_row2)
        export_layout.addWidget(timer_group)  # moved to Export tab
        export_row.setContentsMargins(0, 0, 0, 0)
        self._frames_row = export_row   # holds _w_video in the wide layout
        self._w_frames = QWidget(); self._w_frames.setLayout(export_row)

        # Crop Shape goes on Display tab; Channel Labels goes on Export tab.
        # (The old ann2_row that wrapped them is no longer needed.)

        # Crop constraint group
        crop_con_group = QGroupBox("Crop Shape")
        cc_layout = QHBoxLayout(crop_con_group)
        cc_layout.setContentsMargins(8, 6, 8, 6)
        cc_layout.setSpacing(6)
        self.crop_constraint_combo = QComboBox()
        for _lbl, _val in [
            ("Free",         "free"),
            ("Square 1:1",   "1:1"),
            ("Widescreen 16:9", "16:9"),
            ("Portrait 9:16",  "9:16"),
            ("Standard 4:3",   "4:3"),
            ("Fixed pixels",   "pixels"),
        ]:
            self.crop_constraint_combo.addItem(_lbl, _val)
        self.crop_constraint_combo.setToolTip(
            "Constrain the crop rectangle to a fixed aspect ratio or exact pixel size."
        )
        cc_layout.addWidget(self.crop_constraint_combo)
        self.crop_px_w_spin = QSpinBox()
        self.crop_px_w_spin.setRange(1, 99999); self.crop_px_w_spin.setValue(512)
        self.crop_px_w_spin.setPrefix("W: "); self.crop_px_w_spin.setVisible(False)
        cc_layout.addWidget(self.crop_px_w_spin)
        self.crop_px_h_spin = QSpinBox()
        self.crop_px_h_spin.setRange(1, 99999); self.crop_px_h_spin.setValue(512)
        self.crop_px_h_spin.setPrefix("H: "); self.crop_px_h_spin.setVisible(False)
        cc_layout.addWidget(self.crop_px_h_spin)
        # The shape selector now lives in the crop bar above the image
        # (see _build_crop_bar); this group is no longer shown.
        crop_con_group.setVisible(False)
        self._crop_con_group = crop_con_group   # keep alive: owns crop_px_*_spin

        # Channel labels group
        labels_group = QGroupBox("Channel Labels")
        lbl_layout = QHBoxLayout(labels_group)
        lbl_layout.setContentsMargins(8, 6, 8, 6)
        lbl_layout.setSpacing(6)
        self.labels_cb = QCheckBox("Show")
        lbl_layout.addWidget(self.labels_cb)
        self.label_pos_combo = QComboBox()
        for _lbl, _val in [
            ("Top Left",     "top_left"),    ("Top Right",    "top_right"),
            ("Bottom Left",  "bottom_left"), ("Bottom Right", "bottom_right"),
        ]:
            self.label_pos_combo.addItem(_lbl, _val)
        lbl_layout.addWidget(self.label_pos_combo)
        lbl_layout.addWidget(QLabel("Phase:"))
        self.label_phase_field = QLineEdit("Phase")
        self.label_phase_field.setMaximumWidth(70)
        lbl_layout.addWidget(self.label_phase_field)
        lbl_layout.addWidget(QLabel("GFP:"))
        self.label_gfp_field = QLineEdit("GFP")
        self.label_gfp_field.setMaximumWidth(60)
        lbl_layout.addWidget(self.label_gfp_field)
        lbl_layout.addWidget(QLabel("mCherry:"))
        self.label_mcherry_field = QLineEdit("mCherry")
        self.label_mcherry_field.setMaximumWidth(70)
        lbl_layout.addWidget(self.label_mcherry_field)
        lbl_layout.addWidget(QLabel("Composite:"))
        self.label_composite_field = QLineEdit("Merge")
        self.label_composite_field.setMaximumWidth(80)
        lbl_layout.addWidget(self.label_composite_field)
        lbl_layout.addSpacing(6)
        lbl_layout.addWidget(QLabel("X:"))
        self.label_x_spin = QSpinBox(); self.label_x_spin.setRange(0, 2000)
        self.label_x_spin.setValue(8); self.label_x_spin.setSuffix(" px")
        self.label_x_spin.setMaximumWidth(68)
        lbl_layout.addWidget(self.label_x_spin)
        lbl_layout.addWidget(QLabel("Y:"))
        self.label_y_spin = QSpinBox(); self.label_y_spin.setRange(0, 2000)
        self.label_y_spin.setValue(8); self.label_y_spin.setSuffix(" px")
        self.label_y_spin.setMaximumWidth(68)
        lbl_layout.addWidget(self.label_y_spin)
        export_layout.addWidget(labels_group)  # moved to Export tab
        # (ann2_row is now empty since both groups were re-homed into tabs)

        # Wire up new controls
        self.start_frame_spin.valueChanged.connect(self._on_frame_range_changed)
        self.end_frame_spin.valueChanged.connect(self._on_frame_range_changed)
        self.sb_visible_cb.toggled.connect(self._update_preview)
        self.sb_size_spin.valueChanged.connect(self._update_preview)
        self.sb_pos_combo.currentIndexChanged.connect(self._update_preview)
        self.sb_x_spin.valueChanged.connect(self._update_preview)
        self.sb_y_spin.valueChanged.connect(self._update_preview)
        self.sb_center_cb.toggled.connect(self._update_preview)
        self.sb_above_cb.toggled.connect(self._update_preview)
        self.timer_cb.toggled.connect(self._on_timer_toggled)
        self.timer_fmt_combo.currentIndexChanged.connect(self._on_timer_format_changed)
        self.timer_interval_spin.valueChanged.connect(self._update_preview)
        self.timer_clock_combo.currentIndexChanged.connect(self._update_preview)
        self.timer_t0_spin.valueChanged.connect(self._update_preview)
        self.timer_pos_combo.currentIndexChanged.connect(self._update_preview)
        self.timer_x_spin.valueChanged.connect(self._update_preview)
        self.timer_y_spin.valueChanged.connect(self._update_preview)
        self.timer_decimals_spin.valueChanged.connect(self._update_preview)
        self.crop_constraint_combo.currentIndexChanged.connect(self._on_crop_constraint_changed)
        self.crop_px_w_spin.valueChanged.connect(self._on_crop_constraint_changed)
        self.crop_px_h_spin.valueChanged.connect(self._on_crop_constraint_changed)
        self.labels_cb.toggled.connect(self._update_preview)
        self.label_pos_combo.currentIndexChanged.connect(self._update_preview)
        self.label_x_spin.valueChanged.connect(self._update_preview)
        self.label_y_spin.valueChanged.connect(self._update_preview)
        for _field in (self.label_phase_field, self.label_gfp_field,
                       self.label_mcherry_field, self.label_composite_field):
            _field.textChanged.connect(self._update_preview)

        # Channel toggles + mode buttons
        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("Channels:"))
        self.phase_cb = QCheckBox("Phase")
        self.gfp_cb = QCheckBox("GFP")
        self.mch_cb = QCheckBox("mCherry")
        for cb in (self.phase_cb, self.gfp_cb, self.mch_cb):
            cb.setChecked(True)
            cb.toggled.connect(self._on_visibility_changed)
            chan_row.addWidget(cb)
        chan_row.addSpacing(12)
        chan_row.addWidget(QLabel("Merge:"))
        self.merge_combo = QComboBox()
        for _lbl, _val in MERGE_MODES:
            self.merge_combo.addItem(_lbl, _val)
        self.merge_combo.setToolTip(
            "How GFP and mCherry combine in the merged panel.\n"
            "Additive: overlap adds up and reads yellow (Nikon style).\n"
            "mCherry over GFP: mCherry is painted on top and hides the GFP "
            "beneath it.\nGFP over mCherry: the reverse.\n"
            "Applies to the preview, MP4 export and timeline figures."
        )
        self.merge_combo.currentIndexChanged.connect(self._update_preview)
        chan_row.addWidget(self.merge_combo)
        chan_row.addStretch(1)
        tools_row = QHBoxLayout()
        tools_row.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup(self)
        self.bg_mode_btn = QToolButton(); self.bg_mode_btn.setText("Background ROI")
        self.crop_mode_btn = QToolButton(); self.crop_mode_btn.setText("Crop region")
        self.preview_mode_btn = QToolButton(); self.preview_mode_btn.setText("Preview export")
        for btn, mode in (
            (self.bg_mode_btn, MODE_BG),
            (self.crop_mode_btn, MODE_CROP),
            (self.preview_mode_btn, MODE_PREVIEW),
        ):
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.setProperty("_mode", mode)
            self.mode_group.addButton(btn)
            tools_row.addWidget(btn)
        self.bg_mode_btn.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_button)
        self.clear_bg_btn = QPushButton("Clear BG ROI")
        self.clear_bg_btn.clicked.connect(lambda: self._clear_roi(MODE_BG))
        self.clear_crop_btn = QPushButton("Clear Crop")
        self.clear_crop_btn.clicked.connect(lambda: self._clear_roi(MODE_CROP))
        tools_row.addSpacing(10)
        # The preview-panel choice lives in the View buttons under the time
        # slider; this combo is kept (hidden) as the single source of truth.
        self._view_label = QLabel("View:")
        self.view_combo = QComboBox()
        for _b, _lbl, _val in VIEW_CHOICES:
            self.view_combo.addItem(_lbl, _val)
        tools_row.addWidget(self._view_label)
        tools_row.addWidget(self.view_combo)
        self._view_label.setVisible(False)
        self.view_combo.setVisible(False)
        tools_row.addWidget(QLabel("Rotate:"))
        self.rotate_combo = QComboBox()
        for _lbl, _val in ROTATIONS:
            self.rotate_combo.addItem(_lbl, _val)
        self.rotate_combo.setToolTip(
            "Rotate the whole image (all channels, all frames) before cropping.\n"
            "The crop and background boxes turn with it, and every export —\n"
            "videos, TIFF folders, strips, timeline — uses the rotated image.\n"
            "The choice stays set for the next file you open.")
        tools_row.addWidget(self.rotate_combo)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Zoom to fit the whole image (or double-click the image).")
        tools_row.addWidget(self.fit_btn)
        tools_row.addSpacing(10)
        tools_row.addWidget(self.clear_bg_btn)
        chan_row.setContentsMargins(0, 0, 0, 0)
        self._w_channels = QWidget(); self._w_channels.setLayout(chan_row)
        self._tools_row = tools_row

        # Preview canvas
        self.roi_view = ROIView()
        # The view would otherwise take a file dragged over the image and
        # ignore it; let it fall through to the window, which opens it.
        self.roi_view.setAcceptDrops(False)
        self.roi_view.bg_roi_changed.connect(self._on_bg_roi_changed)
        self.roi_view.crop_roi_changed.connect(self._on_crop_roi_changed)
        self.roi_view.roi_dragging.connect(self._on_roi_dragging)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.rotate_combo.currentIndexChanged.connect(self._on_rotation_changed)
        self.fit_btn.clicked.connect(self.roi_view.fit)
        tools_flow = FlowLayout()
        while self._tools_row.count():
            _it = self._tools_row.takeAt(0)
            if _it.widget() is not None:
                tools_flow.addWidget(_it.widget())
            elif _it.spacerItem() is not None:
                tools_flow.addSpacing(_it.spacerItem().sizeHint().width())
        self._w_tools = QWidget(); self._w_tools.setLayout(tools_flow)
        self._w_crop = self._build_crop_bar()

        # LUT mode selector row: chooses how LUTs vary across time.
        lut_mode_row = QHBoxLayout()
        lut_mode_row.addWidget(QLabel("LUT mode:"))
        self.lut_mode_combo = QComboBox()
        self.lut_mode_combo.addItem("Stack auto (one LUT for all frames)", LUT_STACK)
        self.lut_mode_combo.addItem("Smart per-frame auto", LUT_SMART)
        self.lut_mode_combo.addItem("Manual per-frame (use Next)", LUT_MANUAL)
        self.lut_mode_combo.setToolTip(
            "Stack auto: one LUT computed from the whole stack, applied to every frame.\n"
            "Smart per-frame: each frame gets its own auto-computed LUT (handles photobleaching).\n"
            "Manual per-frame: walk through the timelapse with the Next button, setting LUTs per frame."
        )
        self.lut_mode_combo.currentIndexChanged.connect(self._on_lut_mode_changed)
        lut_mode_row.addWidget(self.lut_mode_combo)
        self.next_frame_btn = QPushButton("Next Frame ▶")
        self.next_frame_btn.setToolTip(
            "Save the current LUTs to this frame, advance to the next frame, "
            "and seed it with the same LUTs as a starting point."
        )
        self.next_frame_btn.setVisible(False)
        self.next_frame_btn.clicked.connect(self._on_next_frame_clicked)
        lut_mode_row.addWidget(self.next_frame_btn)
        lut_mode_row.addStretch(1)
        display_layout.addLayout(lut_mode_row)  # moved to Display tab

        # LUT panels (Phase, GFP, mCherry) + optional reference thumbnail.
        lut_row = QHBoxLayout()
        self._lut_row = lut_row   # direction flips to vertical in side-panel layouts
        # Reference thumbnail: shows the previously-committed frame so the
        # user can match the look in Manual per-frame mode.
        self.ref_box = QGroupBox("Previous frame")
        self.ref_box.setVisible(False)
        ref_layout = QVBoxLayout(self.ref_box)
        ref_layout.setContentsMargins(6, 14, 6, 6)
        self.ref_image_label = QLabel()
        self.ref_image_label.setFixedSize(180, 180)
        self.ref_image_label.setStyleSheet(
            "background-color: #111; border: 1px solid #444;"
        )
        self.ref_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_caption = QLabel("(none yet)")
        self.ref_caption.setStyleSheet("color: #aaa;")
        ref_layout.addWidget(self.ref_image_label)
        ref_layout.addWidget(self.ref_caption)
        ref_layout.addStretch(1)
        self.ref_box.setMaximumWidth(220)
        lut_row.addWidget(self.ref_box)

        self.phase_lut_widget = HistogramLUTWidget("Phase LUT", _ACCENT_PHASE)
        self.gfp_lut_widget = HistogramLUTWidget("GFP LUT", _ACCENT_GFP)
        self.mcherry_lut_widget = HistogramLUTWidget("mCherry LUT", _ACCENT_MCHERRY)
        for w in (self.phase_lut_widget, self.gfp_lut_widget, self.mcherry_lut_widget):
            w.changed.connect(self._on_lut_widget_changed)
            lut_row.addWidget(w)
        display_layout.addLayout(lut_row)  # moved to Display tab
        display_layout.addStretch(1)
        # The tab widget itself goes into the main layout after the preview.

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # Control rows inside the tabs wrap instead of forcing the panel wide.
        for _page in (export_page,):
            self._reflow_rows(_page)
        # Keep spare height below the groups, not spread between their rows.
        for _lay in (export_layout,):
            _lay.addStretch(1)

        self.setStatusBar(QStatusBar())
        settings = QSettings("MicroscopyTimelapse", "microscopy_timelapse")
        saved = settings.value("layout", None)
        if saved not in [v for _l, v in LAYOUTS]:
            # First run: pick by screen size (a 13-16" MacBook is < 1000 pt tall).
            scr = QApplication.primaryScreen()
            saved = ("laptop" if scr is not None
                     and scr.availableGeometry().height() < 1000 else "wide")
        self._layout_key = None
        self.layout_combo.blockSignals(True)
        self.layout_combo.setCurrentIndex(
            next(i for i, (_l, v) in enumerate(LAYOUTS) if v == saved))
        self.layout_combo.blockSignals(False)
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self._apply_layout(saved)
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+\\"), self, activated=self._toggle_side_panel)
        QShortcut(QKeySequence("P"), self, activated=self._toggle_play)
        for _i in range(len(VIEW_CHOICES)):
            QShortcut(QKeySequence(f"Ctrl+{_i + 1}"), self,
                      activated=lambda i=_i: self.view_combo.setCurrentIndex(i))
        self.statusBar().showMessage("Ready.")

    # ---- file load --------------------------------------------------------

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .nd2 file", str(Path.home()), "Nikon ND2 (*.nd2)"
        )
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data = ND2Loader.load(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load file:\n\n{str(e) or type(e).__name__}")
            return
        try:
            # The Rotate setting applies to each file as it is opened.
            self._rotation_deg = int(self.rotate_combo.currentData() or 0)
            data = rotate_nd2(data, rot_k(self._rotation_deg))
            self._install_data(data)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._abandon_load(e)
            return
        QApplication.restoreOverrideCursor()
        msg = "Loaded. Drag a background ROI to begin."
        if self._dropped_boxes:
            msg = (f"Loaded. The {' and '.join(self._dropped_boxes)} from the previous "
                   "file did not fit this image and was removed.")
        self.statusBar().showMessage(msg)

    def _abandon_load(self, exc: Exception) -> None:
        """A file was read but could not be set up, usually because there was
        not enough memory for the copies the preview works on. Empty the
        window rather than leave parts of two files in it."""
        self._stop_play()
        self._data = None
        self._phase_pos = self._gfp_pos = self._mch_pos = None
        self._gfp_sub = self._mch_sub = None
        self.roi_view.clear_image()
        self._set_busy(False)
        self.info_label.setText("No file loaded. Click 'Open .nd2…' to begin.")
        QMessageBox.critical(
            self, "Load failed",
            "The file was read but could not be set up for display, most likely "
            "because there is not enough memory (the app needs about three times "
            f"the size of the file).\n\n{str(exc) or type(exc).__name__}")
        self.statusBar().showMessage("Load failed.")

    def _install_data(self, data: ND2Data) -> None:
        """Make `data` the current dataset and reset everything derived from
        it. Shared by opening a file and appending one. The boxes are kept
        when they fit the new image."""
        self._stop_play()
        self._data = data
        self._phase_pos = None   # drop the previous file's cache until reloaded
        # A box drawn on a larger image would give an empty background mean
        # (NaN) or a crop cut off at the edge.
        self._dropped_boxes = []
        for attr, label in (("_bg_roi", "background box"), ("_crop_roi", "crop box")):
            r = getattr(self.roi_view, attr)
            if r is not None and (r[0] + r[2] > data.width or r[1] + r[3] > data.height):
                setattr(self.roi_view, attr, None)
                self._dropped_boxes.append(label)
        self._position = 0
        self._time = 0
        # Reset LUT mode to stack for a fresh file.
        self._lut_mode = LUT_STACK
        self.lut_mode_combo.blockSignals(True)
        self.lut_mode_combo.setCurrentIndex(0)
        self.lut_mode_combo.blockSignals(False)
        self._per_frame_luts = {"phase": [], "gfp": [], "mcherry": []}
        self._manual_luts = None
        self.next_frame_btn.setVisible(False)
        self.ref_box.setVisible(False)

        # Populate navigation widgets.
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, data.n_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(data.n_frames > 1)
        self.time_slider.blockSignals(False)

        self.pos_combo.blockSignals(True)
        self.pos_combo.clear()
        for p in range(data.n_positions):
            self.pos_combo.addItem(f"{p + 1} of {data.n_positions}")
        self.pos_combo.setEnabled(data.n_positions > 1)
        self.pos_combo.blockSignals(False)

        # Populate export frame range spinboxes.
        self.start_frame_spin.blockSignals(True)
        self.start_frame_spin.setRange(1, data.n_frames)
        self.start_frame_spin.setValue(1)
        self.start_frame_spin.setEnabled(True)
        self.start_frame_spin.blockSignals(False)
        self.end_frame_spin.blockSignals(True)
        self.end_frame_spin.setRange(1, data.n_frames)
        self.end_frame_spin.setValue(data.n_frames)
        self.end_frame_spin.setEnabled(True)
        self.end_frame_spin.blockSignals(False)
        self.timer_t0_spin.setRange(1, data.n_frames)
        self.timer_t0_spin.setValue(1)
        self._on_frame_range_changed()
        # Pre-populate timer interval from ND2 metadata when available.
        if data.frame_interval_s is not None:
            self.timer_interval_spin.setValue(data.frame_interval_s)
        elif data.abs_times_s is not None and data.n_frames > 1:
            # No nominal period in the metadata: use the median real interval.
            # It is taken as a whole number of minutes only when it is within
            # 2 % of one (900.8 s is 15 min, but 450 s is not 8 min).
            iv = float(np.median(np.diff(data.abs_times_s[0])))
            whole_min = round(iv / 60.0) * 60.0
            if iv >= 300 and abs(whole_min - iv) <= 0.02 * iv:
                iv = whole_min
            else:
                iv = round(iv) if iv >= 10 else round(iv, 2)
            self.timer_interval_spin.setValue(iv)

        self._update_time_label()
        self.roi_view._user_zoom = False   # reset zoom when a new file is loaded
        self._load_position(0)
        self._configure_lut_widgets(initial=True)
        self.process_btn.setEnabled(True)
        self.timeline_btn.setEnabled(True)
        self.example_btn.setEnabled(True)
        self.save_meta_btn.setEnabled(True)

        self._set_info_label()
        self._sync_crop_fields(self.roi_view.crop_roi())
        self.append_btn.setEnabled(True)
        self.play_btn.setEnabled(data.n_frames > 1)
        self.timer_clock_combo.setEnabled(data.abs_times_s is not None)
        if data.abs_times_s is None:
            self.timer_clock_combo.setCurrentIndex(0)

    def _load_sources(self, segs: list[dict]) -> bool:
        """Open the file(s) a settings file was saved from. More than one
        means an appended run: each is joined with its recorded shift, so the
        result is the same pixels the export was made from."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            deg = int(self.rotate_combo.currentData() or 0)
            k = rot_k(deg)
            data = rotate_nd2(ND2Loader.load(segs[0]["path"]), k)
            for sg in segs[1:]:
                b = rotate_nd2(ND2Loader.load(sg["path"]), k)
                if (b.width, b.height, b.n_positions) != (data.width, data.height, data.n_positions):
                    raise ValueError(f"{sg['file']} does not match the size of "
                                     f"{segs[0]['file']}")
                shifts = [tuple(v) for v in (sg.get("shift_xy") or [(0, 0)] * b.n_positions)]
                data = append_nd2(data, b, shifts)
        except Exception as exc:         # nothing has been changed yet
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Load failed",
                                 "Could not open the saved file(s):\n\n"
                                 f"{str(exc) or type(exc).__name__}")
            return False
        try:
            self._rotation_deg = deg
            self._install_data(data)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._abandon_load(exc)
            return False
        QApplication.restoreOverrideCursor()
        return True

    def _on_append(self) -> None:
        """Load a continuation .nd2 and join it after the current last frame.

        In memory only — NIS's .nd2 format is not something we can write. The
        result behaves exactly like one long file for preview and every export.
        """
        d = self._data
        if d is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Append .nd2 (continuation of the loaded file)",
            str(d.source_path.parent), "Nikon ND2 (*.nd2)")
        if not path:
            return
        def same_file(a, b) -> bool:     # by identity, not by name: a continuation
            try:                         # may have the same name in another folder
                return Path(a).samefile(b)
            except OSError:
                return False

        if any(same_file(path, sg["path"]) for sg in _segments_of(d)):
            QMessageBox.warning(self, "Already loaded",
                                f"{Path(path).name} is already part of this timelapse.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            b = ND2Loader.load(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Load failed", f"Could not load file:\n\n{e}")
            return
        QApplication.restoreOverrideCursor()
        b = rotate_nd2(b, rot_k(self._rotation_deg))

        problems = []
        if (b.width, b.height) != (d.width, d.height):
            problems.append(f"Image size differs: {d.width}×{d.height} loaded, "
                            f"{b.width}×{b.height} in the new file.")
        if b.n_positions != d.n_positions:
            problems.append(f"Number of XY positions differs: {d.n_positions} vs "
                            f"{b.n_positions}.")
        if d.um_per_px and b.um_per_px and abs(b.um_per_px / d.um_per_px - 1) > 0.01:
            problems.append(f"Pixel size differs ({d.um_per_px:.4f} vs "
                            f"{b.um_per_px:.4f} µm/px) — different objective or binning?")
        if problems:
            QMessageBox.critical(self, "Cannot append", "\n\n".join(problems))
            return

        lines = [f"Append <b>{b.source_path.name}</b> ({b.n_frames} frames) after "
                 f"frame {d.n_frames} of the loaded timelapse?"]
        if tuple(b.channel_names) != tuple(d.channel_names):
            lines.append(f"⚠ Channel names differ: {', '.join(d.channel_names)} "
                         f"vs {', '.join(b.channel_names)}.")
        if d.abs_times_s is not None and b.abs_times_s is not None:
            gap = float(b.abs_times_s[0, 0] - d.abs_times_s[0, -1])
            iv = float(np.median(np.diff(d.abs_times_s[0]))) if d.n_frames > 1 else 0.0
            if gap < 0:
                lines.append(f"⚠ The new file starts {-gap / 60:.1f} min <i>before</i> "
                             f"the loaded one ends — are they in the right order?")
            else:
                lines.append(f"Gap at the join: <b>{gap / 60:.1f} min</b> (usual interval "
                             f"{iv / 60:.1f} min). The timer keeps this real gap.")
        else:
            lines.append("No timestamps in one of the files: the timer will count the "
                         "join as one interval.")

        shifts, resps = [], []
        for pi in range(d.n_positions):
            ref, mov = d.phase[pi, -1], b.phase[pi, 0]
            if not ref.any():   # no phase channel: register on GFP
                ref, mov = d.gfp[pi, -1], b.gfp[pi, 0]
            dx, dy, r = measure_join_shift(ref, mov)
            shifts.append((dx, dy)); resps.append(r)
        worst = max(range(len(shifts)), key=lambda i: abs(shifts[i][0]) + abs(shifts[i][1]))
        dx, dy = shifts[worst]
        reliable = min(resps) >= 0.2
        moved = any(sx or sy for sx, sy in shifts)
        um = d.um_per_px
        if moved:
            where = f" (position {worst + 1})" if d.n_positions > 1 else ""
            dist = f" ≈ {math.hypot(dx, dy) * um:.1f} µm" if um else ""
            lines.append(f"The field moved by <b>{dx:+d}, {dy:+d} px</b>{dist}{where} "
                         f"between the two files (match {min(resps):.2f}"
                         f"{'' if reliable else ' — low, may be wrong'}). "
                         f"<b>Align</b> shifts the new file by whole pixels so the "
                         f"cells don't jump at the join; the uncovered edge is black, "
                         f"like the existing registration padding.")
        else:
            lines.append("The field lines up at the join (no shift needed).")

        box = QMessageBox(self)
        box.setWindowTitle("Append timelapse")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText("<br><br>".join(lines))
        align_btn = box.addButton("Append and align" if moved else "Append",
                                  QMessageBox.ButtonRole.AcceptRole)
        plain_btn = (box.addButton("Append without aligning",
                                   QMessageBox.ButtonRole.AcceptRole) if moved else None)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(align_btn if (reliable or not moved) else plain_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (align_btn, plain_btn) or clicked is None:
            return
        use = shifts if clicked is align_btn else [(0, 0)] * len(shifts)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            join_at = d.n_frames
            joined = append_nd2(d, b, use)
        except Exception as e:           # the loaded file is still intact
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Append failed",
                "The two files could not be joined, most likely because there is "
                f"not enough memory.\n\n{str(e) or type(e).__name__}")
            return
        try:
            del b
            self._data = None; d = None   # free the old stacks before caching
            self._install_data(joined)
            self.time_slider.setValue(join_at)   # show the first appended frame
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._abandon_load(e)
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Appended: {joined.n_frames} frames. Showing frame {join_at + 1}, the "
            f"first of the new file — step back one frame to check the join.")

    def _set_info_label(self) -> None:
        data = self._data
        if data is None:
            return
        scale = (
            f"{data.um_per_px:.4f} µm/px"
            if data.um_per_px
            else "<i>scale missing (scale bar will be skipped)</i>"
        )
        ph, gfp, mch = data.channel_names
        rot = (f", <b>rotated {self._rotation_deg}° clockwise</b>"
               if self._rotation_deg else "")
        if data.segments:
            parts = []
            for i, sg in enumerate(data.segments):
                sh = sg.get("shift_xy")
                sh_txt = ""
                if sh and any(tuple(v) != (0, 0) for v in sh):
                    sx, sy = sh[min(self._position, len(sh) - 1)]
                    sh_txt = f", aligned: moved {-sx:+d}, {-sy:+d} px"
                parts.append(f"{i + 1}. {sg['file']} (frames {sg['start'] + 1}–"
                             f"{sg['start'] + sg['n_frames']}{sh_txt})")
            name = "<b>Joined:</b> " + " + ".join(parts) + "<br>"
        else:
            name = f"<b>{data.source_path.name}</b> &mdash; "
        self.info_label.setText(
            f"{name}"
            f"{data.n_positions} pos × {data.n_frames} frames × {data.width}×{data.height} px, "
            f"{scale}{rot}.<br>"
            f"Channels: phase=<b>{ph}</b>, gfp=<b>{gfp}</b>, mcherry=<b>{mch}</b>.<br>"
            f"<span style='color:#888'>Background ROI / Crop region: drag on the image "
            f"to draw a box, drag inside it to move, drag an edge to resize. View → Phase "
            f"shows one channel large. Preview export shows the final look. Drag the "
            f"histogram handles to tune contrast.</span>"
        )

    # ---- drag-and-drop -------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        # Not while an export is running (Open is disabled then): the new
        # file would replace the data being exported.
        if (self.open_btn.isEnabled() and urls
                and any(u.toLocalFile().lower().endswith(".nd2") for u in urls)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if not self.open_btn.isEnabled():
            event.ignore()
            return
        urls = event.mimeData().urls()
        nd2_paths = [u.toLocalFile() for u in urls if u.toLocalFile().lower().endswith(".nd2")]
        if nd2_paths:
            self._load_file(nd2_paths[0])   # load the first .nd2 file dropped
            event.acceptProposedAction()

    # ---- position/time loading -----------------------------------------------

    def _load_position(self, p: int) -> None:
        """Cache float32 stacks for the chosen position (used by previews)."""
        assert self._data is not None
        self._position = p
        self._phase_pos = self._data.at(p, "phase").astype(np.float32)
        self._gfp_pos = self._data.at(p, "gfp").astype(np.float32)
        self._mch_pos = self._data.at(p, "mcherry").astype(np.float32)
        self._refresh_bg_subtracted()

    def _refresh_bg_subtracted(self) -> None:
        """Cache bg-subtracted stacks for the current position and BG ROI."""
        bg_roi = self.roi_view.bg_roi() if hasattr(self, "roi_view") else None
        if bg_roi is None:
            self._gfp_sub = self._gfp_pos
            self._mch_sub = self._mch_pos
        else:
            self._gfp_sub = self._bg_subtract(self._gfp_pos, bg_roi)
            self._mch_sub = self._bg_subtract(self._mch_pos, bg_roi)

    def _channel_stacks_for_lut(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the (phase, gfp, mcherry) stacks the LUTs operate on.

        Phase is never bg-subtracted; gfp/mcherry use the cached bg-subtracted
        stack if available.
        """
        return self._phase_pos, self._gfp_sub, self._mch_sub

    def _channel_frame(self, ch: str, t: int) -> np.ndarray:
        ph, gfp, mch = self._channel_stacks_for_lut()
        return {"phase": ph, "gfp": gfp, "mcherry": mch}[ch][t]

    def _configure_lut_widgets(self, initial: bool,
                               keep: tuple = ("phase", "gfp", "mcherry")) -> None:
        """(Re)configure each LUT widget for the current data and mode.

        - Stack mode: histogram of the whole (bg-subtracted) stack, single auto.
        - Smart / Manual per-frame: histogram of the *current* frame only.

        In Stack mode a range the user set by hand is kept for the channels in
        `keep` (unless `initial`); only untouched channels follow the new auto
        range. Without this, changing the frame range or switching position
        silently replaced typed display ranges.
        """
        if self._data is None:
            return
        phase_stack, gfp_stack, mch_stack = self._channel_stacks_for_lut()
        if phase_stack is None:
            return  # position stacks not yet loaded (e.g. called before _load_position)
        t = self._time
        widgets = {"phase": self.phase_lut_widget, "gfp": self.gfp_lut_widget,
                   "mcherry": self.mcherry_lut_widget}
        hand_set = {}
        if not initial and self._lut_mode == LUT_STACK:
            hand_set = {ch: w.lut() for ch, w in widgets.items()
                        if ch in keep and w.is_hand_set()}
        if self._lut_mode == LUT_STACK:
            # Auto-LUT is computed from the selected frame range only, so that
            # out-of-range (e.g. out-of-focus) frames don't skew normalisation.
            t_s, t_e = self._export_frame_range()     # ordered, 0-based
            t_s = max(0, t_s)
            t_e = min(self._data.n_frames - 1, t_e)
            data_phase = phase_stack[t_s:t_e + 1]; auto_phase = ChannelLUT.auto_from(data_phase)
            data_gfp   = gfp_stack[t_s:t_e + 1];   auto_gfp   = ChannelLUT.auto_from(data_gfp)
            data_mch   = mch_stack[t_s:t_e + 1];   auto_mch   = ChannelLUT.auto_from(data_mch)
            lut_phase = auto_phase; lut_gfp = auto_gfp; lut_mch = auto_mch
        else:
            # Per-frame: histogram on the current frame; LUT comes from storage.
            data_phase = phase_stack[t]; auto_phase = ChannelLUT.auto_from(data_phase)
            data_gfp   = gfp_stack[t];   auto_gfp   = ChannelLUT.auto_from(data_gfp)
            data_mch   = mch_stack[t];   auto_mch   = ChannelLUT.auto_from(data_mch)
            lut_phase = self._per_frame_luts["phase"][t]
            lut_gfp   = self._per_frame_luts["gfp"][t]
            lut_mch   = self._per_frame_luts["mcherry"][t]

        self._suppress_lut_save = True
        try:
            self.phase_lut_widget.configure(data_phase, auto_phase)
            self.gfp_lut_widget.configure(data_gfp, auto_gfp)
            self.mcherry_lut_widget.configure(data_mch, auto_mch)
            # In per-frame mode, override the freshly-applied auto with the
            # stored LUT for this frame.
            if self._lut_mode != LUT_STACK:
                self.phase_lut_widget.canvas.set_range(
                    lut_phase.vmin, lut_phase.vmax, lut_phase.gamma
                )
                self.phase_lut_widget._refresh_fields(lut_phase.vmin, lut_phase.vmax, lut_phase.gamma)
                self.gfp_lut_widget.canvas.set_range(
                    lut_gfp.vmin, lut_gfp.vmax, lut_gfp.gamma
                )
                self.gfp_lut_widget._refresh_fields(lut_gfp.vmin, lut_gfp.vmax, lut_gfp.gamma)
                self.mcherry_lut_widget.canvas.set_range(
                    lut_mch.vmin, lut_mch.vmax, lut_mch.gamma
                )
                self.mcherry_lut_widget._refresh_fields(lut_mch.vmin, lut_mch.vmax, lut_mch.gamma)
            for ch, lut in hand_set.items():
                widgets[ch].set_lut(lut)
        finally:
            self._suppress_lut_save = False

    # ---- LUT mode handlers ------------------------------------------------

    def _on_lut_mode_changed(self, _index: int) -> None:
        mode = self.lut_mode_combo.currentData()
        # Leaving Manual: remember the ranges set frame by frame, so that
        # switching the mode away and back (one notch of the mouse wheel over
        # the dropdown is enough) does not discard them.
        if (self._lut_mode == LUT_MANUAL and mode != LUT_MANUAL
                and self._data is not None and self._per_frame_luts["phase"]):
            self._export_luts()     # store the current frame's widget values first
            self._manual_luts = {ch: list(v) for ch, v in self._per_frame_luts.items()}
        self._lut_mode = mode
        # Toggle the Manual-only widgets.
        manual = (mode == LUT_MANUAL)
        self.next_frame_btn.setVisible(manual)
        self.ref_box.setVisible(manual)
        if self._data is None:
            return
        # Initialize per-frame storage when entering a non-stack mode.
        if mode != LUT_STACK:
            saved = self._manual_luts if manual else None
            if saved and all(len(saved[ch]) == self._data.n_frames for ch in saved):
                self._per_frame_luts = {ch: list(v) for ch, v in saved.items()}
            else:
                self._init_per_frame_luts(mode)
        self._configure_lut_widgets(initial=True)
        if manual:
            # Clear the reference (nothing has been "committed" yet at the
            # start of a manual session).
            self.ref_image_label.clear()
            self.ref_caption.setText("(no frame committed yet)")
            self.statusBar().showMessage(
                "Manual per-frame: adjust LUTs and click Next Frame to step through."
            )
        elif mode == LUT_SMART:
            self.statusBar().showMessage(
                "Smart per-frame auto: each frame has its own auto LUT."
            )
        else:
            self.statusBar().showMessage("Stack auto: one LUT per channel applied to all frames.")
        self._update_preview()

    def _init_per_frame_luts(self, mode: str) -> None:
        """Populate `_per_frame_luts` for the current data + mode.

        Smart: per-frame auto from each frame's data.
        Manual: copies of the current widget LUTs (stack-level), so the user
        starts from a sensible global default and customizes from there.
        """
        assert self._data is not None
        phase_stack, gfp_stack, mch_stack = self._channel_stacks_for_lut()
        T = self._data.n_frames
        if mode == LUT_SMART:
            self._per_frame_luts["phase"] = [
                ChannelLUT.auto_from(phase_stack[t]) for t in range(T)
            ]
            self._per_frame_luts["gfp"] = [
                ChannelLUT.auto_from(gfp_stack[t]) for t in range(T)
            ]
            self._per_frame_luts["mcherry"] = [
                ChannelLUT.auto_from(mch_stack[t]) for t in range(T)
            ]
        else:  # LUT_MANUAL
            ph_lut = self.phase_lut_widget.lut()
            gfp_lut = self.gfp_lut_widget.lut()
            mch_lut = self.mcherry_lut_widget.lut()
            self._per_frame_luts["phase"] = [
                ChannelLUT(ph_lut.vmin, ph_lut.vmax, ph_lut.gamma) for _ in range(T)
            ]
            self._per_frame_luts["gfp"] = [
                ChannelLUT(gfp_lut.vmin, gfp_lut.vmax, gfp_lut.gamma) for _ in range(T)
            ]
            self._per_frame_luts["mcherry"] = [
                ChannelLUT(mch_lut.vmin, mch_lut.vmax, mch_lut.gamma) for _ in range(T)
            ]

    def _on_lut_widget_changed(self) -> None:
        """A LUT widget reports its values changed.

        Stack mode: just re-render the preview.
        Per-frame mode: also save the new LUT into per-frame storage at the
        current frame.
        """
        if self._suppress_lut_save:
            # Programmatic update (e.g. _configure_lut_widgets) — don't save.
            self._update_preview()
            return
        if self._lut_mode != LUT_STACK and self._data is not None:
            t = self._time
            self._per_frame_luts["phase"][t] = self.phase_lut_widget.lut()
            self._per_frame_luts["gfp"][t] = self.gfp_lut_widget.lut()
            self._per_frame_luts["mcherry"][t] = self.mcherry_lut_widget.lut()
        self._update_preview()

    def _on_next_frame_clicked(self) -> None:
        """Manual mode: commit current LUTs, advance, propagate as starting
        point for the new frame."""
        if self._data is None or self._lut_mode != LUT_MANUAL:
            return
        t = self._time
        T = self._data.n_frames
        # Snapshot the just-committed frame's render for the reference panel.
        self._update_reference_thumbnail(t)
        # Propagate current LUTs to frame t+1 (the "remember" behavior).
        if t + 1 < T:
            for ch, widget in (
                ("phase", self.phase_lut_widget),
                ("gfp", self.gfp_lut_widget),
                ("mcherry", self.mcherry_lut_widget),
            ):
                cur = widget.lut()
                self._per_frame_luts[ch][t + 1] = ChannelLUT(cur.vmin, cur.vmax, cur.gamma)
            self.time_slider.setValue(t + 1)  # triggers _on_time_changed
        else:
            self.statusBar().showMessage("Already at the last frame.")

    def _update_reference_thumbnail(self, t: int) -> None:
        """Render frame `t` with its committed LUTs (post-bg-subtract, no
        crop, full grid) into the reference panel."""
        try:
            grid, _ = self._render_preview_frame(time_override=t, force_full=True)
        except Exception:
            return
        h, w, _ = grid.shape
        qimg = QImage(grid.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.ref_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.ref_image_label.setPixmap(pix)
        self.ref_caption.setText(f"Frame {t + 1} (committed)")

    # ---- navigation handlers ----------------------------------------------

    # ---- playback ----------------------------------------------------------------

    def _play_range(self) -> tuple[int, int]:
        """(first, last) 0-based frames to play: the export range, or the whole
        timelapse if that is a single frame."""
        n = self._data.n_frames
        a, b = self.start_frame_spin.value() - 1, self.end_frame_spin.value() - 1
        lo, hi = max(0, min(a, b)), min(n - 1, max(a, b))
        return (lo, hi) if hi > lo else (0, n - 1)

    def _toggle_play(self, *_args) -> None:
        if self._play_timer.isActive():
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        if self._data is None or self._data.n_frames < 2:
            return
        lo, hi = self._play_range()
        if not (lo <= self._time < hi):        # outside the range, or on its last frame
            self.time_slider.setValue(lo)
        fps = int(self.speed_combo.currentData())
        self._play_timer.start(max(1, round(1000 / fps)))
        self.play_btn.setText("⏸  Pause")
        self.statusBar().showMessage(
            f"Playing frames {lo + 1}–{hi + 1} at {fps} fps"
            f"{' (looping)' if self.loop_cb.isChecked() else ''} — press P or Pause to stop.")

    def _stop_play(self) -> None:
        if not hasattr(self, "_play_timer"):
            return
        was = self._play_timer.isActive()
        self._play_timer.stop()
        self.play_btn.setText("▶  Play")
        if was:
            self.statusBar().showMessage("Paused.")

    def _play_tick(self) -> None:
        if self._data is None:
            self._stop_play()
            return
        lo, hi = self._play_range()
        t = self._time + 1
        if t > hi or t < lo:
            if not self.loop_cb.isChecked() and t > hi:
                self._stop_play()
                return
            t = lo
        self.time_slider.setValue(t)

    def _on_play_speed_changed(self, *_args) -> None:
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, round(1000 / int(self.speed_combo.currentData()))))

    def _on_time_changed(self, value: int) -> None:
        self._time = int(value)
        self._update_time_label()
        if self._lut_mode != LUT_STACK:
            # Load per-frame LUTs + per-frame histogram for the new time.
            self._configure_lut_widgets(initial=False)
        self._update_preview()

    def _on_position_changed(self, index: int) -> None:
        if self._data is None:
            return
        self._load_position(int(index))
        # Re-derive LUT defaults for the new position. (Auto values change.)
        # Ranges set by hand (Stack or Manual per-frame) are kept.
        if self._lut_mode == LUT_SMART:
            self._init_per_frame_luts(LUT_SMART)
        self._configure_lut_widgets(initial=False)
        self._update_preview()

    def _on_visibility_changed(self, _checked: bool) -> None:
        self._update_preview()

    def _on_mode_button(self, btn) -> None:
        mode = btn.property("_mode")
        self.roi_view.set_mode(mode)
        # Preview mode requires a re-render (different image shape if cropped).
        self._update_preview()
        labels = {MODE_BG: "Background: drag to draw a box on empty background; drag "
                           "inside it to move, edges to resize, arrow keys to nudge.",
                  MODE_CROP: "Crop: drag to draw; drag inside to move, edges/corners to "
                             f"resize; arrow keys nudge (Shift ×10, {_ALT_KEY} resizes). Or type X/Y/W/H.",
                  MODE_PREVIEW: "Preview: this is what the export will look like."}
        self.statusBar().showMessage(labels[mode])

    def _on_bg_roi_changed(self, roi) -> None:
        if roi is None:
            self.statusBar().showMessage("Background ROI cleared.")
        else:
            x, y, w, h = roi
            self.statusBar().showMessage(f"Background ROI: x={x} y={y} w={w} h={h}")
        # Refresh cached bg-subtracted stacks.
        self._refresh_bg_subtracted()
        # The bg-subtracted stack has changed → refresh LUT histograms and Auto.
        # Smart mode recomputes its ranges; ranges set frame by frame in
        # Manual mode are the user's work and are left alone.
        if self._lut_mode == LUT_SMART:
            self._init_per_frame_luts(LUT_SMART)
        # Phase is not background-subtracted, so a hand-set phase range still
        # applies; the fluorescence values have shifted, so those follow auto.
        self._configure_lut_widgets(initial=False, keep=("phase",))
        self._update_preview()

    def _on_crop_roi_changed(self, roi) -> None:
        self._sync_crop_fields(roi)
        if roi is None:
            self.statusBar().showMessage("Crop region cleared.")
        else:
            x, y, w, h = roi
            self.statusBar().showMessage(f"Crop region: x={x} y={y} w={w} h={h}")
        self._update_preview()

    def _clear_roi(self, kind: str) -> None:
        if kind == MODE_BG:
            self.roi_view.clear_bg_roi()
        else:
            self.roi_view.clear_crop_roi()

    def _on_frame_range_changed(self) -> None:
        if self._data is None:
            self.export_count_label.setText("")
            return
        t_start = self.start_frame_spin.value()
        t_end = self.end_frame_spin.value()
        count = max(0, abs(t_end - t_start) + 1)
        self.export_count_label.setText(f"({count} frames)")
        # In Stack auto mode the LUT range is derived from the selected frames.
        if self._lut_mode == LUT_STACK:
            self._configure_lut_widgets(initial=False)
            # _configure_lut_widgets emits changed → _on_lut_widget_changed →
            # _update_preview, so no extra call needed here.

    def _on_crop_constraint_changed(self, *_) -> None:
        val = self.crop_constraint_combo.currentData()
        is_pixels = val == "pixels"
        # Fixed size uses the crop bar's W/H fields; the old separate pixel
        # boxes stay hidden.
        self.crop_px_w_spin.setVisible(False)
        self.crop_px_h_spin.setVisible(False)
        aspect_map = {"1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3)}
        if val in aspect_map:
            self.roi_view.set_crop_constraint(aspect=aspect_map[val])
            roi = self.roi_view.crop_roi()
            if roi is not None:          # bring an existing box to the new shape
                aw, ah = aspect_map[val]
                x, y, w, h = roi
                self.roi_view.set_crop_roi((x, y, w, int(round(w * ah / aw))))
        elif is_pixels:
            self.roi_view.set_crop_constraint(
                px_size=(self.crop_w_spin.value(), self.crop_h_spin.value()))
        else:
            self.roi_view.set_crop_constraint()

    def _on_timer_toggled(self, checked: bool) -> None:
        fmt = self.timer_fmt_combo.currentData()
        self.timer_interval_spin.setEnabled(checked and fmt != "frame")
        self._update_preview()

    def _on_timer_format_changed(self, _index: int) -> None:
        fmt = self.timer_fmt_combo.currentData()
        self.timer_interval_spin.setEnabled(self.timer_cb.isChecked() and fmt != "frame")
        self._update_preview()

    def _update_time_label(self) -> None:
        if self._data is None:
            self.time_label.setText("- / -")
        else:
            txt = f"{self._time + 1} / {self._data.n_frames}"
            seg = self._segment_index(self._time)
            if seg is not None:
                txt += f"  (file {seg + 1})"
            self.time_label.setText(txt)

    def _segment_index(self, t: int) -> Optional[int]:
        """Which appended file frame t came from, or None for a single file."""
        d = self._data
        if d is None or not d.segments:
            return None
        for i, sg in enumerate(d.segments):
            if sg["start"] <= t < sg["start"] + sg["n_frames"]:
                return i
        return None

    def _timer_times(self) -> Optional[np.ndarray]:
        """Per-frame seconds for the timer (any origin; the timer subtracts
        T0), or None for the classic frame × interval.

        Fixed interval on an appended dataset: each file counts frames at the
        set interval from its own *real* start, so the join is drawn as the
        gap that actually happened rather than as one more interval.
        """
        d = self._data
        if d is None:
            return None
        p = min(self._position, d.n_positions - 1)
        real = d.abs_times_s[p] if d.abs_times_s is not None else None
        if self.timer_clock_combo.currentData() == "real" and real is not None:
            return real - real[0]
        if not d.segments:
            return None
        iv = self.timer_interval_spin.value()
        out = np.zeros(d.n_frames, dtype=np.float64)
        for i, sg in enumerate(d.segments):
            a, n = sg["start"], sg["n_frames"]
            if i == 0:
                base = 0.0
            elif real is not None:
                base = float(real[a] - real[0])
                if iv >= 60:   # keep the labels round: 20:49:58 -> 20:50:00
                    base = round(base / 60.0) * 60.0
            else:
                base = out[a - 1] + iv
            out[a:a + n] = base + np.arange(n) * iv
        return out

    def _frame_seconds(self) -> np.ndarray:
        """Seconds of every frame relative to the T0 frame, as the timer shows."""
        d = self._data
        t0 = self.timer_t0_spin.value() - 1
        times = self._timer_times()
        if times is None:
            return (np.arange(d.n_frames) - t0) * self.timer_interval_spin.value()
        return times - times[t0]

    # ---- preview rendering -----------------------------------------------

    def _bg_subtract(
        self, stack_f: np.ndarray, roi: tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = roi
        out = stack_f.copy()
        for t in range(out.shape[0]):
            out[t] -= float(out[t, y:y + h, x:x + w].mean())
        return out

    def _render_preview_frame(
        self,
        time_override: Optional[int] = None,
        force_full: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Build the 2x2 grid for the current state.

        `time_override`: render this frame instead of the current time. Used by
        the reference thumbnail in Manual mode.
        `force_full`: ignore the canvas's preview-mode crop (always show the
        full uncropped grid). Used for the reference thumbnail.

        Returns (grid_uint8, (single_panel_width, single_panel_height)).
        """
        assert self._data is not None
        t = self._time if time_override is None else time_override
        bg_roi = self.roi_view.bg_roi()
        crop_roi = self.roi_view.crop_roi()
        is_preview = (self.roi_view._mode == MODE_PREVIEW) and not force_full

        ph_f = self._phase_pos[t].astype(np.float32, copy=True)
        gfp_f = self._gfp_pos[t].astype(np.float32, copy=True)
        mch_f = self._mch_pos[t].astype(np.float32, copy=True)
        if bg_roi is not None:
            x, y, w, h = bg_roi
            gfp_f -= float(gfp_f[y:y + h, x:x + w].mean())
            mch_f -= float(mch_f[y:y + h, x:x + w].mean())

        ph_lut, gfp_lut, mch_lut = self._luts_at(t)
        ph_u8 = ph_lut.apply(ph_f)
        gp_u8 = gfp_lut.apply(gfp_f)
        mc_u8 = mch_lut.apply(mch_f)
        if not self.phase_cb.isChecked():   ph_u8[:] = 0
        if not self.gfp_cb.isChecked():     gp_u8[:] = 0
        if not self.mch_cb.isChecked():     mc_u8[:] = 0

        if is_preview and crop_roi is not None:
            cx, cy, cw, ch = crop_roi
            ph_u8 = ph_u8[cy:cy + ch, cx:cx + cw]
            gp_u8 = gp_u8[cy:cy + ch, cx:cx + cw]
            mc_u8 = mc_u8[cy:cy + ch, cx:cx + cw]

        grid = Pipeline._make_grid(ph_u8, gp_u8, mc_u8, self.merge_combo.currentData())


        if self.labels_cb.isChecked():
            Pipeline._draw_channel_labels(
                grid,
                {
                    "phase":     self.label_phase_field.text(),
                    "gfp":       self.label_gfp_field.text(),
                    "mcherry":   self.label_mcherry_field.text(),
                    "composite": self.label_composite_field.text(),
                },
                self.label_pos_combo.currentData(),
                x_offset=self.label_x_spin.value(),
                y_offset=self.label_y_spin.value(),
            )
        if self._data.um_per_px and self.sb_visible_cb.isChecked():
            Pipeline._draw_scale_bar(
                grid, self._data.um_per_px,
                self.sb_size_spin.value(),
                self.sb_pos_combo.currentData(),
                x_offset=self.sb_x_spin.value(),
                y_offset=self.sb_y_spin.value(),
                text_center=self.sb_center_cb.isChecked(),
                text_above=self.sb_above_cb.isChecked(),
            )
        if self.timer_cb.isChecked():
            Pipeline._draw_timer(
                grid, t,
                self.timer_fmt_combo.currentData(),
                self.timer_interval_spin.value(),
                self.timer_pos_combo.currentData(),
                x_offset=self.timer_x_spin.value(),
                y_offset=self.timer_y_spin.value(),
                decimals=self.timer_decimals_spin.value(),
                t0=self.timer_t0_spin.value() - 1,
                times_s=self._timer_times(),
            )
        return grid, (ph_u8.shape[1], ph_u8.shape[0])

    def _luts_at(self, t: int) -> tuple[ChannelLUT, ChannelLUT, ChannelLUT]:
        """Pick the LUTs to apply at time `t`, based on the current mode."""
        if self._lut_mode == LUT_STACK or not self._per_frame_luts["phase"]:
            return (
                self.phase_lut_widget.lut(),
                self.gfp_lut_widget.lut(),
                self.mcherry_lut_widget.lut(),
            )
        # Per-frame: pull from storage. If t is the current time, the widget
        # values are authoritative (the user may have edited but not yet
        # propagated); otherwise use stored.
        if t == self._time:
            return (
                self.phase_lut_widget.lut(),
                self.gfp_lut_widget.lut(),
                self.mcherry_lut_widget.lut(),
            )
        return (
            self._per_frame_luts["phase"][t],
            self._per_frame_luts["gfp"][t],
            self._per_frame_luts["mcherry"][t],
        )

    def _update_preview(self) -> None:
        # _phase_pos is None while a file is opening: widget signals fire
        # before _load_position() has cached the new file's stacks.
        if self._data is None or self._phase_pos is None:
            return
        grid, img_wh = self._render_preview_frame()
        panel = self.view_combo.currentData()
        if panel is not None:
            # Single-channel view: the chosen quadrant, at the scene origin, so
            # ROI coordinates are the same as on the grid's Phase panel.
            pw, ph = img_wh
            r, c = {"phase": (0, 0), "gfp": (0, 1),
                    "mcherry": (1, 0), "composite": (1, 1)}[panel]
            grid = np.ascontiguousarray(grid[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw])
        # We use show_grid (full rebuild) when geometry might have changed
        # (mode toggles or crop changes); otherwise update_grid is fast.
        self.roi_view.show_grid(grid, img_wh, (self._data.width, self._data.height))

    # ---- export ----------------------------------------------------------

    def _on_process(self) -> None:
        if self._data is None:
            return
        bg_roi = self.roi_view.bg_roi()
        crop_roi = self.roi_view.crop_roi()
        if bg_roi is None:
            ans = QMessageBox.question(
                self, "No background ROI",
                "No background ROI has been set, so no background subtraction "
                "will be applied to the fluorescent channels. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        if self._data.um_per_px is None:
            ans = QMessageBox.question(
                self, "No pixel scale",
                "The .nd2 file did not provide a pixel size; the scale bar will "
                "not be drawn. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        if self.sb_visible_cb.isChecked() and self._data.um_per_px:
            # The video is two panels wide; a longer bar would be left out.
            grid_w = 2 * (crop_roi[2] if crop_roi else self._data.width)
            bar_px = int(round(self.sb_size_spin.value() / self._data.um_per_px))
            if bar_px > grid_w:
                QMessageBox.warning(
                    self, "Scale bar does not fit",
                    f"A {self.sb_size_spin.value():g} µm bar is {bar_px} px, but the "
                    f"video is only {grid_w} px wide. Set a shorter length in "
                    "Export → Scale Bar and try again.")
                return

        # Suggest a filename that includes the position index if multi-position.
        suffix = f"_pos{self._position + 1}" if self._data.n_positions > 1 else ""
        default_out = (
            self._data.source_path.with_name(self._data.source_path.stem + suffix + ".mp4")
        )
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save MP4", str(default_out), "MP4 video (*.mp4)"
        )
        if not out_path:
            return

        visibility = ChannelVisibility(
            phase=self.phase_cb.isChecked(),
            gfp=self.gfp_cb.isChecked(),
            mcherry=self.mch_cb.isChecked(),
        )

        t_export_start = self.start_frame_spin.value() - 1  # convert to 0-indexed
        t_export_end   = self.end_frame_spin.value() - 1
        if t_export_end < t_export_start:
            t_export_start, t_export_end = t_export_end, t_export_start
        n_export = t_export_end - t_export_start + 1

        self._set_busy(True)
        try:
            self.statusBar().showMessage("Processing frames…")
            self.progress.setRange(0, n_export)
            self.progress.setValue(0)
            QApplication.processEvents()

            def on_progress(done, total):
                self.progress.setValue(done)
                QApplication.processEvents()

            luts = self._export_luts()
            phase_lut, gfp_lut, mcherry_lut = (
                luts["phase_lut"], luts["gfp_lut"], luts["mcherry_lut"])
            phase_luts, gfp_luts, mcherry_luts = (
                luts["phase_luts"], luts["gfp_luts"], luts["mcherry_luts"])

            fps = int(self.fps_spin.value())
            frames = Pipeline.process(
                self._data,
                position=self._position,
                bg_roi=bg_roi,
                crop_roi=crop_roi,
                visibility=visibility,
                phase_lut=phase_lut,
                gfp_lut=gfp_lut,
                mcherry_lut=mcherry_lut,
                phase_luts=phase_luts,
                gfp_luts=gfp_luts,
                mcherry_luts=mcherry_luts,
                scale_bar_visible=self.sb_visible_cb.isChecked(),
                scale_bar_um=self.sb_size_spin.value(),
                scale_bar_position=self.sb_pos_combo.currentData(),
                scale_bar_x_offset=self.sb_x_spin.value(),
                scale_bar_y_offset=self.sb_y_spin.value(),
                scale_bar_text_center=self.sb_center_cb.isChecked(),
                scale_bar_text_above=self.sb_above_cb.isChecked(),
                timer_enabled=self.timer_cb.isChecked(),
                timer_format=self.timer_fmt_combo.currentData(),
                timer_interval_s=self.timer_interval_spin.value(),
                timer_t0=self.timer_t0_spin.value() - 1,
                timer_times_s=self._timer_times(),
                timer_position=self.timer_pos_combo.currentData(),
                timer_x_offset=self.timer_x_spin.value(),
                timer_y_offset=self.timer_y_spin.value(),
                timer_decimals=self.timer_decimals_spin.value(),
                channel_labels=(
                    {
                        "phase":     self.label_phase_field.text(),
                        "gfp":       self.label_gfp_field.text(),
                        "mcherry":   self.label_mcherry_field.text(),
                        "composite": self.label_composite_field.text(),
                    } if self.labels_cb.isChecked() else None
                ),
                channel_label_position=self.label_pos_combo.currentData(),
                label_x_offset=self.label_x_spin.value(),
                label_y_offset=self.label_y_spin.value(),
                frame_start=t_export_start,
                frame_end=t_export_end,
                progress=on_progress,
                merge_mode=self.merge_combo.currentData(),
            )
            self.statusBar().showMessage(f"Encoding {len(frames)} frames at {fps} fps to MP4…")
            self.progress.setRange(0, len(frames))
            self.progress.setValue(0)
            QApplication.processEvents()
            crf, pxfmt = self.quality_combo.currentData()
            VideoExporter.write(frames, out_path, fps=fps, crf=crf,
                                pixelformat=pxfmt, progress=on_progress,
                                upscale=self.vscale_combo.currentData())
            note = ("\n\nThis is a 4:4:4 video. It plays in VLC, QuickTime, Premiere "
                    "and ImageJ, but not in Windows' own players or PowerPoint; "
                    "choose the Compatible quality for those."
                    if pxfmt == "yuv444p" else "")
            QMessageBox.information(self, "Done", f"Saved:\n{out_path}{note}")
            self.statusBar().showMessage(f"Saved {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Processing failed", str(e))
            self.statusBar().showMessage(f"Error: {e}")
        finally:
            self._set_busy(False)

    # ---- LUTs for export -----------------------------------------------------

    def _export_luts(self) -> dict:
        """The LUT keyword arguments for Pipeline.process(), as currently set.

        In per-frame modes the *current* widget values are flushed into storage
        at the current time index first, so an unsaved tweak is not lost.
        """
        if self._lut_mode != LUT_STACK:
            t = self._time
            self._per_frame_luts["phase"][t] = self.phase_lut_widget.lut()
            self._per_frame_luts["gfp"][t] = self.gfp_lut_widget.lut()
            self._per_frame_luts["mcherry"][t] = self.mcherry_lut_widget.lut()
            return dict(
                phase_lut=None, gfp_lut=None, mcherry_lut=None,
                phase_luts=self._per_frame_luts["phase"],
                gfp_luts=self._per_frame_luts["gfp"],
                mcherry_luts=self._per_frame_luts["mcherry"],
            )
        return dict(
            phase_lut=self.phase_lut_widget.lut(),
            gfp_lut=self.gfp_lut_widget.lut(),
            mcherry_lut=self.mcherry_lut_widget.lut(),
            phase_luts=None, gfp_luts=None, mcherry_luts=None,
        )

    # ---- example set export --------------------------------------------------

    # (key, folder name)
    EXAMPLE_FOLDERS: tuple[tuple[str, str], ...] = (
        ("phase",        "Phase only"),
        ("gfp",          "GFP only"),
        ("mcherry",      "mCherry only"),
        ("merge",        "Composite GFP + mCherry"),
        ("merge_no_mch", "Composite GFP+Phase"),
    )

    def _build_example_set_group(self) -> QGroupBox:
        grp = QGroupBox("Example set  (toolbar: Export Example Set…)")
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 6, 8, 6); v.setSpacing(4)
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Video FPS:"))
        self.ex_fps_spin = QSpinBox(); self.ex_fps_spin.setRange(1, 60)
        self.ex_fps_spin.setValue(4)
        row.addWidget(self.ex_fps_spin)
        row.addSpacing(10)
        row.addWidget(QLabel("Video size:"))
        self.ex_vscale_combo = _make_video_scale_combo(default=0)
        row.addWidget(self.ex_vscale_combo)
        row.addSpacing(10)
        row.addWidget(QLabel("TIFF DPI:"))
        self.ex_dpi_spin = QSpinBox(); self.ex_dpi_spin.setRange(72, 2400)
        self.ex_dpi_spin.setValue(600)
        row.addWidget(self.ex_dpi_spin)
        row.addSpacing(10)
        self.ex_scale_each_cb = QCheckBox("Scale frame in every folder")
        self.ex_scale_each_cb.setChecked(True)
        self.ex_scale_each_cb.setToolTip(
            "Also put a SCALE_ frame (that folder's own image + scale bar) in each\n"
            "channel folder, next to the frames it belongs with.")
        row.addWidget(self.ex_scale_each_cb)
        row.addStretch(1)
        v.addLayout(row)
        info = QLabel(
            "Writes into the folder you choose:\n"
            "  • Phase only / GFP only / mCherry only / Composite GFP + mCherry /\n"
            "    Composite GFP+Phase  — one TIFF per frame, native pixels\n"
            "  • the custom strip below, if 'Include in Example Set' is ticked\n"
            "  • <name>.mp4 (all 4 panels) and GFP_<name>.mp4 (mCherry off)\n"
            "    — H.264 CRF 0 4:4:4, no scale bar, timer or labels\n"
            "  • SCALE_<name>_frameNNNN.tif — first frame with the scale bar, to trace\n"
            "  • example_settings.json — every setting used\n"
            "Video size: small crops look blurry because players smooth them when\n"
            "enlarging. 'Auto' enlarges with sharp, blocky pixels (no smoothing) to at\n"
            "least 720 px tall. TIFF frames are always native size.\n"
            "Uses the current LUTs, crop, background ROI, Merge mode, export frame\n"
            "range and the Scale Bar length/position below. Channel checkboxes are\n"
            "ignored: the set always contains every channel.")
        info.setWordWrap(True)
        v.addWidget(info)
        return grp

    def _build_strip_group(self) -> QGroupBox:
        grp = QGroupBox("Custom strip  (panels side by side, one TIFF per frame)")
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 6, 8, 6); v.setSpacing(4)
        row1 = QHBoxLayout(); row1.setSpacing(6)
        self.strip_combos: list[QComboBox] = []
        for i, default in enumerate(("merge_no_mch", "gfp", "phase", None)):
            cb = QComboBox()
            for lbl, key in STRIP_CHOICES:
                cb.addItem(lbl, key)
            cb.setCurrentIndex([k for _l, k in STRIP_CHOICES].index(default))
            row1.addWidget(QLabel(f"{i + 1}:"))
            row1.addWidget(cb)
            self.strip_combos.append(cb)
        row1.addStretch(1)
        v.addLayout(row1)
        row2 = QHBoxLayout(); row2.setSpacing(6)
        row2.addWidget(QLabel("Direction:"))
        self.strip_dir_combo = QComboBox()
        self.strip_dir_combo.addItem("Horizontal (left → right)", True)
        self.strip_dir_combo.addItem("Vertical (top → bottom)", False)
        row2.addWidget(self.strip_dir_combo)
        row2.addWidget(QLabel("Gap:"))
        self.strip_gap_spin = QSpinBox(); self.strip_gap_spin.setRange(0, 200)
        self.strip_gap_spin.setValue(0); self.strip_gap_spin.setSuffix(" px")
        row2.addWidget(self.strip_gap_spin)
        row2.addWidget(QLabel("Gap colour:"))
        self.strip_bg_combo = QComboBox()
        self.strip_bg_combo.addItem("White", 255)
        self.strip_bg_combo.addItem("Black", 0)
        row2.addWidget(self.strip_bg_combo)
        row2.addStretch(1)
        v.addLayout(row2)
        row3 = QHBoxLayout(); row3.setSpacing(6)
        self.strip_in_set_cb = QCheckBox("Include in Example Set")
        self.strip_in_set_cb.setChecked(True)
        self.strip_video_cb = QCheckBox("Strip video too")
        self.strip_video_cb.setChecked(True)
        self.strip_video_cb.setToolTip(
            "Also write Strip_<name>.mp4 with the Example-set FPS and video size.")
        self.strip_only_btn = QPushButton("Export Strip Only…")
        self.strip_only_btn.setToolTip(
            "Write just the strip (frames, SCALE frame and optional video) into a "
            "folder you choose, without the rest of the example set.")
        self.strip_only_btn.clicked.connect(self._on_export_strip_only)
        for wdg in (self.strip_in_set_cb, self.strip_video_cb, self.strip_only_btn):
            row3.addWidget(wdg)
        row3.addStretch(1)
        v.addLayout(row3)
        note = QLabel("'Merge (no mCherry)' is phase + GFP only — use it when the "
                      "mCherry channel isn't part of the figure.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        v.addWidget(note)
        return grp

    def _strip_keys(self) -> list[str]:
        return [cb.currentData() for cb in self.strip_combos if cb.currentData()]

    # -- shared export helpers --

    def _export_frame_range(self) -> tuple[int, int]:
        t0 = self.start_frame_spin.value() - 1
        t1 = self.end_frame_spin.value() - 1
        return (t1, t0) if t1 < t0 else (t0, t1)

    def _export_stem(self) -> str:
        d = self._data
        suffix = f"_pos{self._position + 1}" if d.n_positions > 1 else ""
        joined = "_joined" if d.segments else ""
        return d.source_path.stem + joined + suffix

    def _check_scale_bar_fits(self) -> bool:
        d = self._data
        if not d.um_per_px:
            return True
        crop = self.roi_view.crop_roi()
        panel_w = crop[2] if crop else d.width
        bar_px = int(round(self.sb_size_spin.value() / d.um_per_px))
        if bar_px > panel_w:
            QMessageBox.warning(
                self, "Scale bar does not fit",
                f"A {self.sb_size_spin.value():g} µm bar is {bar_px} px, but the crop "
                f"is only {panel_w} px wide ({panel_w * d.um_per_px:.1f} µm). "
                "Set a shorter length in Export → Scale Bar and try again.")
            return False
        return True

    def _choose_export_folder(self, title: str) -> Optional[Path]:
        start = getattr(self, "_example_last_parent", None) or str(self._data.source_path.parent)
        out_dir = QFileDialog.getExistingDirectory(self, title, start)
        if not out_dir:
            return None
        out = Path(out_dir)
        existing = [p for p in out.iterdir() if not p.name.startswith(".")]
        if existing:
            ans = QMessageBox.question(
                self, "Folder not empty",
                f"{out.name} already contains {len(existing)} item(s). Files with the "
                "same names will be overwritten; others are left alone. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return None
        self._example_last_parent = str(out.parent)
        return out

    def _render_export(self, t0: int, t1: int, need_full: bool, need_no_mch: bool, tick):
        """Render the export range through Pipeline.process (the same path as the
        video export), with overlays off. Returns (full, no_mch, panel), where
        panel(key, i) slices one channel out of frame i's 2×2 grid."""
        d = self._data
        common = dict(
            position=self._position,
            bg_roi=self.roi_view.bg_roi(),
            crop_roi=self.roi_view.crop_roi(),
            scale_bar_visible=False, timer_enabled=False, channel_labels=None,
            frame_start=t0, frame_end=t1,
            merge_mode=self.merge_combo.currentData(),
            **self._export_luts(),
        )
        full = no_mch = None
        if need_full:
            self.statusBar().showMessage("Rendering all channels…")
            full = Pipeline.process(
                d, visibility=ChannelVisibility(phase=True, gfp=True, mcherry=True),
                progress=tick, **common)
        if need_no_mch:
            self.statusBar().showMessage("Rendering without mCherry…")
            no_mch = Pipeline.process(
                d, visibility=ChannelVisibility(phase=True, gfp=True, mcherry=False),
                progress=tick, **common)

        def panel(key: str, i: int) -> np.ndarray:
            src = no_mch if (key == "merge_no_mch" or full is None) else full
            g = src[i]
            H, W = g.shape[0] // 2, g.shape[1] // 2
            return {"phase": g[:H, :W], "gfp": g[:H, W:], "mcherry": g[H:, :W],
                    "merge": g[H:, W:], "merge_no_mch": g[H:, W:]}[key]
        return full, no_mch, panel

    def _scale_frame(self, img: np.ndarray) -> np.ndarray:
        img = np.ascontiguousarray(img).copy()
        Pipeline._draw_scale_bar(
            img, self._data.um_per_px, self.sb_size_spin.value(),
            self.sb_pos_combo.currentData(),
            x_offset=self.sb_x_spin.value(), y_offset=self.sb_y_spin.value(),
            text_center=self.sb_center_cb.isChecked(),
            text_above=self.sb_above_cb.isChecked(),
        )
        return img

    def _write_strip(self, out: Path, stem: str, t0: int, n: int, panel, dpi: int,
                     tick, video: bool) -> tuple[str, Optional[str]]:
        """Write the custom strip folder (+ SCALE frame, + optional video).
        Returns (folder name, video file name or None)."""
        keys = self._strip_keys()
        horiz = self.strip_dir_combo.currentData()
        gap = self.strip_gap_spin.value()
        bg = self.strip_bg_combo.currentData()
        folder = "Strip " + " + ".join(STRIP_NAMES[k] for k in keys)
        fdir = out / folder
        fdir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(n):
            strip = compose_strip([panel(k, i) for k in keys], horiz, gap, bg)
            frames.append(strip)
            TimelineExporter.save(strip, fdir / f"{stem}_frame{t0 + i + 1:04d}.tif",
                                  fmt="tif", dpi=dpi)
            tick()
        if self._data.um_per_px:
            # Scale bar on the last panel, which is where a reader looks for it.
            parts = [np.ascontiguousarray(panel(k, 0)) for k in keys]
            parts[-1] = self._scale_frame(parts[-1])
            TimelineExporter.save(compose_strip(parts, horiz, gap, bg),
                                  fdir / f"SCALE_{stem}_frame{t0 + 1:04d}.tif",
                                  fmt="tif", dpi=dpi)
        vname = None
        if video:
            vname = f"Strip_{stem}.mp4"
            VideoExporter.write(frames, out / vname, fps=self.ex_fps_spin.value(),
                                crf=0, pixelformat="yuv444p",
                                upscale=self.ex_vscale_combo.currentData(), progress=tick)
        return folder, vname

    def _strip_settings(self) -> dict:
        return {
            "panels": self._strip_keys(),
            "horizontal": self.strip_dir_combo.currentData(),
            "gap_px": self.strip_gap_spin.value(),
            "gap_value": self.strip_bg_combo.currentData(),
        }

    # -- the two export actions --

    def _on_export_example_set(self) -> None:
        if self._data is None:
            return
        d = self._data
        if d.um_per_px is None:
            ans = QMessageBox.question(
                self, "No pixel scale",
                "The .nd2 file did not provide a pixel size, so no SCALE frame can "
                "be made. Export everything else anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        if not self._check_scale_bar_fits():
            return
        do_strip = self.strip_in_set_cb.isChecked() and bool(self._strip_keys())
        out = self._choose_export_folder(
            "Choose or create the folder for this example set")
        if out is None:
            return

        stem = self._export_stem()
        t0, t1 = self._export_frame_range()
        n = t1 - t0 + 1
        dpi = self.ex_dpi_spin.value()
        fps = self.ex_fps_spin.value()
        upscale = self.ex_vscale_combo.currentData()
        strip_video = do_strip and self.strip_video_cb.isChecked()

        self._set_busy(True)
        try:
            total = 2 * n + 5 * n + 2 * n + (n * (2 if strip_video else 1) if do_strip else 0)
            done = [0]
            self.progress.setRange(0, total); self.progress.setValue(0)

            def tick(_d=None, _t=None):
                done[0] += 1
                self.progress.setValue(min(done[0], total))
                QApplication.processEvents()

            # Record the settings before rendering: the files are made from
            # what is set now.
            meta = MetadataExporter.collect(self)
            merge_mode = self.merge_combo.currentData()
            bar_um = self.sb_size_spin.value()
            full, no_mch, panel = self._render_export(t0, t1, True, True, tick)

            scale_name = f"SCALE_{stem}_frame{t0 + 1:04d}.tif"
            self.statusBar().showMessage("Example set: writing TIFF frames…")
            for key, folder in self.EXAMPLE_FOLDERS:
                fdir = out / folder
                fdir.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    TimelineExporter.save(
                        np.ascontiguousarray(panel(key, i)),
                        fdir / f"{stem}_frame{t0 + i + 1:04d}.tif", fmt="tif", dpi=dpi)
                    tick()
                if d.um_per_px and self.ex_scale_each_cb.isChecked():
                    TimelineExporter.save(self._scale_frame(panel(key, 0)),
                                          fdir / scale_name, fmt="tif", dpi=dpi)
            if d.um_per_px:
                TimelineExporter.save(self._scale_frame(panel("merge", 0)),
                                      out / scale_name, fmt="tif", dpi=dpi)
            strip_folder = strip_vname = None
            if do_strip:
                self.statusBar().showMessage("Example set: writing the custom strip…")
                strip_folder, strip_vname = self._write_strip(
                    out, stem, t0, n, panel, dpi, tick, strip_video)

            # Videos: CRF 0 4:4:4 (max error 1 grey level as decoded by macOS),
            # optionally enlarged with nearest-neighbour pixels.
            self.statusBar().showMessage("Example set: encoding videos…")
            VideoExporter.write(full, out / f"{stem}.mp4", fps=fps, crf=0,
                                pixelformat="yuv444p", upscale=upscale, progress=tick)
            VideoExporter.write(no_mch, out / f"GFP_{stem}.mp4", fps=fps, crf=0,
                                pixelformat="yuv444p", upscale=upscale, progress=tick)
            vfactor = resolve_upscale(upscale, full[0].shape[0], full[0].shape[1])

            meta["example_set"] = {
                "folders": dict(self.EXAMPLE_FOLDERS),
                "frames_1_indexed": [t0 + 1, t1 + 1],
                "tiff_dpi": dpi, "video_fps": fps, "video_crf": 0,
                "video_pixel_format": "yuv444p",
                "video_upscale_nearest": vfactor,
                "merge_mode": merge_mode,
                "scale_bar_um": bar_um if d.um_per_px else None,
                "strip": ({**self._strip_settings(), "folder": strip_folder,
                           "video": strip_vname} if do_strip else None),
            }
            with open(out / "example_settings.json", "w", encoding="utf-8") as fh:
                _json.dump(meta, fh, indent=2, default=str)

            gh, gw = full[0].shape[:2]
            self.statusBar().showMessage(f"Example set saved to {out}")
            QMessageBox.information(
                self, "Example set saved",
                f"Saved to:\n{out}\n\n"
                f"• 5 folders × {n} TIFF frames ({dpi} dpi)\n"
                + (f"• {strip_folder}/ ({n} frames)\n" if strip_folder else "")
                + f"• {stem}.mp4 and GFP_{stem}.mp4"
                + (f" and {strip_vname}" if strip_vname else "")
                + f"\n  ({fps} fps, {gw * vfactor}×{gh * vfactor} px"
                + (f", pixels enlarged {vfactor}× sharp" if vfactor > 1 else "") + ")\n"
                + (f"• {scale_name} ({self.sb_size_spin.value():g} µm bar)\n"
                   if d.um_per_px else "• no SCALE frame (no pixel size in file)\n")
                + "• example_settings.json")
        except Exception:
            import traceback
            QMessageBox.critical(self, "Example set failed", traceback.format_exc())
            self.statusBar().showMessage("Example set export failed")
        finally:
            self._set_busy(False)

    def _on_export_strip_only(self) -> None:
        if self._data is None:
            QMessageBox.warning(self, "No data", "Open an .nd2 file first.")
            return
        keys = self._strip_keys()
        if not keys:
            QMessageBox.warning(self, "Empty strip", "Choose at least one panel.")
            return
        if not self._check_scale_bar_fits():
            return
        out = self._choose_export_folder("Choose or create the folder for this strip")
        if out is None:
            return
        stem = self._export_stem()
        t0, t1 = self._export_frame_range()
        n = t1 - t0 + 1
        video = self.strip_video_cb.isChecked()
        need_no_mch = "merge_no_mch" in keys
        need_full = any(k in ("mcherry", "merge") for k in keys) or not need_no_mch
        self._set_busy(True)
        try:
            total = n * (int(need_full) + int(need_no_mch)) + n * (2 if video else 1)
            done = [0]
            self.progress.setRange(0, total); self.progress.setValue(0)

            def tick(_d=None, _t=None):
                done[0] += 1
                self.progress.setValue(min(done[0], total))
                QApplication.processEvents()

            meta = MetadataExporter.collect(self)      # before rendering
            strip_settings = self._strip_settings()
            dpi = self.ex_dpi_spin.value()
            merge_mode = self.merge_combo.currentData()
            _f, _n, panel = self._render_export(t0, t1, need_full, need_no_mch, tick)
            folder, vname = self._write_strip(out, stem, t0, n, panel, dpi, tick, video)
            meta["strip"] = {**strip_settings, "folder": folder, "video": vname,
                             "frames_1_indexed": [t0 + 1, t1 + 1],
                             "tiff_dpi": dpi,
                             "merge_mode": merge_mode}
            with open(out / folder / "strip_settings.json", "w", encoding="utf-8") as fh:
                _json.dump(meta, fh, indent=2, default=str)
            self.statusBar().showMessage(f"Strip saved to {out / folder}")
            QMessageBox.information(
                self, "Strip saved",
                f"Saved to:\n{out}\n\n• {folder}/ — {n} TIFF frames + SCALE frame\n"
                + (f"• {vname}\n" if vname else ""))
        except Exception:
            import traceback
            QMessageBox.critical(self, "Strip export failed", traceback.format_exc())
        finally:
            self._set_busy(False)

    # ---- crop bar ------------------------------------------------------------

    def _build_crop_bar(self) -> QWidget:
        """Numeric crop fields + shape + zoom, directly above the image."""
        bar = QWidget()
        fl = FlowLayout(bar)
        self._syncing_crop = False
        self.crop_x_spin, self.crop_y_spin = QSpinBox(), QSpinBox()
        self.crop_w_spin, self.crop_h_spin = QSpinBox(), QSpinBox()
        for lbl, sp in (("Crop  X", self.crop_x_spin), ("Y", self.crop_y_spin),
                        ("W", self.crop_w_spin), ("H", self.crop_h_spin)):
            sp.setRange(0, 99999)
            sp.setKeyboardTracking(False)    # apply on Enter / arrows, not per digit
            sp.setMinimumWidth(64)
            sp.valueChanged.connect(self._on_crop_field_changed)
            fl.addWidget(QLabel(lbl + ":"))
            fl.addWidget(sp)
        self.crop_um_label = QLabel("")
        self.crop_um_label.setStyleSheet("color: #888;")
        fl.addWidget(self.crop_um_label)
        fl.addSpacing(8)
        # Move the shape selector here from the old Display-tab group.
        self.crop_constraint_combo.parentWidget().layout().removeWidget(
            self.crop_constraint_combo)
        for i in range(self.crop_constraint_combo.count()):
            if self.crop_constraint_combo.itemData(i) == "pixels":
                self.crop_constraint_combo.setItemText(i, "Fixed size (W × H)")
        fl.addWidget(QLabel("Shape:"))
        fl.addWidget(self.crop_constraint_combo)
        self.zoom_crop_btn = QPushButton("Zoom to crop")
        self.zoom_crop_btn.setToolTip(
            "Fill the view with the crop box so you can fine-tune it. Fit or a "
            "double-click goes back to the whole image.")
        self.zoom_crop_btn.clicked.connect(self.roi_view.zoom_to_crop)
        fl.addWidget(self.zoom_crop_btn)
        fl.addWidget(self.clear_crop_btn)
        return bar

    def _sync_crop_fields(self, roi) -> None:
        if not hasattr(self, "crop_x_spin"):
            return
        self._syncing_crop = True
        try:
            d = self._data
            iw, ih = (d.width, d.height) if d is not None else (99999, 99999)
            self.crop_x_spin.setRange(0, max(0, iw - 1))
            self.crop_y_spin.setRange(0, max(0, ih - 1))
            self.crop_w_spin.setRange(1, iw)
            self.crop_h_spin.setRange(1, ih)
            if roi is None:
                vals = (0, 0, iw, ih) if d is not None else (0, 0, 1, 1)
                self.crop_um_label.setText("full frame (no crop)")
            else:
                vals = roi
                um = d.um_per_px if d is not None else None
                x, y, w, h = roi
                self.crop_um_label.setText(
                    f"= {w * um:.1f} × {h * um:.1f} µm" if um else f"{w} × {h} px")
            for sp, v in zip((self.crop_x_spin, self.crop_y_spin,
                              self.crop_w_spin, self.crop_h_spin), vals):
                sp.setValue(int(v))
        finally:
            self._syncing_crop = False

    def _on_crop_field_changed(self, *_args) -> None:
        if self._syncing_crop or self._data is None:
            return
        x, y = self.crop_x_spin.value(), self.crop_y_spin.value()
        w, h = self.crop_w_spin.value(), self.crop_h_spin.value()
        val = self.crop_constraint_combo.currentData()
        aspect = {"1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3)}.get(val)
        if aspect is not None:
            aw, ah = aspect
            if self.sender() is self.crop_h_spin:
                w = int(round(h * aw / ah))
            else:
                h = int(round(w * ah / aw))
        if val == "pixels":
            self.roi_view.set_crop_constraint(px_size=(w, h))
        self.roi_view.set_crop_roi((x, y, w, h))

    def _on_roi_dragging(self, kind: str, roi) -> None:
        """Live numbers while a box is being dragged (committed on release)."""
        self._stop_play()   # playback rebuilds the scene and would delete the box being dragged
        if kind == "crop":
            self._sync_crop_fields(roi)
        else:
            x, y, w, h = roi
            self.statusBar().showMessage(f"Background ROI: x={x} y={y} w={w} h={h}")

    def _on_view_changed(self, *_args) -> None:
        i = self.view_combo.currentIndex()
        if 0 <= i < len(self.view_buttons) and not self.view_buttons[i].isChecked():
            self.view_buttons[i].setChecked(True)
        self.roi_view.set_single_panel(self.view_combo.currentData())
        self.roi_view._user_zoom = False
        self._update_preview()
        self.roi_view.fit()

    def _on_rotation_changed(self, *_args) -> None:
        """Turn the loaded data by the change in angle (not from the original:
        only one copy of the stacks is kept, so memory does not double)."""
        new = int(self.rotate_combo.currentData() or 0)
        if self._data is None:
            self._rotation_deg = new
            return
        k = rot_k(new - self._rotation_deg)
        if k == 0:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            w, h = self._data.width, self._data.height
            self._data = rotate_nd2(self._data, k)
            self._rotation_deg = new
            self.roi_view.rotate_rois(k, w, h)
            self._phase_pos = None
            self._load_position(self._position)
            self._sync_crop_fields(self.roi_view.crop_roi())
            self._set_info_label()
            self.roi_view._user_zoom = False
            self._update_preview()
            self.roi_view.fit()
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Rotated to {new}° clockwise." if new else "Rotation reset to 0°.")

    # ---- window layouts ------------------------------------------------------

    def _reflow_rows(self, page: QWidget) -> None:
        """Turn the QHBoxLayout control rows under `page` into wrapping rows."""
        from PySide6.QtWidgets import QAbstractScrollArea

        def wraps_badly(row) -> bool:
            for i in range(row.count()):
                w = row.itemAt(i).widget()
                if w is not None and (isinstance(w, (QAbstractScrollArea, QSlider))
                                      or "Canvas" in type(w).__name__):
                    return True
            return False

        targets = []

        def walk(lay) -> None:
            for i in range(lay.count()):
                it = lay.itemAt(i)
                sub, w = it.layout(), it.widget()
                if sub is not None:
                    if isinstance(sub, QHBoxLayout):
                        if not wraps_badly(sub):
                            targets.append(sub)
                    else:
                        walk(sub)
                elif isinstance(w, QGroupBox) and w.layout() is not None:
                    gl = w.layout()
                    if isinstance(gl, QHBoxLayout):
                        if not wraps_badly(gl):
                            targets.append(gl)
                    else:
                        walk(gl)

        if page.layout() is not None:
            walk(page.layout())
        for row in targets:
            reflow_row(row)

    def _on_layout_changed(self, *_args) -> None:
        key = self.layout_combo.currentData()
        QSettings("MicroscopyTimelapse", "microscopy_timelapse").setValue("layout", key)
        self._apply_layout(key)

    def _apply_layout(self, key: str) -> None:
        """Rearrange the same widgets into one of the LAYOUTS.

        Every block is detached first, so replacing the central widget (which
        deletes the previous containers) never deletes a control.
        """
        blocks = (self._w_top, self.info_label, self._w_nav, self._w_frames,
                  self._w_channels, self._w_tools, self._w_crop, self.roi_view,
                  self.tabs, self.progress)
        progress_hidden = self.progress.isHidden()
        for w in blocks:
            w.setParent(None)
        self._toolbar.removeWidget(self._w_video)
        self._frames_row.removeWidget(self._w_video)
        self._export_layout.removeWidget(self._w_video)
        self._w_video.setParent(None)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 6)
        outer.setSpacing(6)
        outer.addWidget(self._w_top)
        self._splitter = None

        if key == "wide":
            # Next to the frame range rather than on the toolbar, which would
            # otherwise be wider than a 13" screen.
            self._frames_row.addWidget(self._w_video)
            self._lut_row.setDirection(QHBoxLayout.Direction.LeftToRight)
            outer.addWidget(self.info_label)
            outer.addWidget(self._w_nav)
            outer.addWidget(self._w_frames)
            row = QHBoxLayout()
            row.addWidget(self._w_channels, 1)
            row.addWidget(self._w_tools)
            outer.addLayout(row)
            outer.addWidget(self._w_crop)
            outer.addWidget(self.roi_view, 1)
            outer.addWidget(self.tabs)
        else:
            self._export_layout.insertWidget(0, self._w_video, 0, Qt.AlignmentFlag.AlignLeft)
            self._lut_row.setDirection(QHBoxLayout.Direction.TopToBottom)
            left = QWidget()
            lv = QVBoxLayout(left)
            lv.setContentsMargins(0, 0, 4, 0); lv.setSpacing(6)
            right = QWidget()
            rv = QVBoxLayout(right)
            rv.setContentsMargins(6, 0, 0, 0); rv.setSpacing(6)
            if key == "focus":
                lv.addWidget(self._w_channels)
            lv.addWidget(self._w_tools)
            lv.addWidget(self._w_crop)
            lv.addWidget(self.roi_view, 1)
            lv.addWidget(self._w_nav)
            lv.addWidget(self._w_frames)
            rv.addWidget(self.info_label)
            if key != "focus":
                rv.addWidget(self._w_channels)
            rv.addWidget(self.tabs, 1)
            sp = QSplitter(Qt.Orientation.Horizontal)
            sp.addWidget(left)
            sp.addWidget(right)
            sp.setChildrenCollapsible(True)
            sp.setStretchFactor(0, 1)
            sp.setStretchFactor(1, 0)
            sp.setHandleWidth(8)
            outer.addWidget(sp, 1)
            self._splitter = sp

        outer.addWidget(self.progress)
        for w in blocks:
            if w is not self.progress:
                w.show()
        self.progress.setVisible(not progress_hidden)
        self._w_video.show()
        keep = self.size()
        self.setCentralWidget(central)
        # Swapping the central widget can make Qt grow the window to its new
        # preferred size; keep the user's size (Qt still enforces the minimum).
        self.resize(keep)

        if self._splitter is not None:
            total = max(self.width() - 20, 800)
            side = 0 if key == "focus" else min(620, max(460, int(total * 0.40)))
            self._side_width = 560 if key == "focus" else side
            self._splitter.setSizes([total - side, side])
        self._layout_key = key
        self.roi_view.fit()

    def _toggle_side_panel(self) -> None:
        """⌘\\ / Ctrl+\\: fold or unfold the controls (tabs in the wide layout)."""
        if self._splitter is None:
            self.tabs.setVisible(self.tabs.isHidden())
            return
        a, b = self._splitter.sizes()
        if b == 0:
            side = getattr(self, "_side_width", 560) or 560
            self._splitter.setSizes([max(200, a + b - side), side])
        else:
            self._side_width = b
            self._splitter.setSizes([a + b, 0])

    def _on_timeline_export(self) -> None:
        dlg = TimelineDialog(self, parent=self)
        dlg.exec()

    def _on_theme_changed(self) -> None:
        name = self.theme_combo.currentData()
        ThemeManager.instance().apply(name)
        theme = ThemeManager.instance().current()
        # This can fire before LUT widgets / roi_view are built, because we
        # connect the signal early in _build_ui to make the initial Y2K theme
        # actually apply. Skip widget repainting in that case — the QSS is
        # already on the app-wide style sheet, and the widgets will pick up
        # the right ThemeColors when they're constructed.
        for attr in ("phase_lut_widget", "gfp_lut_widget", "mcherry_lut_widget"):
            w = getattr(self, attr, None)
            if w is not None:
                w.apply_theme(theme)
        roi = getattr(self, "roi_view", None)
        if roi is not None:
            roi.apply_theme(theme)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self._stop_play()
        self.play_btn.setEnabled(not busy and self._data is not None and self._data.n_frames > 1)
        self.open_btn.setEnabled(not busy)
        self.load_meta_btn.setEnabled(not busy)
        self.append_btn.setEnabled(not busy and self._data is not None)
        self.process_btn.setEnabled(not busy and self._data is not None)
        self.timeline_btn.setEnabled(not busy and self._data is not None)
        self.example_btn.setEnabled(not busy and self._data is not None)
        self.save_meta_btn.setEnabled(not busy and self._data is not None)
        # The window keeps responding while an export runs
        # (the progress bar needs it), so lock everything that would change
        # what is being exported. The time slider stays usable.
        for wdg in (self.roi_view, self.tabs, self._w_frames, self._w_channels,
                    self._w_tools, self._w_crop):
            wdg.setEnabled(not busy)
        self.pos_combo.setEnabled(not busy and self._data is not None
                                  and self._data.n_positions > 1)
        self.progress.setVisible(busy)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    if "--version" in sys.argv[1:]:
        print(f"{APP_NAME} {__version__}")
        return 0
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")  # required for full QSS support on all platforms
    # Preload the default theme so there's no flash of a different style
    # before MainWindow's combo applies its initial value.
    ThemeManager.instance().apply("y2k")
    win = MainWindow()
    win.show()

    # Open a file given on the command line:
    #     python microscopy_timelapse.py FILE.nd2
    # After win.show(), and through a zero-length timer, so that the empty
    # window is painted before the file starts loading.
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        QTimer.singleShot(0, lambda: win._load_file(sys.argv[1]))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
