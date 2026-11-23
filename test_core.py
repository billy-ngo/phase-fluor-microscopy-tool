#!/usr/bin/env python
"""Tests for the image-processing core of microscopy_timelapse.py.

They use small synthetic arrays, so no .nd2 file and no display are needed.
Run with the same Python as the app:

    .venv/bin/python test_core.py            (macOS / Linux)
    .venv\\Scripts\\python.exe test_core.py    (Windows)
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # no window is opened

import numpy as np

import microscopy_timelapse as mt

_checks = 0


def check(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError(msg)


def make_data(n_pos=1, n_frames=4, h=40, w=60, um_per_px=0.5, seed=0):
    """A synthetic acquisition: every pixel value is unique per channel."""
    rng = np.random.default_rng(seed)
    shape = (n_pos, n_frames, h, w)
    return mt.ND2Data(
        phase=rng.integers(100, 4000, shape, dtype=np.uint16),
        gfp=rng.integers(100, 4000, shape, dtype=np.uint16),
        mcherry=rng.integers(100, 4000, shape, dtype=np.uint16),
        um_per_px=um_per_px,
        source_path=Path("synthetic.nd2"),
        channel_names=("Phase", "GFP", "mCherry"),
        frame_interval_s=60.0,
    )


def test_lut_mapping():
    lut = mt.ChannelLUT(100.0, 300.0, 1.0)
    out = lut.apply(np.array([0.0, 100.0, 200.0, 300.0, 5000.0], dtype=np.float32))
    check(out.dtype == np.uint8 and list(out) == [0, 0, 127, 255, 255],
          f"linear mapping clips at both ends, got {list(out)}")
    # ImageJ convention: gamma above 1 brightens the midtones.
    mid = np.array([200.0], dtype=np.float32)
    bright = mt.ChannelLUT(100.0, 300.0, 2.0).apply(mid)[0]
    dark = mt.ChannelLUT(100.0, 300.0, 0.5).apply(mid)[0]
    check(dark < 127 < bright, f"gamma 2 brightens and 0.5 darkens, got {dark}, {bright}")
    check(bright == int(255 * 0.5 ** 0.5), f"gamma 2 is a square root, got {bright}")
    # A zero or negative range must not divide by zero.
    flat = mt.ChannelLUT(50.0, 50.0).apply(np.array([49.0, 50.0, 51.0], dtype=np.float32))
    check(flat.dtype == np.uint8 and flat[0] == 0, f"empty range handled, got {list(flat)}")


def test_auto_lut():
    stack = np.arange(10000, dtype=np.float32).reshape(1, 100, 100)
    lut = mt.ChannelLUT.auto_from(stack)
    check(0 <= lut.vmin < 50 and 9950 < lut.vmax <= 9999 and lut.gamma == 1.0,
          f"auto range sits just inside the data, got {lut}")
    const = mt.ChannelLUT.auto_from(np.full((1, 8, 8), 7.0, dtype=np.float32))
    check(const.vmax > const.vmin, "constant image still gets a non-empty range")


def test_rotation_keeps_boxes_on_the_same_pixels():
    data = make_data(h=40, w=60)
    roi = (7, 5, 20, 12)                      # x, y, w, h
    x, y, w, h = roi
    ref = data.phase[0, 0, y:y + h, x:x + w]
    for deg in (0, 90, 180, 270):
        k = mt.rot_k(deg)
        turned = mt.rotate_nd2(data, k)
        expect_shape = (40, 60) if deg in (0, 180) else (60, 40)
        check(turned.phase.shape[2:] == expect_shape, f"{deg} deg: image shape {turned.phase.shape}")
        rx, ry, rw, rh = mt.rotate_roi(roi, k, 60, 40)
        got = turned.phase[0, 0, ry:ry + rh, rx:rx + rw]
        check(np.array_equal(got, np.rot90(ref, k)), f"{deg} deg: the box frames the same pixels")
        check(turned.phase.flags["C_CONTIGUOUS"], f"{deg} deg: result is contiguous")
    # Clockwise by 90 degrees moves the top-left pixel to the top-right.
    cw = mt.rotate_nd2(data, mt.rot_k(90))
    check(cw.phase[0, 0, 0, -1] == data.phase[0, 0, 0, 0], "90 deg is clockwise")
    check(mt.rotate_roi(None, 1, 60, 40) is None, "no box stays no box")


def test_shift_and_append():
    a = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6) + 1
    for dx, dy in ((0, 0), (2, 1), (-2, 1), (2, -1), (-3, -2)):
        out = mt.shift_frames(a, dx, dy)
        for yy in range(5):
            for xx in range(6):
                sy, sx = yy - dy, xx - dx
                want = a[sy, sx] if 0 <= sy < 5 and 0 <= sx < 6 else 0
                check(out[yy, xx] == want, f"shift ({dx}, {dy}) at ({xx}, {yy})")

    first, second = make_data(n_frames=3, seed=1), make_data(n_frames=2, seed=2)
    joined = mt.append_nd2(first, second, shifts=[(2, -1)])
    check(joined.n_frames == 5, f"frames add up, got {joined.n_frames}")
    check(np.array_equal(joined.gfp[0, :3], first.gfp[0]), "first file unchanged")
    # The measured displacement of the second file is undone.
    check(np.array_equal(joined.gfp[0, 3:], mt.shift_frames(second.gfp[0], -2, 1)),
          "second file shifted back onto the first")
    segs = joined.segments
    check([s["start"] for s in segs] == [0, 3] and [s["n_frames"] for s in segs] == [3, 2],
          f"segments record where each file starts, got {segs}")

    # Rotating a joined pair turns the recorded shift with the image, so that
    # rotating each file first and then joining with that shift gives the
    # same pixels (this is how a settings file reopens a joined pair).
    for deg in (90, 180, 270):
        k = mt.rot_k(deg)
        turned = mt.rotate_nd2(joined, k)
        shift = [tuple(v) for v in turned.segments[1]["shift_xy"]]
        rebuilt = mt.append_nd2(mt.rotate_nd2(first, k), mt.rotate_nd2(second, k), shift)
        check(np.array_equal(rebuilt.gfp, turned.gfp), f"{deg} deg: rebuilt join is identical")
    check(mt.rotate_nd2(mt.rotate_nd2(joined, 1), 3).segments[1]["shift_xy"] == [[2, -1]],
          "a full turn restores the recorded shift")


def test_join_shift_measurement():
    rng = np.random.default_rng(3)
    base = (rng.random((128, 128)) * 1000).astype(np.float32)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 3)
    moved = mt.shift_frames(base, 5, -3)
    dx, dy, resp = mt.measure_join_shift(base, moved)
    check((dx, dy) == (5, -3), f"displacement measured as ({dx}, {dy})")
    # append_nd2 undoes it: content lines up again away from the zero-filled border.
    back = mt.shift_frames(moved, -dx, -dy)
    check(np.allclose(back[10:-10, 10:-10], base[10:-10, 10:-10]), "undoing the shift restores the image")


def test_channel_matching():
    m = mt.ND2Loader._match_channels
    check(m(["ph3", "GFP", "mCherry"], 3) == (0, 1, 2), "usual names")
    check(m(["mCherry", "Phase Contrast", "EGFP"], 3) == (1, 2, 0), "any order")
    check(m(["ph3", "mCherry"], 2) == (0, None, 1), "a missing channel is None")
    check(m(["A", "B", "C"], 3) == (0, 1, 2), "unnamed channels fill the roles in order")
    check(m([], 1) == (0, None, None), "single unnamed channel becomes phase")
    got = m(["GFP", "GFP 2", "ph"], 3)
    check(len({i for i in got if i is not None}) == len([i for i in got if i is not None]),
          f"no channel is used twice, got {got}")


def test_merge_and_grid():
    ph = np.full((4, 6), 10, np.uint8)
    g = np.full((4, 6), 200, np.uint8)
    m = np.full((4, 6), 100, np.uint8)
    add = mt.merge_channels(ph, g, m, "additive")
    check(tuple(add[0, 0]) == (110, 210, 10), f"additive: R=phase+mCherry, G=phase+GFP, B=phase, got {add[0, 0]}")
    sat = mt.merge_channels(np.full((1, 1), 200, np.uint8), np.full((1, 1), 200, np.uint8),
                            np.zeros((1, 1), np.uint8))
    check(tuple(sat[0, 0]) == (200, 255, 200), "sums clip at 255 instead of wrapping")
    over = mt.merge_channels(ph, g, np.full((4, 6), 255, np.uint8), "mch_over_gfp")
    check(over[0, 0, 1] == 10, "an opaque mCherry layer hides GFP")
    try:
        mt.merge_channels(ph, g, m, "nonsense")
        check(False, "unknown merge mode should raise")
    except ValueError:
        check(True, "unknown merge mode raises")

    grid = mt.Pipeline._make_grid(ph, g, m)
    check(grid.shape == (8, 12, 3), f"grid is 2x2 panels, got {grid.shape}")
    check(tuple(grid[0, 0]) == (10, 10, 10), "top-left: phase in grey")
    check(tuple(grid[0, 6]) == (0, 200, 0), "top-right: GFP in green")
    check(tuple(grid[4, 0]) == (100, 0, 0), "bottom-left: mCherry in red")
    check(tuple(grid[4, 6]) == (110, 210, 10), "bottom-right: merge")


def test_scale_bar_length():
    for um_per_px, length_um in ((0.5, 10.0), (0.0866667, 10.0), (0.13, 5.0)):
        frame = np.zeros((400, 600, 3), np.uint8)
        mt.Pipeline._draw_scale_bar(frame, um_per_px, length_um, "bottom_right",
                                    x_offset=20, y_offset=20, text_above=True)
        white = np.all(frame == 255, axis=2)
        rows = white.sum(axis=1)
        bar_rows = np.flatnonzero(rows == rows.max())
        bar_px = int(rows.max())
        want = int(round(length_um / um_per_px))
        check(bar_px == want, f"{length_um} um at {um_per_px} um/px is {want} px, drew {bar_px}")
        cols = np.flatnonzero(white[bar_rows[0]])
        check(cols[-1] == 600 - 20 - 1, f"bar ends 20 px from the right edge, at {cols[-1]}")
    # A bar that does not fit is not drawn: a shortened one would still carry
    # the full length in its label.
    small = np.zeros((100, 50, 3), np.uint8)
    mt.Pipeline._draw_scale_bar(small, 0.1, 10.0)
    check(not small.any(), "a bar longer than the frame is left out")
    # The label clears the bar by the same margin at every frame size.
    for size in (200, 1024, 2048):
        frame = np.zeros((size, size, 3), np.uint8)
        mt.Pipeline._draw_scale_bar(frame, 0.5, 10.0, "bottom_right", x_offset=40, y_offset=40)
        lit = np.flatnonzero(frame.any(axis=(1, 2)))
        gaps = np.flatnonzero(np.diff(lit) > 1)
        check(len(gaps) == 1 and lit[gaps[0] + 1] - lit[gaps[0]] - 1 == 4,
              f"{size} px frame: 4 empty rows between label and bar, rows {lit[0]}..{lit[-1]}")


def test_timer_text():
    f = mt.Pipeline._format_time
    check(f(0, "frame", 60.0) == "Frame 1" and f(9, "frame", 60.0, t0=4) == "Frame 6", "frame numbers")
    check(f(3, "seconds", 30.0) == "t = 90 s", f"seconds, got {f(3, 'seconds', 30.0)}")
    check(f(3, "minutes", 30.0, decimals=1) == "t = 1.5 min", "minutes with decimals")
    check(f(125, "hms", 60.0) == "2:05:00", f"h:mm:ss, got {f(125, 'hms', 60.0)}")
    check(f(0, "hms", 60.0, t0=2) == "-0:02:00", "frames before time zero are negative")
    # Real acquisition times replace the fixed interval (needed across a join).
    times = np.array([1000.0, 1060.0, 1120.0, 5000.0])
    check(f(3, "seconds", 60.0, times_s=times) == "t = 4000 s", "real time gap is kept")
    check(f(3, "seconds", 60.0, t0=1, times_s=times) == "t = 3940 s", "time zero applies to real times")


def test_frame_list():
    p = mt._parse_frame_list
    check(p("1, 5, 10", 20) == [0, 4, 9], "single frames are 1-based")
    check(p("1-4", 20) == [0, 1, 2, 3], "ranges include both ends")
    check(p("1-10:3", 20) == [0, 3, 6, 9], "ranges with a step")
    check(p("3,1,3,2", 20) == [0, 1, 2], "sorted, without duplicates")
    check(p("0, 21, 5, x, 4-", 20) == [4], "out-of-range and malformed entries are ignored")
    check(p("6-3", 20) == [2, 3, 4, 5], "a range typed backwards is accepted")
    check(p("2–4", 20) == [1, 2, 3], "an en dash works like a hyphen")
    check(p("", 20) == [], "empty text")


def test_upscale_and_strip():
    r = mt.resolve_upscale
    check(r(4, 100, 100) == 4, "explicit factor is used as given")
    check(r(0, 94, 94) == 8 and 94 * 8 >= 720, f"auto reaches 720 px, got {r(0, 94, 94)}")
    check(r(0, 1024, 1024) == 1, "large frames are not enlarged")
    check(r(0, 100, 3000) == 1, "auto never exceeds 3840 px wide")

    a = np.full((4, 5, 3), 10, np.uint8)
    b = np.full((4, 5, 3), 20, np.uint8)
    horiz = mt.compose_strip([a, b], True, gap=2, bg=255)
    check(horiz.shape == (4, 12, 3) and horiz[0, 4, 0] == 10 and horiz[0, 5, 0] == 255
          and horiz[0, 7, 0] == 20, "horizontal strip with a white gap")
    vert = mt.compose_strip([a, b], False, gap=1, bg=0)
    check(vert.shape == (9, 5, 3) and vert[4, 0, 0] == 0 and vert[5, 0, 0] == 20, "vertical strip")


def test_pipeline_matches_manual_calculation():
    data = make_data(n_frames=5, h=40, w=60)
    bg = (2, 3, 10, 8)
    crop = (20, 10, 30, 20)
    luts = dict(phase_lut=mt.ChannelLUT(100, 4000), gfp_lut=mt.ChannelLUT(0, 3000, 1.5),
                mcherry_lut=mt.ChannelLUT(0, 2000))
    frames = mt.Pipeline.process(
        data, position=0, bg_roi=bg, crop_roi=crop, visibility=mt.ChannelVisibility(),
        scale_bar_visible=False, frame_start=1, frame_end=3, **luts)
    check(len(frames) == 3, f"frame range 2-4 gives 3 frames, got {len(frames)}")
    check(frames[0].shape == (40, 60, 3) and frames[0].dtype == np.uint8, f"cropped 2x2 grid, got {frames[0].shape}")

    t = 2                                       # second exported frame
    bx, by, bw, bh = bg
    cx, cy, cw, ch = crop
    gfp = data.gfp[0, t].astype(np.float32)
    gfp -= gfp[by:by + bh, bx:bx + bw].mean()   # background box mean, this frame only
    want_gfp = luts["gfp_lut"].apply(gfp)[cy:cy + ch, cx:cx + cw]
    want_phase = luts["phase_lut"].apply(data.phase[0, t].astype(np.float32))[cy:cy + ch, cx:cx + cw]
    grid = frames[1]
    check(np.array_equal(grid[:ch, cw:, 1], want_gfp), "GFP panel: background-subtracted, mapped, cropped")
    check(not grid[:ch, cw:, 0].any() and not grid[:ch, cw:, 2].any(), "GFP panel is pure green")
    check(np.array_equal(grid[:ch, :cw, 0], want_phase), "phase panel has no background subtraction")

    hidden = mt.Pipeline.process(
        data, position=0, bg_roi=None, crop_roi=None,
        visibility=mt.ChannelVisibility(phase=True, gfp=False, mcherry=True),
        scale_bar_visible=False, **luts)
    check(not hidden[0][:40, 60:].any(), "a hidden channel leaves its panel black")
    check(len(hidden) == 5, "the whole stack is exported by default")


def test_video_writer():
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 255, (31, 45, 3), dtype=np.uint8) for _ in range(6)]   # odd size
    import imageio.v2 as imageio
    with tempfile.TemporaryDirectory(prefix="mt_video_") as tmp:
        for name, kwargs, size in (
                ("plain.mp4", dict(pixelformat="yuv420p", crf=18), (46, 32)),   # padded to even
                ("exact.mp4", dict(pixelformat="yuv444p", crf=0), (45, 31)),
                ("big.mp4", dict(pixelformat="yuv444p", crf=0, upscale=4), (180, 124))):
            path = mt.VideoExporter.write(frames, Path(tmp) / name, fps=5, **kwargs)
            reader = imageio.get_reader(str(path))
            try:
                meta = reader.get_meta_data()
                n = sum(1 for _ in reader)
            finally:
                reader.close()
            check(tuple(meta["size"]) == size, f"{name}: size {meta['size']}, expected {size}")
            check(n == len(frames), f"{name}: {n} frames, expected {len(frames)}")
            check(abs(meta["fps"] - 5) < 1e-6, f"{name}: fps {meta['fps']}")
        try:
            mt.VideoExporter.write([], Path(tmp) / "none.mp4")
            check(False, "an empty frame list should raise")
        except ValueError:
            check(True, "an empty frame list raises")


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)} tests, {_checks} checks, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
