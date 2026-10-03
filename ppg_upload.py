"""
Reading and preparing a user-uploaded PPG recording for PulseIQ.

The uploaded signal goes through the same steps as the training data, so the
model sees input in the format it was trained on:
  1. read one column of numbers from a CSV / TXT file
  2. resample to 125 Hz (the training sampling rate)
  3. remove drift (linear detrend) and band-pass filter 0.5-8 Hz
     (4th-order Butterworth, zero-phase), as in the MIMIC-IV pipeline
  4. cut into non-overlapping 12-second windows (1,500 samples)
  5. per window: min-max scale PPG to [0, 1]; VPG and APG are the first and
     second derivatives (np.gradient), each min-max scaled to [0, 1]
  6. quality checks per window (flat signal, separable heartbeats, heart rate)
"""
import io
import re
from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, detrend, resample_poly, find_peaks
from scipy.stats import skew

FS = 125                 # model's sampling rate (Hz)
WIN = 12 * FS            # 1,500 samples
MAX_WINDOWS = 10         # analyse at most 2 minutes (keeps the free server fast)
MAX_BYTES = 5 * 1024 * 1024

PPG_NAMES = ("ppg", "pleth", "pleth_ir", "ir", "signal", "value", "red", "green", "bvp")
TIME_NAMES = ("time", "t", "timestamp", "seconds", "sec", "ms", "millis", "time_s", "time_ms")


class UploadError(ValueError):
    """A problem with the uploaded file, explained in plain words."""


def _split(line):
    for d in (",", ";", "\t"):
        if d in line:
            return [c.strip() for c in line.split(d) if c.strip() != ""]
    return line.split()


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except ValueError:
        return None


def read_signal(raw: bytes, column: str = ""):
    """Return (signal, fs_from_time_column_or_None, column_used)."""
    if len(raw) > MAX_BYTES:
        raise UploadError("The file is larger than 5 MB. Upload a shorter recording (a few minutes is enough).")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise UploadError("The file is empty.")

    header = None
    first = _split(lines[0])
    if any(_num(c) is None for c in first):
        header = [c.strip().strip('"').lower() for c in first]
        lines = lines[1:]
    rows = [[_num(c) for c in _split(l)] for l in lines]
    rows = [r for r in rows if r and all(v is not None for v in r)]
    if not rows:
        raise UploadError("No numbers were found in the file.")
    ncol = max(len(r) for r in rows)
    rows = [r + [np.nan] * (ncol - len(r)) for r in rows]
    M = np.array(rows, dtype=float)

    names = header if header and len(header) == ncol else [f"column {k + 1}" for k in range(ncol)]
    # which column is the PPG?
    col = None
    if column:
        c = column.strip().lower()
        if c in names:
            col = names.index(c)
        elif c.isdigit() and 1 <= int(c) <= ncol:
            col = int(c) - 1
        else:
            raise UploadError(f"Column '{column}' was not found. Columns in the file: {', '.join(names)}.")
    if col is None:
        for k, n in enumerate(names):
            if n in PPG_NAMES or any(n.startswith(p) for p in ("ppg", "pleth")):
                col = k
                break
    time_col = None
    for k, n in enumerate(names):
        if n in TIME_NAMES and k != col:
            time_col = k
            break
    if col is None:
        cands = [k for k in range(ncol) if k != time_col]
        if time_col is None and ncol >= 2:
            d = np.diff(M[:, 0])
            if np.nanmin(d) > 0 and np.nanstd(d) < 0.05 * abs(np.nanmean(d)):   # evenly increasing -> time
                time_col = 0
                cands = list(range(1, ncol))
        if not cands:
            raise UploadError("Could not find a PPG column. Name it 'ppg' in the first row.")
        col = cands[0]

    sig = M[:, col]
    fs_from_time = None
    if time_col is not None:
        dt = np.nanmedian(np.diff(M[:, time_col]))
        if dt and dt > 0:
            # seconds if steps are < 1, otherwise milliseconds
            fs_from_time = 1.0 / dt if dt < 1 else 1000.0 / dt
    return sig, fs_from_time, names[col]


def _fill_gaps(x):
    bad = ~np.isfinite(x)
    if bad.mean() > 0.2:
        raise UploadError("More than 20% of the values are missing.")
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _resample(x, fs_in):
    if abs(fs_in - FS) < 0.5:
        return x
    num, den = int(round(FS * 100)), int(round(fs_in * 100))
    g = gcd(num, den)
    return resample_poly(x, num // g, den // g)


def _bandpass(x, lo=0.5, hi=8.0, order=4):
    b, a = butter(order, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / ((hi - lo) or 1.0)


def model_channels(p):
    """PPG window -> [3, 1500] model input (training convention)."""
    p = _minmax(p)
    v = np.gradient(p)
    a = np.gradient(v)
    return np.stack([p, _minmax(v), _minmax(a)]).astype(np.float32)


def _periodicity(p):
    """Strength of the repeating heartbeat: peak autocorrelation at lags of
    0.33-1.5 s (40-180 bpm). Real pulse waves score about 0.6-0.95; noise scores low."""
    z = p - p.mean()
    ac = np.correlate(z, z, mode="full")[len(z) - 1:]
    ac = ac / (ac[0] or 1.0)
    lo, hi = int(0.33 * FS), int(1.5 * FS)
    return float(ac[lo:hi].max())


def _heart_rate(p):
    z = (p - p.mean()) / (p.std() or 1.0)
    pk, _ = find_peaks(z, distance=int(0.33 * FS), prominence=0.3)
    if len(pk) < 3:
        return None
    return 60.0 * FS / float(np.median(np.diff(pk)))


def prepare(raw: bytes, fs: float = None, column: str = ""):
    """Read, clean and window an uploaded recording.

    Returns dict with X [n, 3, 1500] (good windows), info for every window,
    a short preview of the cleaned signal, and processing notes.
    """
    from beats import segment_beats

    sig, fs_time, col_name = read_signal(raw, column)
    # Optional header lines written by a recording device or exported with the file:
    #   # cuff: 97/50        (cuff reading taken during the first 12 s -> calibration)
    #   # reference: 91/49   (true BP over the rest, only used for comparison)
    head = raw[:2000].decode("utf-8", "ignore").lower()
    def _bp(tag):
        m = re.search(r"#\s*" + tag + r"[^0-9\n]*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", head)
        return (float(m.group(1)), float(m.group(2))) if m else None
    cuff_in_file, ref_in_file = _bp("cuff"), _bp("reference")
    notes = []
    if fs is None or fs <= 0:
        if fs_time:
            fs = fs_time
            notes.append(f"Sampling rate {fs:.1f} Hz was read from the time column.")
        else:
            fs = FS
            notes.append("No sampling rate given, so 125 Hz was assumed.")
    if not (20 <= fs <= 2000):
        raise UploadError(f"A sampling rate of {fs:g} Hz is not plausible for PPG (expected 20-2000 Hz).")

    sig = _fill_gaps(sig)
    dur = len(sig) / fs
    if dur < 12:
        raise UploadError(f"The recording is {dur:.1f} s long. At least 12 seconds are needed.")
    x = _resample(sig, fs)
    if abs(fs - FS) >= 0.5:
        notes.append(f"Resampled from {fs:g} Hz to 125 Hz.")
    x = _bandpass(detrend(x))

    n_total = len(x) // WIN
    n_use = min(n_total, MAX_WINDOWS)
    if n_total > MAX_WINDOWS:
        notes.append(f"The first {MAX_WINDOWS * 12 // 60} minutes ({MAX_WINDOWS} windows) were analysed.")
    wins = [x[k * WIN:(k + 1) * WIN] for k in range(n_use)]

    # Some sensors record the pulse upside down. A normal PPG pulse has a sharp
    # peak and a slow fall, i.e. positive skew; flip if most windows are negative.
    sk = [skew(w) for w in wins]
    if np.median(sk) < 0:
        wins = [-w for w in wins]
        x = -x
        notes.append("The signal looked upside down (common with infrared finger sensors) and was flipped.")

    info, X = [], []
    for k, w in enumerate(wins):
        reason = None
        if np.std(w) < 1e-6:
            reason = "flat signal"
        else:
            hr = _heart_rate(w)
            if hr is None or not (40 <= hr <= 180):
                reason = "heartbeats not clear"
            elif segment_beats(w, FS) is None or _periodicity(w) < 0.5:
                reason = "heartbeats not clear"
        if reason is None and skew(w) <= 0:
            reason = "pulse shape not recognised"
        item = {"index": k, "startS": k * 12, "ok": reason is None, "reason": reason,
                "heartRate": round(hr, 0) if reason is None else None}
        if reason is None:
            item["row"] = len(X)
            X.append(model_channels(w))
        info.append(item)

    preview_n = min(len(x), 30 * FS)            # first 30 s, every 5th sample
    return {
        "X": np.stack(X) if X else np.zeros((0, 3, WIN), np.float32),
        "windows": info,
        "column": col_name,
        "fsInput": round(float(fs), 2),
        "durationS": round(dur, 1),
        "notes": notes,
        "preview": np.round(_minmax(x[:preview_n])[::5], 4).tolist(),
        "cuffInFile": cuff_in_file,
        "referenceInFile": ref_in_file,
    }
