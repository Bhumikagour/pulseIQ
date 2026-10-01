"""Heartbeat segmentation shared by the app (main.py) and tools/ig_faithfulness.py."""
import numpy as np
from scipy.signal import find_peaks


def segment_beats(ppg, fs):
    """
    Split a PPG window into beats. Returns a list of (foot, peak, cut, next_foot)
    sample indices, or None if fewer than 3 clean beats are found.
      rise       foot -> peak
      peak & dip peak -> cut (40% of the way from peak to the next foot)
      slow fall  cut  -> next_foot
    The 40% cut is a fixed rule, not a detected dicrotic notch.
    """
    ppg = np.asarray(ppg, dtype=float)
    z = (ppg - ppg.mean()) / (ppg.std() or 1.0)
    pk, _ = find_peaks(z, distance=int(0.33 * fs), prominence=0.3)
    if len(pk) < 4:
        return None
    feet = []
    for j in range(len(pk)):
        lo = pk[j - 1] if j else max(0, pk[0] - int(0.5 * fs))
        feet.append(lo + int(np.argmin(z[lo:pk[j] + 1])))
    beats = []
    for j in range(len(pk) - 1):
        f0, p0, f1 = feet[j], pk[j], feet[j + 1]
        if not (f0 < p0 < f1):
            continue
        beats.append((f0, p0, p0 + int(round((f1 - p0) * 0.4)), f1))
    if len(beats) < 3:
        return None
    L = np.array([b[3] - b[0] for b in beats], dtype=float)
    med = np.median(L)
    beats = [b for b, l in zip(beats, L) if 0.4 * fs <= l <= 1.6 * fs and abs(l - med) <= 0.35 * med]
    return beats if len(beats) >= 3 else None
