import numpy as np


def health_indicator(Z_history, lam=None, k=200):
    Z = np.asarray(Z_history, dtype=float)
    if Z.size < 10:
        return 0.0, 0.0, 0.0

    rms_sq = float(np.mean(Z[-k:]**2))

    if lam is None or np.sum(np.abs(lam)) < 1e-10:
        return rms_sq, rms_sq, 0.0

    a = np.abs(np.asarray(lam, dtype=float))
    p = a / (a.sum() + 1e-12)
    N = len(p)
    ent = float(-np.sum(p * np.log2(p + 1e-12)) / (np.log2(N) + 1e-12))
    return rms_sq - ent, rms_sq, ent
