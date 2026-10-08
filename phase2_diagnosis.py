import numpy as np
from scipy.signal import savgol_filter

PARAMETRI = ("dU", "dC", "dD", "U2", "K")

DETEKCIJA = ("A1", "U2", "K", "NEH", "RAT")

OPIS = {
    "dU": "Debalans",
    "dC": "Krutost",
    "dD": "Prigusenje",
    "U2": "Nesaosnost",
    "K":  "Trenje",
}


def izvodi(sig, tau):
    dt = float(np.median(np.diff(tau)))
    win = int(round(1.2 / dt))
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(sig) if len(sig) % 2 else len(sig) - 1))
    return (savgol_filter(sig, win, 3, deriv=0, delta=dt),
            savgol_filter(sig, win, 3, deriv=1, delta=dt),
            savgol_filter(sig, win, 3, deriv=2, delta=dt))


def identifikuj(X, Y, tau, p):
    Xs, Xd, Xdd = izvodi(X, tau)
    Ys, Yd, Ydd = izvodi(Y, tau)
    _BH = np.column_stack([np.sin(tau), np.cos(tau),
                           np.sin(2*tau), np.cos(2*tau),
                           np.sin(3*tau), np.cos(3*tau)])
    _c0, *_ = np.linalg.lstsq(_BH, Xs - np.mean(Xs), rcond=None)
    ph = float(np.arctan2(_c0[1], _c0[0])
               + np.arctan2(p["Dx"], p["Cx"] - 1.0))
    R = np.sqrt(Xs ** 2 + Ys ** 2)

    r_x = Xdd + p["Dx"] * Xd + p["Cx"] * Xs - p["Ux"] * np.sin(tau + ph)
    r_y = Ydd + p["Dy"] * Yd + p["Cy"] * Ys - p["Uy"] * np.cos(tau + ph)

    A_x = np.column_stack([np.sin(tau + ph), -Xs, -Xd,
                           np.sin(2 * tau + ph), Ys - Ys * R])
    A_y = np.column_stack([np.cos(tau + ph), -Ys, -Yd,
                           np.cos(2 * tau + ph), -Xs + Xs * R])

    A = np.vstack([A_x, A_y])
    r = np.concatenate([r_x, r_y])
    theta, *_ = np.linalg.lstsq(A, r, rcond=None)
    par = dict(zip(PARAMETRI, theta))

    B1 = np.column_stack([np.sin(tau), np.cos(tau)])
    c_x, *_ = np.linalg.lstsq(_BH, r_x, rcond=None)
    c_y, *_ = np.linalg.lstsq(_BH, r_y, rcond=None)
    par["A1"] = float(np.hypot(c_x[0], c_x[1]) + np.hypot(c_y[0], c_y[1]))

    cxs, *_ = np.linalg.lstsq(_BH, Xs - np.mean(Xs), rcond=None)
    cys, *_ = np.linalg.lstsq(_BH, Ys - np.mean(Ys), rcond=None)
    ax1 = np.hypot(cxs[0], cxs[1])
    ay1 = np.hypot(cys[0], cys[1])
    par["RAT"] = float(ay1 / max(ax1, 1e-12))

    B_s = np.column_stack([np.sin(tau), np.cos(tau),
                           np.sin(2 * tau), np.cos(2 * tau),
                           np.ones_like(tau)])
    ost = uk = 0.0
    for sig in (X, Y):
        c, *_ = np.linalg.lstsq(B_s, sig, rcond=None)
        ost += float(np.sum((sig - B_s @ c) ** 2))
        uk += float(np.sum(sig ** 2))
    par["NEH"] = float(np.sqrt(ost / (uk + 1e-12)))

    par["REZ"] = float(np.linalg.norm(r - A @ theta)
                       / (np.linalg.norm(p["Ux"] * np.sin(tau + ph)) + 1e-12))

    norme = np.linalg.norm(A, axis=0)
    norme[norme < 1e-12] = 1.0
    cond = float(np.linalg.cond(A / norme))

    return par, cond


def referenca(lista_parametara):
    kljucevi = tuple(PARAMETRI) + tuple(DETEKCIJA)
    mu = {k: float(np.mean([d[k] for d in lista_parametara]))
          for k in kljucevi if k in lista_parametara[0]}
    sd = {k: float(np.std([d[k] for d in lista_parametara]))
          for k in kljucevi if k in lista_parametara[0]}
    return mu, sd


def odstupanje(par, ref, pod=1e-4, min_rel=0.15):
    mu, sd = ref
    out = {}
    for k in mu:
        if k not in par:
            continue
        rel = 0.01 if k == "RAT" else min_rel
        s_eff = max(float(sd[k]), rel * abs(float(mu[k])), pod)
        out[k] = (par[k] - mu[k]) / s_eff
    return out


PRAG_NEH = 2.0


def dijagnoza(z, prag=4.0, odnos_neh=None, ref_rat=None):
    aktivni = []

    if odnos_neh is not None and odnos_neh > PRAG_NEH:
        aktivni.append("TRENJE")
    if abs(z.get("U2", 0.0)) > prag:
        aktivni.append("NESAOSNOST")

    if abs(z.get("A1", 0.0)) > prag and "TRENJE" not in aktivni:
        zr = z.get("RAT", 0.0)
        odnos_zdravo = ref_rat if ref_rat is not None else 1.0
        ka_jedinici = 1.0 if odnos_zdravo < 1.0 else -1.0
        if zr * ka_jedinici > 3.0 * prag:
            naznaka = " (Dеbalans)"
        elif abs(zr) < 0.5 * prag:
            naznaka = " (Krutost/Prigusenje)"
        else:
            naznaka = ""
        aktivni.append("1\u03a9 PROMJENA" + naznaka)

    return " + ".join(aktivni) if aktivni else "ZDRAVO"
