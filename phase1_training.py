import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import json
import os

import numpy as np
from hpinn_model import RotorNet, DEVICE
from config import PARAMS_FILE, DOZVOLI_NADKRITICNO

PARAM_KEYS = ("Dx", "Cx", "Ux", "Dy", "Cy", "Uy", "phi")


def sacuvaj_parametre(params, putanja=PARAMS_FILE, dodatno=None):
    p = {k: float(params[k]) for k in PARAM_KEYS if k in params}
    if dodatno:
        p["_napomena"] = dodatno
    with open(putanja, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)
    return putanja


def ucitaj_parametre(putanja=PARAMS_FILE):
    if not os.path.exists(putanja):
        return None
    with open(putanja, "r", encoding="utf-8") as f:
        d = json.load(f)
    p = {k: d[k] for k in PARAM_KEYS if k in d}
    return p if len(p) == len(PARAM_KEYS) else None
from config import MAX_OBRTAJA


def hpinn_train(tau_m, X_m, Y_m, tau_f,
                epochs=3000, lr=1e-3, alpha=1.0, lam_phys=8.0,
                progress_cb=None):
    n_obrt = (tau_m[-1] - tau_m[0]) / (2 * np.pi)
    if n_obrt > MAX_OBRTAJA:
        n_keep = max(32, int(len(tau_m) * MAX_OBRTAJA / n_obrt))
        print(f"  Prozor za obuku skraćen: {n_obrt:.1f} → "
              f"{MAX_OBRTAJA:.1f} obrtaja ({n_keep} od {len(tau_m)} uzoraka)")
        tau_m, X_m, Y_m = tau_m[:n_keep], X_m[:n_keep], Y_m[:n_keep]
        tau_f = np.linspace(tau_m[0], tau_m[-1], len(tau_f))
        n_obrt = (tau_m[-1] - tau_m[0]) / (2 * np.pi)
    if n_obrt < 2.0:
        print(f"\n  ⚠ Samo {n_obrt:.1f} obrtaja — premalo za razdvajanje "
              f"D, C, U. Povećaj N_MEAS ili smanji f_s.")

    if DOZVOLI_NADKRITICNO:
        print("  ⓘ Dozvoljen nadkriticni rad (C moze biti < 1).")

    net = RotorNet(tau_min=float(tau_m[0]), tau_max=float(tau_m[-1])).to(DEVICE)

    log_Dx = nn.Parameter(torch.tensor([-2.0]).to(DEVICE))
    log_Cx = nn.Parameter(torch.tensor([-1.5]).to(DEVICE))
    log_Ux = nn.Parameter(torch.tensor([-2.0]).to(DEVICE))
    log_Dy = nn.Parameter(torch.tensor([-2.0]).to(DEVICE))
    log_Cy = nn.Parameter(torch.tensor([-1.5]).to(DEVICE))
    log_Uy = nn.Parameter(torch.tensor([-2.0]).to(DEVICE))
    phi_p  = nn.Parameter(torch.tensor([0.0]).to(DEVICE))

    C_MIN = 0.0 if DOZVOLI_NADKRITICNO else 1.0

    def get_params():
        Dx = torch.exp(log_Dx)
        Cx = C_MIN + torch.exp(log_Cx)
        Ux = torch.exp(log_Ux)
        Dy = torch.exp(log_Dy)
        Cy = C_MIN + torch.exp(log_Cy)
        Uy = torch.exp(log_Uy)
        return Dx, Cx, Ux, Dy, Cy, Uy

    ode_params = [log_Dx, log_Cx, log_Ux, log_Dy, log_Cy, log_Uy, phi_p]
    optimizer  = optim.Adam(list(net.parameters()) + ode_params, lr=lr)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t_m = torch.tensor(tau_m, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    Xm  = torch.tensor(X_m,   dtype=torch.float32).unsqueeze(1).to(DEVICE)
    Ym  = torch.tensor(Y_m,   dtype=torch.float32).unsqueeze(1).to(DEVICE)
    t_f = (torch.tensor(tau_f, dtype=torch.float32)
           .unsqueeze(1).to(DEVICE).requires_grad_(True))

    history = []
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        Dx, Cx, Ux, Dy, Cy, Uy = get_params()
        phi = phi_p

        pm = net(t_m)
        theta_reg = sum(p.pow(2).sum() for p in net.parameters())
        L_d = (torch.mean((pm[:, 0:1] - Xm)**2)
             + torch.mean((pm[:, 1:2] - Ym)**2)
             + 1e-5 * theta_reg)

        pf   = net(t_f)
        Xf   = pf[:, 0:1];  Yf   = pf[:, 1:2]
        Xfd  = torch.autograd.grad(Xf.sum(),  t_f, create_graph=True)[0]
        Xfdd = torch.autograd.grad(Xfd.sum(), t_f, create_graph=True)[0]
        Yfd  = torch.autograd.grad(Yf.sum(),  t_f, create_graph=True)[0]
        Yfdd = torch.autograd.grad(Yfd.sum(), t_f, create_graph=True)[0]

        rX = Xfdd + Dx*Xfd + Cx*Xf - Ux*torch.sin(t_f + phi)
        rY = Yfdd + Dy*Yfd + Cy*Yf - Uy*torch.cos(t_f + phi)
        L_ode = torch.mean(rX**2) + torch.mean(rY**2)

        L_phys = ((Dx-Dy)**2 + (Cx-Cy)**2 + (Ux-Uy)**2) / 3.0

        loss = L_d + alpha*L_ode + lam_phys*L_phys

        if torch.isnan(loss):
            print("⚠ NaN detektovan u gubitku!")
            break

        loss.backward()
        nn.utils.clip_grad_norm_(list(net.parameters()) + ode_params, 1.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())

        if loss.item() < 1e-5:
            print(f"\n  → Trening prekinut ranije (Epoha {ep}): "
                  f"Loss je pao ispod praga ({loss.item():.6f})")
            break

        if ep > 300:
            recent = history[-200:]
            m = sum(recent) / len(recent)
            s = (sum((h - m)**2 for h in recent) / len(recent)) ** 0.5
            if s < 0.01 * m and m < 1e-3:
                print(f"\n  → Trening prekinut ranije (Epoha {ep}): "
                      f"Gubitak se više ne smanjuje (konvergencija).")
                break

        if progress_cb:
            progress_cb(ep, loss.item())

    final = [p.item() for p in get_params()]
    Dx, Cx, Ux, Dy, Cy, Uy = final
    phi_val = phi_p.item()

    params = dict(Dx=Dx, Cx=Cx, Ux=Ux, Dy=Dy, Cy=Cy, Uy=Uy, phi=phi_val)

    params = dotjeraj_pobudu(tau_m, X_m, Y_m, params)

    try:
        sacuvaj_parametre(params, dodatno={
            "obrtaja_u_prozoru": float((tau_m[-1]-tau_m[0])/(2*np.pi)),
            "n_meas": int(len(tau_m)),
            "epoha": len(history),
        })
        print(f"  ✓ Parametri sacuvani u {PARAMS_FILE}")
    except Exception as e:
        print(f"  ⚠ Neuspjelo cuvanje parametara: {e}")

    print(f"\n  ✓ HPINN završio | Ξx=[{params['Dx']:.4f},{params['Cx']:.4f},"
          f"{params['Ux']:.4f}] | Ξy=[{params['Dy']:.4f},{params['Cy']:.4f},"
          f"{params['Uy']:.4f}]")

    net.eval()
    return net, params, history


def dotjeraj_pobudu(tau, X, Y, params):
    out = dict(params)
    B = np.column_stack([np.sin(tau), np.cos(tau)])

    a, b = np.linalg.lstsq(B, X - np.mean(X), rcond=None)[0]
    A = float(np.hypot(a, b))
    c = float(np.arctan2(b, a))
    out["Ux"] = float(A * np.hypot(out["Cx"] - 1.0, out["Dx"]))
    out["phi"] = float(c + np.arctan2(out["Dx"], out["Cx"] - 1.0))

    a, b = np.linalg.lstsq(B, Y - np.mean(Y), rcond=None)[0]
    A = float(np.hypot(a, b))
    out["Uy"] = float(A * np.hypot(out["Cy"] - 1.0, out["Dy"]))

    return out
