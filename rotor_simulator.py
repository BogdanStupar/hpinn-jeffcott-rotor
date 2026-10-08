import os
import sys
import time

import mysql.connector
import numpy as np

from config import DELTA_UM


KSI = 0.06
OMEGA_X = 0.8
OMEGA_Y = 0.73
U_DISBALANS = 0.19

F_KRITICNA = 31.0


def _iz_omege(Om, ksi=KSI):
    return 2.0 * ksi / Om, 1.0 / Om ** 2


_Dx, _Cx = _iz_omege(OMEGA_X)
_Dy, _Cy = _iz_omege(OMEGA_Y)

STVARNO = dict(
    Dx=_Dx, Cx=_Cx, Ux=U_DISBALANS,
    Dy=_Dy, Cy=_Cy, Uy=U_DISBALANS,
    phi=0.40,
)

FS = 3200.0

F_ROT = 0.5 * (OMEGA_X + OMEGA_Y) * F_KRITICNA
SUM_UM = 0.15


def _amplituda_x():
    return STVARNO["Ux"] / np.sqrt((STVARNO["Cx"] - 1.0) ** 2
                                   + STVARNO["Dx"] ** 2)

DB = dict(host=os.environ.get("MYSQL_HOST", "localhost"),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DATABASE", "rotor_test"))


def amplituda(D, C, U):
    return U / np.sqrt((C - 1.0) ** 2 + D ** 2)


def init_db():
    conn = mysql.connector.connect(**DB)
    cur = conn.cursor()
    print("[SIMULATOR] Osvjezavam strukturu tabele...")
    cur.execute("DROP TABLE IF EXISTS rotor_data")
    cur.execute("""
        CREATE TABLE rotor_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            t DOUBLE, x DOUBLE, y DOUBLE
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def izvod(tau, s, p):
    X, Xd, Y, Yd = s
    return np.array([
        Xd,
        -p["Dx"] * Xd - p["Cx"] * X + p["Ux"] * np.sin(tau + p["phi"]),
        Yd,
        -p["Dy"] * Yd - p["Cy"] * Y + p["Uy"] * np.cos(tau + p["phi"]),
    ])


def rk4(tau, s, h, p):
    k1 = izvod(tau, s, p)
    k2 = izvod(tau + h / 2, s + h / 2 * k1, p)
    k3 = izvod(tau + h / 2, s + h / 2 * k2, p)
    k4 = izvod(tau + h, s + h * k3, p)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_simulator():
    init_db()
    conn = mysql.connector.connect(**DB)
    cur = conn.cursor()

    p = STVARNO
    om = 2 * np.pi * F_ROT
    dt = 1.0 / FS
    h = om * dt
    t = 0.0
    tau = 0.0
    s = np.zeros(4)

    Ax = amplituda(p["Dx"], p["Cx"], p["Ux"])
    Ay = amplituda(p["Dy"], p["Cy"], p["Uy"])

    print("\n" + "=" * 60)
    print("SIMULATOR ZDRAVOG ROTORA — rjesava jed. (2.21)")
    print("-" * 60)
    print(f"  STVARNI parametri (Faza 1 treba da ih pogodi):")
    print(f"     Dx={p['Dx']:.3f}  Cx={p['Cx']:.3f}  Ux={p['Ux']:.3f}")
    print(f"     Dy={p['Dy']:.3f}  Cy={p['Cy']:.3f}  Uy={p['Uy']:.3f}")
    print(f"     phi={p['phi']:.3f}")
    rez = "podkriticno" if p['Cx'] > 1 else "NADKRITICNO"
    print(f"  Omega_x = {1/np.sqrt(p['Cx']):.3f}, "
          f"Omega_y = {1/np.sqrt(p['Cy']):.3f}   ({rez})")
    print(f"  ksi = {KSI},  pojacanje 1Ω: "
          f"{1/np.sqrt((p['Cx']-1)**2+p['Dx']**2):.2f}")
    print(f"  ocekivana amplituda: X {Ax*DELTA_UM:.1f} um, "
          f"Y {Ay*DELTA_UM:.1f} um  (elipticna orbita)")
    spr = FS / F_ROT
    n_pre = int(round(4 * spr))
    print(f"  f_rot = {F_ROT:.1f} Hz ({F_ROT*60:.0f} o/min), "
          f"w_n = {F_KRITICNA:.1f} Hz")
    print(f"  f_s = {FS:.0f} Hz -> {spr:.0f} uzoraka/obrtaju"
          f"{'  <- PREMALO, treba >=32' if spr < 32 else ''}")
    print(f"  PREPORUKA: N_MEAS = {n_pre} u config.py "
          f"(4 obrtaja u prozoru)")
    snr_x = SUM_UM / (Ax * DELTA_UM) * 100
    oz = ""
    if snr_x > 15:
        oz = "   <- VISOK; podaci ce biti sumoviti"
    elif snr_x > 8:
        oz = "   <- granicno, jos upotrebljivo"
    print(f"  sum: {SUM_UM:.3f} um = {snr_x:.0f} % amplitude X{oz}")
    print("=" * 60 + "\n")

    for _ in range(int(30 * 2 * np.pi / h)):
        s = rk4(tau, s, h, p)
        tau += h

    red = 0
    try:
        while True:
            batch = []
            for _ in range(10):
                s = rk4(tau, s, h, p)
                tau += h
                t += dt
                x = s[0] * DELTA_UM + np.random.normal(0, SUM_UM)
                y = s[2] * DELTA_UM + np.random.normal(0, SUM_UM)
                batch.append((t, float(x), float(y)))

            cur.executemany(
                "INSERT INTO rotor_data (t, x, y) VALUES (%s, %s, %s)", batch)
            conn.commit()

            red += 10
            time.sleep(10 * dt)

            if red % 320 == 0:
                sys.stdout.write(f"\r[DB] t = {t:6.2f} s | "
                                 f"X = {x:6.1f} um, Y = {y:6.1f} um   ")
                sys.stdout.flush()

            if red % 1000 == 0:
                cur.execute("SELECT MIN(id) FROM (SELECT id FROM rotor_data "
                            "ORDER BY id DESC LIMIT 5000) AS tmp")
                mid = cur.fetchone()[0]
                if mid:
                    cur.execute("DELETE FROM rotor_data WHERE id < %s", (mid,))
                    conn.commit()

    except KeyboardInterrupt:
        print("\n[SIMULATOR] Zaustavljen.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_simulator()
