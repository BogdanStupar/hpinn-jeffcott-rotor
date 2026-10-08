import os
import time

import numpy as np

from config import OMEGA_NOM, DELTA_UM

IZVOR = os.environ.get("ROTOR_IZVOR", "csv")

MYSQL_CFG = dict(host=os.environ.get("MYSQL_HOST", "localhost"),
                 user=os.environ.get("MYSQL_USER", "root"),
                 password=os.environ.get("MYSQL_PASSWORD", ""),
                 database=os.environ.get("MYSQL_DATABASE", "rotor_test"))
MYSQL_TABELA = "rotor_data"
MYSQL_KOLONE = ("t", "x", "y")

QDB_PROTOKOL = "rest"
QDB_HOST = os.environ.get("QDB_HOST", "localhost")
QDB_REST_URL = f"http://{QDB_HOST}:9000"
QDB_REST_AUTH = None
QDB_TIMEOUT = 30.0

QDB_PROZOR_MIN = 5

QDB_CFG = dict(host=QDB_HOST, port=8812,
               user=os.environ.get("QDB_USER", "admin"),
               password=os.environ.get("QDB_PASSWORD", ""), dbname="qdb",
               connect_timeout=int(QDB_TIMEOUT))
QDB_TABELA = "messages_BVM1"

QDB_KOL_IZVOR = None
QDB_IZVOR = None

QDB_KOL_X = "ch0"
QDB_KOL_Y = "ch1"

QDB_KOL_BROJAC = "scnt"
FS_QDB = 2048.0

QDB_BROJAC_MOD = 65536
QDB_MAX_PRAZNINA = 1024

CSV_PUTANJA = "rotor_data.csv"
CSV_KOLONE = (0, 1, 2)
CSV_KOL_BROJAC = 0
FS_CSV = 2048.0
CSV_PRESKOCI = 1

SKALA_T = 1.0
SKALA_XY = 1.0 / 8.0

SONDA_MV_PO_UM = 8.0

try:
    import mysql.connector
except ImportError:
    mysql = None


def connect_db():
    return mysql.connector.connect(**MYSQL_CFG)


def citaj_mysql(n_samples):
    conn = _mysql_veza()
    cur = conn.cursor()
    t_, x_, y_ = MYSQL_KOLONE
    cur.execute(f"SELECT {t_}, {x_}, {y_} FROM {MYSQL_TABELA} "
                f"ORDER BY id DESC LIMIT %s", (n_samples,))
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None, None, None
    arr = np.array(rows, dtype=float)[::-1]
    return arr[:, 0], arr[:, 1], arr[:, 2]


_VEZA = [None]


def _mysql_veza():
    if _VEZA[0] is None:
        _VEZA[0] = connect_db()
        _VEZA[0].autocommit = True
    return _VEZA[0]


_CSV_KES = {"kljuc": None, "arr": None}


def _csv_putanja():
    if os.path.isabs(CSV_PUTANJA):
        return CSV_PUTANJA
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        CSV_PUTANJA)


def _ucitaj_csv_cijeli():
    put = _csv_putanja()
    if not os.path.exists(put):
        raise FileNotFoundError(f"CSV ne postoji: {put}")
    st = os.stat(put)
    kljuc = (put, st.st_size, st.st_mtime)
    if _CSV_KES["kljuc"] != kljuc:
        try:
            arr = np.loadtxt(put, delimiter=",", skiprows=CSV_PRESKOCI,
                             usecols=CSV_KOLONE, ndmin=2)
        except ValueError as e:
            raise ValueError(f"CSV se ne moze procitati ({put}): {e}")
        _CSV_KES["kljuc"], _CSV_KES["arr"] = kljuc, arr
    return _CSV_KES["arr"]


def _csv_u_txy(arr):
    if CSV_KOL_BROJAC is None:
        t = arr[:, 0]
    else:
        t = _odmotaj_brojac(arr[:, 0], QDB_BROJAC_MOD) / FS_CSV
    return t, arr[:, 1], arr[:, 2]


def citaj_csv(n_samples):
    arr = _ucitaj_csv_cijeli()
    if len(arr) < 2:
        return None, None, None
    return _csv_u_txy(arr[-n_samples:])


_QDB_VEZA = [None]


def _qdb_veza():
    con = _QDB_VEZA[0]
    if con is None or con.closed:
        try:
            import psycopg2
        except ImportError:
            raise RuntimeError("Za QuestDB treba psycopg2:\n"
                               "    pip install psycopg2-binary")
        con = psycopg2.connect(**QDB_CFG)
        con.autocommit = True
        _QDB_VEZA[0] = con
    return con


def _zatvori_qdb():
    con = _QDB_VEZA[0]
    _QDB_VEZA[0] = None
    if con is not None:
        try:
            con.close()
        except Exception:
            pass


def _odmotaj_brojac(n, mod, max_praznina=QDB_MAX_PRAZNINA):
    if mod is None or len(n) < 2:
        return n
    dn = np.diff(n)
    korak = dn + mod
    prelij = (dn < 0) & (korak >= 1) & (korak <= max_praznina)
    return n + mod * np.concatenate(([0], np.cumsum(prelij)))


def _qdb_upit_rest(upit):
    import base64
    import json
    import socket
    import urllib.error
    import urllib.parse
    import urllib.request

    url = QDB_REST_URL.rstrip("/") + "/exec?" + \
        urllib.parse.urlencode({"query": upit})
    req = urllib.request.Request(url)
    if QDB_REST_AUTH:
        tok = base64.b64encode(
            f"{QDB_REST_AUTH[0]}:{QDB_REST_AUTH[1]}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    try:
        with urllib.request.urlopen(req, timeout=QDB_TIMEOUT) as r:
            odg = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            poruka = json.loads(e.read().decode("utf-8")).get("error", e)
        except Exception:
            poruka = e
        raise RuntimeError(f"QuestDB REST ({e.code}): {poruka}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"QuestDB REST nedostupan ({QDB_REST_URL}): "
                           f"{e.reason}") from None
    except (TimeoutError, socket.timeout):
        raise RuntimeError(
            f"QuestDB je primio upit, ali nije odgovorio za "
            f"{QDB_TIMEOUT:.0f} s.\n  Server se VIDI sa ove masine - upit je "
            f"prespor. Probaj:\n"
            f"    - manji QDB_PROZOR_MIN (sada {QDB_PROZOR_MIN})\n"
            f"    - manji n_samples\n"
            f"    - veci QDB_TIMEOUT") from None
    if "error" in odg:
        raise RuntimeError(f"QuestDB REST: {odg['error']}")
    return odg.get("dataset", [])


def _qdb_upit_pg(upit):
    cur = _qdb_veza().cursor()
    try:
        cur.execute(upit)
        return cur.fetchall()
    finally:
        cur.close()


def citaj_questdb(n_samples):
    upit = (f"SELECT {QDB_KOL_BROJAC}, {QDB_KOL_X}, {QDB_KOL_Y} "
            f"FROM {QDB_TABELA} ")
    uslovi = []
    if QDB_PROZOR_MIN:
        uslovi.append(f"ts > dateadd('m', -{int(QDB_PROZOR_MIN)}, now())")
    if QDB_KOL_IZVOR:
        vr = str(QDB_IZVOR).replace("'", "''")
        uslovi.append(f"{QDB_KOL_IZVOR} = '{vr}'")
    if uslovi:
        upit += "WHERE " + " AND ".join(uslovi) + " "
    upit += f"LIMIT -{int(n_samples)}"

    if QDB_PROTOKOL == "rest":
        rows = _qdb_upit_rest(upit)
    else:
        rows = _qdb_upit_pg(upit)

    if not rows:
        return None, None, None

    arr = np.array(rows, dtype=float)
    n = _odmotaj_brojac(arr[:, 0], QDB_BROJAC_MOD)

    unazad = int(np.sum(np.diff(n) <= 0))
    if unazad > 2:
        raise RuntimeError(
            f"{QDB_KOL_BROJAC} ide unazad {unazad} puta u {len(n)} uzoraka "
            f"- ili vise uredjaja pise u {QDB_TABELA} (postavi "
            f"QDB_KOL_IZVOR/QDB_IZVOR), ili QDB_BROJAC_MOD="
            f"{QDB_BROJAC_MOD} ne odgovara brojacu.")

    t = n / FS_QDB
    return t, arr[:, 1], arr[:, 2]


def citaj_custom(n_samples):
    raise NotImplementedError(
        "Postavi IZVOR = 'mysql', 'questdb' ili 'csv', "
        "ili napisi citaj_custom().")


def procitaj(n_samples):
    if IZVOR == "mysql":
        t, x, y = citaj_mysql(n_samples)
    elif IZVOR == "questdb":
        t, x, y = citaj_questdb(n_samples)
    elif IZVOR == "csv":
        t, x, y = citaj_csv(n_samples)
    else:
        t, x, y = citaj_custom(n_samples)
    if t is None:
        return None, None, None
    return (np.asarray(t, float) * SKALA_T,
            np.asarray(x, float) * SKALA_XY,
            np.asarray(y, float) * SKALA_XY)


def fetch_latest(conn=None, n_samples=300, drop_resets=True):
    t, x, y = procitaj(n_samples)
    if t is None:
        return None, None, None

    skokovi = np.where(np.diff(t) <= 0)[0]
    if len(skokovi):
        poc = int(skokovi[-1] + 1)
        if len(t) - poc >= 32:
            print(f"  ⓘ Reset vremena u prozoru — uzeto posljednjih "
                  f"{len(t)-poc} od {len(t)} uzoraka.")
            t, x, y = t[poc:], x[poc:], y[poc:]

    return t, x, y


def izmjeri_omegu(t, x, y, omega_nom=OMEGA_NOM):
    n = len(t)
    if n < 32:
        return omega_nom

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return omega_nom

    sig = (x - np.mean(x)) + (y - np.mean(y))

    spec  = np.abs(np.fft.rfft(sig * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    f_nom = omega_nom / (2 * np.pi)
    maska = (freqs > 0.3 * f_nom) & (freqs < 3.0 * f_nom)
    if not np.any(maska):
        return omega_nom
    omega = 2 * np.pi * freqs[int(np.argmax(np.where(maska, spec, -np.inf)))]

    tt = t - t[0]
    df = freqs[1] - freqs[0]
    raspon = max(0.05, 2 * np.pi * df / omega)
    for _ in range(4):
        mreza = np.linspace(omega*(1-raspon), omega*(1+raspon), 41)
        ocjene = []
        for om in mreza:
            B = np.column_stack([np.sin(tt*om), np.cos(tt*om)])
            koef, *_ = np.linalg.lstsq(B, sig, rcond=None)
            ocjene.append(np.sum((B @ koef)**2))
        omega = float(mreza[int(np.argmax(ocjene))])
        raspon /= 6.0

    return omega


def normalize(t, x, y, omega=None):
    if omega is None:
        omega = OMEGA_NOM

    tau = (t - t[0]) * omega
    X = x / DELTA_UM
    Y = y / DELTA_UM

    k = int(tau[-1] // (2*np.pi))
    puni = tau <= k * 2*np.pi if k >= 1 else slice(None)
    X = X - X[puni].mean()
    Y = Y - Y[puni].mean()

    return tau, X, Y


def realtime_loader(n_samples=300, poll_interval=0.5):
    zadnji = None
    while True:
        try:
            t, x, y = fetch_latest(n_samples=n_samples)
        except Exception as e:
            print(f"[Akvizicija] greska citanja: {e}; pokusavam ponovo...")
            _VEZA[0] = None
            _zatvori_qdb()
            time.sleep(2.0)
            continue

        if t is not None and len(t) >= 32:
            otisak = (float(t[0]), float(t[-1]), len(t))
            if otisak != zadnji:
                zadnji = otisak
                omega = izmjeri_omegu(t, x, y)
                tau, X, Y = normalize(t, x, y, omega)
                yield tau, X, Y, omega

        time.sleep(poll_interval)


def provjeri_izvor(n_samples=300):
    print(f"Izvor: {IZVOR}")
    if IZVOR == "questdb":
        filt = (f", {QDB_KOL_IZVOR} = '{QDB_IZVOR}'"
                if QDB_KOL_IZVOR else "")
        print(f"  tabela {QDB_TABELA}{filt}, x={QDB_KOL_X}, y={QDB_KOL_Y}")
    t, x, y = fetch_latest(n_samples=n_samples)
    if t is None:
        print("  Nema podataka. Provjeri podesavanja na vrhu fajla.")
        return
    dts = np.diff(t)
    dt = float(np.median(dts))
    fs = 1.0 / dt if dt > 0 else float("nan")
    praznine = int(np.sum(dts > 1.5 * dt))
    om = izmjeri_omegu(t, x, y)
    obrt = (t[-1] - t[0]) * om / (2 * np.pi)
    print(f"  redova procitano : {len(t)}")
    print(f"  t raspon         : {t[-1]-t[0]:.4f} s "
          f"({'raste' if t[-1] > t[0] else 'NE RASTE!'})")
    print(f"  f_s              : {fs:.0f} Hz")
    print(f"  praznina u nizu  : {praznine}"
          f"   {'OK' if praznine == 0 else '(izgubljeni uzorci)'}")
    print(f"  omega izmjereno  : {om:.2f} rad/s ({om/(2*np.pi):.2f} Hz)")
    print(f"  uzoraka/obrtaju  : {fs/(om/(2*np.pi)):.0f}"
          f"   {'OK' if fs/(om/(2*np.pi)) >= 32 else 'PREMALO (treba >=32)'}")
    print(f"  obrtaja u prozoru: {obrt:.2f}")

    tt = t - t[0]
    B = np.column_stack([np.sin(tt*om), np.cos(tt*om)])
    xc, yc = x - x.mean(), y - y.mean()
    fx = B @ np.linalg.lstsq(B, xc, rcond=None)[0]
    fy = B @ np.linalg.lstsq(B, yc, rcond=None)[0]
    udio = (np.sum(fx**2) + np.sum(fy**2)) / max(np.sum(xc**2) + np.sum(yc**2), 1e-30)
    print(f"  udio 1X u signalu: {100*udio:.1f} %", end="")
    if udio < 0.10:
        print("   ! NEMA IZRAZENE 1X KOMPONENTE — rotor vjerovatno\n"
              "    miruje (ili sonde ne gledaju rotor). Izmjerena omega je\n"
              "    tada slucajni pik suma, a identifikacija nema smisla.")
    else:
        print("   OK")
    print(f"  x raspon         : {x.min():.2f} .. {x.max():.2f}")
    print(f"  y raspon         : {y.min():.2f} .. {y.max():.2f}")
    amp = max(x.max()-x.min(), y.max()-y.min()) / 2
    dc = max(abs(x.mean()), abs(y.mean()))
    print(f"  DC nivo / amplit.: {dc:.1f} / {amp:.2f}")
    sirovi = dc / SKALA_XY if SKALA_XY else dc
    kao_um = 200 <= sirovi <= 4200
    kao_mv = (2000 <= sirovi <= 18000 and
              200 <= 200 + (sirovi - 2000) / SONDA_MV_PO_UM <= 4200)
    if kao_um and not kao_mv:
        print(f"    baza pise µm (zazor sonde {sirovi:.0f} µm = "
              f"{sirovi/1000:.2f} mm)")
        if SKALA_XY != 1.0:
            print(f"    ! SKALA_XY je {SKALA_XY:g} — treba 1.0")
    elif kao_mv and not kao_um:
        zaz = 200 + (sirovi - 2000) / SONDA_MV_PO_UM
        print(f"    baza pise mV (zazor sonde {zaz:.0f} µm = "
              f"{zaz/1000:.2f} mm)")
        if abs(SKALA_XY - 1.0 / SONDA_MV_PO_UM) > 1e-9:
            print(f"    ! SKALA_XY je {SKALA_XY:g} — treba "
                  f"1/{SONDA_MV_PO_UM:g}")
    else:
        print(f"    ! sirovi DC {sirovi:.0f} ne odgovara ni µm ni mV\n"
              f"      opsegu PPT-280 — provjeri da li je sonda u\n"
              f"      linearnom opsegu.")

    tau2, X, Y = normalize(t, x, y, om)
    r = float(np.max(np.hypot(X, Y)))
    print(f"  orbita / zazor   : {100*r:.1f} %  (DELTA_UM = {DELTA_UM:g} µm)")
    if r > 1.0:
        print("    ! Orbita veca od zazora — fizicki nemoguce. Ili je\n"
              "      SKALA_XY pogresna, ili je DELTA_UM premali.")
    elif r > 0.30:
        print("    ! Vrlo velika orbita za zdrav rotor (obicno par %).\n"
              "      Provjeri SKALA_XY i DELTA_UM prije identifikacije.")
    else:
        print("    OK za zdrav rotor")


def provjeri_mrezu(host=QDB_HOST, portovi=(9000, 8812), timeout=3.0):
    import socket
    print(f"Server {host}:")
    otvoreni = []
    for p in portovi:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((host, p))
            print(f"  port {p:5d}: OTVOREN")
            otvoreni.append(p)
        except socket.timeout:
            print(f"  port {p:5d}: timeout (paketi se gube - firewall ili "
                  f"nema rute do mreze)")
        except OSError as e:
            print(f"  port {p:5d}: odbijen/greska ({e})")
        finally:
            s.close()
    if 9000 in otvoreni:
        print("  -> Koristi QDB_PROTOKOL = 'rest'.")
    elif 8812 in otvoreni:
        print("  -> Koristi QDB_PROTOKOL = 'pg'.")
    else:
        print("  -> Ovaj racunar ne dopire do servera ni na jednom portu.\n"
              "     Najvjerovatnije nije u istoj mrezi (treba LAN kabl u\n"
              "     laboratoriji, VPN, ili pokretanje na racunaru na kojem\n"
              "     web konzola radi).")


def izdvoji_pogon(izlaz="rotor_pogon.csv", prag=0.10, f_min=None,
                  f_max=300.0, prozor_s=1.0, min_trajanje_s=5.0,
                  tol_brzine=0.02):
    arr = _ucitaj_csv_cijeli()
    t, x, y = _csv_u_txy(arr)
    fs = FS_CSV if CSV_KOL_BROJAC is not None else \
        1.0 / float(np.median(np.diff(t)))
    W = int(round(prozor_s * fs))
    nw = len(arr) // W
    if nw < 1:
        print("Fajl je kraci od jednog prozora.")
        return None
    if f_min is None:
        f_min = 3.0 / prozor_s

    nfft = 16 * W
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    maska = (f >= f_min) & (f <= f_max)
    hann = np.hanning(W)
    tw = np.arange(W) / fs
    udio = np.zeros(nw)
    frek = np.zeros(nw)
    for k in range(nw):
        xs = x[k*W:(k+1)*W]; xs = xs - xs.mean()
        ys = y[k*W:(k+1)*W]; ys = ys - ys.mean()
        P = (np.abs(np.fft.rfft(xs * hann, nfft))**2 +
             np.abs(np.fft.rfft(ys * hann, nfft))**2)
        fr = f[int(np.argmax(np.where(maska, P, -1.0)))]
        B = np.column_stack([np.sin(2*np.pi*fr*tw), np.cos(2*np.pi*fr*tw)])
        e = 0.0
        for sg in (xs, ys):
            e += np.sum((B @ np.linalg.lstsq(B, sg, rcond=None)[0])**2)
        udio[k] = e / max(np.sum(xs**2) + np.sum(ys**2), 1e-30)
        frek[k] = fr
    pogon = udio >= prag

    nizovi, poc = [], None
    for k in range(nw + 1):
        if poc is not None:
            ref = float(np.median(frek[poc:k]))
            if k == nw or not pogon[k] or abs(frek[k] - ref) > tol_brzine * ref:
                nizovi.append((poc, k))
                poc = None
        if poc is None and k < nw and pogon[k]:
            poc = k
    ustaljen = np.zeros(nw, bool)
    for a_, b_ in nizovi:
        if (b_ - a_) * prozor_s >= min_trajanje_s:
            ustaljen[a_:b_] = True

    grupa = max(1, int(round(10.0 / prozor_s)))
    def znak(i):
        if ustaljen[i:i+grupa].mean() >= 0.5:
            return "#"
        return "~" if pogon[i:i+grupa].mean() >= 0.5 else "."
    linija = "".join(znak(i) for i in range(0, nw, grupa))
    print(f"Fajl: {_csv_putanja()}  ({nw*prozor_s:.0f} s)")
    print(f"Vremenska linija (znak = {grupa*prozor_s:.0f} s):  "
          f"# ustaljena brzina   ~ zalet/zaustavljanje   . miruje")
    for i in range(0, len(linija), 60):
        print(f"  {i*grupa*prozor_s:6.0f} s  {linija[i:i+60]}")
    print(f"Udio 1X: najveci {100*udio.max():.1f} %, medijan "
          f"{100*np.median(udio):.1f} %  (prag {100*prag:.0f} %)")

    if pogon.any():
        prikaz = pogon.copy()
        prikaz[1:] |= pogon[:-1]
        prikaz[:-1] |= pogon[1:]
        idx = np.flatnonzero(prikaz)[:300]
        print("\nPo sekundi (samo oko dijelova u pogonu):")
        print("     t [s]   brzina [Hz]   o/min   udio 1X")
        pret = None
        for k in idx:
            if pret is not None and k != pret + 1:
                print("       ...")
            oznaka = "" if pogon[k] else "   (miruje)"
            print(f"   {k*prozor_s:7.0f}   {frek[k]:9.2f}   {frek[k]*60:7.0f}"
                  f"   {100*udio[k]:6.1f} %{oznaka}")
            pret = k

    for a_, b_ in nizovi:
        if (b_ - a_) * prozor_s >= min_trajanje_s:
            fm = float(np.median(frek[a_:b_]))
            print(f"  ustaljeno: {a_*prozor_s:6.0f} - {b_*prozor_s:6.0f} s"
                  f"  ~{fm:.2f} Hz ({fm*60:.0f} o/min)")

    kandidati = [(b_ - a_, a_, b_) for a_, b_ in nizovi]
    if not kandidati or max(kandidati)[0] * prozor_s < min_trajanje_s:
        if pogon.any():
            print(f"\n! Rotor se okrece, ali nigdje ustaljenom brzinom "
                  f"(±{100*tol_brzine:.0f} %) najmanje {min_trajanje_s:.0f} s"
                  f" — nista nije snimljeno.\n"
                  f"  Ako tabela iznad pokazuje da brzina samo blago pliva,\n"
                  f"  probaj blazi kriterij:\n"
                  f"    u.izdvoji_pogon(tol_brzine=0.05, min_trajanje_s=2)")
        else:
            print(f"\n! Rotor nigdje ne radi — nista nije snimljeno.\n"
                  f"  Izvezi iz baze period u kojem se rotor okretao.")
        return None
    _, a, b = max(kandidati)
    if b - a > 2:
        a, b = a + 1, b - 1

    f_med = float(np.median(frek[a:b]))
    red = arr[a*W:b*W]
    put = izlaz if os.path.isabs(izlaz) else os.path.join(
        os.path.dirname(_csv_putanja()), izlaz)
    with open(_csv_putanja(), encoding="utf-8", errors="replace") as fu:
        zaglavlje = fu.readline().rstrip("\r\n") if CSV_PRESKOCI else ""
    fmt = ["%d" if CSV_KOL_BROJAC is not None else "%.9g"] + \
          ["%.6f"] * (red.shape[1] - 1)
    np.savetxt(put, red, delimiter=",", fmt=fmt, header=zaglavlje,
               comments="")
    print(f"\nSnimam najduzi ustaljeni dio: {a*prozor_s:.0f} - "
          f"{b*prozor_s:.0f} s ({(b-a)*prozor_s:.0f} s, {len(red)} redova)")
    print(f"Brzina: ~{f_med:.2f} Hz = {f_med*60:.0f} o/min "
          f"(OMEGA_NOM u config.py treba biti blizu "
          f"{2*np.pi*f_med:.1f} rad/s)")
    print(f"Snimljeno: {put}")
    print(f'Sada postavi  CSV_PUTANJA = "{os.path.basename(put)}"')
    return put


def snimi_iz_baze(sekundi=60, izlaz="rotor_data.csv", prozor_min=None):
    global QDB_PROZOR_MIN
    n = int(sekundi * FS_QDB)
    stari = QDB_PROZOR_MIN
    if prozor_min is None:
        prozor_min = max(2, int(sekundi / 60) + 2)
    QDB_PROZOR_MIN = prozor_min
    try:
        t, x, y = citaj_questdb(n)
    finally:
        QDB_PROZOR_MIN = stari
    if t is None:
        print("Baza nije vratila nijedan red (prazan prozor?).")
        return None

    put = izlaz if os.path.isabs(izlaz) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), izlaz)
    scnt = np.round(t * FS_QDB).astype(np.int64)
    with open(put, "w", encoding="utf-8") as f:
        f.write('"scnt","ch0","ch1"\n')
        for a, b, c in zip(scnt, x, y):
            f.write(f"{a},{b:.6f},{c:.6f}\n")
    print(f"Snimljeno {len(t)} redova ({len(t)/FS_QDB:.1f} s) u {put}")
    _CSV_KES["kljuc"] = None
    return put


def prati_rotor(sekundi=60, interval=2.0, n=2048):
    kraj = time.time() + sekundi
    print("vrijeme    brzina      udio 1X   stanje   (Ctrl+C za prekid)")
    while time.time() < kraj:
        try:
            t, x, y = fetch_latest(n_samples=n)
        except Exception as e:
            print(f"  greska: {e}")
            time.sleep(interval)
            continue
        if t is None or len(t) < 32:
            print("  nema podataka")
            time.sleep(interval)
            continue
        om = izmjeri_omegu(t, x, y)
        tt = t - t[0]
        B = np.column_stack([np.sin(tt*om), np.cos(tt*om)])
        xc, yc = x - x.mean(), y - y.mean()
        e = sum(np.sum((B @ np.linalg.lstsq(B, s, rcond=None)[0])**2)
                for s in (xc, yc))
        udio = e / max(np.sum(xc**2) + np.sum(yc**2), 1e-30)
        f_hz = om / (2*np.pi)
        stanje = "RADI" if udio >= 0.10 and f_hz >= 3 else "miruje"
        print(f"{time.strftime('%H:%M:%S')}  {f_hz:7.2f} Hz "
              f"{f_hz*60:7.0f} o/min  {100*udio:5.1f} %   {stanje}")
        time.sleep(interval)


def provjeri_sondu(n_samples=2048):
    t, x, y = procitaj(n_samples)
    if t is None:
        print("Nema podataka.")
        return
    print("PPT-280: zazor 0,2-2,2 mm @ 8 mV/µm ili 0,2-4,2 mm @ 4 mV/µm;"
          "\n         izlaz 2-18 V\n")
    for ime, v in (("ch0/x", float(np.mean(x))), ("ch1/y", float(np.mean(y)))):
        sirovo = abs(v) / SKALA_XY if SKALA_XY else abs(v)
        print(f"  {ime}: DC = {v:.1f}   (sirovo iz baze: {sirovo:.0f})")
        if 200 <= sirovo <= 4200:
            print(f"      kao µm -> zazor {sirovo/1000:.3f} mm   "
                  f"{'u opsegu' if sirovo <= 2200 else 'samo za opseg 4,2 mm'}")
        if 2000 <= sirovo <= 18000:
            for sens, gmax in ((8.0, 2.2), (4.0, 4.2)):
                z = (0.2 + (sirovo - 2000) / sens / 1000.0)
                if z <= gmax:
                    print(f"      kao mV @ {sens:.0f} mV/µm -> zazor "
                          f"{z:.3f} mm (u opsegu 0,2-{gmax} mm)")
        if not (200 <= sirovo <= 18000):
            print("      ! izvan svih ocekivanih opsega — provjeri sondu")
    print(f"\n  Sada: SKALA_XY = {SKALA_XY:g}, "
          f"osjetljivost {SONDA_MV_PO_UM:g} mV/µm.")
    print("\n  NAPOMENA: PPT-280 trazi metu promjera >= 20 mm. Ako je\n"
          "  vratilo tanje, stvarna osjetljivost je manja od katalaske,\n"
          "  pa su stvarni pomaci VECI od izracunatih.")
