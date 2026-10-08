import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import threading
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import config
from config import (BG, PANEL, CARD, C, DIM, ACC, GRN, ERR, GOLD, PRP,
                    state, fault_params, N_MEAS, N_COLLOC,
                    RUB_CONTACT_RADIUS, RUB_CONTACT_REL,
                    KIT_ZAKLJUCANI, N_REF, N_POTVRDA, N_USREDNJI)
from phase1_training import hpinn_train, ucitaj_parametre
from health_indicator import health_indicator
from phase2_diagnosis import (identifikuj, referenca, odstupanje, dijagnoza,
                   PRAG_NEH,
                   PARAMETRI, OPIS)
from data_loader import realtime_loader


class RotorGUI:
    _podesavam = False

    def __init__(self):
        self.fig = plt.figure(figsize=(19, 11), facecolor=BG)
        self.fig.canvas.manager.set_window_title("HPINN Rotor Kit Simulator")
        self._build_layout()
        self._phase1_data = None
        self._trained_net = None
        self._sim_thread  = None
        self._lock        = threading.Lock()
        plt.rcParams.update({"text.color": C})
        self._loss_hist = []
        self._loss_drawn = 0
        self._new_data_ready = False
        self._lam_buffer = []
        self._par_buffer = []
        self._diag_buffer = []
        self._last_diag = "ZDRAVO"
        self._zadnji_otisak = None
        self._ref_napredak = None
        self._R_zdravo = None
        self._Rc_kontakt = RUB_CONTACT_RADIUS
        self._lin_fault_cache = None
        self._omega = None
        self._sim_plot_data = None
        self._training = False

        self._timer = self.fig.canvas.new_timer(interval=1500)
        self._timer.add_callback(self._osvjezi_sve)
        self._timer.start()

    def _build_layout(self):
        gs = gridspec.GridSpec(
            4, 5, figure=self.fig,
            left=0.055, right=0.975, top=0.925, bottom=0.20,
            hspace=0.62, wspace=0.42,
        )

        def ax(r, c, rs=1, cs=1):
            a = self.fig.add_subplot(gs[r:r+rs, c:c+cs])
            a.set_facecolor(CARD)
            for sp in a.spines.values(): sp.set_edgecolor("#2a2f42")
            a.tick_params(colors=DIM, labelsize=7)
            a.xaxis.label.set_color(DIM)
            a.yaxis.label.set_color(DIM)
            a.title.set_color(C)
            a.grid(color="#1e2233", linewidth=0.7)
            return a

        self.ax_data_x  = ax(0, 0, 1, 2)
        self.ax_data_y  = ax(0, 2, 1, 2)
        self.ax_loss    = ax(0, 4)
        self.ax_orbit_h = ax(1, 0)
        self.ax_params  = ax(1, 1)
        self.ax_colloc  = ax(1, 2, 1, 2)
        self.ax_status  = ax(1, 4)
        self.ax_orbit_f = ax(2, 0, 2, 1)
        self.ax_fft     = ax(2, 1, 1, 2)
        self.ax_hi      = ax(3, 1, 1, 2)
        self.ax_forces  = ax(2, 3)
        self.ax_fault_w = ax(3, 3)
        self.ax_info    = ax(2, 4, 2, 1)

        self.fig.text(0.5, 0.972, "Bezdimenzionalni Jeffcott rotor model",
                      ha="center", va="center", color=C, fontsize=11,
                      fontweight="bold", fontfamily="monospace")
        self._add_buttons()
        self._add_sliders()
        self._draw_placeholder_info()

    def _add_buttons(self):
        def mk(pos, label, col, cb):
            a = self.fig.add_axes(pos)
            b = Button(a, label, color="#1e2233", hovercolor="#2a3050")
            b.label.set_color(col); b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        y, h, w, gap, x0 = 0.022, 0.040, 0.122, 0.014, 0.065
        px = lambda i: [x0 + i*(w+gap), y, w, h]
        self.btn_gen   = mk(px(0), "1. Pokretanje",      ACC,  self._on_generate)
        self.btn_train = mk(px(1), "2. Nauci parametre", GRN,  self._on_train)
        self.btn_sim   = mk(px(2), "3. Simulacija",      GOLD, self._on_sim_toggle)
        self.btn_reset = mk(px(3), "Reset sim",          ERR,  self._on_reset_sim)
        self.btn_mode  = mk(px(4), f"Rezim: {config.MODE}", PRP, self._on_mode)
        self.btn_load  = mk(px(5), r"Ucitaj $\Xi$",           C,  self._on_load)

    def _add_sliders(self):
        def mk(col_i, row_i, label, lo, hi, init, color):
            xs = [0.085, 0.405, 0.725]
            ys = [0.145, 0.093]
            a = self.fig.add_axes([xs[col_i], ys[row_i], 0.145, 0.017])
            a.set_facecolor(CARD)
            s = Slider(a, label, lo, hi, valinit=init,
                       color="#2a3050", track_color="#1a1e2e")
            s.label.set_color(color)
            s.label.set_fontsize(7.5)
            s.valtext.set_color(color)
            s.valtext.set_fontsize(7.5)
            s.poly.set_facecolor(color)
            return s

        self.sl_imbalance    = mk(0, 0, "Dеbalans",    0.0, 0.8, 0.0, ACC)
        self.sl_misalignment = mk(1, 0, "Nesaosnost", 0.0, 0.6, 0.0, GOLD)
        self.sl_rub_eta      = mk(2, 0, "Trenje eta",      0.0, 0.5, 0.0, ERR)
        self.sl_rub_kappa    = mk(0, 1, "Trenje kappa",    0.0, 0.3, 0.0, PRP)
        self.sl_stiffness    = mk(1, 1, "Krutost",      0.5, 2.0, 1.0, GRN)
        self.sl_damping      = mk(2, 1, "Prigusenje",   0.3, 2.0, 1.0, C)

        for sl, key in [
            (self.sl_imbalance,    "imbalance"),
            (self.sl_misalignment, "misalignment"),
            (self.sl_rub_eta,      "rub_eta"),
            (self.sl_rub_kappa,    "rub_kappa"),
            (self.sl_stiffness,    "stiffness"),
            (self.sl_damping,      "damping"),
        ]:
            sl.on_changed(lambda val, k=key: self._update_fault(k, val))

        self._slajderi = {
            "imbalance":    self.sl_imbalance,
            "misalignment": self.sl_misalignment,
            "rub_eta":      self.sl_rub_eta,
            "rub_kappa":    self.sl_rub_kappa,
        }
        self._primijeni_rezim()

    def _primijeni_rezim(self):
        kit = (config.MODE == "KIT")
        for key, sl in self._slajderi.items():
            if kit and abs(sl.val) > 1e-9:
                sl.set_val(0.0)
            if kit:
                fault_params[key] = 0.0
            sl.label.set_color(DIM if kit else sl.poly.get_facecolor())
            sl.valtext.set_color(DIM if kit else sl.poly.get_facecolor())

    def _on_load(self, event):
        if self._training:
            self._show_msg(self.ax_status,
                           "Obuka je u toku.\nSacekajte da zavrsi\n"
                           "ili je prekinite.", ERR)
            print("[GUI] Ucitavanje odbijeno: obuka je u toku. Kada zavrsi, "
                  "prepisala bi ucitane parametre.")
            self.fig.canvas.draw_idle()
            return

        p = ucitaj_parametre()
        if p is None:
            self._show_msg(self.ax_status,
                           "Nema sacuvanih\nparametara.\nPokreni dugme 2.", ERR)
            self.fig.canvas.draw_idle()
            return

        state["params"] = p
        state["learned"] = True
        state["baseline_lam"] = None
        self._lam_buffer = []
        self._par_buffer = []
        self._diag_buffer = []
        self._last_diag = "ZDRAVO"
        self._R_zdravo = None
        self._Rc_kontakt = RUB_CONTACT_RADIUS
        self._lin_fault_cache = None

        print(f"[GUI] Ucitani parametri: {p}")
        self._skaliraj_slajdere(p)

        with self._lock:
            data = self._phase1_data
        if data:
            self._draw_phase1_results(data[0], data[1], data[2], p, sol=None)
        else:
            self._show_msg(self.ax_status,
                           "Xi ucitani.\nPokreni 1 pa 3.", GRN)
        self.fig.canvas.draw_idle()

    def _on_mode(self, event):
        config.MODE = "HIL" if config.MODE == "KIT" else "KIT"
        self.btn_mode.label.set_text(f"Rezim: {config.MODE}")
        self._primijeni_rezim()
        self._lin_fault_cache = None
        print(f"[GUI] Rezim -> {config.MODE}")
        self.fig.canvas.draw_idle()

    def _update_fault(self, key, val):
        if self._podesavam:
            return
        if key in ("stiffness", "damping") and state.get("learned"):
            fault_params[key] = float(val)
            self._ogranici_krutost_prigusenje(key)
            fault_params[key] = float(
                self.sl_stiffness.val if key == "stiffness"
                else self.sl_damping.val)
            self._lin_fault_cache = None
            return

        if config.MODE == "KIT" and key in KIT_ZAKLJUCANI:
            if abs(val) > 1e-9:
                self._slajderi[key].set_val(0.0)
            fault_params[key] = 0.0
            return
        fault_params[key] = val
        self._lin_fault_cache = None

    def _draw_placeholder_info(self):
        for ax, txt in [
            (self.ax_data_x,  "Kliknite dugme 1\nza pokretanje"),
            (self.ax_data_y,  "Y kanal"),
            (self.ax_loss,    "Kriva gubitka\n(HPINN obuka)"),
            (self.ax_orbit_h, "Zdrava orbita\nnakon identifikacije"),
            (self.ax_params,  "Identifikovani\nparametri ODE"),
            (self.ax_colloc,  "Mjerne / kolokacijske tacke"),
            (self.ax_status,  "Cekanje na\ninicijalizaciju..."),
            (self.ax_orbit_f, "Live orbita\npokrenite dugme 3"),
            (self.ax_fft,     "Spektar"),
            (self.ax_hi,      "Health Indicator"),
            (self.ax_forces,  "Identifikovani parametri"),
            (self.ax_fault_w, "Tezina kvara [%]"),
            (self.ax_info,    "Slajderi -> injektuj kvar\nposlije ucenja parametara"),
        ]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                    ha="center", va="center", color=DIM, fontsize=8,
                    fontfamily="monospace")

    def _on_generate(self, event):
        if getattr(self, '_live_running', False):
            print("[GUI] Citanje baze je vec pokrenuto.")
            return
        self._live_running = True
        print("\n[Faza 1] Pokretanje citanja baze u stvarnom vremenu...")

        def live_data_loop():
            try:
                for tau_m, X_m, Y_m, omega in realtime_loader(
                        n_samples=N_MEAS, poll_interval=0.5):
                    if not self._live_running:
                        break
                    tau_f = np.linspace(tau_m[0], tau_m[-1], N_COLLOC)
                    with self._lock:
                        self._phase1_data = (tau_m, X_m, Y_m, tau_f)
                        self._omega = omega
                        self._new_data_ready = True
            except Exception as e:
                print(f"[Akvizicija] prekinuta: {e}")
                self._live_running = False

        threading.Thread(target=live_data_loop, daemon=True).start()

    def _draw_phase1_data(self, tau_m, X_m, Y_m, tau_f):
        for ax, sig, col, lbl in ((self.ax_data_x, X_m, GRN, "X"),
                                  (self.ax_data_y, Y_m, PRP, "Y")):
            ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
            ax.plot(tau_m, sig, color=col, lw=1.2, label=f"Senzor {lbl}")
            ax.set_title(f"Mjerne tacke  -  {lbl} pomak", fontsize=8, color=C)
            ax.set_xlabel("tau", fontsize=7); ax.set_ylabel(lbl, fontsize=7)
            ax.legend(fontsize=6, framealpha=0.3)
            ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_colloc
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        ax.scatter(tau_m, np.ones_like(tau_m)*1.5, s=6, color=ACC,
                   alpha=0.5, label=f"Mjerne (n={len(tau_m)})")
        ax.scatter(tau_f, np.ones_like(tau_f)*0.5, s=3, color=ERR,
                   alpha=0.3, label=f"Kolokacijske (n={len(tau_f)})")
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["tau_f", "tau_i"], color=DIM, fontsize=7)
        ax.set_title("Raspored mjernih / kolokacijskih tacaka", fontsize=8, color=C)
        ax.legend(fontsize=6, framealpha=0.3)
        ax.set_ylim(0, 2)

        self._draw_status_telemetry(tau_m, tau_f)

    def _draw_status_telemetry(self, tau_m, tau_f):
        ax = self.ax_status
        ax.cla(); ax.set_facecolor(CARD); ax.axis("off")
        n_obrt = (tau_m[-1]-tau_m[0])/(2*np.pi)
        om = self._omega if self._omega else float("nan")
        lines = [
            ("TELEMETRIJA UZIVO", GRN),
            (f"  Prozor tacaka:  {len(tau_m)}", C),
            (f"  Kolokacijske:   {len(tau_f)}", C),
            (f"  tau u [{tau_m[0]:.1f}, {tau_m[-1]:.1f}]", C),
            (f"  obrtaja (Faza 2): {n_obrt:.2f}",
             C if n_obrt >= 2.0 else ERR),
            (f"  uzoraka/obrtaju:  {len(tau_m)/max(n_obrt,1e-9):.0f}",
             C if len(tau_m)/max(n_obrt,1e-9) >= 25 else ERR),
            (f"  omega = {om:.2f} rad/s ({om/(2*np.pi):.2f} Hz)", C),
            ("", C),
            ("Dovedite rotor u ZDRAVO stanje", DIM),
            ("pa kliknite dugme 2", GOLD),
        ]
        if n_obrt < 2.0:
            lines.append(("  ! premalo obrtaja (treba >= 2)", ERR))
        if len(tau_m)/max(n_obrt, 1e-9) < 25:
            lines.append(("  ! rijetko odabiranje (treba >= 25/obrt)", ERR))
        for i, (txt, col) in enumerate(lines):
            ax.text(0.05, 0.97 - i*0.093, txt, transform=ax.transAxes,
                    color=col, fontsize=7.5, fontfamily="monospace", va="top")
        ax.set_title("Status", fontsize=8, color=C)

    def _on_train(self, event):
        if self._phase1_data is None:
            self._show_msg(self.ax_status, "Prvo pokrenite 1\nda stignu podaci!", ERR)
            self.fig.canvas.draw_idle()
            return

        with self._lock:
            tau_m, X_m, Y_m, tau_f = self._phase1_data

        def run():
            self._training = True
            self.btn_load.label.set_color("#3a3f52")
            self._loss_hist = []; self._loss_drawn = 0

            def show_training_msg():
                self._show_msg(self.ax_loss, "HPINN obuka\nu toku...", ACC)
                self._show_msg(self.ax_orbit_h, "HPINN obuka\nu toku...", ACC)
                self._show_msg(self.ax_params,  "Učim ODE\nparametre...", ACC)
                self.fig.canvas.draw()

            self.fig.canvas.get_tk_widget().after(0, show_training_msg)

            print("\n[Faza 1] HPINN obuka na realnim podacima...")

            def cb(ep, loss_val):
                with self._lock:
                    self._loss_hist.append(loss_val)

            try:
                net_trained, learned, hist = hpinn_train(
                    tau_m, X_m, Y_m, tau_f,
                    epochs=500, lr=1e-3, alpha=1.0, lam_phys=8.0,
                    progress_cb=cb)
            except Exception as e:
                print(f"  Greska u obuci: {e}")
                self._training = False
                return

            self._trained_net = net_trained
            self._training = False
            state["params"]  = learned
            state["learned"] = True
            state["baseline_lam"] = None
            self._lam_buffer = []
            self._diag_buffer = []
            self._last_diag = "ZDRAVO"
            print(f"\n  Identifikovani parametri: {learned}")

            p = learned
            def ode(t, s):
                X, Xd, Y, Yd = s
                Xdd = -p["Dx"]*Xd - p["Cx"]*X + p["Ux"]*np.sin(t+p["phi"])
                Ydd = -p["Dy"]*Yd - p["Cy"]*Y + p["Uy"]*np.cos(t+p["phi"])
                return [Xd, Xdd, Yd, Ydd]

            D_min = min(learned["Dx"], learned["Dy"])
            pre = float(np.clip(5.0 * (2.0 / max(D_min, 1e-6)), 30.0, 500.0))
            t_eval = np.linspace(tau_m[0], tau_m[-1], 1000)
            sol = solve_ivp(ode, (tau_m[0] - pre, tau_m[-1]),
                            [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, rtol=1e-6)

            self.fig.canvas.get_tk_widget().after(
                0, lambda: self._draw_phase1_results(tau_m, X_m, Y_m, learned, sol))
            self.fig.canvas.get_tk_widget().after(
                0, lambda: self._skaliraj_slajdere(learned))

        threading.Thread(target=run, daemon=True).start()

    def _granica_ruba(self, p, cilj=0.95, n_iter=7):
        tau = np.linspace(0.0, 6 * 2 * np.pi, 600)

        def R_max(K):
            def rhs(t, s):
                X, Xd, Y, Yd = s
                Fx = p["Ux"] * np.sin(t)
                Fy = p["Uy"] * np.cos(t)
                R = np.hypot(X, Y)
                if R > self._Rc_kontakt:
                    Fx += K * Y - K * Y * R
                    Fy += -K * X + K * X * R
                return [Xd, -p["Dx"] * Xd - p["Cx"] * X + Fx,
                        Yd, -p["Dy"] * Yd - p["Cy"] * Y + Fy]
            try:
                s = solve_ivp(rhs, (-80.0, tau[-1]), [0.0] * 4,
                              t_eval=tau, rtol=1e-6, max_step=0.1)
                return float(np.max(np.hypot(s.y[0], s.y[2])))
            except Exception:
                return 99.0

        if R_max(0.0) >= cilj:
            return 0.0
        lo, hi = 0.0, 3.0
        if R_max(hi) <= cilj:
            return hi
        for _ in range(n_iter):
            sr = 0.5 * (lo + hi)
            if R_max(sr) <= cilj:
                lo = sr
            else:
                hi = sr
        return lo

    def _amplituda_lin(self, p, cm, dm):
        ax = abs(p["Ux"]) / np.sqrt((p["Cx"] * cm - 1.0) ** 2
                                    + (p["Dx"] * dm) ** 2)
        ay = abs(p["Uy"]) / np.sqrt((p["Cy"] * cm - 1.0) ** 2
                                    + (p["Dy"] * dm) ** 2)
        return float(ax + ay)

    def _ogranici_krutost_prigusenje(self, mijenjan):
        p = state.get("params")
        if not p:
            return
        CILJ = 0.95
        cm = float(self.sl_stiffness.val)
        dm = float(self.sl_damping.val)
        if self._amplituda_lin(p, cm, dm) <= CILJ:
            return

        sl = self.sl_stiffness if mijenjan == "stiffness" else self.sl_damping
        polazna = 1.0
        tekuca = cm if mijenjan == "stiffness" else dm
        lo, hi = polazna, tekuca
        for _ in range(12):
            sr = 0.5 * (lo + hi)
            a = (self._amplituda_lin(p, sr, dm) if mijenjan == "stiffness"
                 else self._amplituda_lin(p, cm, sr))
            if a <= CILJ:
                lo = sr
            else:
                hi = sr
        self._podesavam = True
        try:
            sl.set_val(round(lo, 3))
        finally:
            self._podesavam = False
        fault_params[mijenjan] = float(sl.val)
        print(f"  Klizac '{mijenjan}' ogranicen na {lo:.3f} — dalje bi "
              f"orbita presla {CILJ*100:.0f} % zazora (rezonanca).")

    def _skaliraj_slajdere(self, p):
        CILJ = 0.95

        def pojacanje(k):
            return 1.0 / np.sqrt((p["Cx"] - k ** 2) ** 2
                                 + (k * p["Dx"]) ** 2)

        g1, g2 = pojacanje(1), pojacanje(2)
        A1 = abs(p["Ux"]) * g1

        u1 = float(np.clip(max(CILJ / g1 - abs(p["Ux"]), 0.0), 0.01, 5.0))
        u2 = float(np.clip(max(CILJ - A1, 0.0) / max(g2, 1e-6), 0.01, 5.0))
        krub = float(np.clip(self._granica_ruba(p, CILJ), 0.05, 3.0))

        for sl, umax in ((self.sl_imbalance, u1),
                         (self.sl_misalignment, u2),
                         (self.sl_rub_eta, krub),
                         (self.sl_rub_kappa, krub)):
            sl.valmin, sl.valmax = 0.0, umax
            sl.ax.set_xlim(0.0, umax)
            if sl.val > umax:
                self._podesavam = True
                try:
                    sl.set_val(umax)
                finally:
                    self._podesavam = False

        print(f"  Pojacanje 1Ω={g1:.2f} 2Ω={g2:.2f} | zdrava orbita "
              f"{A1*100:.0f} % zazora")
        print(f"  Opseg klizaca (do {CILJ*100:.0f} % zazora): "
              f"debalans 0-{u1:.2f}, nesaosnost 0-{u2:.2f}, rub 0-{krub:.2f}")
        self.fig.canvas.draw_idle()

    def _osvjezi_sve(self):
        promjena = False
        try:
            promjena |= bool(self._refresh_loss())
            promjena |= bool(self._refresh_data())
            promjena |= bool(self._refresh_sim())
        except Exception as e:
            print(f"  Osvjezavanje: {e}")
        if promjena:
            self.fig.canvas.draw_idle()

    def _refresh_loss(self):
        with self._lock:
            h = list(self._loss_hist)
        n = len(h)
        if n == 0 or n == self._loss_drawn:
            return False
        self._loss_drawn = n
        y = np.asarray(h, float); ep = np.arange(1, n+1)
        ax = self.ax_loss
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        ax.semilogy(ep, y, color=ACC, lw=1.2)
        ax.set_xlabel("Epoha", fontsize=7, color=DIM)
        ax.set_ylabel("Gubitak (log)", fontsize=7, color=DIM)
        ax.set_title(f"Funkcija gubitka  (L={y[-1]:.2e})", fontsize=8, color=C)
        ax.tick_params(which='both', colors=DIM, labelsize=7)
        return True

    def _refresh_data(self):
        if self._new_data_ready and not state.get("sim_running", False):
            with self._lock:
                self._new_data_ready = False
                data = self._phase1_data
            if data:
                self._draw_phase1_data(*data)
                return True
        return False

    def _refresh_sim(self):
        with self._lock:
            args = self._sim_plot_data
            self._sim_plot_data = None

        if args is not None:
            try:
                self._update_sim_plots(*args)
            except Exception as e:
                print(f"  Greska crtanja: {e}")
            self.fig.canvas.draw_idle()

    def _draw_phase1_results(self, tau_m, X_m, Y_m, learned, sol=None):
        ax = self.ax_orbit_h
        ax.cla()
        ax.set_facecolor(CARD)
        ax.grid(color="#1e2233", lw=0.7)

        def ode(t, s):
            X, Xd, Y, Yd = s
            p = learned
            Xdd = -p["Dx"]*Xd - p["Cx"]*X + p["Ux"]*np.sin(t+p["phi"])
            Ydd = -p["Dy"]*Yd - p["Cy"]*Y + p["Uy"]*np.cos(t+p["phi"])
            return [Xd, Xdd, Yd, Ydd]

        D_min = min(learned["Dx"], learned["Dy"])
        pre = float(np.clip(5.0 * (2.0 / max(D_min, 1e-6)), 30.0, 500.0))
        t_eval = np.linspace(tau_m[0], tau_m[-1], 1000)
        sol = solve_ivp(ode, (tau_m[0] - pre, tau_m[-1]),
                        [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, rtol=1e-6)

        ax = self.ax_orbit_h
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        ax.plot(X_m, Y_m, color=ACC, lw=1, alpha=0.5, label="Mjerena")
        ax.plot(sol.y[0], sol.y[2], color=GRN, lw=1.8, label="HPINN")
        ax.set_title("Orbita (zdravo)", fontsize=8, color=C)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.legend(fontsize=6, framealpha=0.3)
        ax.set_aspect("equal")

        ax = self.ax_params
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        pnames = ["Dx", "Cx", "Ux", "Dy", "Cy", "Uy"]
        ax.bar(np.arange(6), [learned[k] for k in pnames], 0.5,
               color=GRN, alpha=0.8)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(pnames, fontsize=7, color=DIM)
        ax.set_title("Identif. bazni parametri", fontsize=8, color=C)

        ax = self.ax_status
        ax.cla(); ax.set_facecolor(CARD); ax.axis("off")
        Om = 1.0/np.sqrt(max(learned["Cx"], 1e-9))
        lines = [
            ("BAZNI MODEL ZAVRSEN", GRN),
            ("  Rotor profilisan", C),
            ("", C),
            (f"  Dx={learned['Dx']:.3f}  Cx={learned['Cx']:.3f}", C),
            (f"  Dy={learned['Dy']:.3f}  Cy={learned['Cy']:.3f}", C),
            (f"  Ux={learned['Ux']:.3f}  Uy={learned['Uy']:.3f}", C),
            (f"  Omega = 1/sqrt(Cx) = {Om:.3f}",
             C if learned["Cx"] > 1.05 else ERR),
            ("", C),
            ("Kliknite 3 za dijagnostiku", GOLD),
        ]
        for i, (txt, col) in enumerate(lines):
            ax.text(0.05, 0.97 - i*0.093, txt, transform=ax.transAxes,
                    color=col, fontsize=7.5, fontfamily="monospace", va="top")
        ax.set_title("Status", fontsize=8, color=C)
        self.fig.canvas.draw_idle()

    def _on_ui_thread(self, fn):
        try:
            self.fig.canvas.get_tk_widget().after(0, fn)
        except Exception:
            fn()

    def _show_msg(self, ax, msg, color=C):
        ax.cla(); ax.set_facecolor(CARD); ax.axis("off")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", color=color,
                fontsize=9, fontfamily="monospace")

    def _on_sim_toggle(self, event):
        if not state["learned"]:
            self._show_msg(self.ax_info, "Prvo nauci parametre\n(dugme 2)!", ERR)
            self.fig.canvas.draw_idle()
            return
        state["sim_running"] = not state["sim_running"]
        if state["sim_running"]:
            def show_simulation_msg():
                self._show_msg(self.ax_orbit_f, "Snimanje reference\nzdravog stanja...", ACC)
                self._show_msg(self.ax_fft, "Snimanje reference\nzdravog stanja...", ACC)
                self._show_msg(self.ax_hi,  "Snimanje reference\nzdravog stanja...", ACC)
                self._show_msg(self.ax_forces, "Snimanje reference\nzdravog stanja...", ACC)
                self._show_msg(self.ax_fault_w, "Snimanje reference\nzdravog stanja...", ACC)
                self._show_msg(self.ax_info,  "Rezultati...", ACC)
                self.fig.canvas.draw()

            self.fig.canvas.get_tk_widget().after(0, show_simulation_msg)

            self.btn_sim.label.set_text("3. Zaustavi")
            self.btn_sim.label.set_color(ERR)
            if self._sim_thread is None or not self._sim_thread.is_alive():
                self._sim_thread = threading.Thread(target=self._simulation_loop,
                                                    daemon=True)
                self._sim_thread.start()
        else:
            self.btn_sim.label.set_text("3. Simulacija")
            self.btn_sim.label.set_color(GOLD)

    def _on_reset_sim(self, event):
        state["sim_running"] = False
        state["sim_t"] = 0.0
        state["orbit_history"] = ([], [])
        state["hi_history"] = []
        state["rms_history"] = []
        state["fault_lam"] = None
        state["baseline_lam"] = None
        self._lam_buffer = []
        self._par_buffer = []
        self._diag_buffer = []
        self._last_diag = "ZDRAVO"
        self._zadnji_otisak = None
        self._R_zdravo = None
        self._Rc_kontakt = RUB_CONTACT_RADIUS
        self._lin_fault_cache = None
        self._sim_plot_data = None

        for sl, pocetna in ((self.sl_imbalance,    0.0),
                            (self.sl_misalignment, 0.0),
                            (self.sl_rub_eta,      0.0),
                            (self.sl_rub_kappa,    0.0),
                            (self.sl_stiffness,    1.0),
                            (self.sl_damping,      1.0)):
            sl.reset()
            sl.set_val(pocetna)

        self.btn_sim.label.set_text("3. Simulacija")
        self.btn_sim.label.set_color(GOLD)
        for ax in [self.ax_orbit_f, self.ax_fft, self.ax_hi,
                   self.ax_forces, self.ax_fault_w]:
            ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        self.fig.canvas.draw_idle()
        print("\n[Faza 2] Simulacija resetovana.")
        self._sim_thread = None

    def _simulation_loop(self):
        while state["sim_running"]:
            if self._phase1_data is None:
                time.sleep(0.1); continue

            with self._lock:
                tau_m, X_raw, Y_raw, _ = self._phase1_data
            p = state["params"]
            fp = {k: float(v) for k, v in fault_params.items()}

            otisak = (float(tau_m[0]), float(tau_m[-1]), len(tau_m),
                      float(X_raw[0]), float(X_raw[-1]),
                      tuple(round(float(v), 5) for v in fp.values()))
            nov_prozor = (otisak != self._zadnji_otisak)
            self._zadnji_otisak = otisak

            if not nov_prozor and state.get("baseline_lam") is not None:
                time.sleep(0.2)
                continue

            dX, dY = self._fault_delta(tau_m, p, fp)
            X_m, Y_m = X_raw + dX, Y_raw + dY

            try:
                par, cond = identifikuj(X_m, Y_m, tau_m, p)

                if cond > 200:
                    time.sleep(0.1)
                    continue

            except Exception as e:
                print(f"  Faza 2: {e}"); time.sleep(0.3); continue

            if state.get("baseline_lam") is None:
                if not nov_prozor:
                    time.sleep(0.2); continue
                self._lam_buffer.append(par)
                n = len(self._lam_buffer)
                if n < N_REF:
                    self._ref_napredak = (n, N_REF)
                    if n % 5 == 0:
                        print(f"  Snimam referencu zdravog stanja: "
                              f"{n}/{N_REF} prozora")
                    time.sleep(0.2); continue
                self._ref_napredak = None
                state["baseline_lam"] = referenca(self._lam_buffer)
                self._R_zdravo = float(np.max(np.sqrt(X_raw**2 + Y_raw**2)))
                self._Rc_kontakt = RUB_CONTACT_REL * self._R_zdravo
                try:
                    self._on_ui_thread(lambda: self._skaliraj_slajdere(p))
                except Exception as e:
                    print(f"  ⚠ Ponovno skaliranje klizaca: {e}")
                print(f"  Max radijus zdrave orbite: {self._R_zdravo:.3f}"
                      f"  -> prag kontakta "
                      f"{RUB_CONTACT_REL*self._R_zdravo:.3f}")
                self._lam_buffer = []
                _mu, _sd = state["baseline_lam"]
                print(f"  Referenca zdravog stanja snimljena "
                      f"({N_REF} prozora).")
                for _k, _min in (("A1", 0.05), ("U2", 0.05), ("NEH", 0.005)):
                    _r = _sd[_k] / max(abs(_mu[_k]), 1e-12)
                    _oz = "  <- rasipanje premalo!" if _r < _min else ""
                    print(f"     {_k:<4} mu={_mu[_k]:+.5f}  sd={_sd[_k]:.5f}"
                          f"  ({_r*100:.0f} %){_oz}")
                time.sleep(0.2); continue

            self._par_buffer.append(par)
            if len(self._par_buffer) > N_USREDNJI:
                self._par_buffer.pop(0)
            par_sr = {k: float(np.mean([q[k] for q in self._par_buffer]))
                      for k in self._par_buffer[0]}
            m = len(self._par_buffer)
            prag = 4.0 / np.sqrt(m)

            z = odstupanje(par_sr, state["baseline_lam"])
            state["fault_lam"] = par_sr
            state["cond"] = cond
            par = par_sr

            mu_ref = state["baseline_lam"][0]
            odnos = (par_sr.get("NEH", 0.0) /
                     max(mu_ref.get("NEH", 1e-9), 1e-9))
            state["neh_odnos"] = odnos

            R_sad = float(np.max(np.sqrt(X_m ** 2 + Y_m ** 2)))
            state["proc"] = {
                "1$\\Omega$": abs(z.get("A1", 0.0)) / prag,
                "nesaosnost":  abs(z.get("U2", 0.0)) / prag,
                "trenje":      max(0.0, (odnos - 1.0) / (PRAG_NEH - 1.0)),
                "zazor":       100.0 * R_sad,
                "porast":      max(0.0, 100.0 * (R_sad / max(
                                   self._R_zdravo or R_sad, 1e-9) - 1.0)),
            }
            diag = dijagnoza(z, prag=prag, odnos_neh=odnos,
                             ref_rat=state["baseline_lam"][0].get("RAT"))

            self._diag_buffer.append(diag)
            if len(self._diag_buffer) > N_POTVRDA:
                self._diag_buffer.pop(0)
            if len(set(self._diag_buffer)) == 1:
                self._last_diag = diag
            diag = self._last_diag

            R = np.sqrt(X_m**2 + Y_m**2)
            hi_v, rms_v, ent = health_indicator(
                R, np.array([z[k] for k in PARAMETRI]))
            state["hi_history"].append(hi_v)
            state["rms_history"].append(rms_v)
            if len(state["hi_history"]) > 500:
                state["hi_history"].pop(0); state["rms_history"].pop(0)

            with self._lock:
                self._sim_plot_data = (X_m, Y_m, tau_m, fp, par, z, cond, diag, prag)
            state["sim_t"] = float(tau_m[-1])
            time.sleep(0.2)

    def _fault_delta(self, tau_m, p, fp):
        key = (tuple(round(float(fp[k]), 4) for k in sorted(fp)),
               round(float(tau_m[0]), 3), round(float(tau_m[-1]), 3),
               len(tau_m), round(float(self._R_zdravo or 0), 3))
        if self._lin_fault_cache is not None and self._lin_fault_cache[0] == key:
            return self._lin_fault_cache[1], self._lin_fault_cache[2]

        zdravo = (abs(fp["stiffness"]-1) < 1e-3 and abs(fp["damping"]-1) < 1e-3
                  and max(fp["imbalance"], fp["misalignment"],
                          fp["rub_eta"], fp["rub_kappa"]) < 1e-6)
        if zdravo:
            z = np.zeros_like(tau_m)
            self._lin_fault_cache = (key, z, z)
            return z, z

        Rc = RUB_CONTACT_RADIUS
        if self._R_zdravo:
            Rc = RUB_CONTACT_REL * self._R_zdravo

        def rhs(t, s, cm, dm, kvar):
            X, Xd, Y, Yd = s
            Fx = p["Ux"]*np.sin(t + p["phi"])
            Fy = p["Uy"]*np.cos(t + p["phi"])
            if kvar:
                Fx += fp["imbalance"]*np.sin(t + p["phi"])
                Fy += fp["imbalance"]*np.cos(t + p["phi"])
                Fx += fp["misalignment"]*np.sin(2*t + p["phi"])
                Fy += fp["misalignment"]*np.cos(2*t + p["phi"])
                R = np.sqrt(X*X + Y*Y)
                if R > Rc:
                    H, K = fp["rub_eta"], fp["rub_kappa"]
                    Fx += -H*X + K*Y + H*X*R - K*Y*R
                    Fy += -K*X - H*Y + K*X*R + H*Y*R
            Xdd = -p["Dx"]*dm*Xd - p["Cx"]*cm*X + Fx
            Ydd = -p["Dy"]*dm*Yd - p["Cy"]*cm*Y + Fy
            return [Xd, Xdd, Yd, Ydd]

        D_min = min(p["Dx"], p["Dy"]) * max(fp["damping"], 1e-3)
        n_obrt = float(np.clip(5.0 * (2.0 / max(D_min, 1e-6)) / (2*np.pi),
                               15.0, 40.0))
        pre = n_obrt * 2*np.pi
        span = (float(tau_m[0]) - pre, float(tau_m[-1]))
        y0 = [0., 0., 0., 0.]
        try:
            sh = solve_ivp(rhs, span, y0, t_eval=tau_m, rtol=1e-5,
                           args=(1.0, 1.0, False))
            sf = solve_ivp(rhs, span, y0, t_eval=tau_m, rtol=1e-5,
                           args=(fp["stiffness"], fp["damping"], True))
            dX, dY = sf.y[0]-sh.y[0], sf.y[2]-sh.y[2]
        except Exception:
            dX = dY = np.zeros_like(tau_m)

        self._lin_fault_cache = (key, dX, dY)
        return dX, dY

    def _update_sim_plots(self, X, Y, T, fp, par, z, cond, diag, prag=4.0):
        if len(X) < 5:
            return
        p = state["params"]
        kvar = diag != "ZDRAVO"
        col_diag = ERR if kvar else GRN

        d = max(1, len(T) // 200)
        for ax, sig, base, lbl in ((self.ax_data_x, X, GRN, "X"),
                                   (self.ax_data_y, Y, PRP, "Y")):
            ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
            ax.plot(T[::d], sig[::d], color=ERR if kvar else base, lw=1.2)
            ax.set_title(f"Signal  -  {lbl}", fontsize=8, color=C)
            ax.set_xlabel("tau", fontsize=7)
            ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_orbit_f
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        m = min(len(X), 250)
        ax.plot(X[-m:], Y[-m:], color=ERR if kvar else GRN, lw=1.3)
        ax.plot(X[-1], Y[-1], "o", color=GOLD, ms=7, zorder=10)
        rx = p["Ux"]/np.sqrt((p["Cx"]-1)**2 + p["Dx"]**2)
        ry = p["Uy"]/np.sqrt((p["Cy"]-1)**2 + p["Dy"]**2)
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(rx*np.cos(th), ry*np.sin(th), color=GRN, lw=0.9,
                linestyle=":", alpha=0.6, label="Zdrav ref.")
        ax.set_title(f"Live orbita  ->  {diag}", fontsize=8.5, color=col_diag)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.legend(fontsize=6, framealpha=0.3); ax.set_aspect("equal")
        ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_fft
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        if len(X) >= 128:
            dt = float(np.median(np.diff(T)))
            f = np.fft.rfftfreq(len(X), d=dt)*2*np.pi
            m = f <= 4.6
            ax.plot(f[m], (np.abs(np.fft.rfft(X-X.mean()))/len(X))[m],
                    color=ACC, lw=1.1, label="X")
            ax.plot(f[m], (np.abs(np.fft.rfft(Y-Y.mean()))/len(Y))[m],
                    color=PRP, lw=1.1, alpha=0.75, label="Y")
            for k, lbl, c in ((1, "1x", GRN), (2, "2x", GOLD), (3, "3x", ERR)):
                ax.axvline(k, color=c, lw=0.8, linestyle="--", alpha=0.65)
                ax.text(k, 0.92, lbl, color=c, fontsize=7, ha="center",
                        transform=ax.get_xaxis_transform())
            ax.set_xlim(0, 4.5)
        ax.set_title("Spektar - red harmonika (x$\\Omega$)   plavo X, ljubicasto Y", fontsize=8, color=C)
        ax.set_xlabel("k", fontsize=7, color=DIM)
        ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_hi
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        if state["hi_history"]:
            hi = np.array(state["hi_history"]); rm = np.array(state["rms_history"])
            dh = max(1, len(hi) // 200)
            hi, rm = hi[::dh], rm[::dh]
            t = np.arange(len(hi)) * dh
            ax.plot(t, hi, color=ACC, lw=1.5, label="HI")
            ax.plot(t, rm, color=GOLD, lw=1, linestyle="--", alpha=0.7,
                    label="RMS2")
        ax.set_title("Health Indicator   plavo HI, zuto RMS$^2$",
                     fontsize=8, color=C)
        ax.set_xlabel("Koraci", fontsize=7, color=DIM)
        ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_forces
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        imena = [OPIS[k] for k in PARAMETRI]
        vals = [par[k] for k in PARAMETRI]
        boje = [ACC, GRN, C, GOLD, ERR]
        bars = ax.barh(range(len(vals)), vals, color=boje, alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(imena, fontsize=7, color=DIM)
        ax.axvline(0, color=DIM, lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y()+b.get_height()/2, f" {v:+.3f}",
                    va="center", color=C, fontsize=6.5)
        ax.set_title("Identifikovani parametri kvara", fontsize=8, color=C)
        ax.tick_params(colors=DIM, labelsize=7)

        ax = self.ax_fault_w
        ax.cla(); ax.set_facecolor(CARD); ax.grid(color="#1e2233", lw=0.7)
        K_SKALA = 100.0 / 45.0 - 1.0
        PRAG_P = 45.0
        pr = state.get("proc", {})
        red = ["1$\\Omega$", "nesaosnost", "trenje"]
        sirovo = [max(float(pr.get(k, 0.0)), 0.0) for k in red]
        vals = [100.0 * s / (s + K_SKALA) for s in sirovo]
        boje_z = [ERR if v > PRAG_P else (GOLD if v > 0.7 * PRAG_P else GRN)
                  for v in vals]
        bars = ax.bar(range(len(vals)), vals, color=boje_z, alpha=0.9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(red, fontsize=7, color=DIM)
        for b, v, s in zip(bars, vals, sirovo):
            ax.text(b.get_x() + b.get_width() / 2, v + 2.5,
                    f"{v:.0f}%", ha="center", color=C, fontsize=7)
        ax.axhline(PRAG_P, color=ERR, lw=1.0, ls="--", alpha=0.9)
        ax.text(len(vals) - 0.45, PRAG_P + 2, "prag", color=ERR, fontsize=6.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 45, 75, 100])
        ax.set_ylabel("tezina [%]", fontsize=7, color=DIM)
        zaz = pr.get("zazor", 0.0)
        boja_n = ERR if (cond >= 100 or zaz > 100) else (
            GOLD if zaz > 85 else C)
        dodatak = "  IZVAN VAZENJA MODELA" if zaz > 100 else ""
        ax.set_title(f"Tezina kvara   zazor {zaz:.0f} %   cond={cond:.0f}"
                     f"{dodatak}", fontsize=8, color=boja_n)
        ax.tick_params(colors=DIM, labelsize=6.5)

        ax = self.ax_info
        ax.cla(); ax.set_facecolor(CARD); ax.axis("off")
        hi_now  = state["hi_history"][-1] if state["hi_history"] else 0.0
        rms_now = state["rms_history"][-1] if state["rms_history"] else 0.0
        kit = (config.MODE == "KIT")
        lines = [
            (f"REZIM: {config.MODE}", PRP), ("-"*20, DIM),
        ]
        if kit:
            lines += [(" Dеbalans / nesaosnost /", DIM),
                      (" Trenje: Fizicki na KIT-u", DIM)]
        else:
            lines += [
                (f" Dеbalans   : {fp['imbalance']:.3f}", ACC),
                (f" Misalignment: {fp['misalignment']:.3f}", GOLD),
                (f" Rub eta     : {fp['rub_eta']:.3f}", ERR),
                (f" Rub kappa   : {fp['rub_kappa']:.3f}", PRP)]
        lines += [
            (f" Krutost x   : {fp['stiffness']:.2f}", GRN),
            (f" Prigusenje x: {fp['damping']:.2f}", C),
            ("", C),
            ("ZDRAV ROTOR", DIM), ("-"*20, DIM),
            (f" Dx={p['Dx']:.3f}  Cx={p['Cx']:.3f}", C),
            (f" Dy={p['Dy']:.3f}  Cy={p['Cy']:.3f}", C),
            (f" Ux={p['Ux']:.3f}  Uy={p['Uy']:.3f}", C),
            ("", C),
            ("ONLINE METRIKE", DIM), ("-"*20, DIM),
            (f" tau  : {state['sim_t']:.1f}", C),
            (f" HI   : {hi_now:+.4f}", ACC),
            (f" RMS2 : {rms_now:.4f}", GOLD),
            (f" NEH  : {state.get('neh_odnos', 1.0):.2f}x  (prag 2.0)",
             ERR if state.get("neh_odnos", 1.0) > 2.0 else C),
            (f" rezid: {par.get('REZ', 0.0):.3f}", DIM),
        ]
        nap = getattr(self, "_ref_napredak", None)
        if nap:
            lines.append((f" SNIMAM REFERENCU {nap[0]}/{nap[1]}", GOLD))
            lines.append((" rotor mora biti ZDRAV", DIM))
        else:
            lines.append((f" {diag}", col_diag))
        for i, (txt, col) in enumerate(lines):
            ax.text(0.04, 0.99 - i*0.045, txt, transform=ax.transAxes,
                    color=col, fontsize=7.5, fontfamily="monospace", va="top")
        ax.set_title("Parametri kvara", fontsize=8, color=GOLD)

    def show(self):
        plt.show()
        self._live_running = False
        state["sim_running"] = False


if __name__ == "__main__":
    RotorGUI().show()
