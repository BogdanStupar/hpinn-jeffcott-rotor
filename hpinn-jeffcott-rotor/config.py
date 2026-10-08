BG    = "#0a0c14"
PANEL = "#12151f"
CARD  = "#1a1e2e"
C     = "#d8dce8"
DIM   = "#6b7280"
ACC   = "#38bdf8"
GRN   = "#34d399"
ERR   = "#f87171"
GOLD  = "#fbbf24"
PRP   = "#a78bfa"

TAU_END    = 30.0
DT         = 0.005
N_MEAS     = 136
N_COLLOC   = 2500

MAX_OBRTAJA = 4.0

RUB_CONTACT_REL = 0.85

RUB_CONTACT_RADIUS = 1.0

DOZVOLI_NADKRITICNO = True

PARAMS_FILE = "hpinn_params.json"

OMEGA_NOM = 376.97
DELTA_UM  = 250

state = {
    "phase": 1,
    "learned": False,
    "params": None,
    "sim_running": False,
    "sim_t": 0.0,
    "sim_state": [0,0,0,0],
    "orbit_history": ([], []),
    "hi_history": [],
    "rms_history": [],
    "fault_lam": None,
}

N_REF = 20

N_POTVRDA = 3

N_USREDNJI = 8

MODE = "KIT"

KIT_ZAKLJUCANI = ("imbalance", "misalignment", "rub_eta", "rub_kappa")

fault_params = {
    "imbalance":    0.0,
    "misalignment": 0.0,
    "rub_eta":      0.0,
    "rub_kappa":    0.0,
    "stiffness":    1.0,
    "damping":      1.0,
}
