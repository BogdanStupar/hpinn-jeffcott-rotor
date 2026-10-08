# HPINN Rotor Monitoring

Condition monitoring and fault detection of a Jeffcott rotor using hierarchical
physics-informed neural networks (HPINN).

This repository contains the code for my master thesis, *Condition Monitoring and
Fault Detection of a Jeffcott Rotor Model Using Hierarchical Physics-Informed Neural
Networks* (University of Belgrade, Faculty of Mechanical Engineering, Department of
Control Engineering, 2026; supervisor Prof. Radiša Jovanović). The method was
validated on synthetic signals, in a hardware-in-the-loop setup and on a physical
Baker Hughes RK4 rotor kit with eddy-current proximity probes.

## Method

The rotor is described by the dimensionless Jeffcott model, with X along the
horizontal and Y along the vertical axis:

```
X'' + Dx·X' + Cx·X = Ux·sin(τ + φ)
Y'' + Dy·Y' + Cy·Y = Uy·cos(τ + φ)
```

where τ = Ωt is dimensionless time, D is damping, C is stiffness, U is unbalance and
φ is the excitation phase. Displacements are normalized by the radial clearance, so
R = √(X² + Y²) = 1 corresponds to rotor–stator contact.

Monitoring runs in two phases.

### Phase 1: identification of the healthy rotor

A fully connected network (5 × 64, tanh) approximates the orbit (X(τ), Y(τ)). The
network weights and the parameter vector Ξ = [D, C, U, φ] are trained together on
three loss terms:

- **L_data:** fit to measured displacements.
- **L_ODE:** residual of the equations of motion at collocation points, with
  derivatives computed by automatic differentiation.
- **L_phys:** coupling between the parameters of the two axes.

Stability is enforced structurally, not as a penalty term. The parameters are
reparametrized as D = e^β_D and C = C_min + e^β_C, so the Routh–Hurwitz conditions
(D > 0, C > 0) hold for any value of β. Setting C_min = 1 additionally restricts the
model to subcritical operation (Ω < 1); C_min = 0 also allows supercritical
operation.

After training, U and φ are refined analytically from the measured 1Ω amplitude and
phase, using the identified C and D. The identified model is saved to
`hpinn_params.json` and frozen.

### Phase 2: fault identification

Each fault appears as a deviation from the frozen healthy model. The residual of the
healthy model is linear in all deviations, so Phase 2 is a single least-squares
regression over physics-derived terms rather than over a generic candidate library:

```
r_x = dD·X' + dC·X + dU·sin(τ+φ) + U₂·sin(2τ+φ) + K·(Y − Y·R)
r_y = dD·Y' + dC·Y + dU·cos(τ+φ) + U₂·cos(2τ+φ) + K·(−X + X·R)
```

| Term | Physical meaning |
|------|------------------|
| dU   | change in unbalance |
| dC   | change in linear stiffness |
| dD   | change in linear damping |
| U₂   | 2Ω excitation (misalignment) |
| K    | rub friction coefficient (cross-coupled term) |

Both equations are stacked into one system, so the coupling between axes in the
physical model acts as a constraint on the solution.

Detection uses statistics relative to a healthy reference recorded on the first
`N_REF` windows:

- **1Ω group (unbalance, stiffness, damping):** detected through the total 1Ω
  residual amplitude A1. The individual dU, dC and dD all produce only 1Ω content
  and differ only in phase, so they are reported together. The change in orbit
  semi-axis ratio is used only as a hint for unbalance.
- **Misalignment:** detected through the 2Ω coefficient U₂.
- **Rub:** detected through NEH, the fraction of signal energy left after the mean,
  1Ω and 2Ω components are removed. What remains are higher harmonics and
  subharmonics, the signature of intermittent rotor–stator contact. The 2Ω component
  is excluded so that misalignment is not reported as rub. Rub is reported when NEH
  rises to more than twice its healthy value.

Deviations are expressed in units of the healthy-state spread (a 4σ threshold).
Estimates are averaged over `N_USREDNJI` consecutive windows, and a diagnosis
changes only after it repeats `N_POTVRDA` times in a row.

A scalar health indicator is also computed (`health_indicator.py`):
HI = RMS² − H_norm(λ), where H_norm is the normalized entropy of the fault
coefficients.

## Main results

- The parameter vector Ξ = [D, C, U] is **not identifiable** from steady-state
  response at a single speed. The identified model still reproduces the response
  amplitude within 0.1%.
- Regression over physics-derived terms lowered the condition number from about
  **6,700 to about 20** compared with sparse regression over a generic library
  (SINDy-type).
- On the physical rotor kit, unbalance, misalignment and rotor–stator rub were all
  detected and correctly classified. The results agreed with a commercial vibration
  monitoring system measuring the same rotor.

## Repository structure

| File | Description |
|------|-------------|
| `main.py` | Entry point, starts the GUI |
| `gui.py` | Desktop application (matplotlib + Tk): acquisition, training, live diagnostics, fault sliders |
| `hpinn_model.py` | Network architecture (`RotorNet`) |
| `phase1_training.py` | Phase 1: training, parameter saving and loading |
| `phase2_diagnosis.py` | Phase 2: fault regression, healthy reference, deviations and diagnosis |
| `health_indicator.py` | Health indicator |
| `data_loader.py` | Data acquisition (CSV, MySQL, QuestDB) and nondimensionalization |
| `rotor_simulator.py` | Healthy rotor simulator that writes noisy synthetic data to MySQL in real time |
| `config.py` | All settings: windows, thresholds, operating mode, clearance |

Code identifiers and GUI labels are in Serbian (for example *faza* = phase,
*kvar* = fault, *ucitavanje podataka* = data loading).

## Installation

Python 3.8 or newer is required.

```bash
git clone https://github.com/BogdanStupar/hpinn-jeffcott-rotor.git
cd hpinn-jeffcott-rotor
pip install -r requirements.txt
```

A CUDA GPU is used when available; otherwise training runs on the CPU.

## Usage

### Data source

The data source is selected with the `ROTOR_IZVOR` environment variable:

| Value | Source | Notes |
|-------|--------|-------|
| `csv` (default) | `rotor_data.csv` next to the code | columns: sample counter, x [µm], y [µm]; one header row; sampling rate `FS_CSV` (2048 Hz) |
| `mysql` | table `rotor_data (t, x, y)` | used with `rotor_simulator.py` |
| `questdb` | QuestDB table over REST (port 9000) or PostgreSQL (port 8812) | live acquisition from the test rig |

Connection settings are read from environment variables (`MYSQL_HOST`, `MYSQL_USER`,
`MYSQL_PASSWORD`, `MYSQL_DATABASE`, `QDB_HOST`, `QDB_USER`, `QDB_PASSWORD`). The unit
scale factor `SKALA_XY` in `data_loader.py` must match the units the source
writes: 1.0 for micrometres, or 1/8 for millivolts from PPT-280 probes at 8 mV/µm.

### Synthetic data

```bash
export ROTOR_IZVOR=mysql MYSQL_PASSWORD=<password>
python rotor_simulator.py     # terminal 1: writes the healthy rotor response
python main.py                # terminal 2: starts the application
```

The simulator prints the true parameters, so the Phase 1 result can be checked
against them.

### Application

1. **1. Pokretanje** starts reading data in real time.
2. **2. Nauci parametre** runs Phase 1. **Ucitaj Ξ** instead loads previously
   identified parameters from `hpinn_params.json`.
3. **3. Simulacija** starts Phase 2. The first `N_REF` windows are recorded as the
   healthy reference, and diagnosis starts after that.
4. Faults are introduced either physically on the rig or with the sliders,
   depending on the mode:
   - **KIT:** unbalance, misalignment and rub are introduced on the rig, and their
     sliders are locked.
   - **HIL:** the healthy state is measured on the rig, and the selected fault is
     superimposed by solving the perturbed equations of motion.

   Stiffness and damping sliders work in both modes.

## Design notes

- **Different windows for the two phases.** The tanh network reliably fits only
  about four revolutions, so Phase 1 trains on the first `MAX_OBRTAJA` revolutions of
  the window. Phase 2 needs at least about 100 samples for a stable regression, so it
  uses the full window.
- **Excitation phase estimated per window.** Each window starts at τ = 0 at an
  arbitrary point of the rotation cycle. The phase is therefore read from the
  measured 1Ω component of each window rather than taken from Phase 1. This is
  consistent with the finding that φ is not identifiable from steady-state response.
- **Reduced regression basis.** The nonlinear stiffness and the diagonal rub term
  from the full model are omitted. For a steady, nearly sinusoidal signal they are
  collinear with dC, and separating them requires a speed change (run-up or
  coast-down).
- **Elliptical orbit.** For a perfectly circular orbit the radius is constant, so
  radially symmetric nonlinear terms cannot be detected. The simulator therefore uses
  anisotropic supports (Ωx ≠ Ωy).

## Limitations

- All measurements are taken at a single operating speed. Stiffness and damping
  faults are detected, but cannot be separated from each other without data at
  several speeds.
- Measurement data from the test rig are not included in this repository.

## Author

Bogdan Stupar, M.Sc. in Mechanical Engineering (Control Engineering), University of
Belgrade. Contact: stuparbogdan000@gmail.com
