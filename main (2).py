"""
Mini-projet RF : exposition electromagnetique BTS 5G/6G - modelisation + optimisation PSO
=========================================================================================
Chaine (PDF) : X -> F -> E -> (tete multicouche) -> SAR -> dT (Pennes) ; E -> I_EM -> dVm
               et en parallele  E -> Pr -> SINR -> Rb ; puis J(X) sous contraintes.

Corrections par rapport a la v1 :
  1. Modele multicouche de la tete (peau/graisse/os/LCR/cerveau), Gamma_i, T_i = 1+Gamma_i,
     E_cerveau = E0 * prod(T_i * exp(-alpha_i e_i))                      [Question 2]
  2. Equation de Pennes multicouche resolue (volumes finis, Euler implicite),
     continuite de T et du flux assuree par les conductances aux interfaces.
  3. Polarisation : Cp = |ep.et|^2 dans le SAR, |cos(theta)| dans I_EM, et facteur de
     desadaptation de polarisation (PLF) dans le lien radio.
  4. Proprietes dielectriques dependantes de f (interpolation log-log).
  5. Chaine F -> E -> Pr : meme champ pour l'exposition et le lien radio
     (Pr = Friis avec gain d'element x |AF|^2) -> vrai compromis SAR / debit.
  6. Convention unique : E_rms pour la propagation, amplitude crete pour SAR = sigma*Cp*E^2/(2 rho).
  7. Limite adaptee a f : SAR <= 2 W/kg (<= 6 GHz) ; densite de puissance absorbee Sab (> 6 GHz).
  8. dVm en regime sinusoidal : |dVm| = |I_EM| * Rm / sqrt(1+(w Rm Cm)^2)
     (-> I*Rm quand f -> 0 comme dans le PDF ; negligeable aux GHz, comme le dit le PDF).

ATTENTION : les tables dielectriques / thermiques sont des valeurs APPROXIMATIVES de type
Gabriel / IT'IS. Les remplacer par celles du cours avant le rendu final. Les seuils
Sab, dT_max, E_int_max sont a verifier dans l'annexe / ICNIRP 2020.

Execution :  python -m pip install numpy matplotlib ;  python main.py [--no-plot]
"""

import sys
import numpy as np

# =============================================================================
# PARTIE 1 - CONSTANTES, CONFIGURATION, VARIABLES DE DECISION
# =============================================================================

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4*np.pi*1e-7
ETA0 = np.sqrt(MU0/EPS0)

CONFIG = {
    # PSO
    "seed": 21, "n_particles": 30, "max_iter": 80,
    "w_max": 0.85, "w_min": 0.40, "c1": 1.7, "c2": 1.7, "penalty_weight": 1.0e5,

    # Bornes des variables de decision
    "N_min": 2, "N_max": 4,
    "Pt_min_W": 1.0, "Pt_max_W": 40.0,            # puissance TOTALE emise par la BTS
    "f_min_Hz": 3.5e9, "f_max_Hz": 15.0e9,        # f_min = f_max -> frequence fixee
    "alpha_min": 0.1, "alpha_max": 1.0,           # amplitudes relatives a_n
    "position_min_m": 0.0, "position_max_m": 2.0, # abscisses x_n des elements
    "hBTS_min_m": 10.0, "hBTS_max_m": 40.0,
    "rBTS_min_m": 5.0, "rBTS_max_m": 500.0,       # distance horizontale BTS-utilisateur
    "dmin_factor": 0.5,                           # d_min = 0.5 * lambda

    # Utilisateur / radio
    "user_h_m": 1.5,
    "Ge_linear": 10**(5/10),                      # gain d'un element (5 dBi)
    "Gr_linear": 1.0,                             # gain terminal (0 dBi)
    "bandwidth_Hz": 20e6,
    "noise_W": 1.38e-23*290*20e6*10**(7/10),      # kTB * NF(7 dB)
    "interference_W": 4e-13,
    "term_pol_rad": np.deg2rad(60.0),             # polarisation du terminal (fixe, tenu a l'oreille)
    "tissue_dir_rad": 0.0,                        # direction preferentielle du tissu e_t

    # Thermique
    "depth_total_m": 0.05, "dz_m": 0.5e-3, "dt_s": 5.0,
    "exposure_s": 360.0,                          # 6 min (moyenne ICNIRP)
    "h_conv_W_m2K": 10.0,
    "rho_blood": 1050.0, "c_blood": 3617.0,

    # Membrane (cellule neuronale sphere r = 10 um)
    "cell_radius_m": 10e-6,
    "cm_area_F_m2": 0.01,                         # 1 uF/cm2
    "rm_area_ohm_m2": 0.1,                        # 1000 ohm.cm2

    # Energie / cout (PDF : emission seule ; Caut ~ N*Celem)
    "PA_efficiency": 0.35, "duration_h": 1.0,
    "cost_per_element": 8.0,
    "full_cost": False,                           # True : + C_RF + C_ctrl + C_inst
    "cost_RF_chain": 4.0, "cost_control": 3.0, "cost_installation": 10.0,

    # Seuils (a verifier : annexe du cours / ICNIRP 2020)
    "f_SAR_limit_Hz": 6e9,
    "SAR_max_W_kg": 2.0,          # 100 kHz - 6 GHz, tete/tronc
    "Sab_max_W_m2": 20.0,         # > 6 GHz, public, 4 cm2  (a verifier)
    "dT_max_K": 2.0,              # tete (annexe : 5 C / 2 C selon tissu)
    "dVm_max_V": 1e-3,            # valeur de demonstration
    "Eint_max_V_m": 5.0,          # valeur de demonstration (E_int_max(f))
    "SINR_min_linear": 10.0, "Rb_min_bps": 50e6,
    "Ptotal_max_W": 40.0, "Econs_max_Wh": 200.0,

    # Normalisation de J (niveaux de reference ~ 1 % de la limite)
    "SAR0": 0.02, "dT0": 0.02, "dVm0": 1e-5, "E0": 100.0, "Rb0": 100e6, "C0": 40.0,
    "W1": 1.0, "W2": 1.0, "W3": 1.0, "W4": 1.0, "W5": 1.0, "W6": 1.0,
}

NMAX = CONFIG["N_max"]
# X = [Pt, f, N, a_1..a_Nmax, beta_1..beta_Nmax, x_1..x_Nmax, pol, orientation, h_BTS, r_BTS]
IDX_PT, IDX_F, IDX_N = 0, 1, 2
IDX_A = 3
IDX_BETA = IDX_A + NMAX
IDX_XPOS = IDX_BETA + NMAX
IDX_POL = IDX_XPOS + NMAX
IDX_ORIENT = IDX_POL + 1
IDX_H = IDX_ORIENT + 1
IDX_R = IDX_H + 1
DIM = IDX_R + 1


def create_bounds():
    c = CONFIG
    low = np.array([c["Pt_min_W"], c["f_min_Hz"], c["N_min"]]
                   + [c["alpha_min"]]*NMAX + [0.0]*NMAX + [c["position_min_m"]]*NMAX
                   + [0.0, 0.0, c["hBTS_min_m"], c["rBTS_min_m"]], dtype=float)
    high = np.array([c["Pt_max_W"], c["f_max_Hz"], c["N_max"]]
                    + [c["alpha_max"]]*NMAX + [2*np.pi]*NMAX + [c["position_max_m"]]*NMAX
                    + [np.pi, np.pi, c["hBTS_max_m"], c["rBTS_max_m"]], dtype=float)
    return low, high


LOWER, UPPER = create_bounds()


def decode(X):
    X = np.asarray(X, dtype=float)
    N = int(np.clip(np.rint(X[IDX_N]), CONFIG["N_min"], CONFIG["N_max"]))
    return {"Pt_W": float(X[IDX_PT]), "f_Hz": float(X[IDX_F]), "N": N,
            "a": X[IDX_A:IDX_A+N].copy(),
            "beta": X[IDX_BETA:IDX_BETA+N].copy(),
            "xpos": X[IDX_XPOS:IDX_XPOS+N].copy(),
            "pol_rad": float(X[IDX_POL]), "orientation_rad": float(X[IDX_ORIENT]),
            "h_BTS_m": float(X[IDX_H]), "r_BTS_m": float(X[IDX_R])}


def repair(X):
    """Bornes + N entier + d_min = 0.5*lambda entre elements (positions triees)."""
    X = np.clip(np.asarray(X, dtype=float), LOWER, UPPER).copy()
    n = int(np.clip(np.rint(X[IDX_N]), CONFIG["N_min"], CONFIG["N_max"]))
    X[IDX_N] = n
    dmin = CONFIG["dmin_factor"]*C0/X[IDX_F]
    x = np.sort(X[IDX_XPOS:IDX_XPOS+n])
    for i in range(1, n):
        x[i] = max(x[i], x[i-1]+dmin)
    if x[-1] > CONFIG["position_max_m"]:
        x -= x[-1]-CONFIG["position_max_m"]
    if x[0] < CONFIG["position_min_m"]:
        x += CONFIG["position_min_m"]-x[0]
    X[IDX_XPOS:IDX_XPOS+n] = x
    return X


def make_X(Pt, f, N, a, beta, x, pol, orient, h, r):
    X = np.zeros(DIM)
    X[IDX_PT], X[IDX_F], X[IDX_N] = Pt, f, N
    X[IDX_A:IDX_A+N] = a
    X[IDX_BETA:IDX_BETA+N] = beta
    X[IDX_XPOS:IDX_XPOS+N] = x
    X[IDX_A+N:IDX_A+NMAX] = 1.0
    X[IDX_POL], X[IDX_ORIENT], X[IDX_H], X[IDX_R] = pol, orient, h, r
    return X


# =============================================================================
# PARTIE 2 - QUESTION 1 : CHAMP RAYONNE PAR LA BTS AU NIVEAU DU PORTABLE
#            X -> F -> E     (Friis / espace libre)
# =============================================================================

def calculate_field(X, verbose=False):
    """
    E_rms = sqrt(30 * Pt * Ge * |AF|^2) / d   (Friis, equivalent a Pr = Pt Gt Gr (lambda/4 pi d)^2)
    AF = sum_n w_n exp(j(beta_n + k x_n cos(theta))), w_n = a_n / sqrt(sum a^2)  (sum w^2 = 1,
    donc Pt = puissance TOTALE emise et |AF|^2 <= N : gain de reseau).
    """
    p = decode(X)
    lam = C0/p["f_Hz"]
    k = 2*np.pi/lam
    d = np.hypot(p["r_BTS_m"], p["h_BTS_m"]-CONFIG["user_h_m"])      # |r_B - r_U|
    w = p["a"]/np.sqrt(np.sum(p["a"]**2))
    AF_c = np.sum(w*np.exp(1j*(p["beta"] + k*p["xpos"]*np.cos(p["orientation_rad"]))))
    AF = abs(AF_c)
    G_tot = CONFIG["Ge_linear"]*AF**2
    E_rms = np.sqrt(30*p["Pt_W"]*G_tot)/d
    S_inc = E_rms**2/ETA0
    if verbose:
        print("\n[Q1] Champ rayonne par la BTS au niveau du portable")
        print(f"  lambda = {lam*1e3:.3f} mm | d = {d:.2f} m | N = {p['N']}")
        print(f"  |AF| = {AF:.4f} (max {np.sqrt(p['N']):.4f}) | gain total = {10*np.log10(G_tot):.2f} dBi")
        print(f"  E_rms = {E_rms:.4e} V/m | E_crete = {np.sqrt(2)*E_rms:.4e} V/m")
        print(f"  Densite de puissance incidente = {S_inc:.4e} W/m2")
    return {"lambda_m": lam, "k_rad_m": k, "distance_m": d, "AF": AF, "G_tot": G_tot,
            "E_rms_V_m": E_rms, "E_peak_V_m": np.sqrt(2)*E_rms, "S_inc_W_m2": S_inc}


# =============================================================================
# PARTIE 3 - QUESTION 2 : TETE MULTICOUCHE  E_i(z) = E_i(0) exp(-alpha_i z)
# =============================================================================

# (nom, epaisseur m, rho kg/m3, c J/kg/K, k W/m/K, perfusion 1/s) - valeurs typiques approx.
LAYERS = [
    ("peau",    0.002, 1109.0, 3391.0, 0.37, 0.0010),
    ("graisse", 0.003,  911.0, 2348.0, 0.21, 0.0004),
    ("os",      0.006, 1908.0, 1313.0, 0.32, 0.0003),
    ("LCR",     0.002, 1007.0, 4096.0, 0.57, 0.0),
    ("cerveau", None,  1045.0, 3696.0, 0.51, 0.0105),
]
# Proprietes dielectriques approximatives (er, sigma[S/m]) a f = 0.9, 2.45, 5.8, 10, 15 GHz
DIEL_F = np.array([0.9, 2.45, 5.8, 10.0, 15.0])*1e9
DIEL = {
    "peau":    ([41.4, 38.0, 35.1, 31.3, 26.5], [0.87, 1.46, 3.72, 8.0, 12.5]),
    "graisse": ([5.46, 5.28, 4.95, 4.6, 4.3],   [0.051, 0.104, 0.26, 0.50, 0.85]),
    "os":      ([12.45, 11.4, 9.6, 8.0, 6.9],   [0.143, 0.394, 1.0, 2.2, 3.5]),
    "LCR":     ([68.7, 66.2, 62.0, 52.5, 43.0], [2.41, 3.46, 8.0, 15.0, 22.0]),
    "cerveau": ([52.7, 48.9, 43.5, 38.0, 31.5], [0.94, 1.81, 4.96, 11.0, 17.5]),
}


def dielectric(name, f):
    """er(f), sigma(f) par interpolation log-log (Phase 2 : proprietes dependantes de f)."""
    er_t, sg_t = DIEL[name]
    lf = np.log(f)
    er = np.exp(np.interp(lf, np.log(DIEL_F), np.log(er_t)))
    sg = np.exp(np.interp(lf, np.log(DIEL_F), np.log(sg_t)))
    return er, sg


def build_grid():
    dz, L = CONFIG["dz_m"], CONFIG["depth_total_m"]
    n = int(round(L/dz))
    z = (np.arange(n)+0.5)*dz
    ends = np.cumsum([l[1] for l in LAYERS[:-1]])
    starts = np.concatenate(([0.0], ends))
    idx = np.searchsorted(ends, z, side="right")
    return z, idx, starts


Z, LIDX, LSTART = build_grid()


def layer_field(E_peak_inc, f, cp_pol=1.0):
    """Cascade du PDF : Gamma_i=(eta_{i+1}-eta_i)/(eta_{i+1}+eta_i), T_i=1+Gamma_i,
    E_cerveau = E0 * prod T_i exp(-alpha_i e_i). Retourne aussi E(z), SAR(z) sur la grille."""
    w = 2*np.pi*f
    eta, alpha, sigma = [ETA0], [], []
    for name, *_ in LAYERS:
        er, sg = dielectric(name, f)
        den = sg + 1j*w*EPS0*er
        alpha.append(np.sqrt(1j*w*MU0*den).real)
        eta.append(np.sqrt(1j*w*MU0/den))
        sigma.append(sg)
    Gam = [(eta[k+1]-eta[k])/(eta[k+1]+eta[k]) for k in range(len(LAYERS))]
    T = [abs(1+g) for g in Gam]
    thick = [l[1] for l in LAYERS[:-1]]
    E_in = [E_peak_inc*T[0]]
    for k in range(1, len(LAYERS)):
        E_in.append(E_in[-1]*np.exp(-alpha[k-1]*thick[k-1])*T[k])
    alpha_a, sigma_a, E_in_a = map(np.array, (alpha, sigma, E_in))
    Ez = E_in_a[LIDX]*np.exp(-alpha_a[LIDX]*(Z-LSTART[LIDX]))
    rho = np.array([l[2] for l in LAYERS])[LIDX]
    SARz = sigma_a[LIDX]*cp_pol*Ez**2/(2*rho)              # W/kg  (E crete)
    return {"alpha": alpha_a, "sigma": sigma_a, "Gamma": Gam, "T": T, "E_in": E_in_a,
            "Ez": Ez, "SARz": SARz, "E_brain": E_in[-1], "Gamma0": Gam[0]}


# =============================================================================
# PARTIE 4 - SAR, Sab ET EQUATION DE PENNES MULTICOUCHE
# =============================================================================

def _build_thermal():
    dz, dt, n = CONFIG["dz_m"], CONFIG["dt_s"], len(Z)
    rho = np.array([l[2] for l in LAYERS])[LIDX]
    c = np.array([l[3] for l in LAYERS])[LIDX]
    k = np.array([l[4] for l in LAYERS])[LIDX]
    wb = np.array([l[5] for l in LAYERS])[LIDX]
    Cap = rho*c*dz
    Perf = CONFIG["rho_blood"]*CONFIG["c_blood"]*wb*dz
    G = 1.0/(dz/(2*k[:-1]) + dz/(2*k[1:]))        # conductance d'interface : continuite du flux
    K = np.zeros((n, n))
    for i in range(n-1):
        K[i, i] += G[i]; K[i+1, i+1] += G[i]
        K[i, i+1] -= G[i]; K[i+1, i] -= G[i]
    K[0, 0] += 1.0/(1.0/CONFIG["h_conv_W_m2K"] + dz/(2*k[0]))   # surface : convection
    K[-1, -1] += 2*k[-1]/dz                                     # fond : theta = 0
    A = np.diag(Cap/dt) + K + np.diag(Perf)
    return rho, Cap, np.linalg.inv(A)


RHO_N, CAP_N, A_INV = _build_thermal()


def solve_pennes(SARz, return_history=False):
    """rho*c dT/dt = d/dz(k dT/dz) + rho*SAR - rho_b c_b w_b (T - T_b), theta = T - T_b."""
    dt, dz = CONFIG["dt_s"], CONFIG["dz_m"]
    nsteps = int(round(CONFIG["exposure_s"]/dt))
    q = RHO_N*SARz*dz
    theta = np.zeros(len(Z))
    hist = [theta.copy()] if return_history else None
    for _ in range(nsteps):
        theta = A_INV @ (CAP_N/dt*theta + q)
        if return_history:
            hist.append(theta.copy())
    return (theta, np.array(hist)) if return_history else theta


def calculate_sar_temperature(lf, S_inc, verbose=False):
    SAR = float(np.max(lf["SARz"]))                        # SAR local max (sans moyenne 10 g : conservatif)
    Sab = (1-abs(lf["Gamma0"])**2)*S_inc                   # puissance absorbee surfacique
    theta = solve_pennes(lf["SARz"])
    dT = float(np.max(theta))
    if verbose:
        print("\n[4] SAR, Sab et temperature (Pennes multicouche)")
        print(f"  SAR local max = {SAR:.4e} W/kg | Sab = {Sab:.4e} W/m2")
        print(f"  dT max apres {CONFIG['exposure_s']:.0f} s = {dT:.4e} K")
    return {"SAR_W_kg": SAR, "Sab_W_m2": Sab, "dT_K": dT, "theta": theta}


# =============================================================================
# PARTIE 5 - MODELE MEMBRANAIRE  Cm dVm/dt + Vm/Rm = I_EM ,  I_EM = sigma E A_eff |cos(theta)|
# =============================================================================

def calculate_membrane(lf, f, pol_rad, verbose=False):
    r = CONFIG["cell_radius_m"]
    Acell = 4*np.pi*r**2
    Cm = CONFIG["cm_area_F_m2"]*Acell
    Rm = CONFIG["rm_area_ohm_m2"]/Acell
    A_eff = np.pi*r**2
    cos_t = abs(np.cos(pol_rad-CONFIG["tissue_dir_rad"]))
    E_int = lf["E_brain"]
    I_EM = lf["sigma"][-1]*E_int*A_eff*cos_t
    w = 2*np.pi*f
    dVm = I_EM*Rm/np.sqrt(1+(w*Rm*Cm)**2)   # regime sinusoidal ; -> I*Rm quand w -> 0
    if verbose:
        print("\n[5] Membrane (cerveau)")
        print(f"  tau = Rm Cm = {Rm*Cm:.3e} s | I_EM = {I_EM:.3e} A | |dVm| = {dVm:.3e} V")
    return {"I_EM_A": I_EM, "dVm_V": dVm, "E_int_V_m": E_int, "tau_s": Rm*Cm}


# =============================================================================
# PARTIE 6 - LIEN RADIO  E -> Pr -> SINR -> Rb
# =============================================================================

def calculate_radio(fld, X, verbose=False):
    p = decode(X)
    PLF = np.cos(p["pol_rad"]-CONFIG["term_pol_rad"])**2
    Pr = fld["S_inc_W_m2"]*CONFIG["Gr_linear"]*fld["lambda_m"]**2/(4*np.pi)*PLF   # = Friis
    SINR = Pr/(CONFIG["interference_W"]+CONFIG["noise_W"])
    Rb = CONFIG["bandwidth_Hz"]*np.log2(1+SINR)
    if verbose:
        print("\n[6] Lien radio")
        print(f"  PLF = {PLF:.3f} | Pr = {Pr:.3e} W | SINR = {10*np.log10(max(SINR,1e-30)):.2f} dB | Rb = {Rb/1e6:.2f} Mbit/s")
    return {"PLF": PLF, "Pr_W": Pr, "SINR_linear": SINR, "Rb_bps": Rb}


# =============================================================================
# PARTIE 7 - ENERGIE ET COUT
# =============================================================================

def calculate_energy_cost(X, verbose=False):
    p = decode(X)
    Pcons = p["Pt_W"]/CONFIG["PA_efficiency"]          # emission seule (PDF) : P_PA domine
    Econs = Pcons*CONFIG["duration_h"]
    Caut = p["N"]*CONFIG["cost_per_element"]
    if CONFIG["full_cost"]:
        Caut += CONFIG["cost_RF_chain"]+CONFIG["cost_control"]+CONFIG["cost_installation"]
    if verbose:
        print("\n[7] Energie / cout")
        print(f"  Pcons = {Pcons:.2f} W | Econs = {Econs:.2f} Wh | Caut = {Caut:.1f}")
    return {"Pcons_W": Pcons, "Econs_Wh": Econs, "Caut": Caut, "P_RF_total_W": p["Pt_W"]}


# =============================================================================
# PARTIE 8 - J(X) ET CONTRAINTES
# =============================================================================

def evaluate(X, verbose=False):
    X = repair(X)
    p = decode(X)
    fld = calculate_field(X, verbose)
    cp = np.cos(p["pol_rad"]-CONFIG["tissue_dir_rad"])**2
    lf = layer_field(fld["E_peak_V_m"], p["f_Hz"], cp)
    bio = calculate_sar_temperature(lf, fld["S_inc_W_m2"], verbose)
    mem = calculate_membrane(lf, p["f_Hz"], p["pol_rad"], verbose)
    rad = calculate_radio(fld, X, verbose)
    ec = calculate_energy_cost(X, verbose)
    c = CONFIG

    terms = {"SAR": c["W1"]*bio["SAR_W_kg"]/c["SAR0"],
             "DeltaT": c["W2"]*bio["dT_K"]/c["dT0"],
             "DeltaVm": c["W3"]*mem["dVm_V"]/c["dVm0"],
             "Econs": c["W4"]*ec["Econs_Wh"]/c["E0"],
             "Debit": -c["W5"]*rad["Rb_bps"]/c["Rb0"],
             "Cout": c["W6"]*ec["Caut"]/c["C0"]}
    J = sum(terms.values())

    low_f = p["f_Hz"] <= c["f_SAR_limit_Hz"]
    exposure_ratio = bio["SAR_W_kg"]/c["SAR_max_W_kg"] if low_f else bio["Sab_W_m2"]/c["Sab_max_W_m2"]
    viol = {
        "Exposition": max(0.0, exposure_ratio-1.0),
        "DeltaT": max(0.0, bio["dT_K"]-c["dT_max_K"])/c["dT_max_K"],
        "DeltaVm": max(0.0, mem["dVm_V"]-c["dVm_max_V"])/c["dVm_max_V"],
        "E_int": max(0.0, mem["E_int_V_m"]-c["Eint_max_V_m"])/c["Eint_max_V_m"],
        "SINR": max(0.0, c["SINR_min_linear"]-rad["SINR_linear"])/c["SINR_min_linear"],
        "Rb": max(0.0, c["Rb_min_bps"]-rad["Rb_bps"])/c["Rb_min_bps"],
        "Ptotal": max(0.0, ec["P_RF_total_W"]-c["Ptotal_max_W"])/c["Ptotal_max_W"],
        "Econs": max(0.0, ec["Econs_Wh"]-c["Econs_max_Wh"])/c["Econs_max_Wh"],
    }
    penalty = c["penalty_weight"]*sum(v*v for v in viol.values())
    feasible = all(v <= 1e-12 for v in viol.values())
    if verbose:
        print("\n[8] J(X) et contraintes")
        for k_, v_ in terms.items():
            print(f"  {k_:9s}: {v_: .6f}")
        print(f"  J = {J:.6f} | penalite = {penalty:.4f}")
        lim = "SAR/2 W/kg" if low_f else "Sab/Sab_max"
        print(f"  Exposition = {100*exposure_ratio:.4f} % de la limite ({lim})")
        for k_, v_ in viol.items():
            print(f"    {k_:10s}: {'OK' if v_ <= 1e-12 else 'VIOLATION'}")
        print(f"  Faisable : {feasible}")
    return {"J": J, "fitness": J+penalty, "feasible": feasible, "violations": viol,
            "terms": terms, "exposure_ratio": exposure_ratio,
            "metrics": {**fld, **bio, **mem, **rad, **ec}, "layer": lf, "X": X.copy()}


# =============================================================================
# PARTIE 9 - PSO
# =============================================================================

def run_pso(verbose=True):
    rng = np.random.default_rng(CONFIG["seed"])
    n = CONFIG["n_particles"]
    X = np.array([repair(x) for x in rng.uniform(LOWER, UPPER, size=(n, DIM))])
    vmax = 0.15*(UPPER-LOWER)
    V = rng.uniform(-vmax, vmax, size=(n, DIM))
    pbest = X.copy()
    pbest_fit = np.array([evaluate(x)["fitness"] for x in X])
    g = int(np.argmin(pbest_fit)); gbest, gfit = pbest[g].copy(), pbest_fit[g]
    history = [gfit]
    if verbose:
        print("="*76 + "\nOPTIMISATION PSO\n" + "="*76)
        print(f"Particules={n} | iterations={CONFIG['max_iter']} | dim={DIM}")
    for t in range(CONFIG["max_iter"]):
        w = CONFIG["w_max"]-(CONFIG["w_max"]-CONFIG["w_min"])*t/max(CONFIG["max_iter"]-1, 1)
        for i in range(n):
            r1, r2 = rng.random(DIM), rng.random(DIM)
            V[i] = np.clip(w*V[i]+CONFIG["c1"]*r1*(pbest[i]-X[i])+CONFIG["c2"]*r2*(gbest-X[i]), -vmax, vmax)
            X[i] = repair(X[i]+V[i])
            fit = evaluate(X[i])["fitness"]
            if fit < pbest_fit[i]:
                pbest[i], pbest_fit[i] = X[i].copy(), fit
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gfit:
            gbest, gfit = pbest[g].copy(), pbest_fit[g]
        history.append(gfit)
        if verbose and (t == 0 or (t+1) % 10 == 0):
            print(f"  it {t+1:03d}/{CONFIG['max_iter']} : fitness = {gfit:.6f}")
    return gbest, history


# =============================================================================
# PARTIE 10 - ETUDES, SWEEP EN FREQUENCE, GRAPHIQUES, MAIN
# =============================================================================

def question2_profile(X):
    """Module de E(z) dans la tete et decroissance par couche."""
    p = decode(repair(X))
    fld = calculate_field(X)
    lf = layer_field(fld["E_peak_V_m"], p["f_Hz"])
    print("\n[Q2] Champ dans la tete (multicouche)")
    print(f"  E_crete incident = {fld['E_peak_V_m']:.4e} V/m")
    print(f"  {'couche':8s} {'|Gamma|':>8s} {'|T|':>7s} {'alpha[Np/m]':>12s} {'E(entree)[V/m]':>15s}")
    for k, l in enumerate(LAYERS):
        print(f"  {l[0]:8s} {abs(lf['Gamma'][k]):8.3f} {lf['T'][k]:7.3f} {lf['alpha'][k]:12.2f} {lf['E_in'][k]:15.4e}")
    print(f"  E_cerveau / E_incident = {lf['E_brain']/fld['E_peak_V_m']:.4e}")
    return lf


def frequency_sweep(X):
    """Phase 2 : a configuration geometrique fixee, evolution de la physique avec f."""
    print("\n[Phase 2] Balayage en frequence (meme X, f variable)")
    print(f"  {'f[GHz]':>7s} {'E_rms[V/m]':>11s} {'SAR[W/kg]':>11s} {'Sab[W/m2]':>10s} {'dT[K]':>10s} {'E_cerv/E_inc':>13s}")
    for f in [0.9e9, 2.45e9, 3.5e9, 5.8e9, 10e9, 15e9]:
        Xf = X.copy(); Xf[IDX_F] = f
        r = evaluate(Xf)
        m = r["metrics"]
        print(f"  {f/1e9:7.2f} {m['E_rms_V_m']:11.3e} {m['SAR_W_kg']:11.3e} {m['Sab_W_m2']:10.3e} {m['dT_K']:10.3e} {r['layer']['E_brain']/m['E_peak_V_m']:13.3e}")


def make_plots(final, history, path="resultats_rf.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib absent : pas de graphique)")
        return
    lf = final["layer"]
    theta, hist = solve_pennes(lf["SARz"], return_history=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    zmm = Z*1e3
    ax[0, 0].semilogy(zmm, lf["Ez"]); ax[0, 0].set_title("|E(z)| dans la tete (V/m, crete)")
    ax[0, 1].semilogy(zmm, np.maximum(lf["SARz"], 1e-30)); ax[0, 1].set_title("SAR(z) (W/kg)")
    for t_idx in [len(hist)//6, len(hist)//2, -1]:
        ax[1, 0].plot(zmm, hist[t_idx], label=f"t={t_idx*CONFIG['dt_s'] if t_idx>=0 else CONFIG['exposure_s']:.0f} s")
    ax[1, 0].set_title("Elevation de temperature (K)"); ax[1, 0].legend()
    ax[1, 1].plot(history); ax[1, 1].set_title("Convergence PSO (fitness)")
    ends = np.cumsum([l[1] for l in LAYERS[:-1]])*1e3
    for a in ax.ravel()[:3]:
        for e in ends:
            a.axvline(e, color="gray", lw=0.5, ls=":")
        a.set_xlabel("z (mm)")
    ax[1, 1].set_xlabel("iteration")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"Graphiques sauvegardes : {path}")


def main():
    lam = C0/3.5e9
    X_ref = make_X(Pt=20.0, f=3.5e9, N=4, a=np.ones(4), beta=np.zeros(4),
                   x=np.arange(4)*lam/2, pol=np.deg2rad(30), orient=np.pi/2, h=25.0, r=100.0)
    print("="*76 + "\nCONFIGURATION DE REFERENCE (Q1, Q2, Phase 2)\n" + "="*76)
    calculate_field(X_ref, verbose=True)
    question2_profile(X_ref)
    frequency_sweep(X_ref)

    best, hist = run_pso()
    print("\n" + "="*76 + "\nMEILLEURE CONFIGURATION X*\n" + "="*76)
    final = evaluate(best, verbose=True)
    p = decode(final["X"])
    print("\n[X*]")
    print(f"  Pt = {p['Pt_W']:.2f} W | f = {p['f_Hz']/1e9:.3f} GHz | N = {p['N']}")
    print(f"  a = {np.round(p['a'], 3)} | beta = {np.round(p['beta'], 3)} rad")
    print(f"  x = {np.round(p['xpos'], 3)} m | pol = {np.degrees(p['pol_rad']):.1f} deg | orient = {np.degrees(p['orientation_rad']):.1f} deg")
    print(f"  h_BTS = {p['h_BTS_m']:.1f} m | r_BTS = {p['r_BTS_m']:.1f} m | J = {final['J']:.5f}")
    if "--no-plot" not in sys.argv:
        make_plots(final, hist)
    print("\nRappel : tables dielectriques/thermiques et seuils = valeurs approximatives a verifier.")


if __name__ == "__main__":
    main()
