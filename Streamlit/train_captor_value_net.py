"""
Backward induction (T periods) entrenando una red de valor CaptorValueNet que
aproxima V^K_tau(theta | mu, R, beta_tilde(theta), zona, riqueza, victima, estado,
T_mad, T0, eta_cal, c_bar, H_ratio, lambda_4, eta_1, eta_2, beta_R).

Reescribe, de forma standalone (sin Streamlit), la logica ya verificada en Tab 3 de
app_DL.py: outcome_probs (Block B/C), p_det (Block D), p_cap_eff (Block F), p_pay_eff
(Block F), C_ops/C_maint (interior-optimum, recalibrados), IC^K/IR^K/IR^F (Appendix_3.tex
eq:gamma-factible), Delta H (Bernal_H.tex eq:bayes-update, registro minimo).

Uso:
    python3 train_captor_value_net.py --T 10 --N 2000 --out captor_value_net_T10.pt
"""
import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

TIPOS = ["DC", "PAR", "ELN", "FARC"]

# ── Constantes estructurales fijas (verificadas contra app.py / Bernal_H.tex) ──
BETAS_K = {
    "FARC": {"Pago": -0.70, "Muerte": -0.85, "Rescate": 0.90},
    "ELN": {"Pago": 1.10, "Muerte": 0.20, "Rescate": -0.65},
    "PAR": {"Pago": -0.25, "Muerte": 1.35, "Rescate": 0.15},
    "DC": {"Pago": 1.55, "Muerte": -0.40, "Rescate": -0.95},
}
BETAS_Z = {
    "Metropolis": 0.00, "Andean": -0.45, "Caribbean": -0.70,
    "Pacific / Red Zone": -0.20, "Eastern Plains/Jungle": -0.32,
}
ZETAS_POLITICA = {
    "DC": {"za": 0.2409, "zg": 0.5450}, "PAR": {"za": 0.2183, "zg": 0.5848},
    "ELN": {"za": 0.2101, "zg": 0.5532}, "FARC": {"za": 0.2087, "zg": 0.6197},
}
ETA_0_PDET = {"DC": -1.5, "PAR": -2.0, "ELN": -2.5, "FARC": -2.8}
LAMBDAS_0 = {"Pago": 0.012, "Muerte": 0.002, "Rescate": 0.008}
# C0_CAP: recalibrated for ALL 4 types (extension APPROVED by the user -- not a paper value,
# p_cap_eff_grid's baseline heterogeneity constant is free-calibrated in the app throughout).
# Route 3 (exact indifference point), ITERATION 5: attempt 4 (anchored to (0.5,0.5)) missed,
# because the State's OWN cost-minimizing (alpha*,gamma*) landed at its natural (0.68,0.63),
# not the (0.5,0.5) planning anchor -- feasibility was too loose to constrain it there. THIS
# iteration re-anchors the exact-equality target directly to (0.68,0.63) (the State's actual
# revealed preference), using v_next values BACK-CALCULATED from attempt 4's real retrain
# (not carried over from an even-earlier, differently-calibrated attempt). At the exact tie,
# IR^K's native tolerance (unchanged, -1e-9 below) reads it as satisfied ("Release-equivalent"),
# while the Captor's tie-break order (train_captor_true_type_net.py) resolves the SAME tie to
# "Continue". No formula changed -- only these constants, plus that tie-break order.
C0_CAP = {"DC": -7.70, "PAR": -6.32, "ELN": -8.16, "FARC": -15.00}
CALPHA_CAP = {"DC": 0.80, "PAR": 1.00, "ELN": 1.10, "FARC": 1.30}
CGAMMA_CAP = {"DC": 1.00, "PAR": 1.20, "ELN": 1.30, "FARC": 1.50}
CS_CAP = {"Strict": 0.40, "Lax": -0.40}
DELTA_A_CAP = {"Rescue": 0.60, "Negotiate": -0.20}
ALPHA_LETH = {"DC": -5.25, "PAR": -5.15, "ELN": -5.05, "FARC": -4.95}
KAPPA_REL = {"DC": 2.367, "PAR": 14.273, "ELN": 4.163, "FARC": 1.471}
# ETA_REP: recalibrated for ALL 4 types (extension APPROVED by the user), Route 3 iteration 5.
# Re-derived for the much lower p_cap of this iteration (previous ETA_REP values were only
# safe at the higher p_cap of iteration 4 -- at p_cap->0, U_kill->ETA_REP directly, so DC's
# old +0.34 would have let Kill beat Release outright). Keeps U_kill >=1.5 below U_rel/V_cont.
ETA_REP = {"DC": -3.760, "PAR": -14.740, "ELN": -5.530, "FARC": -2.970}
F_CAP = {"DC": 40.704, "PAR": 85.224, "ELN": 55.968, "FARC": 29.256}
# PHI_COST/KAPPA_C/NU_COST: recalibrated for ALL 4 types (extension APPROVED by the user --
# no number is given in Bernal_H.tex/Working_paper_eng.tex for phi/kappa_c/nu, only the
# functional form phi*exp(kappa_c*gamma)+nu, eq:cost-function-kidnapper). A PRIOR attempt this
# session recalibrated these (together with an iota_tau term inside p_cap) and was REVERTED
# because the user required exact paper equations, no iota term. THIS recalibration is
# narrower: constants only, same exact formulas throughout (p_cap_eff_grid, eq:kidnapper-cont
# unchanged) -- combined with C0_CAP and ETA_REP above, targets the Route-3 exact-indifference
# point for all 4 types (uniform across types for simplicity).
PHI_COST = {"DC": 0.02, "PAR": 0.02, "ELN": 0.02, "FARC": 0.02}
KAPPA_C = {"DC": 0.02, "PAR": 0.02, "ELN": 0.02, "FARC": 0.02}
NU_COST = {"DC": 0.010, "PAR": 0.010, "ELN": 0.010, "FARC": 0.010}
PHI_F = {"High wealth": 0.06, "Low wealth": 0.06}
KAPPA_F = {"High wealth": 3.60, "Low wealth": 3.00}
NU_F = {"High wealth": 0.02, "Low wealth": 0.02}
# varrho: prima de riesgo de la colusion, U_col = -E[p_pay]*R*(1+varrho). Sin esto (varrho=0),
# IR^F es estructuralmente imposible de satisfacer para CUALQUIER phi_F/nu_F>0, porque
# U_coop - U_col = -e_tau < 0 siempre. No estaba calibrado en ningun lado de app.py; elegido
# aqui para dar una region factible genuina (ni vacia, ni trivial) — ver diagnostico en el chat.
VARRHO_COLLUDE = 2.0
# V_L, F_col: parametros de la formula EXACTA del paper para la decision propia de la
# Familia (eq:f-coop/eq:f-col, Bernal_H.tex), separada del IR^F simplificado (p_pay+varrho)
# que ya usa solve_state_problem como restriccion de Gamma_tau(mu_tau) sobre el Estado (no
# se toca aqui). Sondeo de calibracion (N=4000 escenarios aleatorios, alpha,gamma,mu,R,tau,
# theta_F uniformes en los rangos de Tab 1): con p_rel_mu ~ lambda_4/total (canal de
# liberacion EXOGENA, ~0.0005, esencialmente independiente de alpha/gamma) y R tipicamente
# 1-100 millones, el termino "-R" de U_col domina para CUALQUIER V_L,F_col en un rango
# economicamente razonable (10-200) -- Cooperar domina a Colusion en el 100% de los
# escenarios muestreados. No es un error de calibracion: es un hallazgo estructural (pagar
# el rescate completo vs. un costo institucional pequeno rara vez es la mejor opcion), analogo
# al hallazgo ya documentado de que "Liberar" domina a "Continuar" para el Secuestrador.
# V_L=200 reusa la misma escala que OMEGA_K (peso de "valor de supervivencia" del Estado);
# F_col=20 es un orden de magnitud comparable al rango tipico de R.
V_L_FAMILY = 200.0
F_COL = 20.0
C_OPS = {
    "DC": (1.00, -0.25, 1.0, -0.20, 1.0, 0.10), "PAR": (1.00, -0.40, 1.0, -0.35, 1.0, 0.10),
    "ELN": (1.00, -0.55, 1.0, -0.50, 1.0, 0.10), "FARC": (1.00, -0.70, 1.0, -0.65, 1.0, 0.10),
}
C_MAINT = {
    "DC": (3.00, -6.00, 10.0, -0.80, 6.0, 0.10), "PAR": (3.00, -6.50, 10.0, -1.10, 6.0, 0.10),
    "ELN": (3.00, -7.00, 10.0, -1.40, 6.0, 0.10), "FARC": (3.00, -7.50, 10.0, -1.70, 6.0, 0.10),
}
OMEGA_P, OMEGA_K = 0.15, 200.0
CHI_ALPHA, CHI_GAMMA = 0.8, 0.5
PSI_H = 25.0
ZONES = ["Metropolis", "Andean", "Caribbean", "Pacific / Red Zone", "Eastern Plains/Jungle"]

GRID_N = 101
G = np.linspace(0.0, 1.0, GRID_N)
GA, GG = np.meshgrid(G, G)


@dataclass
class Params:
    R_millions: float
    beta_tilde: dict
    cov_zone: str
    cov_wealth: str
    cov_vict: str
    cov_state: str
    T_mad: float
    T0: float
    eta_cal: float
    c_bar: float
    H_ratio: float
    lambda_4: float
    eta_1: float
    eta_2: float
    beta_R: float


def sample_params(rng: np.random.Generator) -> Params:
    return Params(
        R_millions=float(rng.uniform(1.0, 100.0)),
        beta_tilde={th: float(rng.uniform(0.50, 0.99)) for th in TIPOS},
        cov_zone=str(rng.choice(ZONES)),
        cov_wealth=str(rng.choice(["Standard", "High"])),
        cov_vict=str(rng.choice(["Private", "Public sector"])),
        cov_state=str(rng.choice(["Strict", "Lax"])),
        T_mad=float(rng.uniform(1.0, 30.0)),
        T0=float(rng.uniform(0.1, 2.0)),
        eta_cal=float(rng.uniform(0.01, 0.20)),
        c_bar=float(rng.uniform(0.00, 0.20)),
        H_ratio=float(rng.uniform(0.0, 1.0)),
        lambda_4=float(rng.uniform(0.0001, 0.0100)),
        eta_1=float(rng.uniform(0.5, 3.0)),
        eta_2=float(rng.uniform(0.5, 3.0)),
        beta_R=float(rng.uniform(1.0, 10.0)),
    )


def m_t(tau: int, p: Params) -> float:
    """Maturation filter M(t) = min(1, (t/T_mad)^2) (eq:hj, Bernal_H.tex/Working_paper.tex),
    scaling the strategic hazards lambda_j (j=1,2,3) that feed outcome_probs_grid. NOT the
    MDG noise temperature T_t (eq:temperature) -- that is a different object, driven by
    T0_S/T0_F/T0_K (per-player, see MDG_MULT_* in run_period.py), unaffected by this fix."""
    return float(min(1.0, (tau / max(1e-9, p.T_mad)) ** 2))


def outcome_probs_grid(alpha, gamma, tipo, p: Params, mt: float):
    beta_K1, beta_K2, beta_K3 = BETAS_K[tipo]["Pago"], BETAS_K[tipo]["Muerte"], BETAS_K[tipo]["Rescate"]
    beta_z = BETAS_Z[p.cov_zone]
    beta_F1 = 0.80 if p.cov_wealth == "High" else 0.00
    beta_V1 = 1.36 if p.cov_vict == "Public sector" else 0.00
    beta_S = 0.50 if p.cov_state == "Lax" else 0.00
    za, zg = ZETAS_POLITICA[tipo]["za"], ZETAS_POLITICA[tipo]["zg"]
    eta0 = ETA_0_PDET[tipo]
    u_det = eta0 + p.eta_1 * alpha + p.eta_2 * gamma
    p_det = 1.0 / (1.0 + np.exp(-u_det))
    idx_pago = (beta_K1 + beta_z + beta_F1 - beta_V1 + beta_S) - za * alpha - zg * gamma - 0.18 * p_det
    idx_muerte = (beta_K2 + beta_z + beta_S) + za * alpha + zg * gamma - 0.18 * p_det
    idx_rescate = (-beta_S + beta_K3 + beta_z) + za * alpha + zg * gamma + 0.18 * p_det
    lam_pay = mt * LAMBDAS_0["Pago"] * np.exp(idx_pago)
    lam_kill = mt * LAMBDAS_0["Muerte"] * np.exp(idx_muerte)
    lam_res = mt * LAMBDAS_0["Rescate"] * np.exp(idx_rescate)
    total = lam_pay + lam_kill + lam_res + p.lambda_4
    p_cont = np.exp(-total)
    q = 1.0 - p_cont
    probs = {
        "Cont": p_cont, "1": q * lam_pay / total, "2": q * lam_kill / total,
        "3": q * lam_res / total, "4": q * p.lambda_4 / total,
    }
    return probs, p_det


def p_cap_eff_grid(alpha, gamma, tipo, p: Params, p_rescue: float, p_nego: float):
    """eq:p-cap (Bernal_H.tex), formula exacta -- sin termino de identificacion (un intento
    anterior de esta sesion agrego un desplazamiento dependiente de iota_tau; se revirtio
    porque el paper no lo tiene). Compartida por el Estado (solve_state_problem) y el
    Secuestrador (train_captor_true_type_net.py)."""
    c0, ca, cg = C0_CAP[tipo], CALPHA_CAP[tipo], CGAMMA_CAP[tipo]
    cs = CS_CAP[p.cov_state]
    u_r = DELTA_A_CAP["Rescue"] + c0 + ca * alpha + cg * gamma + cs
    u_n = DELTA_A_CAP["Negotiate"] + c0 + ca * alpha + cg * gamma + cs
    return p_rescue / (1 + np.exp(-u_r)) + p_nego / (1 + np.exp(-u_n))


def p_pay_eff_grid(alpha, gamma, tipo, p: Params, p_det, mt: float, p_coop: float, p_col: float):
    beta_K1 = BETAS_K[tipo]["Pago"]
    beta_z = BETAS_Z[p.cov_zone]
    beta_F1 = 0.80 if p.cov_wealth == "High" else 0.00
    beta_V1 = 1.36 if p.cov_vict == "Public sector" else 0.00
    beta_S = 0.50 if p.cov_state == "Lax" else 0.00
    za, zg = ZETAS_POLITICA[tipo]["za"], ZETAS_POLITICA[tipo]["zg"]
    idx_coop = (beta_K1 + beta_z + beta_F1 - beta_V1 + beta_S) - za * alpha - zg * gamma - 0.18 * p_det - 1.15
    idx_pay = idx_coop + 3.20
    lam_pay_coop = mt * LAMBDAS_0["Pago"] * np.exp(idx_coop)
    lam_pay_pay = mt * LAMBDAS_0["Pago"] * np.exp(idx_pay)
    idx_muerte = (BETAS_K[tipo]["Muerte"] + beta_z + beta_S) + za * alpha + zg * gamma - 0.18 * p_det
    idx_rescate = (-beta_S + BETAS_K[tipo]["Rescate"] + beta_z) + za * alpha + zg * gamma + 0.18 * p_det
    lam_kill = mt * LAMBDAS_0["Muerte"] * np.exp(idx_muerte)
    lam_res = mt * LAMBDAS_0["Rescate"] * np.exp(idx_rescate)
    total = lam_pay_coop + lam_kill + lam_res + p.lambda_4
    p_cont = np.exp(-(mt * LAMBDAS_0["Pago"] * np.exp(idx_coop) + lam_kill + lam_res + p.lambda_4))
    q = 1.0 - p_cont
    h1_coop = q * lam_pay_coop / total
    h1_pay = q * lam_pay_pay / total
    return p_coop * h1_coop + p_col * h1_pay


def p_surv_raw(tipo, theta_hat, p: Params, iota: float = 1.0):
    """eq:p-surv (Bernal_H.tex): Lambda(alpha_leth(theta) + beta_R * iota * 1{theta_hat=theta}).
    iota defaults to 1.0 for backward compatibility with call sites that don't yet pass the
    tau-specific informational precision (iota_tau = max_theta mu_tau(theta))."""
    u = ALPHA_LETH[tipo] + p.beta_R * iota * (1.0 if tipo == theta_hat else 0.0)
    return 1.0 / (1.0 + np.exp(-u))


def c_ops_grid(gamma, alpha, tipo):
    c0, c1, c2, c3, c4, c5 = C_OPS[tipo]
    return c0 + c1 * gamma + 0.5 * c2 * gamma ** 2 + c3 * alpha + 0.5 * c4 * alpha ** 2 + c5 * gamma * alpha


def c_maint_grid(gamma, alpha, tipo):
    m0, m1, m2, m3, m4, m5 = C_MAINT[tipo]
    return m0 + m1 * gamma + 0.5 * m2 * gamma ** 2 + m3 * alpha + 0.5 * m4 * alpha ** 2 + m5 * gamma * alpha


def entropy(mu: dict) -> float:
    return float(-sum(v * np.log(v) for v in mu.values() if v > 1e-12))


class CaptorValueNet(nn.Module):
    """input: mu(4) + tau_norm(1) + theta_onehot(4) + R_norm(1) + beta_tilde(4) +
    zone_onehot(5) + wealth_onehot(2) + vict_onehot(2) + state_onehot(2) + Tmad_norm(1)
    + T0(1) + eta_cal(1) + c_bar(1) + Hratio(1) + lambda4_norm(1) + eta1_norm(1)
    + eta2_norm(1) + betaR_norm(1) = 35 dims"""

    def __init__(self, in_dim: int = 34, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_input(theta, mu, tau, T, p: Params):
    zone_oh = [1.0 if p.cov_zone == z else 0.0 for z in ZONES]
    wealth_oh = [1.0 if p.cov_wealth == "Standard" else 0.0, 1.0 if p.cov_wealth == "High" else 0.0]
    vict_oh = [1.0 if p.cov_vict == "Private" else 0.0, 1.0 if p.cov_vict == "Public sector" else 0.0]
    state_oh = [1.0 if p.cov_state == "Strict" else 0.0, 1.0 if p.cov_state == "Lax" else 0.0]
    theta_oh = [1.0 if theta == t else 0.0 for t in TIPOS]
    beta_tilde_vec = [p.beta_tilde[t] for t in TIPOS]
    vec = (
        [mu[t] for t in TIPOS] + [tau / T] + theta_oh + [p.R_millions / 100.0] + beta_tilde_vec
        + zone_oh + wealth_oh + vict_oh + state_oh
        + [p.T_mad / 30.0, p.T0 / 2.0, p.eta_cal / 0.20, p.c_bar / 0.20, p.H_ratio]
        + [p.lambda_4 / 0.01, p.eta_1 / 3.0, p.eta_2 / 3.0, p.beta_R / 10.0]
    )
    return np.array(vec, dtype=np.float32)


def encode_input_grid(theta, mu_grid: dict, tau, T, p: Params) -> np.ndarray:
    """Igual que encode_input pero mu_grid[t] es un array (101,101) — un mu distinto por
    punto de la malla (alpha,gamma). Devuelve (10201, 34): una fila por punto de malla."""
    n = mu_grid[TIPOS[0]].size
    zone_oh = [1.0 if p.cov_zone == z else 0.0 for z in ZONES]
    wealth_oh = [1.0 if p.cov_wealth == "Standard" else 0.0, 1.0 if p.cov_wealth == "High" else 0.0]
    vict_oh = [1.0 if p.cov_vict == "Private" else 0.0, 1.0 if p.cov_vict == "Public sector" else 0.0]
    state_oh = [1.0 if p.cov_state == "Strict" else 0.0, 1.0 if p.cov_state == "Lax" else 0.0]
    theta_oh = [1.0 if theta == t else 0.0 for t in TIPOS]
    beta_tilde_vec = [p.beta_tilde[t] for t in TIPOS]
    static_vec = (
        [tau / T] + theta_oh + [p.R_millions / 100.0] + beta_tilde_vec
        + zone_oh + wealth_oh + vict_oh + state_oh
        + [p.T_mad / 30.0, p.T0 / 2.0, p.eta_cal / 0.20, p.c_bar / 0.20, p.H_ratio]
        + [p.lambda_4 / 0.01, p.eta_1 / 3.0, p.eta_2 / 3.0, p.beta_R / 10.0]
    )
    mu_cols = np.stack([mu_grid[t].reshape(-1) for t in TIPOS], axis=1)  # (n, 4)
    static_cols = np.tile(np.array(static_vec, dtype=np.float32), (n, 1))  # (n, 30)
    return np.concatenate([mu_cols, static_cols], axis=1).astype(np.float32)  # (n, 34)


def solve_state_problem(
    mu: dict, tau: int, T: int, p: Params, v_next_fn, tipo_real: str, theta_f: str,
    p_rescue_prev: float = 0.5, p_nego_prev: float = 0.5,
):
    """Oraculo: grid search 101x101 restringido por Gamma_tau(mu), ambas ramas.
    p_rescue_prev/p_nego_prev: mezcla de p_cap (eq:p-cap) que representa la accion del
    Estado -- protocolo de consistencia temporal: en produccion (app_DL.py), el llamador
    pasa el valor DEGENERADO real de tau-1 (act_s: (1,0) si Rescue, (0,1) si Negotiate);
    en entrenamiento (trayectorias sinteticas, sin registro real de un periodo anterior) se
    deja el default neutro (0.5,0.5)."""
    mt = m_t(tau, p)
    p_rescue, p_nego = p_rescue_prev, p_nego_prev
    p_coop, p_col = 0.5, 0.5

    probs_by_type, pdet_by_type = {}, {}
    for th in TIPOS:
        probs_by_type[th], pdet_by_type[th] = outcome_probs_grid(GA, GG, th, p, mt)

    # Delta H_tau(alpha,gamma)
    mu_arr = {th: mu[th] for th in TIPOS}
    H0 = entropy(mu_arr)
    delta_h_acc = np.zeros_like(GA)
    for m_key in ["Cont", "1", "2", "3", "4"]:
        for d in (0, 1):
            w = {th: mu_arr[th] * probs_by_type[th][m_key] * (pdet_by_type[th] if d == 1 else (1 - pdet_by_type[th])) for th in TIPOS}
            z = sum(w.values())
            z_safe = np.where(z > 1e-15, z, 1.0)
            mu2 = {th: w[th] / z_safe for th in TIPOS}
            h = np.zeros_like(GA)
            for th in TIPOS:
                pth = mu2[th]
                h -= np.where(pth > 1e-12, pth * np.log(np.where(pth > 1e-12, pth, 1.0)), 0.0)
            delta_h_acc += z * h
    delta_H = H0 - delta_h_acc

    p_cap_by_type = {th: p_cap_eff_grid(GA, GG, th, p, p_rescue, p_nego) for th in TIPOS}
    p_pay_by_type = {th: p_pay_eff_grid(GA, GG, th, p, pdet_by_type[th], mt, p_coop, p_col) for th in TIPOS}

    # V_next esperado via las 10 ramas (m,d) -> mu_{tau+1}, evaluado con la red (o 0 terminal)
    v_next_acc = {th: np.zeros_like(GA) for th in TIPOS}
    if v_next_fn is not None:
        for m_key in ["Cont", "1", "2", "3", "4"]:
            for d in (0, 1):
                w = {th: mu_arr[th] * probs_by_type[th][m_key] * (pdet_by_type[th] if d == 1 else (1 - pdet_by_type[th])) for th in TIPOS}
                z = sum(w.values())
                z_safe = np.where(z > 1e-15, z, 1.0)
                mu2 = {th: w[th] / z_safe for th in TIPOS}
                v_pred = v_next_fn(mu2, tau + 1, p)  # dict theta -> array(101,101)
                for th in TIPOS:
                    v_next_acc[th] += z * v_pred[th]

    U_rel = {th: -KAPPA_REL[th] for th in TIPOS}
    U_kill = {th: (1 - p_cap_by_type[th]) * ETA_REP[th] - p_cap_by_type[th] * F_CAP[th] for th in TIPOS}
    C_tau = {th: PHI_COST[th] * np.exp(KAPPA_C[th] * GG) + NU_COST[th] for th in TIPOS}
    V_cont = {
        th: (
            p_pay_by_type[th] * p.R_millions * (1.0 - GA) - C_tau[th] - p_cap_by_type[th] * F_CAP[th]
            + p.beta_tilde[th] * (1.0 - p_cap_by_type[th]) * v_next_acc[th]
        )
        for th in TIPOS
    }
    best_k = {th: np.maximum(np.maximum(U_rel[th], U_kill[th]), V_cont[th]) for th in TIPOS}
    stacked = {th: np.stack([np.full_like(GA, U_rel[th]), U_kill[th], V_cont[th]]) for th in TIPOS}
    branch_idx = {th: np.argmax(stacked[th], axis=0) for th in TIPOS}  # 0=rel,1=kill,2=cont

    ic_gain_min = np.full_like(GA, np.inf)
    for th_j in TIPOS:
        gain_j = np.zeros_like(GA)
        for th_i in TIPOS:
            u_i_under_j = np.select(
                [branch_idx[th_j] == 0, branch_idx[th_j] == 1, branch_idx[th_j] == 2],
                [np.full_like(GA, U_rel[th_i]), U_kill[th_i], V_cont[th_i]],
            )
            gain_j += mu_arr[th_i] * (best_k[th_i] - u_i_under_j)
        ic_gain_min = np.minimum(ic_gain_min, gain_j)
    ic_k_grid = ic_gain_min >= -1e-9

    ir_k_gap = sum(mu_arr[th] * (U_rel[th] - np.maximum(V_cont[th], U_kill[th])) for th in TIPOS)
    ir_k_grid = ir_k_gap >= -1e-9
    ir_k_true_gap_grid = U_rel[tipo_real] - np.maximum(V_cont[tipo_real], U_kill[tipo_real])

    # IR^F: formula EXACTA del paper (eq:f-coop/eq:f-col/eq:ir-family), reemplaza la
    # simplificacion ad-hoc anterior (p_pay + VARRHO_COLLUDE, sin base en el paper). p_surv_tau
    # y p_rel_tau (celda "4", liberacion exogena) no dependen de la mezcla de acciones del
    # Estado (Q_tau^{Coop}/Q_tau^{Col} son esperanzas triviales sobre formulas que no usan la
    # accion realizada) -- unicamente dependen de (alpha,gamma,mu_tau,M_tau); no hay nada que
    # "congelar" desde tau=0.
    _theta_hat_arr = max(mu_arr, key=mu_arr.get)
    _iota_arr = max(mu_arr.values())
    p_surv_mu = sum(mu_arr[th] * p_surv_raw(th, _theta_hat_arr, p, _iota_arr) for th in TIPOS)
    p_rel_mu = sum(mu_arr[th] * probs_by_type[th]["4"] for th in TIPOS)
    p_det_mu = sum(mu_arr[th] * pdet_by_type[th] for th in TIPOS)
    e_tau_f = PHI_F[theta_f] * np.exp(KAPPA_F[theta_f] * GG) + NU_F[theta_f]
    U_coop_f = p_surv_mu * V_L_FAMILY - e_tau_f
    U_col_f = p_rel_mu * V_L_FAMILY - p.R_millions - p_det_mu * F_COL
    ir_f_grid = U_coop_f >= U_col_f

    feasible = ic_k_grid & ir_k_grid & ir_f_grid

    # 1. Calculate per-type benchmarks for the deviation premium (anchor)
    benchmarks = {}
    for th in TIPOS:
        # V_R benchmark for type th under information certainty
        p_surv_th = p_surv_raw(th, th, p)
        V_R_th = OMEGA_K * (1.0 - p_surv_th) + c_ops_grid(GG, GA, th)
        
        # V_N benchmark for type th under information certainty
        probs_th, pdet_th = outcome_probs_grid(GA, GG, th, p, mt)
        V_N_th = OMEGA_P * p.R_millions * (1.0 - GA) + OMEGA_K * probs_th["2"] + c_maint_grid(GG, GA, th)
        
        # Secuestrador IR^K under degenerate belief mu_th = 1.0
        U_rel_th = -KAPPA_REL[th]
        p_cap_th = p_cap_eff_grid(GA, GG, th, p, 0.5, 0.5)
        U_kill_th = (1 - p_cap_th) * ETA_REP[th] - p_cap_th * F_CAP[th]
        C_tau_th = PHI_COST[th] * np.exp(KAPPA_C[th] * GG) + NU_COST[th]
        p_pay_th = p_pay_eff_grid(GA, GG, th, p, pdet_th, mt, 0.5, 0.5)
        
        if v_next_fn is not None:
            mu_deg = {th_temp: np.ones_like(GA) if th_temp == th else np.zeros_like(GA) for th_temp in TIPOS}
            v_next_deg = v_next_fn(mu_deg, tau + 1, p)
            V_cont_next_th = v_next_deg[th]
        else:
            V_cont_next_th = np.zeros_like(GA)
            
        V_cont_th = (
            p_pay_th * p.R_millions * (1.0 - GA) - C_tau_th - p_cap_th * F_CAP[th]
            + p.beta_tilde[th] * (1.0 - p_cap_th) * V_cont_next_th
        )
        ir_k_feasible_th = (U_rel_th - np.maximum(V_cont_th, U_kill_th)) >= -1e-9
        
        # Familia IR^F under degenerate belief mu_th = 1.0
        _wealth_key = "High wealth" if p.cov_wealth == "High" else "Low wealth"
        e_tau_f_th = PHI_F[_wealth_key] * np.exp(KAPPA_F[_wealth_key] * GG) + NU_F[_wealth_key]
        U_coop_f_th = p_surv_th * V_L_FAMILY - e_tau_f_th
        U_col_f_th = probs_th["4"] * V_L_FAMILY - p.R_millions - pdet_th * F_COL
        ir_f_feasible_th = U_coop_f_th >= U_col_f_th
        
        feasible_th = ir_k_feasible_th & ir_f_feasible_th
        
        if np.any(feasible_th):
            V_R_masked_th = np.where(feasible_th, V_R_th, np.inf)
            V_N_masked_th = np.where(feasible_th, V_N_th, np.inf)
            idx_R_th = np.unravel_index(np.argmin(V_R_masked_th), V_R_masked_th.shape)
            idx_N_th = np.unravel_index(np.argmin(V_N_masked_th), V_N_masked_th.shape)
        else:
            idx_R_th = np.unravel_index(np.argmin(V_R_th), V_R_th.shape)
            idx_N_th = np.unravel_index(np.argmin(V_N_th), V_N_th.shape)
            
        benchmarks[th] = {
            "alpha_R": float(GA[idx_R_th]),
            "gamma_R": float(GG[idx_R_th]),
            "alpha_N": float(GA[idx_N_th]),
            "gamma_N": float(GG[idx_N_th]),
        }

    # 2. Calculate belief-weighted averages of benchmarks (anchors)
    alpha_R_mu = sum(mu_arr[th] * benchmarks[th]["alpha_R"] for th in TIPOS)
    gamma_R_mu = sum(mu_arr[th] * benchmarks[th]["gamma_R"] for th in TIPOS)
    alpha_N_mu = sum(mu_arr[th] * benchmarks[th]["alpha_N"] for th in TIPOS)
    gamma_N_mu = sum(mu_arr[th] * benchmarks[th]["gamma_N"] for th in TIPOS)

    # 3. Solve objective functions with the correct deviation premium
    p_surv_const = sum(mu_arr[th] * (1.0 - p_surv_raw(th, _theta_hat_arr, p, _iota_arr)) for th in TIPOS)
    C_ops_mix = sum(mu_arr[th] * c_ops_grid(GG, GA, th) for th in TIPOS)
    
    # Rescue with dynamic belief-weighted penalty
    V_R = OMEGA_K * p_surv_const + C_ops_mix + CHI_ALPHA * (GA - alpha_R_mu) ** 2 + CHI_GAMMA * (GG - gamma_R_mu) ** 2 - PSI_H * delta_H
    
    # Negotiation with dynamic belief-weighted penalty
    C_maint_mix = sum(mu_arr[th] * c_maint_grid(GG, GA, th) for th in TIPOS)
    h2_mix = sum(mu_arr[th] * probs_by_type[th]["2"] for th in TIPOS)
    V_N = OMEGA_P * p.R_millions * (1.0 - GA) + OMEGA_K * h2_mix + C_maint_mix + CHI_ALPHA * (GA - alpha_N_mu) ** 2 + CHI_GAMMA * (GG - gamma_N_mu) ** 2 - PSI_H * delta_H

    V_R_masked = np.where(feasible, V_R, np.inf)
    V_N_masked = np.where(feasible, V_N, np.inf)
    idx_r = np.unravel_index(np.argmin(V_R_masked), V_R_masked.shape)
    idx_n = np.unravel_index(np.argmin(V_N_masked), V_N_masked.shape)
    floor_r, floor_n = V_R_masked[idx_r], V_N_masked[idx_n]
    any_feasible = bool(np.any(feasible))

    if not any_feasible:
        # Fallback to the unconstrained minimums of V_R and V_N instead of arbitrary (0.5, 0.5)
        idx_r_unconstrained = np.unravel_index(np.argmin(V_R), V_R.shape)
        idx_n_unconstrained = np.unravel_index(np.argmin(V_N), V_N.shape)
        floor_r_unconstrained = V_R[idx_r_unconstrained]
        floor_n_unconstrained = V_N[idx_n_unconstrained]
        
        if floor_r_unconstrained <= floor_n_unconstrained:
            a_star = float(GA[idx_r_unconstrained])
            g_star = float(GG[idx_r_unconstrained])
            a_S = "Rescue"
            idx_sel = idx_r_unconstrained
        else:
            a_star = float(GA[idx_n_unconstrained])
            g_star = float(GG[idx_n_unconstrained])
            a_S = "Negotiate"
            idx_sel = idx_n_unconstrained
            
        v_tau_by_type = {th: float(best_k[th][idx_sel]) for th in TIPOS}
        extra = {
            "H_mu": H0,
            "delta_H": float(delta_H[idx_sel]),
            "ir_k_true_gap": float(ir_k_true_gap_grid[idx_sel]),
            "floor_selected": float(floor_r_unconstrained if a_S == "Rescue" else floor_n_unconstrained),
            "benchmarks_by_type": benchmarks,
            "alpha_R_mu": alpha_R_mu,
            "gamma_R_mu": gamma_R_mu,
            "alpha_N_mu": alpha_N_mu,
            "gamma_N_mu": gamma_N_mu,
        }
        return a_star, g_star, a_S, v_tau_by_type, False, extra

    if floor_r <= floor_n:
        a_star, g_star, a_S = float(GA[idx_r]), float(GG[idx_r]), "Rescue"
        idx_sel = idx_r
    else:
        a_star, g_star, a_S = float(GA[idx_n]), float(GG[idx_n]), "Negotiate"
        idx_sel = idx_n

    v_tau_by_type = {th: float(best_k[th][idx_sel]) for th in TIPOS}
    extra = {
        "H_mu": H0,
        "delta_H": float(delta_H[idx_sel]),
        "ir_k_true_gap": float(ir_k_true_gap_grid[idx_sel]),
        "floor_selected": float(floor_r if a_S == "Rescue" else floor_n),
        "benchmarks_by_type": benchmarks,
        "alpha_R_mu": alpha_R_mu,
        "gamma_R_mu": gamma_R_mu,
        "alpha_N_mu": alpha_N_mu,
        "gamma_N_mu": gamma_N_mu,
    }
    return a_star, g_star, a_S, v_tau_by_type, True, extra


def simulate_trajectories(N: int, T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(N):
        p = sample_params(rng)
        mu = rng.dirichlet(np.ones(4))
        mu = {th: float(mu[i]) for i, th in enumerate(TIPOS)}
        path = [mu]
        for tau in range(1, T):
            mt = m_t(tau, p)
            probs_by_type, pdet_by_type = {}, {}
            a_ref, g_ref = 0.4, 0.4
            for th in TIPOS:
                probs_by_type[th], pdet_by_type[th] = outcome_probs_grid(a_ref, g_ref, th, p, mt)
            m_keys = ["Cont", "1", "2", "3", "4"]
            m_probs = [sum(mu[th] * probs_by_type[th][mk] for th in TIPOS) for mk in m_keys]
            m_probs = np.array(m_probs) / sum(m_probs)
            m_draw = rng.choice(m_keys, p=m_probs)
            d_prob = sum(mu[th] * pdet_by_type[th] for th in TIPOS)
            d_draw = 1 if rng.random() < d_prob else 0
            w = {th: mu[th] * probs_by_type[th][m_draw] * (pdet_by_type[th] if d_draw == 1 else (1 - pdet_by_type[th])) for th in TIPOS}
            z = sum(w.values())
            mu = {th: (w[th] / z if z > 1e-15 else 0.25) for th in TIPOS}
            path.append(mu)
        trajs.append((p, path))
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--out", type=str, default="captor_value_net_T10.pt")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    T, N = args.T, args.N
    print(f"[{time.strftime('%H:%M:%S')}] Simulando {N} trayectorias Monte Carlo, T={T}...", flush=True)
    t0 = time.time()
    trajs = simulate_trajectories(N, T, seed=42)
    print(f"[{time.strftime('%H:%M:%S')}] Simulacion lista en {time.time()-t0:.1f}s", flush=True)

    net = CaptorValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    device = "cpu"

    def v_next_fn_factory(net, T):
        def fn(mu2: dict, tau_next: int, p: Params):
            if tau_next > T:
                return {th: np.zeros_like(GA) for th in TIPOS}
            out = {}
            with torch.no_grad():
                for th in TIPOS:
                    x = encode_input_grid(th, mu2, tau_next, T, p)  # (10201, 34)
                    v = net(torch.from_numpy(x)).numpy()  # (10201,)
                    out[th] = v.reshape(GA.shape)
            return out
        return fn

    loss_history = []
    for tau in range(T, 0, -1):
        t_level0 = time.time()
        v_next_fn = None if tau == T else v_next_fn_factory(net, T)
        X_list, y_list = [], []
        oracle_time = 0.0
        for i, (p, path) in enumerate(trajs):
            mu_tau = path[tau - 1]
            ot0 = time.time()
            tipo_real = max(mu_tau, key=mu_tau.get)
            theta_f = "High wealth" if p.cov_wealth == "High" else "Low wealth"
            _, _, _, v_by_type, _, _ = solve_state_problem(mu_tau, tau, T, p, v_next_fn, tipo_real, theta_f)
            oracle_time += time.time() - ot0
            for th in TIPOS:
                X_list.append(encode_input(th, mu_tau, tau, T, p))
                y_list.append(v_by_type[th])
            if (i + 1) % 500 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] tau={tau}: {i+1}/{N} puntos, oraculo acumulado {oracle_time:.1f}s", flush=True)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        for epoch in range(args.epochs):
            opt.zero_grad()
            pred = net(X)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
        loss_history.append({"tau": tau, "final_loss": float(loss.item()), "n_points": len(X_list),
                              "oracle_seconds": oracle_time, "level_seconds": time.time() - t_level0})
        print(f"[{time.strftime('%H:%M:%S')}] tau={tau} DONE: loss={loss.item():.6f}, "
              f"oraculo={oracle_time:.1f}s, nivel_total={time.time()-t_level0:.1f}s", flush=True)

    torch.save({"state_dict": net.state_dict(), "T": T, "N": N,
                "loss_history": loss_history}, args.out)
    with open(args.out + ".json", "w") as f:
        json.dump({"T": T, "N": N, "loss_history": loss_history, "total_seconds": time.time() - t0}, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Entrenamiento completo. Guardado en {args.out}. Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
