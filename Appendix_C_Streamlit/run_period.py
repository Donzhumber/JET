"""Multi-period simulation engine (tau=1 -> T_max), generalizing the single-period
(tau=1) logic already live in app_DL.py to run tau=2,3,...,T_max in a loop.

Design choices (extensions APPROVED by the user, no formula changed vs. the already-
validated tau=1 code -- only WHICH tau each formula is evaluated at, and how the
trained networks are queried beyond their T=10 training horizon):

1. Per-player MDG temperatures (MDG_MULT_STATE/FAMILY/CAPTOR) and the literal eq:m-t
   (real H(mu_tau)/H(mu_0), not the H_ratio proxy) are the SAME formulas already used
   for tau=1 in app_DL.py -- duplicated here (not imported from app_DL.py, which runs
   Streamlit UI code at import time) so this module can run standalone / be imported
   by a script.
2. v_next queries beyond the trained horizon (tau_next > T=10): CLAMPED to tau_next=T
   (the last trained level), evaluated with the REAL/current mu of that period -- not
   the "return 0.0" terminal convention used inside training. Approved by the user:
   "siempre hacerlo en cada tau con T=10 ... sin importar el tau en que se encuentre".
   Only touches how the ALREADY-TRAINED networks are queried; nothing is retrained.
3. mu_{tau+1} (eq:bayes-update) uses the GENERIC cvn.outcome_probs_grid/p_cap_eff_grid
   (tau-parameterized) for the likelihood terms -- NOT the tau=0-specific Tab1-slider
   closures (_outcome_probs_tab3 etc. in app_DL.py, which read session_state["act_f"]
   fixed at tau=0 and are only valid for the tau=0->1 transition).
4. d_tau (detection) draw: NEW, generalizes Tab 1 Block D's tau=0-only "Draw Detection
   Signal" button to every period (needed so mu can recurse from tau to tau+1).
5. Stopping time (eq:stopping-time, Bernal_H.tex): if m_tau != Cont, the episode closes
   at that tau. The caller decides whether to keep looping past that point in
   "counterfactual absorbing-state" mode (same convention the paper's own calibration
   figures use) -- this module just reports `m_closes_episode`, it does not stop itself.
"""
import numpy as np
import torch

import train_captor_value_net as cvn
import train_captor_true_type_net as cttn
import family_optimization as famopt

try:
    from rational_behavior import (
        draw_voice_indicator,
        sample_voice_observation,
        communication_likelihood_LC,
    )
    _RB_AVAILABLE = True
except Exception:
    _RB_AVAILABLE = False

T_TRAINED = 10  # horizon both .pt networks were actually trained on

# Same multipliers as app_DL.py's MDG_MULT_STATE/FAMILY/CAPTOR (extension approved,
# NOT literal in Bernal_H.tex/Working_paper_eng.tex -- see app_DL.py's own comment).
# T0 (the "T0 (Initial)" base value) moved OUT of these dicts and became a direct,
# UI-adjustable per-player argument (T0_S/T0_F/T0_K below, Tab 1 Block A) -- approved by
# the user, replacing the old "shared p.T0 * multiplier" scheme with 3 independent base
# values. c_bar RAISED (0.5/1.0/1.5 -> 14.0/18.0/22.0) so the long-run noise floor stays
# meaningfully dispersed even at large tau (previously it decayed back to ~deterministic
# by tau~20 regardless of T0, since the floor term p.c_bar*mult was too small).
MDG_MULT_STATE = {"eta_cal": 1.2, "c_bar": 14.0}
MDG_MULT_FAMILY = {"eta_cal": 1.0, "c_bar": 18.0}
MDG_MULT_CAPTOR = {"eta_cal": 0.8, "c_bar": 22.0}
# Default per-player T0 (same values used for tau=0 in app_DL.py Block A's new sliders).
T0_S_DEFAULT = 0.90
T0_F_DEFAULT = 1.30
T0_K_DEFAULT = 1.80


def _mdg_temp_player(T0_i: float, mult: dict, h_ratio_real: float, p, tau: int) -> float:
    """Player-specific literal T_t (eq:m-t), same functional form as the generic
    M_tau, using T0_i (direct per-player base, Tab 1 Block A slider) and mult['eta_cal']/
    ['c_bar'] on top of the shared p.eta_cal/p.c_bar -- see app_DL.py._mdg_temp_player.
    tau enters the exp(-eta_cal*mult*tau) term (generalized from the tau=1-only version)."""
    return float(T0_i * max(
        h_ratio_real * np.exp(-p.eta_cal * mult["eta_cal"] * tau), p.c_bar * mult["c_bar"]
    ))


def make_v_next_fn_grid_extended(net, T_trained: int = T_TRAINED):
    """Grid-shaped v_next query, CLAMPED to tau_next<=T_trained (see module docstring
    point 2) -- replaces the 'return 0.0 beyond T' convention used during training."""
    def fn(mu2: dict, tau_next: int, p):
        tau_query = min(tau_next, T_trained)
        out = {}
        with torch.no_grad():
            for th in cvn.TIPOS:
                x = cvn.encode_input_grid(th, mu2, tau_query, T_trained, p)
                v = net(torch.from_numpy(x)).numpy()
                out[th] = v.reshape(cvn.GA.shape)
        return out
    return fn


def make_v_next_fn_scalar_extended(net, T_trained: int = T_TRAINED):
    """Scalar v_next query, CLAMPED to tau_next<=T_trained -- same idea, for the
    Captor's own true-type continuation (solve_captor_true_type_continuation)."""
    def fn(theta: str, mu: dict, tau_next: int, p):
        tau_query = min(tau_next, T_trained)
        x = cvn.encode_input(theta, mu, tau_query, T_trained, p)
        with torch.no_grad():
            v = net(torch.from_numpy(x[None, :])).numpy()
        return float(v[0])
    return fn


def entropy(mu: dict) -> float:
    return float(-sum(v * np.log(v) for v in mu.values() if v > 1e-12))


# Detection-sensitivity coefficient in the kappa_h index formulas (zeta_d), SAME value as
# app_DL.py Block D's `zd` slider default and as cvn.outcome_probs_grid's hardcoded 0.18.
ZD_DETECTION_KH = 0.18
# kappa_h's own p_det term uses FIXED eta_1=eta_2=1.0 (app_DL.py's ETA_1_PDET/ETA_2_PDET
# module constants), NOT the Tab-1 eta_1/eta_2 sliders (p.eta_1/p.eta_2) used everywhere
# else (outcome_probs_grid, draw_d, etc). This is how the already-validated tau=1 kappa_h
# code (app_DL.py:2124) is written -- replicated as-is, not "fixed", since it wasn't part
# of the phi_F correction the user approved. Only diverges from p.eta_1/p.eta_2 if those
# sliders are moved away from their default value of 1.0.
_ETA_1_PDET_KH = 1.0
_ETA_2_PDET_KH = 1.0


def compute_benchmarks_tau(theta: str, p, mt_tau: float, v_next_grid=None, tau: int = 1) -> dict:
    """Per-type benchmarks (gamma_R, alpha_R, gamma_N, alpha_N) at period tau,
    RESTRICTED to the feasibility set Gamma_t(mu_theta) under degenerate belief mu_theta = 1.0.
    """
    p_surv = cvn.p_surv_raw(theta, theta, p)
    V_R = cvn.OMEGA_K * (1.0 - p_surv) + cvn.c_ops_grid(cvn.GG, cvn.GA, theta)

    probs, pdet = cvn.outcome_probs_grid(cvn.GA, cvn.GG, theta, p, mt_tau)
    h2 = probs["2"]
    V_N = (
        cvn.OMEGA_P * p.R_millions * (1.0 - cvn.GA)
        + cvn.OMEGA_K * h2
        + cvn.c_maint_grid(cvn.GG, cvn.GA, theta)
    )

    # Evaluate Feasibility under degenerate belief mu_theta = 1.0 (for this type)
    # Restricción del Secuestrador (IR^K)
    U_rel = -cvn.KAPPA_REL[theta]
    p_cap = cvn.p_cap_eff_grid(cvn.GA, cvn.GG, theta, p, 0.5, 0.5)
    U_kill = (1 - p_cap) * cvn.ETA_REP[theta] - p_cap * cvn.F_CAP[theta]
    C_tau = cvn.PHI_COST[theta] * np.exp(cvn.KAPPA_C[theta] * cvn.GG) + cvn.NU_COST[theta]
    p_pay = cvn.p_pay_eff_grid(cvn.GA, cvn.GG, theta, p, pdet, mt_tau, 0.5, 0.5)
    
    if v_next_grid is not None:
        mu_degenerate = {th: np.ones_like(cvn.GA) if th == theta else np.zeros_like(cvn.GA) for th in cvn.TIPOS}
        v_next_acc = v_next_grid(mu_degenerate, tau + 1, p)
        V_cont_next = v_next_acc[theta]
    else:
        V_cont_next = np.zeros_like(cvn.GA)

    V_cont = (
        p_pay * p.R_millions * (1.0 - cvn.GA) - C_tau - p_cap * cvn.F_CAP[theta]
        + p.beta_tilde[theta] * (1.0 - p_cap) * V_cont_next
    )
    
    ir_k_feasible = (U_rel - np.maximum(V_cont, U_kill)) >= -1e-9

    # Restricción de la Familia (IR^F)
    _wealth_key = "High wealth" if p.cov_wealth == "High" else "Low wealth"
    e_tau_f = cvn.PHI_F[_wealth_key] * np.exp(cvn.KAPPA_F[_wealth_key] * cvn.GG) + cvn.NU_F[_wealth_key]
    U_coop_f = p_surv * cvn.V_L_FAMILY - e_tau_f
    U_col_f = probs["4"] * cvn.V_L_FAMILY - p.R_millions - pdet * cvn.F_COL
    ir_f_feasible = U_coop_f >= U_col_f

    # Feasible mask
    feasible_mask = ir_k_feasible & ir_f_feasible

    # Find restricted argmin
    if np.any(feasible_mask):
        V_R_masked = np.where(feasible_mask, V_R, np.inf)
        V_N_masked = np.where(feasible_mask, V_N, np.inf)
        idx_R = np.unravel_index(np.argmin(V_R_masked), V_R_masked.shape)
        idx_N = np.unravel_index(np.argmin(V_N_masked), V_N_masked.shape)
    else:
        # Fallback to unconstrained
        idx_R = np.unravel_index(np.argmin(V_R), V_R.shape)
        idx_N = np.unravel_index(np.argmin(V_N), V_N.shape)

    return {
        "gamma_R": float(cvn.GG[idx_R]), "alpha_R": float(cvn.GA[idx_R]),
        "gamma_N": float(cvn.GG[idx_N]), "alpha_N": float(cvn.GA[idx_N]),
    }


def compute_belief_weighted(benchmarks_by_type: dict, mu_tau: dict) -> dict:
    """alpha_R^mu, gamma_R^mu, alpha_N^mu, gamma_N^mu: mu_tau-weighted sums of the per-type
    benchmarks above -- same formula as app_DL.py's tau=1 alpha_R_mu/gamma_R_mu/etc rows,
    generalized by using THIS period's mu_tau instead of mu_1."""
    return {
        "alpha_R_mu": float(sum(mu_tau[th] * benchmarks_by_type[th]["alpha_R"] for th in mu_tau)),
        "gamma_R_mu": float(sum(mu_tau[th] * benchmarks_by_type[th]["gamma_R"] for th in mu_tau)),
        "alpha_N_mu": float(sum(mu_tau[th] * benchmarks_by_type[th]["alpha_N"] for th in mu_tau)),
        "gamma_N_mu": float(sum(mu_tau[th] * benchmarks_by_type[th]["gamma_N"] for th in mu_tau)),
    }


def compute_kappa_h_tau(theta: str, alpha_star: float, gamma_star: float, mt_tau: float,
                         act_f: str, act_k: str, act_s: str, p) -> float:
    """kappa_h(theta,tau) (`eq:kappa-c`, Bernal_H.tex/Working_paper_eng.tex -- explicitly
    t-indexed and traced across the full simulated horizon in the papers' own calibration
    figures, not a tau=0/1-only construct). SAME formula as app_DL.py's `_lam123_tau1`
    closure (lines 2105-2143), generalized to any tau via (alpha_star,gamma_star,mt_tau,
    act_f/k/s) of THAT period -- consistent with C_t(theta_K)'s own definition in both
    papers, which conditions on the CONTEMPORANEOUS realized actions of period t (not
    lagged; see eq:hj / def:desenlace-fisico-mt: "the material state ... tilde a_t^F,
    tilde a_t^K, tilde a_t^S ... feeds a block of competing risks that determines m_t").

    CORRECTED vs. the original tau=1 code (approved by the user after checking the papers):
    the phi_F indicator now checks act_f=='Cooperate' (the real tau>=1 Family action-space
    label) instead of the stale =='Pay'' check (only valid for tau=0's exogenous Block-C/D
    simulator), which silently never fired for tau>=1 -- phi_F was always 0.00. This fix is
    ALSO applied retroactively to app_DL.py's own tau=1 inline block, not just here, so both
    stay consistent."""
    beta_K_1 = cvn.BETAS_K[theta]["Pago"]
    beta_K_2 = cvn.BETAS_K[theta]["Muerte"]
    beta_K_3 = cvn.BETAS_K[theta]["Rescate"]
    beta_z = cvn.BETAS_Z[p.cov_zone]
    beta_F_1 = 0.80 if p.cov_wealth == "High" else 0.00
    beta_V_1 = 1.36 if p.cov_vict == "Public sector" else 0.00
    beta_S = 0.50 if p.cov_state == "Lax" else 0.00
    phi_F_1 = 3.20 if act_f == "Cooperate" else 0.00
    phi_K_1 = -1.15 if act_k == "Continue" else 0.00
    phi_F_2 = -1.50 if act_f == "Cooperate" else 0.00
    phi_K_kill_2 = 4.00 if act_k == "Kill" else 0.00
    phi_K_cont_2 = 0.50 if act_k == "Continue" else 0.00
    zeta_R_3 = 2.50 if act_s == "Rescue" else 0.00
    phi_F_3 = -1.00 if act_f == "Cooperate" else 0.00
    phi_K_3 = 0.50 if act_k == "Continue" else 0.00

    za = cvn.ZETAS_POLITICA[theta]["za"]
    zg = cvn.ZETAS_POLITICA[theta]["zg"]
    eta_0_t = cvn.ETA_0_PDET[theta]
    u_det_t = eta_0_t + _ETA_1_PDET_KH * alpha_star + _ETA_2_PDET_KH * gamma_star
    p_det_t = 1.0 / (1.0 + np.exp(-u_det_t))

    idx_pago = (
        beta_K_1 + beta_z + beta_F_1 - beta_V_1 + beta_S
        - za * alpha_star - zg * gamma_star - ZD_DETECTION_KH * p_det_t + phi_F_1 + phi_K_1
    )
    idx_muerte = (
        beta_K_2 + beta_z + beta_S
        + za * alpha_star + zg * gamma_star - ZD_DETECTION_KH * p_det_t - phi_F_2 + phi_K_kill_2 + phi_K_cont_2
    )
    idx_rescate = (
        -beta_S + beta_K_3 + beta_z
        + za * alpha_star + zg * gamma_star + ZD_DETECTION_KH * p_det_t + zeta_R_3 - phi_F_3 + phi_K_3
    )

    lam_pay = mt_tau * cvn.LAMBDAS_0["Pago"] * np.exp(idx_pago)
    lam_kill = mt_tau * cvn.LAMBDAS_0["Muerte"] * np.exp(idx_muerte)
    lam_res = mt_tau * cvn.LAMBDAS_0["Rescate"] * np.exp(idx_rescate)

    return float(zg * (lam_kill + lam_res - lam_pay))


def draw_m(alpha_star: float, gamma_star: float, tipo_real: str, p, mt: float, rng: np.random.Generator):
    """Physical-outcome draw (eq:LH-compacta ~ eq:hj/eq:pCont/eq:xi), same mechanism
    already used for tau=1's `m` row in app_DL.py, generalized to any (alpha,gamma,mt)."""
    probs, _pdet = cvn.outcome_probs_grid(alpha_star, gamma_star, tipo_real, p, mt)
    v = float(rng.random())
    b_cont = probs["Cont"]
    b1 = b_cont + probs["1"]
    b2 = b1 + probs["2"]
    b3 = b2 + probs["3"]
    if v <= b_cont:
        key = "Cont"
    elif v <= b1:
        key = "1"
    elif v <= b2:
        key = "2"
    elif v <= b3:
        key = "3"
    else:
        key = "4"
    label_map = {
        "Cont": "Continue Captivity (cont)", "1": "Ransom Paid (j=1)",
        "2": "Victim Deceased (j=2)", "3": "Tactical Rescue (j=3)", "4": "Exogenous Release (j=4)",
    }
    return {
        "v": v, "outcome_key": key, "outcome": label_map[key], "probs": dict(probs),
        "closes_episode": key != "Cont",
    }


def draw_d(alpha_star: float, gamma_star: float, tipo_real: str, p, rng: np.random.Generator):
    """Detection draw (eq:detection), NEW: generalizes Tab 1 Block D's tau=0-only
    button to every period tau -- needed so mu can recurse tau -> tau+1."""
    eta0 = cvn.ETA_0_PDET[tipo_real]
    u_det = eta0 + p.eta_1 * alpha_star + p.eta_2 * gamma_star
    p_det = float(1.0 / (1.0 + np.exp(-u_det)))
    u = float(rng.random())
    d = 1 if u <= p_det else 0
    return {"u": u, "d": d, "p_det": p_det}


def bayes_update_mu(mu_tau: dict, alpha_star: float, gamma_star: float, mt: float, p,
                     m_key: str, d_realized: int, act_k_realized: str,
                     voice_L_C_by_type: dict) -> dict:
    """mu_{tau+1}(theta) ~ mu_tau(theta) * L_I,K(theta) * Pr(m|theta) * Pr(d|theta) *
    L_C(theta) -- eq:bayes-update, GENERIC version (cvn.outcome_probs_grid/p_cap_eff_grid,
    tau-parameterized), not the tau=0-specific Tab1-slider closures in app_DL.py."""
    branch_to_action = {"rel": "Release", "kill": "Kill", "cont": "Continue"}
    w = {}
    for th in cvn.TIPOS:
        probs_th, pdet_th = cvn.outcome_probs_grid(alpha_star, gamma_star, th, p, mt)
        pr_m = probs_th[m_key]
        pr_d = pdet_th if d_realized == 1 else (1.0 - pdet_th)

        # L_I,K(theta): implementation likelihood of the REALIZED act_k under theta's
        # OWN rational branch (same 3-way logit as the Captor's own tilde a_K draw).
        U_rel = -cvn.KAPPA_REL[th]
        p_cap_th = float(cvn.p_cap_eff_grid(alpha_star, gamma_star, th, p, 0.5, 0.5))
        U_kill = (1 - p_cap_th) * cvn.ETA_REP[th] - p_cap_th * cvn.F_CAP[th]
        C_tau_th = cvn.PHI_COST[th] * np.exp(cvn.KAPPA_C[th] * gamma_star) + cvn.NU_COST[th]
        # myopic V_cont proxy for the likelihood only (no v_next dependency needed here,
        # consistent with how tau=0's implementation likelihood already ignores continuation)
        V_cont_th = -C_tau_th - p_cap_th * cvn.F_CAP[th]
        branch_vals_th = {"rel": U_rel, "kill": U_kill, "cont": V_cont_th}
        branch_th = max(branch_vals_th, key=branch_vals_th.get)
        a_star_th = branch_to_action[branch_th]
        num = {a: (np.exp(1.0 / mt) if a == a_star_th else np.exp(0.0)) for a in ("Continue", "Release", "Kill")}
        denom = sum(num.values())
        l_i_k = float(num.get(act_k_realized, 1.0) / denom)

        l_c = voice_L_C_by_type.get(th, 1.0)
        w[th] = mu_tau[th] * l_i_k * pr_m * pr_d * l_c
    z = sum(w.values())
    if z > 1e-15:
        return {th: w[th] / z for th in w}
    return {th: 1.0 / len(cvn.TIPOS) for th in cvn.TIPOS}


def run_one_period(
    tau: int, mu_tau: dict, H_mu0: float, p_rescue_prev: float, p_nego_prev: float,
    cov_perp: str, theta_f: str, p,
    net_state, net_captor,
    pi_call_realized: dict, voz_params: dict, omega_voz: float,
    rng: np.random.Generator,
    T0_S: float = T0_S_DEFAULT, T0_F: float = T0_F_DEFAULT, T0_K: float = T0_K_DEFAULT,
):
    """Runs ONE full period tau (tau>=1): State optimizes, Family optimizes, Captor
    (true type) optimizes, the 3 MDG draws (S/F/K), the m/d/voice draws, kappa_h, and
    mu_{tau+1}. Returns a dict with everything needed to report this period AND to
    seed period tau+1 (mu_next, p_rescue_next, p_nego_next)."""
    v_next_grid = make_v_next_fn_grid_extended(net_state)
    v_next_scalar = make_v_next_fn_scalar_extended(net_captor)

    a_star, g_star, aS_star, v_by_type, feasible, extra = cvn.solve_state_problem(
        mu_tau, tau, T_TRAINED, p, v_next_grid, cov_perp, theta_f,
        p_rescue_prev=p_rescue_prev, p_nego_prev=p_nego_prev,
    )

    H_mu_tau = float(extra["H_mu"])
    H_ratio_real = (H_mu_tau / H_mu0) if H_mu0 > 1e-12 else 0.0
    mt_generic = cvn.m_t(tau, p)
    mt_S = _mdg_temp_player(T0_S, MDG_MULT_STATE, H_ratio_real, p, tau)
    mt_F = _mdg_temp_player(T0_F, MDG_MULT_FAMILY, H_ratio_real, p, tau)
    mt_K = _mdg_temp_player(T0_K, MDG_MULT_CAPTOR, H_ratio_real, p, tau)

    # tilde a_S (eq:logit-hybrid, 2 branches)
    num_r = np.exp(1.0 / mt_S) if aS_star == "Rescue" else np.exp(0.0)
    num_n = np.exp(1.0 / mt_S) if aS_star == "Negotiate" else np.exp(0.0)
    p_rescue = float(num_r / (num_r + num_n))
    p_nego = float(num_n / (num_r + num_n))
    u_s = float(rng.random())
    act_s = "Rescue" if u_s <= p_rescue else "Negotiate"

    # Family (eq:f-coop/eq:f-col/eq:ir-family, exact)
    U_coop_f, U_col_f, ir_f_gap, a_F_star, fam_extras = famopt.family_utilities(
        mu_tau, theta_f, a_star, g_star, tau, p
    )
    num_coop = np.exp(1.0 / mt_F) if a_F_star == "Cooperate" else np.exp(0.0)
    num_col = np.exp(1.0 / mt_F) if a_F_star == "Collude" else np.exp(0.0)
    p_coop = float(num_coop / (num_coop + num_col))
    p_col = float(num_col / (num_coop + num_col))
    u_f = float(rng.random())
    act_f = "Cooperate" if u_f <= p_coop else "Collude"

    # Captor, true type (eq:k-bellman/eq:k-rel/eq:k-kill/eq:kidnapper-cont)
    branch_true, U_rel_true, U_kill_true, V_cont_true, k_extras = (
        cttn.solve_captor_true_type_continuation(
            cov_perp, a_star, g_star, mu_tau, tau, p, v_next_scalar, p_rescue, p_nego,
        )
    )
    branch_to_action = {"rel": "Release", "kill": "Kill", "cont": "Continue"}
    a_K_star = branch_to_action[branch_true]
    num_cont = np.exp(1.0 / mt_K) if a_K_star == "Continue" else np.exp(0.0)
    num_rel = np.exp(1.0 / mt_K) if a_K_star == "Release" else np.exp(0.0)
    num_kill = np.exp(1.0 / mt_K) if a_K_star == "Kill" else np.exp(0.0)
    denom_k = num_cont + num_rel + num_kill
    p_cont_k = float(num_cont / denom_k)
    p_rel_k = float(num_rel / denom_k)
    p_kill_k = float(num_kill / denom_k)
    u_k = float(rng.random())
    if u_k <= p_cont_k:
        act_k = "Continue"
    elif u_k <= p_cont_k + p_rel_k:
        act_k = "Release"
    else:
        act_k = "Kill"

    # m draw (eq:LH-compacta), using the generic mt (system-level, not a player temp)
    m_result = draw_m(a_star, g_star, cov_perp, p, mt_generic, rng)

    # d draw (NEW, generalized eq:detection)
    d_result = draw_d(a_star, g_star, cov_perp, p, rng)

    # voice draw (eq:voz-descomp/eq:LC), reusing rational_behavior.py, illustrative
    voice_result = None
    voice_L_C_by_type = {th: 1.0 for th in cvn.TIPOS}
    if _RB_AVAILABLE and pi_call_realized is not None:
        V_tau = draw_voice_indicator(pi_call_realized[cov_perp], rng)
        x_obs = sample_voice_observation(cov_perp, voz_params, rng) if V_tau == 1 else None
        L_C_true, L_voz_true = communication_likelihood_LC(
            cov_perp, V_t=V_tau, omega_voz=omega_voz, pi_call=pi_call_realized,
            x_obs=x_obs, voz_params_by_theta=voz_params,
        )
        voice_result = {
            "V_t": int(V_tau), "x_obs": x_obs.tolist() if x_obs is not None else None,
            "L_voz": float(L_voz_true) if V_tau == 1 else None, "L_C": float(L_C_true),
        }
        for th in cvn.TIPOS:
            l_c_th, _ = communication_likelihood_LC(
                th, V_t=V_tau, omega_voz=omega_voz, pi_call=pi_call_realized,
                x_obs=x_obs, voz_params_by_theta=voz_params,
            )
            voice_L_C_by_type[th] = float(l_c_th)

    # mu_{tau+1} (eq:bayes-update, GENERIC -- see bayes_update_mu docstring)
    mu_next = bayes_update_mu(
        mu_tau, a_star, g_star, mt_generic, p,
        m_result["outcome_key"], d_result["d"], act_k, voice_L_C_by_type,
    )

    # Per-type benchmarks (eq:kappa-c's tilde-lambda ingredients + rescue/negotiation argmins)
    # and kappa_h(theta,tau), generalized to THIS tau -- see compute_benchmarks_tau/
    # compute_kappa_h_tau docstrings. Reuses the benchmarks already calculated in solve_state_problem
    # to avoid redundant grid searches.
    benchmarks_by_type = extra["benchmarks_by_type"]
    belief_weighted = {
        "alpha_R_mu": extra["alpha_R_mu"],
        "gamma_R_mu": extra["gamma_R_mu"],
        "alpha_N_mu": extra["alpha_N_mu"],
        "gamma_N_mu": extra["gamma_N_mu"],
    }
    neg_sign_kappa_h = {}
    for th in cvn.TIPOS:
        kh_th = compute_kappa_h_tau(th, a_star, g_star, mt_generic, act_f, act_k, act_s, p)
        neg_sign_kappa_h[th] = -int(np.sign(kh_th))

    return {
        "tau": tau, "mu_tau": dict(mu_tau), "mu_next": mu_next,
        "H_mu": H_mu_tau, "H_ratio_real": H_ratio_real,
        "alpha": a_star, "gamma": g_star, "a_S": aS_star, "feasible": feasible,
        "ir_k_true_gap": extra["ir_k_true_gap"], "delta_H": extra["delta_H"],
        "T_generic": mt_generic, "T_S": mt_S, "T_F": mt_F, "T_K": mt_K,
        "act_s": act_s, "u_s": u_s, "p_rescue": p_rescue, "p_nego": p_nego,
        "a_F_star": a_F_star, "U_coop_f": U_coop_f, "U_col_f": U_col_f,
        "ir_f_gap": ir_f_gap, "act_f": act_f, "u_f": u_f, "p_coop": p_coop, "p_col": p_col,
        "a_K_star": a_K_star, "branch_true": branch_true,
        "U_rel_true": U_rel_true, "U_kill_true": U_kill_true, "V_cont_true": V_cont_true,
        "act_k": act_k, "u_k": u_k, "p_cont_k": p_cont_k, "p_rel_k": p_rel_k, "p_kill_k": p_kill_k,
        "m": m_result, "d": d_result, "voice": voice_result,
        "p_rescue_next": p_rescue, "p_nego_next": p_nego,   # protocolo de consistencia temporal
        "benchmarks_by_type": benchmarks_by_type, "neg_sign_kappa_h": neg_sign_kappa_h,
        **belief_weighted,
    }
