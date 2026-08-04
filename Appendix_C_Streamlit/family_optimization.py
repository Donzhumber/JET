"""
Problema PROPIO de la Familia en tau: formula EXACTA de Bernal_H.tex/Working_paper_eng.tex
(eq:f-coop, eq:f-col, eq:ir-family). Es un problema ESTATICO por diseno -- no tiene termino
de continuacion en el paper (a diferencia del Secuestrador), asi que no requiere ninguna red
entrenada; un calculo cerrado por periodo es exacto.

Toma (alpha_tau*, gamma_tau*, mu_tau) ya resueltos por el Estado (train_captor_value_net.py)
como insumos fijos. No valida IC^K/IR^K/IR^F como restriccion -- ese gate solo aplica a la
busqueda del Estado sobre (alpha,gamma); aqui es un arg-max directo de utilidad.

Uso: from family_optimization import family_utilities
"""
import numpy as np

from train_captor_value_net import (
    TIPOS, Params, m_t, outcome_probs_grid, p_surv_raw, PHI_F, KAPPA_F, NU_F,
    V_L_FAMILY, F_COL,
)


def family_utilities(mu_tau, theta_f, alpha_star, gamma_star, tau, p: Params):
    """eq:f-coop / eq:f-col / eq:ir-family (Bernal_H.tex), formula EXACTA (no la
    simplificacion p_pay+varrho que usaba antes Gamma_tau internamente para el Estado -- esa
    tambien se reemplazo por esta misma formula, ver train_captor_value_net.py). Todas las
    probabilidades se ponderan por mu_tau (la Familia no observa theta_K, solo su creencia).
    Devuelve (U_coop, U_col, gap=U_coop-U_col, a_F_star, extras-dict)."""
    mt = m_t(tau, p)
    theta_hat = max(mu_tau, key=mu_tau.get)
    iota = max(mu_tau.values())
    p_surv_mu = sum(mu_tau[th] * p_surv_raw(th, theta_hat, p, iota) for th in TIPOS)
    p_rel_mu, p_det_mu = 0.0, 0.0
    for th in TIPOS:
        probs, pdet = outcome_probs_grid(alpha_star, gamma_star, th, p, mt)
        p_rel_mu += mu_tau[th] * probs["4"]
        p_det_mu += mu_tau[th] * pdet
    e_tau = PHI_F[theta_f] * np.exp(KAPPA_F[theta_f] * gamma_star) + NU_F[theta_f]
    U_coop = p_surv_mu * V_L_FAMILY - e_tau
    U_col = p_rel_mu * V_L_FAMILY - p.R_millions - p_det_mu * F_COL
    gap = U_coop - U_col
    a_F_star = "Cooperate" if gap >= -1e-9 else "Collude"
    extras = {
        "p_surv_mu": p_surv_mu, "p_rel_mu": p_rel_mu, "p_det_mu": p_det_mu, "e_tau": e_tau,
    }
    return float(U_coop), float(U_col), float(gap), a_F_star, extras
