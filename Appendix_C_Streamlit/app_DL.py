import os
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Optional: trained CaptorValueNet for the tau=1 dynamic State optimization button
# (Tab 3). Both files live in this same folder; torch/the trained weights may be
# absent (e.g. a fresh checkout before running train_captor_value_net.py), so this
# import is guarded and the button simply stays disabled with an explanatory message.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch
    import train_captor_value_net as cvn
    import family_optimization as famopt
    import train_captor_true_type_net as cttn
    import run_period as rp
    _CVN_AVAILABLE = True
except Exception:
    _CVN_AVAILABLE = False

# Optional: reuse app.py's ALREADY-VALIDATED voice/acoustic generative mechanism
# (rational_behavior.py, one directory up) for the tau=1+ V(voz) draw in Tab 3 --
# instead of re-deriving it. Guarded the same way as the CVN import above; if the
# module or its expected functions are absent, the voice draw simply stays disabled.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rational_behavior import (
        sample_incident_pi_call_realized,
        draw_voice_indicator,
        sample_voice_observation,
        communication_likelihood_LC,
    )
    _RB_AVAILABLE = True
except Exception:
    _RB_AVAILABLE = False

# 1. Page Configuration (Standard clean theme, no overriding CSS that hides text)
st.set_page_config(
    page_title="Section 4.3 Probabilistic Technology - Bernal_H.tex",
    layout="wide"
)

# Compact typography: reduce base font sizes app-wide (headers, body text, metrics,
# tabs) without touching colors/theme, so more content fits without scrolling.
st.markdown(
    """
    <style>
    .stApp, .stMarkdown p, .stMarkdown li { font-size: 0.85rem !important; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    h4, h5 { font-size: 0.95rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 6px 14px; }
    .stSlider label, .stSelectbox label { font-size: 0.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for materialized actions and simulation draw values
if "act_s" not in st.session_state:
    st.session_state["act_s"] = "Negotiate"
    st.session_state["u_s"] = 0.7500
    st.session_state["p_rescue_draw"] = 0.5000
    st.session_state["p_nego_draw"] = 0.5000
if "act_k" not in st.session_state:
    st.session_state["act_k"] = "Continue"
    st.session_state["u_k"] = 0.2500
    st.session_state["p_cont_draw"] = 0.3333
    st.session_state["p_rel_draw"] = 0.3333
    st.session_state["p_kill_draw"] = 0.3333
if "act_f" not in st.session_state:
    st.session_state["act_f"] = "Cooperate"
    st.session_state["u_f"] = 0.4000
    st.session_state["p_coop_draw"] = 0.5000
    st.session_state["p_col_draw"] = 0.5000
# Voice trajectory (tau=1,2,3,... one entry appended per "Run State Optimization" click).
# voice_pi_call_realized is drawn ONCE (first click) and reused for every later period --
# an incident-level trait, not something redrawn each period.
if "voice_path" not in st.session_state:
    st.session_state["voice_path"] = []
if "voice_pi_call_realized" not in st.session_state:
    st.session_state["voice_pi_call_realized"] = None
# Physical-outcome trajectory (tau=1,2,3,... one entry per click), reusing
# cvn.outcome_probs_grid (eq:hj/eq:pCont/eq:xi = eq:LH-compacta) unmodified.
if "m_path" not in st.session_state:
    st.session_state["m_path"] = []

# Native Streamlit headers
st.title("1. Probabilistic technology")
st.markdown("### Visualizing the stochastic architecture, implementation noise, competing risks, and outcome materialization (Section 4.3)")
st.markdown("---")

# =====================================================================
# CALIBRATED STRUCTURAL PARAMETERS FROM app.py & model_logic.py
# =====================================================================
# Riesgos basales (lambda_j0)
LAMBDAS_0 = {
    "Pago": 0.012,
    "Muerte": 0.002,
    "Rescate": 0.008
}

# Coeficientes por tipo de secuestrador (beta_K,j)
BETAS_K = {
    "FARC": {"Pago": -0.70, "Muerte": -0.85, "Rescate": 0.90},
    "ELN":  {"Pago": 1.10,  "Muerte": 0.20,  "Rescate": -0.65},
    "PAR":  {"Pago": -0.25, "Muerte": 1.35,  "Rescate": 0.15},
    "DC":   {"Pago": 1.55,  "Muerte": -0.40, "Rescate": -0.95}
}

# Coeficientes geográficos (beta_z)
BETAS_Z = {
    "Metropolis": 0.00,
    "Andean": -0.45,
    "Caribbean": -0.70,
    "Pacific / Red Zone": -0.20,
    "Eastern Plains/Jungle": -0.32
}

# Coeficientes de sensibilidad a políticas (zeta_alpha, zeta_gamma)
ZETAS_POLITICA = {
    "DC":   {"zeta_alpha": 0.2409, "zeta_gamma": 0.5450},
    "PAR":  {"zeta_alpha": 0.2183, "zeta_gamma": 0.5848},
    "ELN":  {"zeta_alpha": 0.2101, "zeta_gamma": 0.5532},
    "FARC": {"zeta_alpha": 0.2087, "zeta_gamma": 0.6197}
}

# Parámetros de detección (eta_0, eta_1, eta_2)
ETA_0_PDET = {
    "DC": -1.5,
    "PAR": -2.0,
    "ELN": -2.5,
    "FARC": -2.8
}
ETA_1_PDET = 1.0
ETA_2_PDET = 1.0

# Parámetros de supervivencia en rescate focal (alpha_leth, beta_R)
ALPHA_LETH = {
    "DC": -5.25,
    "PAR": -5.15,
    "ELN": -5.05,
    "FARC": -4.95
}
BETA_R_SURV = 7.0

# Technical capture probability p_cap(a, theta_i, theta_S, alpha*, gamma*) (eq. p-cap, Bernal_H.tex)
DELTA_A_CAP = {"Rescue": 0.60, "Negotiate": -0.20}          # impact of the executed intervention mode
C0_CAP = {"DC": -0.30, "PAR": -0.10, "ELN": 0.10, "FARC": 0.30}   # baseline heterogeneity by type
CALPHA_CAP = {"DC": 0.80, "PAR": 1.00, "ELN": 1.10, "FARC": 1.30}  # financial-block sensitivity
CGAMMA_CAP = {"DC": 1.00, "PAR": 1.20, "ELN": 1.30, "FARC": 1.50}  # operational-pressure sensitivity
CS_CAP = {"Strict": 0.40, "Lax": -0.40}                     # institutional capacity of the state

# Voice/acoustic evidence parameters (eq. voz-descomp / Lvoz / LC, Bernal_H.tex)
# NOTE: these three (X_TRUE_VOZ/SIGMA_VOZ/PI_CALL) are the LEGACY scalar simplification
# used ONLY by tau=0's static Block F (Tab 1: fixed x_obs slider, not a genuine draw).
X_TRUE_VOZ = {"DC": -1.0, "PAR": -0.3, "ELN": 0.3, "FARC": 1.0}   # reference acoustic pattern x_bar(theta_K)
SIGMA_VOZ = {"DC": 0.8, "PAR": 0.9, "ELN": 0.9, "FARC": 1.0}       # dispersion sigma_tilde(theta_K)
PI_CALL = {"DC": 0.15, "PAR": 0.25, "ELN": 0.35, "FARC": 0.45}     # contact probability pi_call(theta_K)

# Full vector calibration (4 acoustic traits x (x_bar, sigma_L, sigma_S) per type), copied
# VERBATIM from app.py's _default_cal_voz_params() so both apps share one calibration.
# Used ONLY by the tau=1+ genuine voice draw below (Table 5.2's V(voz) column, tau=1+) --
# NOT by tau=0's Block F, which keeps its own separate scalar simplification untouched.
VOZ_PARAMS_DEFAULT = {
    "DC": {"x": [158.0, 0.22, 0.17, 0.62], "sigma_L": [8.5, 0.052, 0.032, 0.085], "sigma_S": [6.0, 0.042, 0.022, 0.065]},
    "PAR": {"x": [171.0, 0.30, 0.11, 0.53], "sigma_L": [7.8, 0.049, 0.029, 0.078], "sigma_S": [5.3, 0.037, 0.021, 0.056]},
    "ELN": {"x": [167.0, 0.28, 0.12, 0.51], "sigma_L": [7.5, 0.047, 0.027, 0.074], "sigma_S": [5.5, 0.036, 0.020, 0.059]},
    "FARC": {"x": [176.0, 0.37, 0.13, 0.43], "sigma_L": [7.0, 0.044, 0.025, 0.070], "sigma_S": [4.9, 0.033, 0.018, 0.051]},
}
# kappa for the Beta-realized pi_call draw (sample_incident_pi_call_realized) -- same
# constant app.py uses; not a paper-mandated value, an inherited calibration choice.
VOZ_KAPPA_REALIZED = 30.0

# Per-player MDG temperature multipliers (eq:logit-hybrid / eq:m-t). Bernal_H.tex and
# Working_paper_eng.tex define ONE shared "system temperature" T_t (Working_paper_eng.tex:
# "T_t=T_t(mu_t,t)>0 represents the system temperature") -- NOT one per player; T_0, eta_cal,
# c_bar carry no player index anywhere in either text. This is therefore a DEVIATION from the
# literal text, explicitly requested and approved by the user (not a fidelity correction).
# Applied ONLY to the three MDG-executed-action draws (tilde a_S/F/K); the M_t row (Table 5.2
# Sec.28), kappa_h, the m draw, and the benchmarks all keep using the generic, unmultiplied
# system-level M_{tau=1} (T0_mult=eta_mult=c_mult=1.0), unchanged.
# T0 ya NO vive en estos dicts -- ahora es un valor directo por jugador (sliders T0_S/T0_F/
# T0_K, Tab 1 Block A), reemplazando el viejo esquema "p.T0 * multiplicador compartido"
# (aprobado por el usuario). c_bar SUBIDO (0.5/1.0/1.5 -> 14.0/18.0/22.0) para que el piso de
# ruido de largo plazo no vuelva a colapsar a casi-determinista en tau grandes.
MDG_MULT_STATE = {"eta_cal": 1.2, "c_bar": 14.0}    # institutional actor: faster maturation
MDG_MULT_FAMILY = {"eta_cal": 1.0, "c_bar": 18.0}   # baseline
MDG_MULT_CAPTOR = {"eta_cal": 0.8, "c_bar": 22.0}   # least institutional actor: slower maturation, highest persistent noise floor
T0_S_DEFAULT = 0.90
T0_F_DEFAULT = 1.30
T0_K_DEFAULT = 1.80


def _mdg_temp_player(T0_i: float, mult: dict, h_ratio_real: float, p) -> float:
    """Player-specific literal T_t (eq:m-t), same functional form as the generic
    M_{tau=1}, using T0_i (direct per-player base, Tab 1 Block A slider) and mult['eta_cal']/
    ['c_bar'] on top of the shared eta_cal/c_bar. h_ratio_real = H(mu_tau)/H(mu_0), tau=1 fixed."""
    return float(T0_i * max(
        h_ratio_real * np.exp(-p.eta_cal * mult["eta_cal"] * 1.0), p.c_bar * mult["c_bar"]
    ))

# --- PERSISTENT STORAGE HELPERS (8 SLOTS) ---
SAVED_RUNS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs.json")

PARAM_KEYS = [
    # Sliders and selectboxes Tab 1
    "T0_S_slider", "T0_F_slider", "T0_K_slider",
    "eta_cal_slider", "c_bar_slider", "H_ratio_slider",
    "cov_perp_selectbox", "cov_vict_selectbox", "cov_wealth_selectbox", "cov_zone_selectbox", "cov_state_selectbox",
    "T_mad_slider", "alpha_slider", "gamma_slider", "ransom_R_slider",
    "beta_tilde_dc_slider", "beta_tilde_par_slider", "beta_tilde_eln_slider", "beta_tilde_farc_slider",
    "lambda_4_slider", "eta_1_slider", "eta_2_slider",
    # Input params Tab 3
    "t_max_input", "counterfactual_ext_input", "tau_view_selector",
    # Simulation outputs / states
    "tau_history", "tau_history_normalized", "tau_display_max", "tau_closed_at", "tau1_state_opt_result",
    # tau=0 draws & actions (Tab 1 / st.session_state)
    "act_s", "act_k", "act_f", "u_s", "u_k", "u_f",
    "p_rescue_draw", "p_nego_draw", "p_cont_draw", "p_rel_draw", "p_kill_draw", "p_coop_draw", "p_col_draw",
    "m_tau0_draw", "m_tau0_outcome", "m_path",
    "d_tau0_draw", "d_tau0_pdet", "d_tau0_realized",
    "voice_path", "voice_pi_call_realized"
]

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def load_saved_runs():
    if not os.path.exists(SAVED_RUNS_FILE):
        return {}
    try:
        with open(SAVED_RUNS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return {}

def save_runs_to_file(data):
    try:
        with open(SAVED_RUNS_FILE, "w") as f:
            json.dump(make_json_serializable(data), f, indent=2)
    except Exception as e:
        st.error(f"Error saving runs: {e}")

def prepare_loaded_runs(run_data):
    # Convert keys of dictionaries that should be ints back to int keys
    if "tau_history" in run_data and isinstance(run_data["tau_history"], dict):
        run_data["tau_history"] = {int(k): v for k, v in run_data["tau_history"].items()}
    if "tau_history_normalized" in run_data and isinstance(run_data["tau_history_normalized"], dict):
        run_data["tau_history_normalized"] = {int(k): v for k, v in run_data["tau_history_normalized"].items()}
    return run_data

def save_current_run_to_slot(slot_idx):
    runs = load_saved_runs()
    
    # Check if we have active run
    if "tau_history" not in st.session_state or not st.session_state["tau_history"]:
        st.error("No active simulation run to save. Please run the simulation first.")
        return
        
    # Gather run data
    run_data = {}
    for k in PARAM_KEYS:
        if k in st.session_state:
            run_data[k] = st.session_state[k]
            
    # Generate metadata
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get custom name from session state
    custom_name = st.session_state.get(f"name_input_{slot_idx}", "").strip()
    
    # Auto-generate name if blank
    if not custom_name:
        perp = st.session_state.get("cov_perp_selectbox", "Unknown")
        zone = st.session_state.get("cov_zone_selectbox", "Unknown")
        state_type = st.session_state.get("cov_state_selectbox", "Unknown")
        t_max = st.session_state.get("t_max_input", 10)
        custom_name = f"{perp} | {zone} | {state_type} (T_max={t_max})"
        
    runs[str(slot_idx)] = {
        "name": custom_name,
        "timestamp": timestamp,
        "data": run_data
    }
    
    save_runs_to_file(runs)
    # Use toast or warning session state to keep message persistent across rerun if needed,
    # but a simple toast or print is also fine. We will set a session message.
    st.session_state["saved_run_message"] = f"Run successfully saved to Slot {slot_idx}!"

def load_run_from_slot(slot_idx):
    runs = load_saved_runs()
    slot_key = str(slot_idx)
    if slot_key not in runs:
        st.error(f"Slot {slot_idx} is empty.")
        return
        
    run_data = runs[slot_key]["data"]
    run_data = prepare_loaded_runs(run_data)
    
    # Write to session state
    for k in PARAM_KEYS:
        if k in run_data:
            st.session_state[k] = run_data[k]
        else:
            # Pop if not present in saved run, to keep state clean
            st.session_state.pop(k, None)
            
    st.session_state["saved_run_message"] = f"Slot {slot_idx} loaded successfully!"

def delete_run_from_slot(slot_idx):
    runs = load_saved_runs()
    slot_key = str(slot_idx)
    if slot_key in runs:
        del runs[slot_key]
        save_runs_to_file(runs)
        st.session_state["saved_run_message"] = f"Slot {slot_idx} cleared!"


# =====================================================================
# 3. TAB DEFINITION
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Probabilistic technology", "2. Rational Behavior", "3. Results", "4. Graphs", "5. Description"]
)

with tab1:
    # ---------------------------------------------------------------------
    # Block A: The Mano de Dios-Guadalupe (MDG) Frictional Process
    # ---------------------------------------------------------------------
    st.markdown("## 🅰️ The Mano de Dios-Guadalupe (MDG) Frictional Process")
    st.markdown(
        """
        The MDG process models institutional friction. It transforms each player's latent strategic intention 
        $a_t^{i*}$ (for $i \\in \\{S \\text{ (State)}, K \\text{ (Captor)}, F \\text{ (Family)}\\}$) into an executed action $\\tilde{a}_t^i$ via a logit distribution centered on the plan, 
        ensuring full support over the action space.
        """
    )

    # 1. Render Equations
    st.latex(r"T_t = T_0 \max\left\{\frac{H(\mu_t)}{H(\mu_0)} e^{-\eta_{\text{cal}} t},\,\underline{c}\right\}")
    st.latex(r"\mathbb{P}_I^i(\tilde{a}_t^i = a \mid a_t^{i*}, X_t) = \frac{\exp\left(\mathbb{I}_{\{a = a_t^{i*}\}} / T_t\right)}{\sum_{a' \in \mathcal{A}^i} \exp\left(\mathbb{I}_{\{a' = a_t^{i*}\}} / T_t\right)}")

    # t (dias transcurridos, Block A) fijado en 0 -- eliminado como slider (aprobado por el
    # usuario): tau=0 representa el caso recien iniciado, sin madurar (M_t=0 en Block B/C/D,
    # T_t sin decaimiento exponencial). No afecta tau>=1 (run_period.py tiene su propio reloj
    # `tau`, independiente de esta variable de solo-tau=0).
    t_days = 0

    st.markdown("##### 🔧 Modifiable Parameters for Block A:")
    # Render sliders with independent bold markdown labels above to avoid truncation in narrow columns
    c_t0s, c_t0f, c_t0k = st.columns(3)
    with c_t0s:
        st.markdown("**$T_0^S$ (State)**")
        T0_S = st.slider("T0_S", 0.1, 3.0, T0_S_DEFAULT, step=0.05, key="T0_S_slider", label_visibility="collapsed")
    with c_t0f:
        st.markdown("**$T_0^F$ (Family)**")
        T0_F = st.slider("T0_F", 0.1, 3.0, T0_F_DEFAULT, step=0.05, key="T0_F_slider", label_visibility="collapsed")
    with c_t0k:
        st.markdown("**$T_0^K$ (Captor)**")
        T0_K = st.slider("T0_K", 0.1, 3.0, T0_K_DEFAULT, step=0.05, key="T0_K_slider", label_visibility="collapsed")
    # T_0 (generic/illustrative): FIXED at 1.0 (approved by the user) -- no longer a UI
    # slider. It has no live role in the mechanism (m/lambda_j use M(t)=min(1,(t/T_mad)^2)
    # via cvn.m_t, not T_0); it only survives internally as Params.T0, a legacy NN input
    # feature (encode_input) shared with the two trained value nets, kept fixed so their
    # already-trained semantics are undisturbed.
    T_0 = 1.0

    c_t3, c_t4, c_t5 = st.columns(3)
    with c_t3:
        st.markdown(r"**Decay Rate ($\eta_{\text{cal}}$)**")
        eta_cal = st.slider("eta_cal", 0.01, 0.20, 0.075, step=0.005, key="eta_cal_slider", label_visibility="collapsed")
    with c_t4:
        st.markdown(r"**Noise Floor ($\underline{c}$)**")
        c_bar = st.slider("c_bar", 0.00, 0.20, 0.05, step=0.01, key="c_bar_slider", label_visibility="collapsed")
    with c_t5:
        st.markdown(r"**Entropy Ratio ($H(\mu_t)/H(\mu_0)$)**")
        H_ratio = st.slider("H_ratio", 0.0, 1.0, 0.80, step=0.05, key="H_ratio_slider", label_visibility="collapsed")

    # Computations for Block A temperature (t=0 fijo -> sin decaimiento exponencial).
    # T_t_S/F/K: por jugador, MISMA formula (eq:m-t) que _mdg_temp_player, evaluada con
    # T0_S/F/K y los multiplicadores MDG_MULT_STATE/FAMILY/CAPTOR (extension aprobada, ya
    # usada para tau=1/tau>=2) -- estas SI alimentan el ruido de ejecucion de acciones
    # (tilde a^S/F/K), sin cambios por esta correccion. El T_t generico/ilustrativo (antes
    # aqui, con slider T_0) se retiro de la UI: no alimentaba nada real (m/lambda_j usan
    # M(t)=min(1,(t/T_mad)^2), eq:hj, ver Block B y cvn.m_t) y su comparativa visual era
    # el unico consumidor -- decision aprobada por el usuario.
    T_t_S = float(T0_S * max(H_ratio * np.exp(-eta_cal * MDG_MULT_STATE["eta_cal"] * t_days), c_bar * MDG_MULT_STATE["c_bar"]))
    T_t_F = float(T0_F * max(H_ratio * np.exp(-eta_cal * MDG_MULT_FAMILY["eta_cal"] * t_days), c_bar * MDG_MULT_FAMILY["c_bar"]))
    T_t_K = float(T0_K * max(H_ratio * np.exp(-eta_cal * MDG_MULT_CAPTOR["eta_cal"] * t_days), c_bar * MDG_MULT_CAPTOR["c_bar"]))
    st.caption(f"$T_t^S={T_t_S:.4f}$ &ensp; $T_t^F={T_t_F:.4f}$ &ensp; $T_t^K={T_t_K:.4f}$ (individual temperatures per player — these DO feed into $\\tilde a^{{S,F,K}}$)")

    st.markdown("##### 🎭 Latent Plans and Executed Action Distributions for each player:")
    
    # 3 Columns for State, Captor, Family
    col_s, col_k, col_f = st.columns(3)
    
    # State (S)
    with col_s:
        st.markdown("### 🏛️ State (S)")
        st.markdown("**Latent Intention ($a^{S*}$):**")
        a_S_star = st.selectbox("State Latent Intention", ["Rescue", "Negotiate"], index=1, key="a_S_star_selectbox", label_visibility="collapsed")
        
        # Logit probabilities
        num_rescue = np.exp(1.0 / T_t_S) if a_S_star == "Rescue" else np.exp(0.0)
        num_nego = np.exp(1.0 / T_t_S) if a_S_star == "Negotiate" else np.exp(0.0)
        denom_s = num_rescue + num_nego
        p_rescue = num_rescue / denom_s
        p_nego = num_nego / denom_s
        
        st.markdown("**Executed Probabilities:**")
        with st.container(height=90, border=False):
            st.markdown(f"*   $\\mathbb{{P}}(\\tilde{{a}}^S = \\text{{Rescue}})$ = **{p_rescue:.4f}**\n*   $\\mathbb{{P}}(\\tilde{{a}}^S = \\text{{Negotiate}})$ = **{p_nego:.4f}**")

        fig_s = go.Figure(data=[go.Bar(x=["Rescue", "Negotiate"], y=[p_rescue, p_nego], marker_color=["#4F46E5", "#64748B"], text=[f"{p_rescue:.2%}", f"{p_nego:.2%}"], textposition='auto')])
        fig_s.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1]), font=dict(size=11))
        st.plotly_chart(fig_s, use_container_width=True)

    # Captor (K)
    with col_k:
        st.markdown("### 🦹 Captor (K)")
        st.markdown("**Latent Intention ($a^{K*}$):**")
        a_K_star = st.selectbox("Captor Latent Intention", ["Continue", "Release", "Kill"], index=0, key="a_K_star_selectbox", label_visibility="collapsed")
        
        # Logit probabilities
        num_cont = np.exp(1.0 / T_t_K) if a_K_star == "Continue" else np.exp(0.0)
        num_rel = np.exp(1.0 / T_t_K) if a_K_star == "Release" else np.exp(0.0)
        num_kill = np.exp(1.0 / T_t_K) if a_K_star == "Kill" else np.exp(0.0)
        denom_k = num_cont + num_rel + num_kill
        p_cont_k = num_cont / denom_k
        p_rel_k = num_rel / denom_k
        p_kill_k = num_kill / denom_k
        
        st.markdown("**Executed Probabilities:**")
        with st.container(height=90, border=False):
            st.markdown(f"*   $\\mathbb{{P}}(\\tilde{{a}}^K = \\text{{Continue}})$ = **{p_cont_k:.4f}**\n*   $\\mathbb{{P}}(\\tilde{{a}}^K = \\text{{Release}})$ = **{p_rel_k:.4f}**\n*   $\\mathbb{{P}}(\\tilde{{a}}^K = \\text{{Kill}})$ = **{p_kill_k:.4f}**")

        fig_k = go.Figure(data=[go.Bar(x=["Continue", "Release", "Kill"], y=[p_cont_k, p_rel_k, p_kill_k], marker_color=["#0284C7", "#F59E0B", "#E11D48"], text=[f"{p_cont_k:.2%}", f"{p_rel_k:.2%}", f"{p_kill_k:.2%}"], textposition='auto')])
        fig_k.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1]), font=dict(size=11))
        st.plotly_chart(fig_k, use_container_width=True)

    # Family (F)
    with col_f:
        st.markdown("### 👪 Family (F)")
        st.markdown("**Latent Intention ($a^{F*}$):**")
        a_F_star = st.selectbox("Family Latent Intention", ["Cooperate", "Collude"], index=0, key="a_F_star_selectbox", label_visibility="collapsed")
        
        # Logit probabilities
        num_coop = np.exp(1.0 / T_t_F) if a_F_star == "Cooperate" else np.exp(0.0)
        num_col = np.exp(1.0 / T_t_F) if a_F_star == "Collude" else np.exp(0.0)
        denom_f = num_coop + num_col
        p_coop = num_coop / denom_f
        p_col = num_col / denom_f
        
        st.markdown("**Executed Probabilities:**")
        with st.container(height=90, border=False):
            st.markdown(f"*   $\\mathbb{{P}}(\\tilde{{a}}^F = \\text{{Cooperate}})$ = **{p_coop:.4f}**\n*   $\\mathbb{{P}}(\\tilde{{a}}^F = \\text{{Collude}})$ = **{p_col:.4f}**")

        fig_f = go.Figure(data=[go.Bar(x=["Cooperate", "Collude"], y=[p_coop, p_col], marker_color=["#10B981", "#D97706"], text=[f"{p_coop:.2%}", f"{p_col:.2%}"], textposition='auto')])
        fig_f.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1]), font=dict(size=11))
        st.plotly_chart(fig_f, use_container_width=True)

    # Live Action Materialization Simulator
    st.markdown("🎲 **Simulate Executed Action Profile Materialization**")
    if st.button("Draw MDG Action Realizations"):
        u_s = np.random.default_rng().random()
        u_k = np.random.default_rng().random()
        u_f = np.random.default_rng().random()
        
        st.session_state["u_s"] = u_s
        st.session_state["u_k"] = u_k
        st.session_state["u_f"] = u_f
        
        st.session_state["p_rescue_draw"] = p_rescue
        st.session_state["p_nego_draw"] = p_nego
        
        st.session_state["p_cont_draw"] = p_cont_k
        st.session_state["p_rel_draw"] = p_rel_k
        st.session_state["p_kill_draw"] = p_kill_k
        
        st.session_state["p_coop_draw"] = p_coop
        st.session_state["p_col_draw"] = p_col
        
        st.session_state["act_s"] = "Rescue" if u_s <= p_rescue else "Negotiate"
        st.session_state["act_k"] = "Continue" if u_k <= p_cont_k else ("Release" if u_k <= p_cont_k + p_rel_k else "Kill")
        st.session_state["act_f"] = "Pay" if u_f > p_coop else "Cooperate"
        st.rerun()

    # Detailed simulation results explaining the why
    st.markdown(
        f"### 📊 Current Executed Profile ($\\tilde{{a}}_t$): $\\tilde{{a}}_S$ = **{st.session_state['act_s']}** | $\\tilde{{a}}_K$ = **{st.session_state['act_k']}** | $\\tilde{{a}}_F$ = **{st.session_state['act_f']}**"
    )
    with st.expander("🔍 Why did this profile materialize? (Logit Partition & Uniform Draws)"):
        st.markdown(
            f"""
            *   **State (S):** Draw $U_S$ = **{st.session_state["u_s"]:.6f}**
                *   *Threshold:* $\\mathbb{{P}}(\\text{{Rescue}}) = {st.session_state["p_rescue_draw"]:.4f}$ (Interval: $[0, {st.session_state["p_rescue_draw"]:.4f}]$)
                *   *Result:* Since $U_S$ fell in the **{st.session_state["act_s"]}** region, $\\tilde{{a}}_S$ became **{st.session_state["act_s"]}**.
            *   **Captor (K):** Draw $U_K$ = **{st.session_state["u_k"]:.6f}**
                *   *Thresholds:* Continue (Interval: $[0, {st.session_state["p_cont_draw"]:.4f}]$) | Release (Interval: $({st.session_state["p_cont_draw"]:.4f}, {st.session_state["p_cont_draw"] + st.session_state["p_rel_draw"]:.4f}]$) | Kill (Interval: $({st.session_state["p_cont_draw"] + st.session_state["p_rel_draw"]:.4f}, 1]$)
                *   *Result:* Since $U_K$ fell in the **{st.session_state["act_k"]}** region, $\\tilde{{a}}_K$ became **{st.session_state["act_k"]}**.
            *   **Family (F):** Draw $U_F$ = **{st.session_state["u_f"]:.6f}**
                *   *Threshold:* $\\mathbb{{P}}(\\text{{Cooperate}}) = {st.session_state["p_coop_draw"]:.4f}$ (Interval: $[0, {st.session_state["p_coop_draw"]:.4f}]$)
                *   *Result:* Since $U_F$ fell in the **{st.session_state["act_f"]}** region, $\\tilde{{a}}_F$ became **{st.session_state["act_f"]}**.

            *(Click 'Draw MDG Action Realizations' above to draw a new profile and update the thresholds)*
            """
        )

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block B: Competing Risks & Effective Intensities (lambda_tilde)
    # ---------------------------------------------------------------------
    st.markdown(r"## 🅱️ Competing Risks & Effective Intensities ($\tilde{\lambda}_j$)")
    st.markdown(
        """
        Strategic risk propensities scale dynamically based on policies, organizational tech parameters, 
        and the maturation filter $M(t)$ isolating initial logistical inertia.
        """
    )

    # 1. Render Equations (Section 4.2.3 Riesgos competitivos y supervivencia)
    st.latex(r"M(t) = \min\left\{1, \left(\frac{t}{T_{\text{mad}}}\right)^2\right\}")
    st.latex(r"\tilde{\lambda}_j(t \mid \mathcal{C}_t) = M(t) \lambda_{j}(t \mid \mathcal{C}_t)\quad (j=1,2,3),\qquad \tilde{\lambda}_4(t) = \lambda_4")
    
    st.markdown("##### 🔧 Modifiable Covariates ($\\mathcal{C}_t$) from Section 4.2.3 Equations:")
    
    c_c1, c_c2, c_c3 = st.columns(3)
    with c_c1:
        st.markdown(r"**Captor Type ($\theta_K$)**")
        cov_perp = st.selectbox("Perpetrator", ["FARC", "ELN", "PAR", "DC"], index=0, key="cov_perp_selectbox")
    with c_c2:
        st.markdown(r"**Victim Profile ($\theta_V$)**")
        cov_vict = st.selectbox("Victim profile", ["Private", "Public sector"], index=0, key="cov_vict_selectbox")
    with c_c3:
        st.markdown(r"**Family Wealth ($\theta_F$)**")
        cov_wealth = st.selectbox("Family Wealth", ["Standard", "High"], index=0, key="cov_wealth_selectbox")

    c_c4, c_c5 = st.columns(2)
    with c_c4:
        st.markdown("**Geographic Zone ($z$)**")
        cov_zone = st.selectbox("Geographic zone", ["Metropolis", "Andean", "Caribbean", "Pacific / Red Zone", "Eastern Plains/Jungle"], index=0, key="cov_zone_selectbox")
    with c_c5:
        st.markdown(r"**State Type ($\theta_S$)**")
        cov_state = st.selectbox("State type", ["Strict", "Lax"], index=0, key="cov_state_selectbox")

    st.markdown(r"##### 🎭 Executed Actions ($\tilde{a}_t$) Drawn by Simulator:")
    c_c6, c_c7, c_c8 = st.columns(3)
    with c_c6:
        st.markdown("**Executed Family Action ($\\tilde{a}^F$)**")
        st.info(st.session_state["act_f"])
    with c_c7:
        st.markdown("**Executed Captor Action ($\\tilde{a}^K$)**")
        st.info(st.session_state["act_k"])
    with c_c8:
        st.markdown("**Executed State Action ($\\tilde{a}^S$)**")
        st.info(st.session_state["act_s"])

    st.markdown("##### 🔧 Policy Instruments & Baseline Parameters:")
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        st.markdown(r"**$T_{\text{mad}}$ (days)**")
        T_mad = st.slider("T_mad", 1.0, 30.0, 5.0, step=1.0, key="T_mad_slider", label_visibility="collapsed")
    with c_p2:
        st.markdown(r"**$\alpha^*$ (Interdiction)**")
        alpha_val = st.slider("alpha", 0.0, 1.0, 0.20, step=0.05, key="alpha_slider", label_visibility="collapsed")
    with c_p3:
        st.markdown(r"**$\gamma^*$ (Pressure)**")
        gamma_val = st.slider("gamma", 0.0, 1.0, 0.90, step=0.05, key="gamma_slider", label_visibility="collapsed")

    c_p4, _c_p5, _c_p6 = st.columns(3)
    with c_p4:
        st.markdown(r"**$R$ (Ransom Value, millions COP)**")
        ransom_R_millions = st.slider("R (millions COP)", 1.0, 100.0, 20.0, step=1.0, key="ransom_R_slider", label_visibility="collapsed")
        ransom_R = float(ransom_R_millions) * 1_000_000.0

    st.markdown(r"**$\tilde\beta(\theta_K)$ (Discount Factor, by Captor Type)**")
    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    with c_b1:
        st.markdown("DC")
        beta_tilde_dc = st.slider("beta_tilde_DC", 0.50, 0.99, 0.92, step=0.01, key="beta_tilde_dc_slider", label_visibility="collapsed")
    with c_b2:
        st.markdown("PAR")
        beta_tilde_par = st.slider("beta_tilde_PAR", 0.50, 0.99, 0.92, step=0.01, key="beta_tilde_par_slider", label_visibility="collapsed")
    with c_b3:
        st.markdown("ELN")
        beta_tilde_eln = st.slider("beta_tilde_ELN", 0.50, 0.99, 0.92, step=0.01, key="beta_tilde_eln_slider", label_visibility="collapsed")
    with c_b4:
        st.markdown("FARC")
        beta_tilde_farc = st.slider("beta_tilde_FARC", 0.50, 0.99, 0.92, step=0.01, key="beta_tilde_farc_slider", label_visibility="collapsed")
    BETA_TILDE_TAB1 = {"DC": beta_tilde_dc, "PAR": beta_tilde_par, "ELN": beta_tilde_eln, "FARC": beta_tilde_farc}

    # Computations for Detección p_det
    eta_0 = ETA_0_PDET[cov_perp]
    u_det = eta_0 + ETA_1_PDET * alpha_val + ETA_2_PDET * gamma_val
    p_det = 1.0 / (1.0 + np.exp(-u_det))

    # Load baseline zetas for perpetrator
    zetas_p = ZETAS_POLITICA[cov_perp]
    za = zetas_p["zeta_alpha"]
    zg = zetas_p["zeta_gamma"]
    zd = 0.18 # default detection sensitivity from app.py
    
    # 1. Pago Intensity lambda_1
    beta_K_1 = BETAS_K[cov_perp]["Pago"]
    beta_z_1 = BETAS_Z[cov_zone]
    beta_F_1 = 0.80 if cov_wealth == "High" else 0.00
    beta_V_1 = 1.36 if cov_vict == "Public sector" else 0.00
    beta_S_1 = 0.50 if cov_state == "Lax" else 0.00
    phi_F_1 = 3.20 if st.session_state["act_f"] == "Pay" else 0.00
    phi_K_1 = -1.15 if st.session_state["act_k"] == "Continue" else 0.00
    
    idx_pago = (beta_K_1 + beta_z_1 + beta_F_1 - beta_V_1 + beta_S_1 
                 - za * alpha_val - zg * gamma_val - zd * p_det 
                 + phi_F_1 + phi_K_1)
    lambda_pay_raw = LAMBDAS_0["Pago"] * np.exp(idx_pago)
    
    # 2. Muerte Intensity lambda_2
    beta_K_2 = BETAS_K[cov_perp]["Muerte"]
    beta_z_2 = BETAS_Z[cov_zone]
    beta_S_2 = 0.50 if cov_state == "Lax" else 0.00
    phi_F_2 = -1.50 if st.session_state["act_f"] == "Pay" else 0.00
    phi_K_kill_2 = 4.00 if st.session_state["act_k"] == "Kill" else 0.00
    phi_K_cont_2 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00
    
    idx_muerte = (beta_K_2 + beta_z_2 + beta_S_2 
                  + za * alpha_val + zg * gamma_val - zd * p_det 
                  - phi_F_2 + phi_K_kill_2 + phi_K_cont_2)
    lambda_kill_raw = LAMBDAS_0["Muerte"] * np.exp(idx_muerte)
    
    # 3. Rescate Intensity lambda_3
    beta_K_3 = BETAS_K[cov_perp]["Rescate"]
    beta_z_3 = BETAS_Z[cov_zone]
    beta_S_3 = 0.50 if cov_state == "Lax" else 0.00
    zeta_R_3 = 2.50 if st.session_state["act_s"] == "Rescue" else 0.00
    phi_F_3 = -1.00 if st.session_state["act_f"] == "Pay" else 0.00
    phi_K_3 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00
    
    idx_rescate = (- beta_S_3 + beta_K_3 + beta_z_3 
                   + za * alpha_val + zg * gamma_val + zd * p_det 
                   + zeta_R_3 - phi_F_3 + phi_K_3)
    lambda_res_raw = LAMBDAS_0["Rescate"] * np.exp(idx_rescate)

    # Maturation multiplier M_t
    M_t = float(min(1.0, (t_days / max(1e-9, T_mad)) ** 2))
    
    # Effective intensities
    lambda_pay_eff = M_t * lambda_pay_raw
    lambda_kill_eff = M_t * lambda_kill_raw
    lambda_res_eff = M_t * lambda_res_raw

    # Detailed view: equations, baseline parameters by captor type, and the
    # substituted numeric computation for the currently selected covariates
    with st.expander(r"📐 View each $\lambda_j$: equations, parameters & by-captor-type comparison"):
        st.markdown("**Structural log-linear index equations (Section 4.2.3):**")
        st.latex(r"\text{idx}_{\text{Ransom}} = \beta_{K,1} + \beta_z + \beta_F\,\mathbb{1}\{\theta_F=\text{High}\} - \beta_V\,\mathbb{1}\{\theta_V=\text{Public}\} + \beta_S\,\mathbb{1}\{\theta_S=\text{Lax}\} - \zeta_\alpha \alpha^* - \zeta_\gamma \gamma^* - \zeta_d\, p_{\text{det}} + \phi_F^{\text{Ransom}} + \phi_K^{\text{Ransom}}")
        st.latex(r"\text{idx}_{\text{Death}} = \beta_{K,2} + \beta_z + \beta_S\,\mathbb{1}\{\theta_S=\text{Lax}\} + \zeta_\alpha \alpha^* + \zeta_\gamma \gamma^* - \zeta_d\, p_{\text{det}} - \phi_F^{\text{Death}} + \phi_K^{\text{Kill}} + \phi_K^{\text{Cont}}")
        st.latex(r"\text{idx}_{\text{Rescue}} = -\beta_S\,\mathbb{1}\{\theta_S=\text{Lax}\} + \beta_{K,3} + \beta_z + \zeta_\alpha \alpha^* + \zeta_\gamma \gamma^* + \zeta_d\, p_{\text{det}} + \zeta_R\,\mathbb{1}\{\tilde{a}^S=\text{Rescue}\} - \phi_F^{\text{Rescue}} + \phi_K^{\text{Cont}}")
        st.latex(r"\lambda_j(\mathcal{C}_t) = \lambda_{j0} \cdot \exp\bigl(\text{idx}_j\bigr)")

        st.markdown(r"**Baseline coefficients by captor type ($\beta_{K,j}$, Section 4.2.3):**")
        df_betas_k = pd.DataFrame(BETAS_K).T[["Pago", "Muerte", "Rescate"]].rename(columns={"Pago": "Ransom", "Muerte": "Death", "Rescate": "Rescue"})
        st.dataframe(df_betas_k, use_container_width=True)

        st.markdown(r"**Policy-sensitivity coefficients by captor type ($\zeta_\alpha, \zeta_\gamma$):**")
        df_zetas = pd.DataFrame(ZETAS_POLITICA).T.rename(columns={"zeta_alpha": "ζ_α", "zeta_gamma": "ζ_γ"})
        st.dataframe(df_zetas, use_container_width=True)

        st.markdown(f"**Substituted computation for the current selection ({cov_perp}, {cov_zone}):**")
        st.markdown(
            f"""
            *   $\\lambda_{{1,0}}$ (Ransom) = {LAMBDAS_0['Pago']:.4f}, $\\beta_{{K,1}}$ = {beta_K_1:.2f}, $\\text{{idx}}_{{\\text{{Ransom}}}}$ = {idx_pago:.3f} → $\\lambda_1$ = **{lambda_pay_raw:.5f}**
            *   $\\lambda_{{2,0}}$ (Death) = {LAMBDAS_0['Muerte']:.4f}, $\\beta_{{K,2}}$ = {beta_K_2:.2f}, $\\text{{idx}}_{{\\text{{Death}}}}$ = {idx_muerte:.3f} → $\\lambda_2$ = **{lambda_kill_raw:.5f}**
            *   $\\lambda_{{3,0}}$ (Rescue) = {LAMBDAS_0['Rescate']:.4f}, $\\beta_{{K,3}}$ = {beta_K_3:.2f}, $\\text{{idx}}_{{\\text{{Rescue}}}}$ = {idx_rescate:.3f} → $\\lambda_3$ = **{lambda_res_raw:.5f}**
            *   Maturation filter $M(t)$ = {M_t:.4f} scales all three into the effective $\\tilde{{\\lambda}}_j$ shown in the chart below.
            """
        )

    # Exogenous rate lambda_4
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.markdown(r"**$\lambda_4$ (Exogenous)**")
        lambda_4 = st.slider("lambda_4", 0.0001, 0.0100, 0.0005, step=0.0001, format="%.4f", key="lambda_4_slider", label_visibility="collapsed")
    with c_e2:
        st.markdown("**Structural Intensities Summary:**")
        st.markdown(f"*   **Maturity Multiplier $M(t)$:** {M_t:.4f}\n*   **Detection Prob ($p_{{\\text{{det}}}}$):** {p_det:.4%}")

    # Plotly bar chart for effective intensities
    fig_int = go.Figure(data=[
        go.Bar(
            x=["Ransom (λ₁)", "Death (λ₂)", "Rescue (λ₃)", "Exogenous Release (λ₄)"],
            y=[lambda_pay_eff, lambda_kill_eff, lambda_res_eff, lambda_4],
            marker_color=["#0284C7", "#E11D48", "#4F46E5", "#F59E0B"],
            text=[f"{lambda_pay_eff:.5f}", f"{lambda_kill_eff:.5f}", f"{lambda_res_eff:.5f}", f"{lambda_4:.5f}"],
            textposition='auto',
        )
    ])
    fig_int.update_layout(
        title=f"Theoretical Structural Risk Intensities (Type: {cov_perp})",
        yaxis=dict(title="Intensity Rate (λⱼ)"),
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=11)
    )
    st.plotly_chart(fig_int, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block C: Survival Dynamics & Outcome Materialization (m_t)
    # ---------------------------------------------------------------------
    st.markdown(r"## 🇨️ Survival Dynamics & Outcome Materialization ($m_t$)")
    st.markdown(
        """
        The dynamic duration depends on aggregate intensities, and the mode of closure 
        is drawn according to the relative hazard shares.
        """
    )

    # 1. Render Equations
    st.latex(r"p_{\text{Cont},t} = \exp\left(-\sum_{j=1}^{4}\tilde{\lambda}_j(t)\right)")
    st.latex(r"q(t) = 1 - p_{\text{Cont},t},\qquad \xi_j(t) = \frac{\tilde{\lambda}_j(t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_{\ell}(t)}")
    st.latex(r"\bar{h}_j(t) = q(t)\xi_j(t)")

    # Computations for Block C
    total_eff_intensity = lambda_pay_eff + lambda_kill_eff + lambda_res_eff + lambda_4
    p_cont = float(np.exp(-total_eff_intensity))
    q_t = 1.0 - p_cont

    if total_eff_intensity > 1e-12:
        xi_1 = lambda_pay_eff / total_eff_intensity
        xi_2 = lambda_kill_eff / total_eff_intensity
        xi_3 = lambda_res_eff / total_eff_intensity
        xi_4 = lambda_4 / total_eff_intensity
    else:
        xi_1, xi_2, xi_3, xi_4 = 0.25, 0.25, 0.25, 0.25

    h_1 = q_t * xi_1
    h_2 = q_t * xi_2
    h_3 = q_t * xi_3
    h_4 = q_t * xi_4

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Duration Probabilities:**")
        st.markdown(f"*   $p_{{\\mathrm{{Cont}},t}}$ (Continuation) = **{p_cont:.5f}**\n*   $q(t)$ (Daily Closure) = **{q_t:.5f}**")
    with c2:
        st.markdown(r"**Relative Risk Shares ($\xi_j$):**")
        st.markdown(f"*   $\\xi_1$ (Ransom Share) = **{xi_1:.3f}**\n*   $\\xi_2$ (Death Share) = **{xi_2:.3f}**\n*   $\\xi_3$ (Rescue Share) = **{xi_3:.3f}**\n*   $\\xi_4$ (Exogenous Share) = **{xi_4:.3f}**")
    with c3:
        st.markdown(r"**Daily Marginal Hazards ($h_j$):**")
        st.markdown(f"*   $h_1$ (Ransom Hazard) = **{h_1:.5f}**\n*   $h_2$ (Death Hazard) = **{h_2:.5f}**\n*   $h_3$ (Rescue Hazard) = **{h_3:.5f}**\n*   $h_4$ (Exogenous Hazard) = **{h_4:.5f}**")

    # Plotly Horizontal Stacked Partition Chart representing Inverse Transform interval
    # Cumulative bounds of each segment along the G_t partition of [0, 1)
    bound_0 = 0.0
    bound_cont = p_cont
    bound_h1 = bound_cont + h_1
    bound_h2 = bound_h1 + h_2
    bound_h3 = bound_h2 + h_3
    bound_h4 = 1.0

    fig_partition = go.Figure()
    fig_partition.add_trace(go.Bar(
        y=["Interval Partition"], x=[p_cont], name="Continue", orientation='h',
        marker=dict(color="#94A3B8"), text=[f"{p_cont:.2%}"], textposition='inside',
        customdata=[[bound_0, bound_cont]],
        hovertemplate="<b>Continue</b><br>Interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<extra></extra>"
    ))
    fig_partition.add_trace(go.Bar(
        y=["Interval Partition"], x=[h_1], name="Ransom (j=1)", orientation='h',
        marker=dict(color="#0284C7"), text=[f"{h_1:.2%}"], textposition='inside',
        customdata=[[bound_cont, bound_h1]],
        hovertemplate="<b>Ransom (j=1)</b><br>Interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<extra></extra>"
    ))
    fig_partition.add_trace(go.Bar(
        y=["Interval Partition"], x=[h_2], name="Death (j=2)", orientation='h',
        marker=dict(color="#E11D48"), text=[f"{h_2:.2%}"], textposition='inside',
        customdata=[[bound_h1, bound_h2]],
        hovertemplate="<b>Death (j=2)</b><br>Interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<extra></extra>"
    ))
    fig_partition.add_trace(go.Bar(
        y=["Interval Partition"], x=[h_3], name="Rescue (j=3)", orientation='h',
        marker=dict(color="#4F46E5"), text=[f"{h_3:.2%}"], textposition='inside',
        customdata=[[bound_h2, bound_h3]],
        hovertemplate="<b>Rescue (j=3)</b><br>Interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<extra></extra>"
    ))
    fig_partition.add_trace(go.Bar(
        y=["Interval Partition"], x=[h_4], name="Exogenous (j=4)", orientation='h',
        marker=dict(color="#F59E0B"), text=[f"{h_4:.2%}"], textposition='inside',
        customdata=[[bound_h3, bound_h4]],
        hovertemplate="<b>Exogenous (j=4)</b><br>Interval: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<extra></extra>"
    ))

    fig_partition.update_layout(
        barmode='stack',
        height=160,
        xaxis=dict(range=[0, 1], title="Inverse Transform Map [0, 1]"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=11)
    )
    st.plotly_chart(fig_partition, use_container_width=True)

    # Live Outcome Materialization Simulator (Inverse Transform Sampling Rule G_t)
    with st.expander(r"🎲 Simulate Competing Risk Realization ($\mathcal{G}_t$ Inverse Transform)"):
        st.markdown(
            "Formal materialization rule (Appendix 3, Definition of the outcome kernel; "
            "Working Paper eq. `m-law-inverse`): draw $v_t\\sim\\text{Unif}(0,1)$, fix the total "
            "order $\\mathcal{M}=\\{\\text{Cont},1,2,3,4\\}$, and set"
        )
        st.latex(
            r"m_t=r \iff \sum_{\ell<r}\mathbb{P}(m_t=\ell\mid\theta_K,\mathcal{C}_t)\;\le\; v_t \;<\;\sum_{\ell\le r}\mathbb{P}(m_t=\ell\mid\theta_K,\mathcal{C}_t)"
        )
        st.markdown("Instantiated with the current $p_{\\text{Cont},t}$ and $\\bar h_j(t)$, the partition of $[0,1)$ used by $\\mathcal{G}_t$ is:")
        st.latex(
            r"\mathcal{G}_t(v_t)=\begin{cases}"
            r"\text{Cont} & v_t\in[0,\,p_{\text{Cont},t})\\"
            r"1 & v_t\in[p_{\text{Cont},t},\,p_{\text{Cont},t}+\bar h_1(t))\\"
            r"2 & v_t\in[p_{\text{Cont},t}+\bar h_1(t),\,p_{\text{Cont},t}+\bar h_1(t)+\bar h_2(t))\\"
            r"3 & v_t\in[\,\cdot\,,\,p_{\text{Cont},t}+\bar h_1(t)+\bar h_2(t)+\bar h_3(t))\\"
            r"4 & v_t\in[\,\cdot\,,\,1)"
            r"\end{cases}"
        )
        if st.button("Draw Physical Outcome"):
            u_draw_m = np.random.default_rng().random()

            # Map draw to interval (concrete instance of G_t above)
            if u_draw_m <= p_cont:
                outcome = "Continue Captivity (cont)"
            elif u_draw_m <= p_cont + h_1:
                outcome = "Ransom Paid (j=1)"
            elif u_draw_m <= p_cont + h_1 + h_2:
                outcome = "Victim Deceased (j=2)"
            elif u_draw_m <= p_cont + h_1 + h_2 + h_3:
                outcome = "Tactical Rescue (j=3)"
            else:
                outcome = "Exogenous Release (j=4)"

            st.session_state["m_tau0_draw"] = float(u_draw_m)
            st.session_state["m_tau0_outcome"] = str(outcome)

            st.info(f"Uniform Draw: $v_t$ = {u_draw_m:.6f} | Materialized Outcome: **{outcome}**")

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block D: Collusion Detection Probability (p_det)
    # ---------------------------------------------------------------------
    st.markdown(r"## 4️⃣ Collusion Detection Probability ($p_{\text{det},t}$)")
    st.markdown(
        """
        The probability that collusion is observed. It increases logistically in response 
        to financial interdiction and pressure.
        """
    )

    # 1. Render Equations
    st.latex(r"p_{\text{det},t}(\theta_K) = \Lambda\left(\eta_0(\theta_K) + \eta_1\alpha_t^* + \eta_2\gamma_t^*\right)")
    st.latex(r"\Lambda(u) = \frac{1}{1 + e^{-u}}")

    st.markdown("##### 🔧 Modifiable Parameters for Block D:")
    # Render Sliders inside the Tab
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown(r"**Interdiction Sensitivity ($\eta_1$)**")
        eta_1 = st.slider("eta_1", 0.5, 3.0, 1.0, step=0.1, key="eta_1_slider", label_visibility="collapsed")
    with c_d2:
        st.markdown(r"**Pressure Sensitivity ($\eta_2$)**")
        eta_2 = st.slider("eta_2", 0.5, 3.0, 1.0, step=0.1, key="eta_2_slider", label_visibility="collapsed")

    # Computations for Block D using modified sliders
    u_det_slider = eta_0 + eta_1 * alpha_val + eta_2 * gamma_val
    p_det_computed = 1.0 / (1.0 + np.exp(-u_det_slider))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detection Settings:**")
        st.markdown(f"*   $\\eta_0({cov_perp})$ (Baseline detectability) = **{eta_0:.2f}**\n*   $\\eta_1$ = **{eta_1:.1f}**, $\\eta_2$ = **{eta_2:.1f}**\n*   $\\alpha^*$ = **{alpha_val:.2f}**, $\\gamma^*$ = **{gamma_val:.2f}**")
    with c2:
        # Render dynamic detection probability using st.metric to guarantee visual accessibility
        st.metric(label="Computed Detection Probability (p_det)", value=f"{p_det_computed:.4%}")

    # Gauge visualization for detection
    fig_det = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = p_det_computed * 100.0,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Detection Probability (%)", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#06B6D4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray"
        }
    ))
    fig_det.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), font=dict(size=11))
    st.plotly_chart(fig_det, use_container_width=True)

    if st.button("Draw Detection Signal"):
        u_draw_d = np.random.default_rng().random()
        d_realized = 1 if u_draw_d <= p_det_computed else 0
        st.session_state["d_tau0_draw"] = float(u_draw_d)
        st.session_state["d_tau0_realized"] = int(d_realized)
        st.session_state["d_tau0_pdet"] = float(p_det_computed)
        st.info(f"Uniform Draw: $v_t$ = {u_draw_d:.6f} | Realized $d_0$ = **{d_realized}**")

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block E: Tactical Rescue Survival Probability (p_surv)
    # ---------------------------------------------------------------------
    st.markdown(r"## 5️⃣ Tactical Rescue Survival Probability ($p_{\text{surv},t}$)")
    st.markdown(
        r"""
        Survival under focal rescue attempts. It is calibrated dynamically by intelligence precision 
        and collapses to base lethality in the event of identification error.
        """
    )

    # 1. Render Equations
    st.latex(r"p_{\text{surv},t}(\iota_t, \hat{\theta}_t, \theta_K) = \Lambda\left(\alpha_{\text{leth}}(\theta_K) + \beta_R \cdot \iota_t \cdot \mathbb{I}_{\{\hat{\theta}_t = \theta_K\}}\right)")

    st.markdown("##### 🔧 Modifiable Parameters for Block E:")
    # Render sliders inside the Tab
    c_e_s1, c_e_s2, c_e_s3 = st.columns(3)
    with c_e_s1:
        st.markdown(r"**Information Precision ($\iota_t$)**")
        iota_t = st.slider("iota_t", 0.25, 1.0, 0.70, step=0.05, key="iota_t_slider", label_visibility="collapsed")
    with c_e_s2:
        st.markdown(r"**Information Productivity ($\beta_R$)**")
        beta_R = st.slider("beta_R", 1.0, 10.0, 7.0, step=0.5, key="beta_R_slider", label_visibility="collapsed")
    with c_e_s3:
        st.markdown("**Captor Identification**")
        correct_id = st.checkbox("Correct Identification", value=True, key="correct_id_checkbox")

    # Computations for Block E using exact app.py default parameters
    alpha_leth_val = ALPHA_LETH[cov_perp]
    indicator_id = 1.0 if correct_id else 0.0
    u_surv = alpha_leth_val + beta_R * iota_t * indicator_id
    p_surv = 1.0 / (1.0 + np.exp(-u_surv))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rescue Parameters:**")
        st.markdown(f"*   $\\alpha_{{\\text{{leth}}}}({cov_perp})$ (Base Lethality) = **{alpha_leth_val:.2f}**\n*   $\\beta_R$ = **{beta_R:.2f}**, $\\iota_t$ = **{iota_t:.2f}**\n*   $\\mathbb{{I}}(\\hat{{\\theta}}_t = \\theta_K)$ = **{indicator_id:.1f}**")
    with c2:
        # Render dynamic survival probability using st.metric to guarantee visual accessibility
        st.metric(label="Computed Survival Probability (p_surv)", value=f"{p_surv:.4%}")

    # Gauge visualization for rescue survival
    fig_surv = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = p_surv * 100.0,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Rescue Survival Probability (%)", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#10B981"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray"
        }
    ))
    fig_surv.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), font=dict(size=11))
    st.plotly_chart(fig_surv, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block F: Capture Probability, Voice Evidence & Effective Probabilities
    # (Bernal_H.tex, eqs. p-cap/p-cap-tilde/voz-descomp/Lvoz/LC; Working Paper
    #  "Effective probabilities" section, eqs. 37-41)
    # ---------------------------------------------------------------------
    st.markdown(r"## 6️⃣ Capture, Voice Evidence & Effective Probabilities")
    st.markdown(
        """
        Completes the probability objects of Bernal_H.tex not yet covered above: the technical
        capture probability, the acoustic/voice evidence block, and the four **effective
        probabilities** that feed the Bellman value functions of the captor and the family
        (Section "Effective probabilities"; Working Paper, eqs. 38-41).
        """
    )

    st.markdown("##### 📡 Technical Capture Probability")
    st.latex(r"p_{\text{cap}}(a,\theta_i,\theta_S,\alpha_t^*,\gamma_t^*) = \Lambda\bigl(\delta_a + c_0(\theta_i) + c_\alpha(\theta_i)\,\alpha_t^* + c_\gamma(\theta_i)\,\gamma_t^* + c_S(\theta_S)\bigr)")

    delta_a_rescue = DELTA_A_CAP["Rescue"]
    delta_a_negotiate = DELTA_A_CAP["Negotiate"]
    c0_i = C0_CAP[cov_perp]
    ca_i = CALPHA_CAP[cov_perp]
    cg_i = CGAMMA_CAP[cov_perp]
    cs_theta = CS_CAP[cov_state]

    u_cap_rescue = delta_a_rescue + c0_i + ca_i * alpha_val + cg_i * gamma_val + cs_theta
    u_cap_negotiate = delta_a_negotiate + c0_i + ca_i * alpha_val + cg_i * gamma_val + cs_theta
    p_cap_rescue = 1.0 / (1.0 + np.exp(-u_cap_rescue))
    p_cap_negotiate = 1.0 / (1.0 + np.exp(-u_cap_negotiate))

    st.markdown(
        f"*   $p_{{\\text{{cap}}}}(\\text{{Rescue}}, {cov_perp}, {cov_state})$ = **{p_cap_rescue:.4f}**\n"
        f"*   $p_{{\\text{{cap}}}}(\\text{{Negotiate}}, {cov_perp}, {cov_state})$ = **{p_cap_negotiate:.4f}**"
    )

    st.markdown(r"##### 🎯 Effective Capture Probability (Working Paper, eq. 38)")
    st.latex(
        r"\tilde{p}_{\text{cap},t}(\theta_i) := \mathbb{E}_{\tilde{a}_t^S \mid \mathcal{Q}_t^{\text{Cap}}}"
        r"\bigl[p_{\text{cap}}(\tilde{a}_t^S,\theta_i,\theta_S,\alpha_t^*,\gamma_t^*)\bigr]"
        r" = \sum_{a\in\mathcal{A}^S}\mathbb{P}_I^S(a\mid a_t^{S*},X_t)\cdot p_{\text{cap}}(a,\theta_i,\theta_S,\alpha_t^*,\gamma_t^*)"
    )
    p_cap_eff = p_rescue * p_cap_rescue + p_nego * p_cap_negotiate
    st.metric(label="Effective Capture Probability (p̃ₐₚ,ₜ)", value=f"{p_cap_eff:.4%}")
    st.caption(
        f"Weighted by the State's MDG-realized action probabilities from Block A: "
        f"$\\mathbb{{P}}(\\tilde a^S=\\text{{Rescue}})$={p_rescue:.4f}, "
        f"$\\mathbb{{P}}(\\tilde a^S=\\text{{Negotiate}})$={p_nego:.4f}."
    )

    st.markdown("---")
    st.markdown("##### 🎙️ Acoustic/Voice Evidence Block")
    st.markdown(
        """
        Third source of Bayesian evidence (alongside implementation and competing risks): a voice
        window, or its absence, is informative about $\\theta_K$ through the acoustic pattern and
        the contact propensity.
        """
    )
    st.info(f"🦹 Captor type ($\\theta_K$) for this block: **{cov_perp}** — same selection as *Perpetrator* in the covariates above.")
    st.latex(r"x_t^{obs} = x_t^{true}(\theta_K) + \varepsilon_L + \varepsilon_S,\qquad \varepsilon_L\sim\mathcal{N}(0,\Sigma_L),\ \varepsilon_S\sim\mathcal{N}(0,\Sigma_S),\ \varepsilon_L\perp\varepsilon_S")
    st.latex(r"\mathcal{L}_{\text{voz},t}(\theta_K) \propto \exp\left(-\frac{1}{2}\sum_{i=1}^{k}\frac{\bigl(x_{t,i}^{obs}-\bar{x}_i(\theta_K)\bigr)^2}{\tilde{\sigma}_i(\theta_K)^2}\right)")
    st.latex(
        r"\mathcal{L}_{C,t}(\theta_K\mid V_t) = \begin{cases}"
        r"\bigl[\mathcal{L}_{\text{voz},t}(\theta_K)\,\pi_{\text{call}}(\theta_K)\bigr]^{\omega_{\text{voz}}}, & V_t=1\\"
        r"\bigl[1-\pi_{\text{call}}(\theta_K)\bigr]^{\omega_{\text{voz}}}, & V_t=0"
        r"\end{cases}"
    )

    c_v1, c_v2, c_v3 = st.columns(3)
    with c_v1:
        st.markdown(r"**Voice Emitted ($V_t$)**")
        voice_emitted = st.checkbox("Voice window observed", value=True, key="voice_emitted_checkbox")
        include_voice_posterior = st.checkbox("Include voice in posterior calculations", value=True, key="include_voice_posterior_checkbox")
    with c_v2:
        st.markdown(r"**Observed Acoustic Feature ($x_t^{obs}$, 1-D illustration)**")
        x_obs = st.slider("x_obs", -3.0, 3.0, 0.3, step=0.1, key="x_obs_slider", label_visibility="collapsed")
    with c_v3:
        st.markdown(r"**Learning Weight ($\omega_{\text{voz}}$)**")
        omega_voz = st.slider("omega_voz", 0.0, 1.0, 0.5, step=0.05, key="omega_voz_slider", label_visibility="collapsed")

    x_true_i = X_TRUE_VOZ[cov_perp]
    sigma_i = SIGMA_VOZ[cov_perp]
    pi_call_i = PI_CALL[cov_perp]

    if include_voice_posterior:
        L_voz = float(np.exp(-0.5 * ((x_obs - x_true_i) ** 2) / (sigma_i ** 2)))
        if voice_emitted:
            L_C = (L_voz * pi_call_i) ** omega_voz
        else:
            L_C = (1.0 - pi_call_i) ** omega_voz
        st.markdown(
            f"*   $\\bar{{x}}({cov_perp})$ = **{x_true_i:.2f}**, $\\tilde{{\\sigma}}({cov_perp})$ = **{sigma_i:.2f}**, "
            f"$\\pi_{{\\text{{call}}}}({cov_perp})$ = **{pi_call_i:.2f}**\n"
            f"*   $\\mathcal{{L}}_{{\\text{{voz}},t}}({cov_perp})$ = **{L_voz:.4f}**"
        )
        st.metric(label="Communication Likelihood (L_C,t)", value=f"{L_C:.4f}")
    else:
        L_voz = 1.0
        L_C = 1.0
        st.markdown(
            f"*   $\\bar{{x}}({cov_perp})$ = **{x_true_i:.2f}**, $\\tilde{{\\sigma}}({cov_perp})$ = **{sigma_i:.2f}**, "
            f"$\\pi_{{\\text{{call}}}}({cov_perp})$ = **{pi_call_i:.2f}** (Ignored in posterior)\n"
            f"*   $\\mathcal{{L}}_{{\\text{{voz}},t}}({cov_perp})$ = **1.0000** (Ignored)"
        )
        st.metric(label="Communication Likelihood (L_C,t)", value="1.0000 (Ignored)")

    st.markdown("---")
    st.markdown(r"##### 📐 Remaining Effective Probabilities (Working Paper, eqs. 39-41)")
    st.latex(r"\tilde{p}_{\text{pay},t}(\theta_i) := \mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\text{Cont}}}\bigl[\mathbb{P}(m_t=\text{pay}\mid \tilde{A}_t, X_t', \theta_K=\theta_i)\bigr]")
    st.latex(r"\tilde{p}_{\text{surv},t}(\theta_i) := \mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\text{Coop}}}\bigl[p_{\text{surv},t}(\iota_t,\hat{\theta}_t,\theta_K=\theta_i)\bigr]")
    st.latex(r"\tilde{p}_{\text{rel},t}(\theta_i) := \mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\text{Col}}}\bigl[\mathbb{P}(m_t=\text{rel}\mid \tilde{A}_t, R, \theta_K=\theta_i)\bigr]")

    st.caption(
        "Simplified, transparent instantiation reusing quantities already computed in Blocks A-C-E "
        "(not a full re-derivation of the conditioning tuple $\\mathcal{Q}_t^b$): $\\tilde p_{\\text{pay},t}$ "
        "fixes the captor's branch action at *Continue* and averages the ransom hazard over the family's "
        "MDG-realized action (Cooperate/Pay); $\\tilde p_{\\text{surv},t}$ weights $p_{\\text{surv},t}$ "
        "(Block E) by the State's realized rescue probability; $\\tilde p_{\\text{rel},t}$ weights the "
        "exogenous-release hazard $\\bar h_4(t)$ (Block C) by the family's realized collude probability."
    )

    # p_pay: branch fixes act_k = "Continue"; average over the family's MDG action (Cooperate/Pay)
    phi_K_1_cont_branch = -1.15
    idx_pago_coop = (beta_K_1 + beta_z_1 + beta_F_1 - beta_V_1 + beta_S_1
                      - za * alpha_val - zg * gamma_val - zd * p_det
                      + 0.00 + phi_K_1_cont_branch)
    idx_pago_pay = (beta_K_1 + beta_z_1 + beta_F_1 - beta_V_1 + beta_S_1
                     - za * alpha_val - zg * gamma_val - zd * p_det
                     + 3.20 + phi_K_1_cont_branch)
    lambda_pay_coop_raw = LAMBDAS_0["Pago"] * np.exp(idx_pago_coop)
    lambda_pay_pay_raw = LAMBDAS_0["Pago"] * np.exp(idx_pago_pay)
    lambda_pay_coop_eff = M_t * lambda_pay_coop_raw
    lambda_pay_pay_eff = M_t * lambda_pay_pay_raw
    h1_coop = q_t * (lambda_pay_coop_eff / total_eff_intensity) if total_eff_intensity > 1e-12 else 0.25
    h1_pay = q_t * (lambda_pay_pay_eff / total_eff_intensity) if total_eff_intensity > 1e-12 else 0.25
    p_pay_eff = p_coop * h1_coop + p_col * h1_pay

    # p_surv: survival only at risk if the State's realized action is Rescue
    p_surv_eff = p_rescue * p_surv

    # p_rel: exogenous-release hazard weighted by the family's realized Collude probability
    p_rel_eff = p_col * h_4

    c_eff1, c_eff2, c_eff3 = st.columns(3)
    with c_eff1:
        st.metric(label="p̃ₚₐᵧ,ₜ (Effective Payment)", value=f"{p_pay_eff:.4%}")
    with c_eff2:
        st.metric(label="p̃ₛᵤᵣᵥ,ₜ (Effective Survival)", value=f"{p_surv_eff:.4%}")
    with c_eff3:
        st.metric(label="p̃ᵣₑₗ,ₜ (Effective Release)", value=f"{p_rel_eff:.4%}")

    fig_eff = go.Figure(data=[
        go.Bar(
            x=["Capture", "Payment", "Survival", "Release"],
            y=[p_cap_eff, p_pay_eff, p_surv_eff, p_rel_eff],
            marker_color=["#7C3AED", "#0284C7", "#10B981", "#F59E0B"],
            text=[f"{p_cap_eff:.2%}", f"{p_pay_eff:.2%}", f"{p_surv_eff:.2%}", f"{p_rel_eff:.2%}"],
            textposition='auto',
        )
    ])
    fig_eff.update_layout(
        title="Summary of the Four Effective Probabilities",
        yaxis=dict(title="Probability", range=[0, 1]),
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=11)
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Block G: Initial Bayesian Prior mu_0 over Captor Type
    # (Bernal_H.tex: mu_0(theta | z, theta_V); calibration mu_0^ELN=(.25,.35,.05,.35),
    #  mu_0^PAR=(.55,.05,.35,.05))
    # ---------------------------------------------------------------------
    st.markdown(r"## 7️⃣ Initial Bayesian Prior ($\mu_0$) over Captor Type")
    st.markdown(
        """
        Set the initial belief $\\mu_0(\\theta_K \\mid z,\\theta_V)\\in\\Delta(\\Theta_K)$ over the four
        captor types. Enter the priors for DC, PAR, and ELN manually (each constrained to $[0,1]$);
        the FARC prior is computed automatically so that the four probabilities sum to 1, exactly as
        in the calibration of Bernal_H.tex (e.g. $\\mu_0^{\\text{ELN}}=(0.25,0.35,0.05,0.35)$).
        """
    )
    st.latex(r"\mu_0(\theta \mid z,\theta_V) = \mathbb{P}(\theta_K=\theta \mid z,\theta_V) \in \Delta(\Theta_K), \qquad \sum_{\theta\in\Theta_K}\mu_0(\theta)=1")

    c_mu1, c_mu2, c_mu3 = st.columns(3)
    with c_mu1:
        st.markdown(r"**$\mu_0(\text{DC})$**")
        mu0_dc = st.number_input("mu0_dc", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f", key="mu0_dc_input", label_visibility="collapsed")
    with c_mu2:
        st.markdown(r"**$\mu_0(\text{PAR})$**")
        mu0_par = st.number_input("mu0_par", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f", key="mu0_par_input", label_visibility="collapsed")
    with c_mu3:
        st.markdown(r"**$\mu_0(\text{ELN})$**")
        mu0_eln = st.number_input("mu0_eln", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f", key="mu0_eln_input", label_visibility="collapsed")

    # FARC is computed automatically so the four priors sum to 1
    mu0_sum_first3 = mu0_dc + mu0_par + mu0_eln
    mu0_farc = 1.0 - mu0_sum_first3

    if mu0_farc < 0:
        st.error(
            f"⚠️ $\\mu_0(\\text{{DC}})+\\mu_0(\\text{{PAR}})+\\mu_0(\\text{{ELN}})$ = {mu0_sum_first3:.4f} > 1. "
            "Reduce one of the three inputs so that the automatic FARC prior stays non-negative."
        )
        mu0_farc = 0.0
    else:
        st.success(
            f"$\\mu_0(\\text{{FARC}}) = 1-({mu0_dc:.2f}+{mu0_par:.2f}+{mu0_eln:.2f})=$ **{mu0_farc:.4f}** (computed automatically)"
        )

    c_mu4, c_mu5 = st.columns([1, 2])
    with c_mu4:
        st.metric(label="μ₀(FARC) — automatic", value=f"{mu0_farc:.4f}")
    with c_mu5:
        st.markdown(
            f"*   $\\mu_0(\\text{{DC}})$ = **{mu0_dc:.4f}**\n"
            f"*   $\\mu_0(\\text{{PAR}})$ = **{mu0_par:.4f}**\n"
            f"*   $\\mu_0(\\text{{ELN}})$ = **{mu0_eln:.4f}**\n"
            f"*   $\\mu_0(\\text{{FARC}})$ = **{mu0_farc:.4f}**\n"
            f"*   $\\sum_\\theta \\mu_0(\\theta)$ = **{(mu0_dc + mu0_par + mu0_eln + mu0_farc):.4f}**"
        )

    fig_mu0 = go.Figure(data=[go.Bar(
        x=["DC", "PAR", "ELN", "FARC"],
        y=[mu0_dc, mu0_par, mu0_eln, mu0_farc],
        marker_color=["#64748B", "#0284C7", "#AB63FA", "#E11D48"],
        text=[f"{v:.2%}" for v in [mu0_dc, mu0_par, mu0_eln, mu0_farc]],
        textposition='auto',
    )])
    fig_mu0.update_layout(
        title="Initial Prior μ₀(θₖ) over Captor Type",
        yaxis=dict(title="Probability", range=[0, 1]),
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=11)
    )
    st.plotly_chart(fig_mu0, use_container_width=True)

with tab2:
    st.markdown("## Rational Behavior and Best Responses")
    st.markdown(
        "This tab presents **only the equations** that define each player's optimization "
        "problem at a generic decision period $\\tau$, taking Tab 1 as the state at $\\tau=0$ "
        "(latent beliefs $\\mu_\\tau$, the captor's true type $\\theta_K$ hidden from the Family "
        "and the State). **No computation or solving is performed here** — this is a pure "
        "presentation of the theoretical structure, following Bernal_H.tex, "
        "*\"Rational behavior and best responses\"* (§ subsec:bellman), and supported by the "
        "explicit functional forms in Working_paper_eng.tex."
    )

    st.markdown("---")

    # -----------------------------------------------------------------
    # A. CAPTOR'S PROBLEM
    # -----------------------------------------------------------------
    st.markdown("### A. Captor's Problem $(\\theta_K)$")
    st.markdown(
        "At each $\\tau$, the Captor of true type $\\theta_K$ compares three continuation "
        "values — releasing, killing, or continuing the negotiation — and chooses the one "
        "with highest value (Bellman equation):"
    )
    st.latex(r"""
    V^K_\tau(\theta_K) \;=\; \max\Big\{\, U^K_{rel,\tau}(\theta_K),\;\; U^K_{kill,\tau}(\theta_K,\theta_S),\;\; V^K_{cont,\tau}(\theta_K) \,\Big\}
    """)

    st.markdown("**Release payoff:**")
    st.latex(r"""
    U^K_{rel}(\theta_K) \;=\; -\,\kappa_{rel}(\theta_K)
    """)

    st.markdown("**Kill payoff** (net of the effective capture risk $\\tilde p_{cap,\\tau}$ and its associated penalty $F_{cap}$):")
    st.latex(r"""
    U^K_{kill}(\theta_K,\theta_S) \;=\; \big(1-\tilde p_{cap,\tau}(\theta_K)\big)\,\eta(\theta_K) \;-\; \tilde p_{cap,\tau}(\theta_K)\, F_{cap}(\theta_K,\theta_S)
    """)

    st.markdown(
        "**Continuation value** — current flow (expected ransom net of the State's blocking "
        "instrument $\\alpha_\\tau^{*}$, less the institutional cost $C_\\tau$ and the expected "
        "capture penalty) plus the type-discounted future value, discounted only by the "
        "probability of **not** being captured this period:"
    )
    st.latex(r"""
    V^K_{cont,\tau}(\theta_K) \;=\; \tilde p_{pay,\tau}(\theta_K)\, R\,(1-\alpha_\tau^{*}) \;-\; C_\tau(\gamma_\tau^{*},\theta_K) \;-\; \tilde p_{cap,\tau}(\theta_K)\, F_{cap}(\theta_K,\theta_S) \;+\; \tilde\beta(\theta_K)\big(1-\tilde p_{cap,\tau}(\theta_K)\big)\, \mathbb{E}\big[\, V^K_{\tau+1}\big(\mu_{\tau+1}(a_{cont})\big) \,\big]
    """)

    st.markdown(
        "with the institutional / operational cost of maintaining the kidnapping under the "
        "State's instruments $(\\alpha_\\tau,\\gamma_\\tau)$ made explicit in Working_paper_eng.tex "
        "as a convex-exponential form (eq. cost-function-kidnapper):"
    )
    st.latex(r"""
    C_\tau(\gamma_\tau,\theta_K) \;=\; \phi(\theta_K)\,\exp\big(\kappa_c(\theta_K)\,\gamma_\tau\big) \;+\; \nu(\theta_K)
    """)

    st.markdown("**Captor's compact best-response (argmax form):**")
    st.latex(r"""
    a_K^{*}(\theta_K) \;=\; \arg\max_{a_K \in \{Release,\,Kill,\,Continue\}} \; U^K_{a_K,\tau}(\theta_K)
    """)

    st.markdown("**Numeric calibration** (baseline, $\\theta_S=$ Strict — verified against `app.py`, `rational_behavior.py`):")
    _captor_rows = [
        (r"$\kappa_{rel}$ (release disutility)",            2.367, 14.273, 4.163, 1.471),
        (r"$\eta$ (reputational kill benefit)",              0.340,  3.348, 2.360, -0.580),
        (r"$F_{cap}$ (capture penalty, $\theta_S=$Strict)", 40.704, 85.224, 55.968, 29.256),
        (r"$\phi$ (cost scale, $C_\tau$)",                  33.00,  35.00,  38.00,  40.00),
        (r"$\kappa_c$ (cost curvature, $C_\tau$)",           2.61,   2.63,   2.70,   2.70),
        (r"$\nu$ (cost shift, $C_\tau$)",                    0.750,  0.500,  0.250,  0.000),
    ]
    _captor_table_md = "| Coefficient | DC | PAR | ELN | FARC |\n|---|---|---|---|---|\n" + "\n".join(
        r"| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(*row) for row in _captor_rows
    )
    _captor_table_md += "\n| $\\tilde\\beta$ (discount factor; Tab 1 slider, per type) | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
        BETA_TILDE_TAB1["DC"], BETA_TILDE_TAB1["PAR"], BETA_TILDE_TAB1["ELN"], BETA_TILDE_TAB1["FARC"]
    )
    _r_fmt_tab2 = "{:,.0f}".format(ransom_R)
    _captor_table_md += "\n| $R$ (ransom value, COP; Tab 1 slider) | {r} | {r} | {r} | {r} |".format(r=_r_fmt_tab2)
    st.markdown(_captor_table_md)
    st.caption("φ, κ_c, ν: canonical Table 12/15 calibration (app.py `_TAB15_FIXED_COST_COEFFS`). κ_rel, η, F_cap: baseline defaults from `derive_kidnapper_structural_params`, executed and verified directly. β̃ is now type-specific and editable live in Tab 1 (Block B, default 0.92 for all types, matching the previous global constant). R is common to all types (app.py: \"R common to all types\") and is Tab 1's live slider value (default 20,000,000 COP, matching app.py's `_R_default`) — it updates if you change the slider in Tab 1.")

    st.markdown("---")

    # -----------------------------------------------------------------
    # B. FAMILY'S PROBLEM
    # -----------------------------------------------------------------
    st.markdown("### B. Family's Problem $(\\theta_F)$")
    st.markdown(
        "The Family, holding belief $\\mu_\\tau$ over the Captor's type (never observing "
        "$\\theta_K$ directly), chooses between **Cooperate** (pay through official channels) "
        "and **Collude** (private/parallel payment), each carrying its own institutional cost "
        "and expected continuation value:"
    )

    st.markdown(
        "**Cooperate payoff** — matches Bernal_H.tex `eq:f-coop` exactly (this is the formula "
        "`family_utilities(...)` in its own `family_optimization.py` computes for $a_F^{1,*}$ "
        "in Table 5.2, and the *same* formula the State's own $\\Gamma_\\tau(\\mu_\\tau)$ gate "
        "uses internally for $IR^F$ — unified, no separate simplified version):"
    )
    st.latex(r"""
    U^F_{coop,\tau}(\theta_F) \;=\; \Big(\sum_{\theta\in\Theta_K}\mu_\tau(\theta)\,\tilde p_{surv,\tau}(\theta)\Big)\, V_L \;-\; e_\tau(\gamma_\tau,\theta_F)
    """)

    st.markdown(
        "with the Family's institutional cost of cooperating under State pressure $\\gamma_\\tau$ "
        "given the explicit convex-exponential form (Working_paper_eng.tex, eq. family-institutional-cost):"
    )
    st.latex(r"""
    e_\tau(\gamma_\tau,\theta_F) \;=\; \phi_F(\theta_F)\,\exp\big(\kappa_F(\theta_F)\,\gamma_\tau\big) \;+\; \nu_F(\theta_F)
    """)

    st.markdown(
        "**Collude payoff** (`eq:f-col`) — private payment channel: pays the full ransom $R$ "
        "directly, banking on the (structurally rare) exogenous-release channel "
        "$\\tilde p_{rel,\\tau}$ instead of $\\tilde p_{surv,\\tau}$, and risks the detection "
        "penalty $F_{col}$ instead of the institutional cost $e_\\tau$:"
    )
    st.latex(r"""
    U^F_{col,\tau}(\theta_F) \;=\; \Big(\sum_{\theta\in\Theta_K}\mu_\tau(\theta)\,\tilde p_{rel,\tau}(\theta)\Big)\, V_L \;-\; R \;-\; p_{det,\tau}\, F_{col}
    """)
    st.markdown(
        "$\\tilde p_{surv,\\tau}$ reuses the State's rescue-branch survival probability "
        "(`eq:p-surv`, Block C below), evaluated at $\\iota_\\tau=\\max_\\theta\\mu_\\tau(\\theta)$. "
        "$\\tilde p_{rel,\\tau}$ is the exogenous-release cell ($m_\\tau=4$, rate $\\lambda_4$) of "
        "the same competing-hazards outcome block used throughout — it does **not** depend on "
        "which action the State realizes (the wrapper expectation over $\\tilde A_\\tau$ in the "
        "paper's `eq:p-rel-eff` is trivial, since neither inner formula takes the realized "
        "action as an argument)."
    )

    st.markdown("**Family's compact best-response (argmax form):**")
    st.latex(r"""
    a_F^{*}(\theta_F) \;=\; \arg\max_{a_F \in \{Cooperate,\,Collude\}} \; U^F_{a_F,\tau}(\theta_F)
    """)

    st.markdown("**Numeric calibration**:")
    _family_rows = [
        (r"$\phi_F$ (cost scale, $e_\tau$)",      0.0600, 0.0600),
        (r"$\kappa_F$ (cost curvature, $e_\tau$)", 3.6000, 3.0000),
        (r"$\nu_F$ (cost shift, $e_\tau$)",        0.0200, 0.0200),
    ]
    _family_table_md = "| Coefficient | High wealth | Low wealth |\n|---|---|---|\n" + "\n".join(
        r"| {} | {:.4f} | {:.4f} |".format(*row) for row in _family_rows
    )
    _family_table_md += "\n| $V_L$ (value of life, common to all types) | 200.00 | 200.00 |"
    _family_table_md += "\n| $F_{col}$ (collusion-detection penalty, common) | 20.00 | 20.00 |"
    st.markdown(_family_table_md)
    st.caption(
        "φ_F, κ_F, ν_F recalibrated (down from app.py's `_rb_family_phi_kappa_nu`, which gave "
        "φ_F=1.8, ν_F=1.2 — see history). $V_L=200$ (same order of magnitude as the State's "
        "$\\omega_k$) and $F_{col}=20$ (comparable to typical $R$) are new — $\\varrho$ (collusion "
        "risk premium) and the $\\tilde p_{pay}$-based formula previously shown here were an "
        "ad-hoc stand-in with no basis in Bernal_H.tex/Working_paper_eng.tex; both are now "
        "removed, replaced by the exact `eq:f-coop`/`eq:f-col` above. **Calibration finding** "
        "(sondeo of 4000 random scenarios): $\\tilde p_{rel,\\tau}\\sim\\lambda_4/\\text{total}"
        "\\approx0.0005$ is structurally tiny next to $R$ (typically 1–100 millions COP), so "
        "Cooperate weakly dominates Collude in essentially 100% of sampled scenarios for any "
        "$V_L,F_{col}$ in a economically reasonable range — paying the full ransom directly "
        "rarely beats a small institutional cost. This mirrors the already-documented finding "
        "that Release dominates Continue for the Captor: a genuine structural result of this "
        "calibration, not a bug."
    )

    st.markdown("---")

    # -----------------------------------------------------------------
    # C. STATE'S PROBLEM
    # -----------------------------------------------------------------
    st.markdown("### C. State's Problem $(\\theta_S)$")
    st.markdown(
        "The State does **not** observe $\\theta_K$ either — it only holds the posterior belief "
        "$\\mu_\\tau$ over captor types, and must choose the discrete action $a_\\tau^S$ "
        "(Rescue / Negotiate) jointly with the continuous instruments $(\\alpha_\\tau,\\gamma_\\tau)$ "
        "— financial blocking and operational pressure — to minimize an expected social loss "
        "that combines human risk, economic transfer, operational cost, and the value of learning."
    )

    st.markdown("**Conditional loss by branch** (eq. state-loss):")
    st.latex(r"""
    L_\tau(a_\tau^S,\alpha_\tau,\gamma_\tau,\theta_K,\iota_\tau) \;=\;
    \begin{cases}
    V_\tau^R(\iota_\tau,\hat\theta_\tau,\theta_K,\alpha_\tau,\gamma_\tau), & \text{if } a_\tau^S = \text{Rescue},\\[4pt]
    V_\tau^N(\theta_K,\alpha_\tau,\gamma_\tau), & \text{if } a_\tau^S = \text{Negotiate},
    \end{cases}
    """)

    st.markdown(
        "where, under **tactical rescue**, the State bears the cost of a lethal failure of the "
        "operation and the deployment cost; and under **negotiation**, it bears the net criminal "
        "transfer, the lethality of captivity, and the cost of maintaining the containment ring:"
    )
    st.latex(r"""
    V_\tau^R \;:=\; \omega_k\big[1-\tilde p_{surv,\tau}(\iota_\tau,\hat\theta_\tau,\theta_K)\big] \;+\; C_{ops}(\gamma_\tau,\alpha_\tau;\theta_K)
    """)
    st.latex(r"""
    V_\tau^N \;:=\; \omega_p\, R\,(1-\alpha_\tau) \;+\; \omega_k\, \bar h_2(\tau\mid\theta_K,\mathcal{C}_\tau) \;+\; C_{maint}(\gamma_\tau,\alpha_\tau;\theta_K)
    """)

    st.markdown(
        "with the operational cost $C_{ops}$ (rescue branch) and maintenance cost $C_{maint}$ "
        "(negotiation branch) given the explicit quadratic, continuous, and convex forms of "
        "Working_paper_eng.tex (eqs. cops-quadratic / cmaint-quadratic):"
    )
    st.latex(r"""
    C_{ops}(\gamma_\tau,\alpha_\tau;\theta_K) \;=\; c_0(\theta_K) + c_1(\theta_K)\gamma_\tau + \tfrac{c_2(\theta_K)}{2}\gamma_\tau^2 + c_3(\theta_K)\alpha_\tau + \tfrac{c_4(\theta_K)}{2}\alpha_\tau^2 + c_5(\theta_K)\gamma_\tau\alpha_\tau
    """)
    st.latex(r"""
    C_{maint}(\gamma_\tau,\alpha_\tau;\theta_K) \;=\; m_0(\theta_K) + m_1(\theta_K)\gamma_\tau + \tfrac{m_2(\theta_K)}{2}\gamma_\tau^2 + m_3(\theta_K)\alpha_\tau + \tfrac{m_4(\theta_K)}{2}\alpha_\tau^2 + m_5(\theta_K)\gamma_\tau\alpha_\tau
    """)

    st.markdown("**State's dual-control problem** (subject to the feasibility set $\\Gamma_\\tau(\\mu_\\tau)$, which embeds the belief-weighted IC/IR constraints of Block D below; eq. state-dual-control):")
    st.latex(r"""
    \big(a_\tau^{S*},\, \alpha_\tau^{*},\, \gamma_\tau^{*}\big) \;\in\; \arg\min_{(a_\tau^S,\,\alpha_\tau,\,\gamma_\tau)\,\in\,\Gamma_\tau(\mu_\tau)} \;\Big\{\, \sum_{\theta\in\Theta_K}\mu_\tau(\theta)\, L_\tau(a_\tau^S,\alpha_\tau,\gamma_\tau,\theta,\iota_\tau) \;+\; \Pi^S_{\tau,a^S}(\alpha_\tau,\gamma_\tau;\mu_\tau) \;-\; \psi_H\,\Delta H_\tau(\alpha_\tau,\gamma_\tau) \,\Big\}
    """)

    st.markdown(
        "**Deviation premium** $\\Pi^S_{\\tau,b}$ — charges a penalty when the candidate "
        "instruments $(\\alpha_\\tau,\\gamma_\\tau)$ move away from the belief-weighted Bayesian "
        "center of branch $b\\in\\{R,N\\}$, $(\\alpha_{\\tau,b}^{\\mu},\\gamma_{\\tau,b}^{\\mu})$ "
        "(eq. prima-desviacion):"
    )
    st.latex(r"""
    \Pi^S_{\tau,b}(\alpha_\tau,\gamma_\tau;\mu_\tau) \;=\; \chi_\alpha\big(\alpha_\tau-\alpha_{\tau,b}^{\mu}\big)^2 \;+\; \chi_\gamma\big(\gamma_\tau-\gamma_{\tau,b}^{\mu}\big)^2, \qquad \chi_\alpha,\chi_\gamma \ge 0
    """)

    st.markdown("**Entropy-linked exploration term** $\\Delta H_\\tau$ — the expected reduction in the entropy of beliefs induced by the candidate policy, operationalizing active learning (eq. delta-H):")
    st.latex(r"""
    \Delta H_\tau(\alpha_\tau,\gamma_\tau) \;=\; H(\mu_\tau) \;-\; \sum_{m}\sum_{d\in\{0,1\}} \Pr(m_\tau=m,d_\tau=d\mid\mu_\tau,\alpha_\tau,\gamma_\tau,\mathcal{C}_\tau)\, H\big(\mu_{\tau+1}^{m,d}\big)
    """)

    st.markdown(
        "**Branch loss floors** — the continuous instruments are minimized within each branch "
        "first, incorporating the deviation penalty and the informational subsidy (eqs. piso-R / piso-N):"
    )
    st.latex(r"""
    \widetilde V_\tau^{R*}(\iota_\tau,\hat\theta_\tau,\mu_\tau) \;=\; \min_{(\alpha_\tau,\gamma_\tau)\in\Gamma_\tau^R(\mu_\tau)} \Big\{\, \sum_{\theta\in\Theta_K}\mu_\tau(\theta)\,V_\tau^R(\iota_\tau,\hat\theta_\tau,\theta,\alpha_\tau,\gamma_\tau) \;+\; \Pi^S_{\tau,R}(\alpha_\tau,\gamma_\tau;\mu_\tau) \;-\; \psi_H\,\Delta H_\tau(\alpha_\tau,\gamma_\tau) \,\Big\}
    """)
    st.latex(r"""
    \widetilde V_\tau^{N*}(\mu_\tau) \;=\; \min_{(\alpha_\tau,\gamma_\tau)\in\Gamma_\tau^N(\mu_\tau)} \Big\{\, \sum_{\theta\in\Theta_K}\mu_\tau(\theta)\,V_\tau^N(\theta,\alpha_\tau,\gamma_\tau) \;+\; \Pi^S_{\tau,N}(\alpha_\tau,\gamma_\tau;\mu_\tau) \;-\; \psi_H\,\Delta H_\tau(\alpha_\tau,\gamma_\tau) \,\Big\}
    """)

    st.markdown("**State's optimal social cost** — the lower of the two branch floors (eq. state-expected-loss):")
    st.latex(r"""
    \mathcal{L}_\tau^{S*} \;=\; \min\Big\{\, \widetilde V_\tau^{R*}(\iota_\tau,\hat\theta_\tau,\mu_\tau),\;\; \widetilde V_\tau^{N*}(\mu_\tau) \,\Big\}
    """)

    st.markdown("**State's discrete decision rule** — rescue prevails on ties (eq. state-discrete-rule):")
    st.latex(r"""
    \big(a_\tau^{S*},\alpha_\tau^{*},\gamma_\tau^{*}\big) \;=\;
    \begin{cases}
    (\text{Rescue},\,\alpha_{res}^{*},\,\gamma_{res}^{*}), & \text{if } \widetilde V_\tau^{R*} \le \widetilde V_\tau^{N*},\\[4pt]
    (\text{Negotiate},\,\alpha_{neg}^{*},\,\gamma_{neg}^{*}), & \text{if } \widetilde V_\tau^{R*} > \widetilde V_\tau^{N*}.
    \end{cases}
    """)

    st.markdown(
        "**Numeric calibration** — recalibrated **per-type** coefficients (this app's own "
        "baseline, distinct from `app.py`'s raw session defaults). $c_1,c_3$ (resp. "
        "$m_1,m_3$) are set **negative**, with positive quadratic terms, so that "
        "$C_{ops}(\\theta_K)$ and $C_{maint}(\\theta_K)$ each have a genuine **interior** "
        "minimum in $(0,1)^2$ for every $\\theta_K$ — a purely positive-coefficient cost "
        "(as `app.py`'s raw defaults would give) is monotonic in the box and forces the "
        "perfect-info benchmark to the corner $(\\alpha,\\gamma)=(0,0)$, which is not "
        "informative. See Tab 3 for the resulting $(\\alpha^{\\theta_K,*},\\gamma^{\\theta_K,*})$ "
        "per type:"
    )
    _state_cost_rows = [
        (r"$c_0$",                 1.00, 1.00, 1.00, 1.00),
        (r"$c_1$ ($\gamma$ linear)", -0.25, -0.40, -0.55, -0.70),
        (r"$c_2$ ($\gamma$ quadratic, $\div 2$)", 1.00, 1.00, 1.00, 1.00),
        (r"$c_3$ ($\alpha$ linear)", -0.20, -0.35, -0.50, -0.65),
        (r"$c_4$ ($\alpha$ quadratic, $\div 2$)", 1.00, 1.00, 1.00, 1.00),
        (r"$c_5$ ($\alpha\gamma$ cross)", 0.10, 0.10, 0.10, 0.10),
    ]
    _state_cost_table_md = "| $C_{ops}$ Coefficient | DC | PAR | ELN | FARC |\n|---|---|---|---|---|\n" + "\n".join(
        r"| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(*row) for row in _state_cost_rows
    )
    st.markdown(_state_cost_table_md)

    _state_maint_rows = [
        (r"$m_0$",                 3.00, 3.00, 3.00, 3.00),
        (r"$m_1$ ($\gamma$ linear)", -6.00, -6.50, -7.00, -7.50),
        (r"$m_2$ ($\gamma$ quadratic, $\div 2$)", 10.00, 10.00, 10.00, 10.00),
        (r"$m_3$ ($\alpha$ linear)", -0.80, -1.10, -1.40, -1.70),
        (r"$m_4$ ($\alpha$ quadratic, $\div 2$)", 6.00, 6.00, 6.00, 6.00),
        (r"$m_5$ ($\alpha\gamma$ cross)", 0.10, 0.10, 0.10, 0.10),
    ]
    _state_maint_table_md = "| $C_{maint}$ Coefficient | DC | PAR | ELN | FARC |\n|---|---|---|---|---|\n" + "\n".join(
        r"| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(*row) for row in _state_maint_rows
    )
    st.markdown(_state_maint_table_md)

    st.markdown("**Numeric calibration** — loss weights and deviation-penalty coefficients (global, not type-specific):")
    _state_weight_rows = [
        (r"$\omega_p$ (ransom-transfer weight, applied to $R$ in millions COP)", 0.15),
        (r"$\omega_k$ (lethality weight)",     200.0),
        (r"$\chi_\alpha$ ($\alpha$ deviation penalty)", 0.8),
        (r"$\chi_\gamma$ ($\gamma$ deviation penalty)", 0.5),
        (r"$\psi_H$ (informational-subsidy weight, $-\psi_H\Delta H_\tau$; fixed for every $\tau$)", 25.0),
    ]
    _state_weight_table_md = "| Coefficient | Value |\n|---|---|\n" + "\n".join(
        r"| {} | {:.2f} |".format(*row) for row in _state_weight_rows
    )
    st.markdown(_state_weight_table_md)
    st.caption("Rescaled from app.py's raw ω_p=15/ω_k=200,000 (which are calibrated for R in raw pesos, ~2×10⁷) to ω_p=0.15/ω_k=200 paired with R in millions of COP — otherwise the ransom-transfer term mechanically dominates C_ops/C_maint by several orders of magnitude and collapses every type to the same corner solution, regardless of the cost function's shape. ψ_H=25 matches app.py's `t52_entropy_info_weight` default; will be used once Tab 3's τ=1 constrained State optimization (under Γ_t(μ_t)) is rewritten.")

    st.markdown("---")

    # -----------------------------------------------------------------
    # D. INCENTIVE AND PARTICIPATION CONSTRAINTS
    # -----------------------------------------------------------------
    st.markdown("### D. Incentive and Participation Constraints")
    st.markdown(
        "These constraints define the feasibility set $\\Gamma_\\tau(\\mu_\\tau)$ used in the "
        "State's problem above: the mechanism must not give any captor type an incentive to "
        "misreport its behavior, and every player must be willing to participate rather than "
        "opt out."
    )

    st.markdown(
        "**Captor's incentive compatibility** (`eq:ic-kidnapper`) — belief-weighted: no true "
        "type should want to mimic the branch chosen by another type. Exact form used inside "
        "`solve_state_problem`'s $\\Gamma_\\tau(\\mu_\\tau)$ mask (pooling/mimicry comparison "
        "over the three branches):"
    )
    st.latex(r"""
    \min_{\theta_j\in\Theta_K}\;\sum_{\theta_i\in\Theta_K}\mu_\tau(\theta_i)\Big[\,V^K_\tau(\theta_i)-U^K\big(b(\theta_j)\mid\theta_i,\alpha_\tau,\gamma_\tau\big)\,\Big]\;\ge\;0,\qquad b(\theta_j)=\arg\max\{U^K_{rel}(\theta_j),U^K_{kill}(\theta_j),V^K_{cont,\tau}(\theta_j)\}
    """)

    st.markdown(
        "**Captor's individual rationality** (`eq:ir-K`) — belief-weighted, Release as the "
        "outside option:"
    )
    st.latex(r"""
    \sum_{\theta\in\Theta_K}\mu_\tau(\theta)\Big[\,U^K_{rel}(\theta)-\max\{V^K_{cont,\tau}(\theta,\alpha_\tau,\gamma_\tau),\,U^K_{kill}(\theta,\theta_S)\}\,\Big]\;\ge\;0
    """)

    st.markdown(
        "**Family's individual rationality** (`eq:ir-family`) — now the *same* exact formula "
        "used for $a_F^{1,*}$ in Block B above (unified, see caption there):"
    )
    st.latex(r"""
    U^F_{coop,\tau}(\theta_F)\;\ge\;U^F_{col,\tau}(\theta_F)
    """)

    st.markdown(
        "**Captor's own dynamic problem, true type** (added for $\\tau=1$ in Tab 3): once "
        "$(\\alpha_1^*,\\gamma_1^*)$ are solved above, $V^K_{cont,1}(\\theta_K^{true})$ is "
        "**re-evaluated** (not reused as-is) by `solve_captor_true_type_continuation(...)`, "
        "now in its own `train_captor_true_type_net.py` with its own trained network "
        "(`captor_true_type_value_net_T10.pt`): $\\tilde p_{cap,1}$ uses the *real* MDG "
        "probabilities from the $\\tilde a_S$ draw (not the neutral $0.5/0.5$ mixture "
        "`solve_state_problem` uses internally while searching the grid), and the 10-branch "
        "$(m,d)$ continuation expectation — at **every** level of that network's own backward-"
        "induction training, not just as a $\\tau=1$ patch — is weighted by "
        "$\\Pr(m,d\\mid\\theta_K^{true})$, the true type's own hazard rates, instead of the "
        "$\\mu_1$-marginal weight the State's network uses at every level of *its* training. "
        "Both are legitimate, distinct objects, each with its own consistent network; Tab 3 "
        "reports both."
    )

    st.markdown("---")
    st.caption(
        "Source: Bernal_H.tex, § *Rational behavior and best responses* (subsec:bellman) — "
        "eqs. k-bellman, k-kill, k-rel, kidnapper-cont, k-argmax, f-coop, f-col, f-argmax, "
        "state-loss, state-dual-control, prima-desviacion, delta-H, piso-R, piso-N, "
        "state-expected-loss, state-discrete-rule, ic-kidnapper, ir-K, ir-family. "
        "Explicit functional forms (cost-function-kidnapper, family-institutional-cost, "
        "cops-quadratic, cmaint-quadratic) from Working_paper_eng.tex. This tab now shows the "
        "*exact* forms used by the code (Tab 3's τ=1 button and the internal Γ_τ(μ_τ) gate), "
        "not a simplified stand-in — verified line-by-line, see chat report."
    )

with tab3:
    _tab3_title_col, _tab3_reset_col = st.columns([5, 1])
    with _tab3_title_col:
        st.markdown("## Results")
    with _tab3_reset_col:
        if st.button(
            "🔄 Reset", key="reset_tab3_button",
            help="Full clean slate: clears the τ=1…T_max trajectory AND τ=0's exogenous draws "
                 "(Tab 1: MDG action realizations, m/d/voice draws) so the next \"Run State "
                 "Optimization\" starts from a genuinely fresh τ=0, not stale draws from a "
                 "previous run.",
        ):
            for _reset_key_tab3 in [
                # tau=1...T_max trajectory (Results tab's own computed state)
                "tau1_state_opt_result", "tau_history", "tau_history_normalized",
                "tau_display_max", "tau_closed_at", "tau_view_selector",
                # tau=0 exogenous draws (Tab 1) -- popped so their `if key not in
                # st.session_state` init blocks re-seed clean defaults on rerun
                "act_s", "act_k", "act_f", "u_s", "u_k", "u_f",
                "p_rescue_draw", "p_nego_draw", "p_cont_draw", "p_rel_draw", "p_kill_draw",
                "p_coop_draw", "p_col_draw",
                "m_tau0_draw", "m_tau0_outcome", "m_path",
                "d_tau0_draw", "d_tau0_pdet", "d_tau0_realized",
                "voice_path", "voice_pi_call_realized",
            ]:
                st.session_state.pop(_reset_key_tab3, None)
            st.rerun()

    if _CVN_AVAILABLE and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "captor_value_net_T10.pt")):
        # T_max (extension APPROVED por el usuario, no esta en Bernal_H.tex/Working_paper.tex):
        # el ciclo siempre inicia en tau=1 (se muestra solo tau=1 si T_max=1, comportamiento
        # identico al original); con T_max>1 se sigue con tau=2...T_max reusando run_period.py
        # (misma formula que tau=1, generalizada) hasta el maximo elegido (tope 1000).
        _t_max_col1, _t_max_col2, _t_max_col3 = st.columns([1.6, 0.9, 1.1])
        with _t_max_col1:
            _run_state_opt_clicked = st.button(
                "▶️ Run State Optimization (τ=1…T_max, dynamic — grid search + IC/IR masks + trained value net, T=10)"
            )
        with _t_max_col2:
            st.number_input(
                "T_max", min_value=1, max_value=1000, value=20, step=1, key="t_max_input",
                help="The cycle always starts at τ=1; with T_max>1 it continues τ=2…T_max reusing run_period.py.",
            )
        with _t_max_col3:
            st.checkbox(
                "Counterfactual extension", value=True, key="counterfactual_ext_input",
                help="If the episode closes (m_τ≠Cont) before T_max: keep going under the "
                     "absorbing state (same treatment the paper itself uses in its calibration "
                     "figures) instead of stopping the cycle there.",
            )
    else:
        _run_state_opt_clicked = False
        st.caption(
            "*(State optimization button disabled — `captor_value_net_T10.pt` not found or PyTorch "
            "unavailable. Run `train_captor_value_net.py --T 10` first.)*"
        )

    st.markdown(
        "This tab reports **Results** using the same row/variable structure as Table 5.2 of "
        "`app.py` — *\"Equilibrium by τ: optimal actions, MDG implementation and outcome\"* — "
        "starting at the $\\tau=0$ column. Rows fixed by Tab 1's exogenous inputs (Block A–G) "
        "show their current value; the State's **perfect-info benchmarks** "
        "($\\alpha_R^{\\theta_K,*},\\gamma_R^{\\theta_K,*},\\alpha_N^{\\theta_K,*},\\gamma_N^{\\theta_K,*}$) "
        "are computed **for each captor type** by solving $\\arg\\min_{(\\alpha,\\gamma)\\in[0,1]^2} V_0^R$ "
        "and $\\arg\\min_{(\\alpha,\\gamma)\\in[0,1]^2} V_0^N$ (Tab 2, Section C) via a fine grid search "
        "over $[0,1]^2$ (101×101 points) — traditional numerical optimization, not deep learning: "
        "these are one-off, 2-dimensional, box-constrained problems with a fully closed-form "
        "objective, so a deterministic, reproducible grid search is both simpler and more "
        "transparent than training a function approximator. Rows that still require solving the "
        "Captor's/Family's full Bayesian problem (belief-update direction, feasibility, IC/IR) "
        "remain **pending**. A $\\tau=1$ column starts here with $\\mu_1(\\theta_K)$ — the full "
        "Bayes update (Bernal_H.tex eq:bayes-update) using $\\tau=0$'s **realized** signals "
        "$(\\tilde a_K, m, d, V_t)$; requires drawing $m$ (Tab 1, \"Draw Physical Outcome\") and "
        "$d$ (Tab 1 Block D, \"Draw Detection Signal\") first. Remaining $\\tau=1$ rows are "
        "future work."
    )

    _mu0_tab3 = {"DC": float(mu0_dc), "PAR": float(mu0_par), "ELN": float(mu0_eln), "FARC": float(mu0_farc)}
    _iota_tab3 = float(max(_mu0_tab3.values())) if _mu0_tab3 else float("nan")
    _theta_hat_tab3 = max(_mu0_tab3, key=_mu0_tab3.get) if _mu0_tab3 else cov_perp
    _H_mu0_tab3 = float(-sum(p * np.log(p) for p in _mu0_tab3.values() if p > 1e-12))

    # ── Perfect-info benchmarks (α*, γ*) per type — grid search over [0,1]² ──
    # Solves argmin_{(alpha,gamma)} V_0^R / V_0^N (Working_paper_eng.tex eqs.
    # benchmark-rescue-type / benchmark-neg-type) for tau=0, for each theta_K,
    # using the exact structural formulas already computed in Tab 1 Blocks B/D/E
    # (idx_pago/idx_muerte/idx_rescate, p_det, p_surv) and the baseline C_ops/
    # C_maint coefficients reported in Tab 2. Traditional numerical optimization
    # (fine grid search, deterministic and reproducible; no new dependency) —
    # see Tab 2/Tab 3 methodology note below for why deep learning is not used here.
    # Recalibrated (interior-solution) per-type coefficients: c1,c3 (resp. m1,m3) are
    # NEGATIVE so C_ops/C_maint have a genuine interior minimum in (0,1)^2 for every
    # theta_K (a purely-positive-coefficient cost, as used previously, forces a corner
    # solution at (0,0) since the function is then monotonic in the box).
    _C_OPS_COEF_TAB3 = {
        "DC":   (1.00, -0.25, 1.0, -0.20, 1.0, 0.10),
        "PAR":  (1.00, -0.40, 1.0, -0.35, 1.0, 0.10),
        "ELN":  (1.00, -0.55, 1.0, -0.50, 1.0, 0.10),
        "FARC": (1.00, -0.70, 1.0, -0.65, 1.0, 0.10),
    }  # c0..c5 per theta_K
    _C_MAINT_COEF_TAB3 = {
        "DC":   (3.00, -6.00, 10.0, -0.80, 6.0, 0.10),
        "PAR":  (3.00, -6.50, 10.0, -1.10, 6.0, 0.10),
        "ELN":  (3.00, -7.00, 10.0, -1.40, 6.0, 0.10),
        "FARC": (3.00, -7.50, 10.0, -1.70, 6.0, 0.10),
    }  # m0..m5 per theta_K

    def _c_ops_tab3(gamma: np.ndarray, alpha: np.ndarray, tipo: str) -> np.ndarray:
        c0, c1, c2, c3, c4, c5 = _C_OPS_COEF_TAB3[tipo]
        return c0 + c1 * gamma + 0.5 * c2 * gamma**2 + c3 * alpha + 0.5 * c4 * alpha**2 + c5 * gamma * alpha

    def _c_maint_tab3(gamma: np.ndarray, alpha: np.ndarray, tipo: str) -> np.ndarray:
        m0, m1, m2, m3, m4, m5 = _C_MAINT_COEF_TAB3[tipo]
        return m0 + m1 * gamma + 0.5 * m2 * gamma**2 + m3 * alpha + 0.5 * m4 * alpha**2 + m5 * gamma * alpha

    def _h2_tab3(alpha: np.ndarray, gamma: np.ndarray, tipo: str) -> np.ndarray:
        """Death hazard h̄_2(0|θ_K,C_0), reusing Block B's idx_muerte formula generalized to any θ_K."""
        beta_K_2 = BETAS_K[tipo]["Muerte"]
        beta_K_1 = BETAS_K[tipo]["Pago"]
        beta_K_3 = BETAS_K[tipo]["Rescate"]
        beta_z = BETAS_Z[cov_zone]
        beta_F_1 = 0.80 if cov_wealth == "High" else 0.00
        beta_V_1 = 1.36 if cov_vict == "Public sector" else 0.00
        beta_S = 0.50 if cov_state == "Lax" else 0.00
        phi_F_1 = 3.20 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_1 = -1.15 if st.session_state["act_k"] == "Continue" else 0.00
        phi_F_2 = -1.50 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_kill_2 = 4.00 if st.session_state["act_k"] == "Kill" else 0.00
        phi_K_cont_2 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00
        zeta_R_3 = 2.50 if st.session_state["act_s"] == "Rescue" else 0.00
        phi_F_3 = -1.00 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_3 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00

        za = ZETAS_POLITICA[tipo]["zeta_alpha"]
        zg = ZETAS_POLITICA[tipo]["zeta_gamma"]
        eta_0_t = ETA_0_PDET[tipo]
        u_det_t = eta_0_t + ETA_1_PDET * alpha + ETA_2_PDET * gamma
        p_det_t = 1.0 / (1.0 + np.exp(-u_det_t))

        idx_pago_t = (beta_K_1 + beta_z + beta_F_1 - beta_V_1 + beta_S
                      - za * alpha - zg * gamma - zd * p_det_t + phi_F_1 + phi_K_1)
        idx_muerte_t = (beta_K_2 + beta_z + beta_S
                         + za * alpha + zg * gamma - zd * p_det_t - phi_F_2 + phi_K_kill_2 + phi_K_cont_2)
        idx_rescate_t = (-beta_S + beta_K_3 + beta_z
                          + za * alpha + zg * gamma + zd * p_det_t + zeta_R_3 - phi_F_3 + phi_K_3)

        lam_pay_t = M_t * LAMBDAS_0["Pago"] * np.exp(idx_pago_t)
        lam_kill_t = M_t * LAMBDAS_0["Muerte"] * np.exp(idx_muerte_t)
        lam_res_t = M_t * LAMBDAS_0["Rescate"] * np.exp(idx_rescate_t)
        total_t = lam_pay_t + lam_kill_t + lam_res_t + lambda_4
        p_cont_t = np.exp(-total_t)
        q_t_t = 1.0 - p_cont_t
        xi_2_t = np.where(total_t > 1e-12, lam_kill_t / np.maximum(total_t, 1e-12), 0.25)
        return q_t_t * xi_2_t

    def _outcome_probs_tab3(alpha: float, gamma: float, tipo: str) -> tuple[dict[str, float], float]:
        """Pr(m|theta,alpha,gamma,C_0) for m in {Cont,1,2,3,4}, and p_det(theta,alpha,gamma) —
        same Block B/C/D formulas as _h2_tab3, but returning all 5 outcome probabilities
        (not just h_2) plus p_det, for the Bayes update used in Delta H_0."""
        beta_K_1 = BETAS_K[tipo]["Pago"]
        beta_K_2 = BETAS_K[tipo]["Muerte"]
        beta_K_3 = BETAS_K[tipo]["Rescate"]
        beta_z = BETAS_Z[cov_zone]
        beta_F_1 = 0.80 if cov_wealth == "High" else 0.00
        beta_V_1 = 1.36 if cov_vict == "Public sector" else 0.00
        beta_S = 0.50 if cov_state == "Lax" else 0.00
        phi_F_1 = 3.20 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_1 = -1.15 if st.session_state["act_k"] == "Continue" else 0.00
        phi_F_2 = -1.50 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_kill_2 = 4.00 if st.session_state["act_k"] == "Kill" else 0.00
        phi_K_cont_2 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00
        zeta_R_3 = 2.50 if st.session_state["act_s"] == "Rescue" else 0.00
        phi_F_3 = -1.00 if st.session_state["act_f"] == "Pay" else 0.00
        phi_K_3 = 0.50 if st.session_state["act_k"] == "Continue" else 0.00

        za = ZETAS_POLITICA[tipo]["zeta_alpha"]
        zg = ZETAS_POLITICA[tipo]["zeta_gamma"]
        eta_0_t = ETA_0_PDET[tipo]
        u_det_t = eta_0_t + ETA_1_PDET * alpha + ETA_2_PDET * gamma
        p_det_t = float(1.0 / (1.0 + np.exp(-u_det_t)))

        idx_pago_t = (beta_K_1 + beta_z + beta_F_1 - beta_V_1 + beta_S
                      - za * alpha - zg * gamma - zd * p_det_t + phi_F_1 + phi_K_1)
        idx_muerte_t = (beta_K_2 + beta_z + beta_S
                         + za * alpha + zg * gamma - zd * p_det_t - phi_F_2 + phi_K_kill_2 + phi_K_cont_2)
        idx_rescate_t = (-beta_S + beta_K_3 + beta_z
                          + za * alpha + zg * gamma + zd * p_det_t + zeta_R_3 - phi_F_3 + phi_K_3)

        lam_pay_t = M_t * LAMBDAS_0["Pago"] * np.exp(idx_pago_t)
        lam_kill_t = M_t * LAMBDAS_0["Muerte"] * np.exp(idx_muerte_t)
        lam_res_t = M_t * LAMBDAS_0["Rescate"] * np.exp(idx_rescate_t)
        total_t = lam_pay_t + lam_kill_t + lam_res_t + lambda_4
        p_cont_t = float(np.exp(-total_t))
        q_t_t = 1.0 - p_cont_t
        return {
            "Cont": p_cont_t,
            "1": float(q_t_t * lam_pay_t / total_t),
            "2": float(q_t_t * lam_kill_t / total_t),
            "3": float(q_t_t * lam_res_t / total_t),
            "4": float(q_t_t * lambda_4 / total_t),
            "lam_1": float(lam_pay_t), "lam_2": float(lam_kill_t), "lam_3": float(lam_res_t),
        }, p_det_t

    def _entropy_tab3(mu: dict[str, float]) -> float:
        return float(-sum(p * np.log(p) for p in mu.values() if p > 1e-12))

    # ── Delta H_0(alpha_0,gamma_0): expected entropy reduction (eq. delta-H) ──
    # For each of the 5x2=10 (m,d) pairs, Bayes-update mu_0 -> mu_1^{m,d} using the
    # "minimal record" rule (Bernal_H.tex eq:bayes-update / Working_paper_eng.tex
    # eq:posterior-counterfactual-entropy: no voice/implementation evidence, only
    # the physical outcome m and the detection signal d), then weight H(mu_1^{m,d})
    # by its own predictive probability Pr(m,d) and sum (closed-form, no optimization).
    _outcome_probs_by_type_tab3: dict[str, dict[str, float]] = {}
    _p_det_by_type_tab3: dict[str, float] = {}
    for _th_dh in ["DC", "PAR", "ELN", "FARC"]:
        _outcome_probs_by_type_tab3[_th_dh], _p_det_by_type_tab3[_th_dh] = _outcome_probs_tab3(
            float(alpha_val), float(gamma_val), _th_dh
        )

    _delta_h_weighted_sum = 0.0
    _delta_h_prob_check = 0.0
    for _m_dh in ["Cont", "1", "2", "3", "4"]:
        for _d_dh in (0, 1):
            _w_dh = {
                _th_dh: (
                    _mu0_tab3[_th_dh]
                    * _outcome_probs_by_type_tab3[_th_dh][_m_dh]
                    * (_p_det_by_type_tab3[_th_dh] if _d_dh == 1 else (1.0 - _p_det_by_type_tab3[_th_dh]))
                )
                for _th_dh in _mu0_tab3
            }
            _z_dh = float(sum(_w_dh.values()))
            _delta_h_prob_check += _z_dh
            if _z_dh > 1e-15:
                _mu1_dh = {_th_dh: _w_dh[_th_dh] / _z_dh for _th_dh in _w_dh}
                _delta_h_weighted_sum += _z_dh * _entropy_tab3(_mu1_dh)

    _delta_H_tab3 = _H_mu0_tab3 - _delta_h_weighted_sum

    # ── kappa_h(theta_K,0) and -sgn(kappa_h) per type (Bernal_H.tex line 1207 / ──
    # Appendix_3.tex eq:kappa-c, Proposition + proof). zeta_{gamma,1}=zeta_{gamma,2}=
    # zeta_{gamma,3}=zeta_gamma(theta_K) in this calibration (same |slope|, sign baked
    # into idx_pago vs idx_muerte/idx_rescate), so kappa_h reduces to a simple
    # combination of the raw lambda_j's already computed above for Delta H.
    _kappa_h_by_type_tab3: dict[str, float] = {}
    _neg_sign_kappa_h_by_type_tab3: dict[str, int] = {}
    for _th_kh in ["DC", "PAR", "ELN", "FARC"]:
        _zg_kh = ZETAS_POLITICA[_th_kh]["zeta_gamma"]
        _probs_kh = _outcome_probs_by_type_tab3[_th_kh]
        _kappa_h_by_type_tab3[_th_kh] = float(
            _zg_kh * (_probs_kh["lam_2"] + _probs_kh["lam_3"] - _probs_kh["lam_1"])
        )
        _neg_sign_kappa_h_by_type_tab3[_th_kh] = -int(np.sign(_kappa_h_by_type_tab3[_th_kh]))

    # ── Gamma_t(mu_t) bajo EV and IR^K(true type): substitute (mu_0, alpha_0, gamma_0) ──
    # into IR^K, IC^K, IR^F (Tab 2, Section D). Generalizes Block F's p_cap/p_pay/p_surv
    # (previously only computed for cov_perp) to all 4 types. V^K_cont and U^F_coop use a
    # MYOPIC tau=0 approximation (continuation value ~ 0) — the true dynamic continuation
    # value requires solving the infinite-horizon Bellman equation, out of scope here; this
    # is a one-period substitution check, not the full dynamic IR/IC verification of app.py.
    KAPPA_REL_TAB3 = {"DC": 2.367, "PAR": 14.273, "ELN": 4.163, "FARC": 1.471}
    ETA_TAB3 = {"DC": 0.340, "PAR": 3.348, "ELN": 2.360, "FARC": -0.580}
    F_CAP_TAB3 = {"DC": 40.704, "PAR": 85.224, "ELN": 55.968, "FARC": 29.256}  # theta_S = Strict
    PHI_TAB3 = {"DC": 33.00, "PAR": 35.00, "ELN": 38.00, "FARC": 40.00}
    KAPPA_C_TAB3 = {"DC": 2.61, "PAR": 2.63, "ELN": 2.70, "FARC": 2.70}
    NU_TAB3 = {"DC": 0.750, "PAR": 0.500, "ELN": 0.250, "FARC": 0.000}
    PHI_F_TAB3 = {"High wealth": 0.06, "Low wealth": 0.06}
    KAPPA_F_TAB3 = {"High wealth": 3.60, "Low wealth": 3.00}
    NU_F_TAB3 = {"High wealth": 0.02, "Low wealth": 0.02}
    # varrho: prima de riesgo de colusion. Con varrho=0, IR^F es estructuralmente imposible
    # de satisfacer para cualquier phi_F/nu_F>0 (U_coop-U_col=-e_tau<0 siempre). No calibrado
    # en app.py; elegido para dar una region factible genuina, ver diagnostico en el chat.
    VARRHO_COLLUDE_TAB3 = 2.0
    _theta_f_tab3 = "High wealth" if cov_wealth == "High" else "Low wealth"

    def _p_cap_eff_tab3(alpha: float, gamma: float, tipo: str) -> float:
        c0_t, ca_t, cg_t = C0_CAP[tipo], CALPHA_CAP[tipo], CGAMMA_CAP[tipo]
        cs_t = CS_CAP[cov_state]
        u_r = DELTA_A_CAP["Rescue"] + c0_t + ca_t * alpha + cg_t * gamma + cs_t
        u_n = DELTA_A_CAP["Negotiate"] + c0_t + ca_t * alpha + cg_t * gamma + cs_t
        p_r = 1.0 / (1.0 + np.exp(-u_r))
        p_n = 1.0 / (1.0 + np.exp(-u_n))
        return float(p_rescue * p_r + p_nego * p_n)

    def _p_surv_eff_tab3(tipo: str) -> float:
        u_s = ALPHA_LETH[tipo] + beta_R * iota_t * (1.0 if correct_id else 0.0)
        p_s = 1.0 / (1.0 + np.exp(-u_s))
        return float(p_rescue * p_s)

    def _p_pay_eff_tab3(alpha: float, gamma: float, tipo: str, p_det_t: float) -> float:
        beta_K_1_t = BETAS_K[tipo]["Pago"]
        beta_z_1 = BETAS_Z[cov_zone]
        beta_F_1 = 0.80 if cov_wealth == "High" else 0.00
        beta_V_1 = 1.36 if cov_vict == "Public sector" else 0.00
        beta_S_1 = 0.50 if cov_state == "Lax" else 0.00
        za_t, zg_t = ZETAS_POLITICA[tipo]["zeta_alpha"], ZETAS_POLITICA[tipo]["zeta_gamma"]
        idx_pago_coop = (beta_K_1_t + beta_z_1 + beta_F_1 - beta_V_1 + beta_S_1
                          - za_t * alpha - zg_t * gamma - zd * p_det_t + 0.00 - 1.15)
        idx_pago_pay = (beta_K_1_t + beta_z_1 + beta_F_1 - beta_V_1 + beta_S_1
                         - za_t * alpha - zg_t * gamma - zd * p_det_t + 3.20 - 1.15)
        lam_pay_coop = M_t * LAMBDAS_0["Pago"] * np.exp(idx_pago_coop)
        lam_pay_pay = M_t * LAMBDAS_0["Pago"] * np.exp(idx_pago_pay)
        # Recompute total_t/q_t_t at THIS (alpha,gamma) — must not reuse the tau=0 dict,
        # otherwise this function silently ignores the alpha/gamma it was passed whenever
        # they differ from Tab 1's exogenous slider point (e.g. when reused at tau=1).
        _probs_t, _ = _outcome_probs_tab3(alpha, gamma, tipo)
        total_t = _probs_t["lam_1"] + _probs_t["lam_2"] + _probs_t["lam_3"] + lambda_4
        q_t_t = 1.0 - _probs_t["Cont"]
        h1_coop = q_t_t * lam_pay_coop / total_t if total_t > 1e-12 else 0.25
        h1_pay = q_t_t * lam_pay_pay / total_t if total_t > 1e-12 else 0.25
        return float(p_coop * h1_coop + p_col * h1_pay)

    _p_cap_by_type_tab3 = {th: _p_cap_eff_tab3(float(alpha_val), float(gamma_val), th) for th in _mu0_tab3}
    _p_surv_by_type_tab3 = {th: _p_surv_eff_tab3(th) for th in _mu0_tab3}
    _p_pay_by_type_tab3 = {
        th: _p_pay_eff_tab3(float(alpha_val), float(gamma_val), th, _p_det_by_type_tab3[th])
        for th in _mu0_tab3
    }

    _U_rel_tab3 = {th: -KAPPA_REL_TAB3[th] for th in _mu0_tab3}
    _U_kill_tab3 = {
        th: (1.0 - _p_cap_by_type_tab3[th]) * ETA_TAB3[th] - _p_cap_by_type_tab3[th] * F_CAP_TAB3[th]
        for th in _mu0_tab3
    }
    _C_tau_tab3 = {
        th: PHI_TAB3[th] * np.exp(KAPPA_C_TAB3[th] * float(gamma_val)) + NU_TAB3[th]
        for th in _mu0_tab3
    }
    # Myopic V^K_cont (continuation value ~ 0); R in millions COP, same convention as Tab 3's V_N.
    _V_cont_myopic_tab3 = {
        th: (
            _p_pay_by_type_tab3[th] * float(ransom_R_millions) * (1.0 - float(alpha_val))
            - _C_tau_tab3[th]
            - _p_cap_by_type_tab3[th] * F_CAP_TAB3[th]
        )
        for th in _mu0_tab3
    }
    _best_k_tab3 = {
        th: max(_U_rel_tab3[th], _U_kill_tab3[th], _V_cont_myopic_tab3[th]) for th in _mu0_tab3
    }
    _branch_k_tab3 = {
        th: max(
            [("rel", _U_rel_tab3[th]), ("kill", _U_kill_tab3[th]), ("cont", _V_cont_myopic_tab3[th])],
            key=lambda x: x[1],
        )[0]
        for th in _mu0_tab3
    }

    _ir_k_gap_by_type_tab3 = {
        th: _U_rel_tab3[th] - max(_V_cont_myopic_tab3[th], _U_kill_tab3[th]) for th in _mu0_tab3
    }
    _ir_k_ev_gap_tab3 = float(sum(_mu0_tab3[th] * _ir_k_gap_by_type_tab3[th] for th in _mu0_tab3))
    _ir_k_ev_tab3 = bool(_ir_k_ev_gap_tab3 >= -1e-9)
    _ir_k_true_gap_tab3 = float(_ir_k_gap_by_type_tab3[cov_perp])
    _ir_k_true_tab3 = bool(_ir_k_true_gap_tab3 >= -1e-9)

    _ic_k_gains_tab3 = []
    for _th_j in _mu0_tab3:
        _b_j = _branch_k_tab3[_th_j]
        _u_branch_j = {"rel": _U_rel_tab3, "kill": _U_kill_tab3, "cont": _V_cont_myopic_tab3}[_b_j]
        _gain_j = float(sum(_mu0_tab3[_th_i] * (_best_k_tab3[_th_i] - _u_branch_j[_th_i]) for _th_i in _mu0_tab3))
        _ic_k_gains_tab3.append(_gain_j)
    _ic_k_tab3 = bool(all(g >= -1e-9 for g in _ic_k_gains_tab3))

    _e_tau_family_tab3 = (
        PHI_F_TAB3[_theta_f_tab3] * np.exp(KAPPA_F_TAB3[_theta_f_tab3] * float(gamma_val))
        + NU_F_TAB3[_theta_f_tab3]
    )
    _e_mu_p_pay_tab3 = float(sum(_mu0_tab3[th] * _p_pay_by_type_tab3[th] for th in _mu0_tab3))
    # varrho (collude risk premium) not calibrated anywhere in app.py/rational_behavior.py — set to 0.
    _U_coop_family_tab3 = -_e_mu_p_pay_tab3 * float(ransom_R_millions) - float(_e_tau_family_tab3)
    _U_col_family_tab3 = -_e_mu_p_pay_tab3 * float(ransom_R_millions) * (1.0 + VARRHO_COLLUDE_TAB3)
    _ir_f_gap_tab3 = float(_U_coop_family_tab3 - _U_col_family_tab3)
    _ir_f_tab3 = bool(_ir_f_gap_tab3 >= -1e-9)

    _gamma_t_ev_tab3 = bool(_ir_k_ev_tab3 and _ic_k_tab3 and _ir_f_tab3)

    # ── mu_1(theta_K): full Bayes update using tau=0's REALIZED signals ──
    # (Bernal_H.tex eq:bayes-update / eq:LF-joint; Working_paper_eng.tex eq:bayes-unif /
    # eq:LH-joint). mu_1(theta) proportional to mu_0(theta) * L_I,K(theta) * Pr(m|theta) *
    # Pr(d|theta) * L_C(theta|V_t) — the full record (implementation + physical outcome +
    # detection + voice), NOT the minimal (m,d)-only record used for Delta H.
    _m_tau0_key_map = {
        "Continue Captivity (cont)": "Cont",
        "Ransom Paid (j=1)": "1",
        "Victim Deceased (j=2)": "2",
        "Tactical Rescue (j=3)": "3",
        "Exogenous Release (j=4)": "4",
    }
    _m_realized_key_tab3 = _m_tau0_key_map.get(str(st.session_state.get("m_tau0_outcome", "")))
    _d_realized_tab3 = st.session_state.get("d_tau0_realized")

    _branch_to_action_tab3 = {"rel": "Release", "kill": "Kill", "cont": "Continue"}
    _act_k_realized_tab3 = str(st.session_state.get("act_k", "—"))

    def _voice_likelihood_tab3(tipo: str) -> float:
        if not st.session_state.get("include_voice_posterior_checkbox", True):
            return 1.0
        x_true_t = X_TRUE_VOZ[tipo]
        sigma_t = SIGMA_VOZ[tipo]
        pi_call_t = PI_CALL[tipo]
        l_voz_t = float(np.exp(-0.5 * ((x_obs - x_true_t) ** 2) / (sigma_t ** 2)))
        if voice_emitted:
            return float((l_voz_t * pi_call_t) ** omega_voz)
        return float((1.0 - pi_call_t) ** omega_voz)

    def _implementation_likelihood_k_tab3(tipo: str) -> float:
        a_star_t = _branch_to_action_tab3[_branch_k_tab3[tipo]]
        # CORREGIDO: usa T_t_K (misma temperatura que genero el sorteo real de a_tilde_K
        # arriba), no el T_t generico -- consistencia entre el sorteo y su propia
        # verosimilitud de implementacion L_I,K(theta) en el Bayes update tau=0->1.
        num_cont_t = np.exp(1.0 / T_t_K) if a_star_t == "Continue" else np.exp(0.0)
        num_rel_t = np.exp(1.0 / T_t_K) if a_star_t == "Release" else np.exp(0.0)
        num_kill_t = np.exp(1.0 / T_t_K) if a_star_t == "Kill" else np.exp(0.0)
        denom_t = num_cont_t + num_rel_t + num_kill_t
        probs_t = {"Continue": num_cont_t / denom_t, "Release": num_rel_t / denom_t, "Kill": num_kill_t / denom_t}
        return float(probs_t.get(_act_k_realized_tab3, 1.0 / 3.0))

    _mu1_tab3: dict[str, float] = {}
    _mu1_ready_tab3 = _m_realized_key_tab3 is not None and _d_realized_tab3 is not None
    if _mu1_ready_tab3:
        _w_mu1 = {}
        for _th_mu1 in _mu0_tab3:
            _l_i_k = _implementation_likelihood_k_tab3(_th_mu1)
            _pr_m = _outcome_probs_by_type_tab3[_th_mu1][_m_realized_key_tab3]
            _p_det_th = _p_det_by_type_tab3[_th_mu1]
            _pr_d = _p_det_th if int(_d_realized_tab3) == 1 else (1.0 - _p_det_th)
            _l_c = _voice_likelihood_tab3(_th_mu1)
            _w_mu1[_th_mu1] = _mu0_tab3[_th_mu1] * _l_i_k * _pr_m * _pr_d * _l_c
        _z_mu1 = float(sum(_w_mu1.values()))
        _mu1_tab3 = (
            {th: _w_mu1[th] / _z_mu1 for th in _w_mu1}
            if _z_mu1 > 1e-15
            else {th: 1.0 / len(_mu0_tab3) for th in _mu0_tab3}
        )

    if _run_state_opt_clicked:
        if not _mu1_ready_tab3:
            st.warning(
                "μ₁ is not ready yet — draw both $m$ (Tab 1, \"Draw Physical Outcome\") and "
                "$d$ (Tab 1 Block D, \"Draw Detection Signal\") before running the optimization."
            )
        else:
            with st.spinner("Solving Γ_τ(μ_τ)-constrained grid search for τ=1 (T=10 trained value net)..."):

                @st.cache_resource
                def _load_captor_value_net():
                    _ckpt = torch.load(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "captor_value_net_T10.pt"),
                        map_location="cpu", weights_only=False,
                    )
                    _net = cvn.CaptorValueNet()
                    _net.load_state_dict(_ckpt["state_dict"])
                    _net.eval()
                    return _net, int(_ckpt["T"])

                _net_tab3, _T_trained_tab3 = _load_captor_value_net()

                @st.cache_resource
                def _load_captor_true_type_value_net():
                    _path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "captor_true_type_value_net_T10.pt"
                    )
                    if not os.path.exists(_path):
                        return None, None
                    _ckpt = torch.load(_path, map_location="cpu", weights_only=False)
                    _net = cvn.CaptorValueNet()
                    _net.load_state_dict(_ckpt["state_dict"])
                    _net.eval()
                    return _net, int(_ckpt["T"])

                _net_true_type_tab3, _T_true_type_trained_tab3 = _load_captor_true_type_value_net()

                _p_opt_tab3 = cvn.Params(
                    R_millions=float(ransom_R_millions),
                    beta_tilde=dict(BETA_TILDE_TAB1),
                    cov_zone=cov_zone, cov_wealth=cov_wealth, cov_vict=cov_vict, cov_state=cov_state,
                    T_mad=float(T_mad), T0=float(T_0), eta_cal=float(eta_cal), c_bar=float(c_bar),
                    H_ratio=float(H_ratio), lambda_4=float(lambda_4), eta_1=float(eta_1),
                    eta_2=float(eta_2), beta_R=float(beta_R),
                )

                def _v_next_fn_tab3(mu2, tau_next, p):
                    if tau_next > _T_trained_tab3:
                        return {th: np.zeros_like(cvn.GA) for th in cvn.TIPOS}
                    out = {}
                    with torch.no_grad():
                        for th in cvn.TIPOS:
                            x = cvn.encode_input_grid(th, mu2, tau_next, _T_trained_tab3, p)
                            v = _net_tab3(torch.from_numpy(x)).numpy()
                            out[th] = v.reshape(cvn.GA.shape)
                    return out

                _theta_f_opt_tab3 = "High wealth" if cov_wealth == "High" else "Low wealth"
                # Protocolo de consistencia temporal para (p_rescue,p_nego) dentro de p_cap
                # (eq:p-cap): en produccion se ancla al desenlace YA EJECUTADO por el Estado en
                # tau=0 (act_s, Tabla 5.2) -- degenerado (1,0) si Rescue, (0,1) si Negotiate --
                # en vez del neutro 0.5/0.5 que usan los scripts de entrenamiento (sin registro
                # real de un periodo anterior en sus trayectorias sinteticas).
                _act_s_tau0_tab3 = str(st.session_state.get("act_s", "Negotiate"))
                _p_rescue_prev_tab3 = 1.0 if _act_s_tau0_tab3 == "Rescue" else 0.0
                _p_nego_prev_tab3 = 1.0 - _p_rescue_prev_tab3
                _a_star, _g_star, _aS_star, _v_by_type, _feasible, _extra = cvn.solve_state_problem(
                    _mu1_tab3, 1, _T_trained_tab3, _p_opt_tab3, _v_next_fn_tab3, cov_perp, _theta_f_opt_tab3,
                    p_rescue_prev=_p_rescue_prev_tab3, p_nego_prev=_p_nego_prev_tab3,
                )

                # ── M_{tau=1}: LITERAL eq:hj (Bernal_H.tex/Working_paper.tex) maturation filter,
                # M(t)=min(1,(t/T_mad)^2). CORRECTED (approved by the user): this used to reuse
                # the eq:temperature formula (T_0*max(H_ratio*exp(-eta_cal*t), c_bar)), which is
                # a DIFFERENT object (T_t, the MDG action-noise temperature) -- the paper's own
                # m/lambda_j scaling depends only on T_mad, not on T0/eta_cal/c_bar/H_ratio. Both
                # trained nets (captor_value_net_T10.pt, captor_true_type_value_net_T10.pt) were
                # retrained under this corrected cvn.m_t so their learned V(.) stays consistent.
                _H_mu1_tab3 = float(_extra.get("H_mu", 0.0))
                _H_ratio_real_tau1 = (_H_mu1_tab3 / _H_mu0_tab3) if _H_mu0_tab3 > 1e-12 else 0.0
                _mt_1_tab3 = cvn.m_t(1, _p_opt_tab3)
                # ── Per-player MDG temperatures (extension APPROVED by the user -- NOT literal
                # in Bernal_H.tex/Working_paper_eng.tex, which define ONE shared "system
                # temperature"; see MDG_MULT_* comment above). Used ONLY for the three
                # MDG-executed-action draws below (tilde a_S/F/K); everything else (kappa_h, m,
                # benchmarks, the M_t row itself) keeps using the generic _mt_1_tab3.
                _mt_1_S_tab3 = _mdg_temp_player(T0_S, MDG_MULT_STATE, _H_ratio_real_tau1, _p_opt_tab3)
                _mt_1_F_tab3 = _mdg_temp_player(T0_F, MDG_MULT_FAMILY, _H_ratio_real_tau1, _p_opt_tab3)
                _mt_1_K_tab3 = _mdg_temp_player(T0_K, MDG_MULT_CAPTOR, _H_ratio_real_tau1, _p_opt_tab3)
                # ── tilde a_S (tau=1): MDG-executed realization of the State's action,
                # generated in this SAME click, using a_S^{1,*} (just solved above) as the
                # latent intention and M_{tau=1}^S (State's own literal eq:m-t, real
                # H(mu_1)/H(mu_0), MDG_MULT_STATE) as the noise temperature -- mirrors Block A's
                # logit mechanism exactly.
                _num_r_1 = np.exp(1.0 / _mt_1_S_tab3) if _aS_star == "Rescue" else np.exp(0.0)
                _num_n_1 = np.exp(1.0 / _mt_1_S_tab3) if _aS_star == "Negotiate" else np.exp(0.0)
                _denom_s_1 = _num_r_1 + _num_n_1
                _p_rescue_1 = float(_num_r_1 / _denom_s_1)
                _p_nego_1 = float(_num_n_1 / _denom_s_1)
                _u_s_1 = float(np.random.default_rng().random())
                _act_s_tau1 = "Rescue" if _u_s_1 <= _p_rescue_1 else "Negotiate"

                # ── Familia (tau=1): formula EXACTA del paper (eq:f-coop/eq:f-col/eq:ir-family).
                # Es la MISMA formula (V_L, F_col) que ahora usa internamente solve_state_problem
                # como restriccion de Gamma_1(mu_1) sobre el Estado (unificadas -- ya no hay una
                # version "simplificada" en paralelo). Esto es la decision PROPIA de la Familia,
                # reportada aparte. mu_1, alpha_1*, gamma_1* ya resueltos. Vive en su propio
                # archivo (family_optimization.py) -- sin red, problema estatico por diseno.
                _U_coop_f1, _U_col_f1, _ir_f_gap1, _a_F_star1, _fam_extras1 = famopt.family_utilities(
                    _mu1_tab3, _theta_f_opt_tab3, _a_star, _g_star, 1, _p_opt_tab3
                )

                # tilde a_F (tau=1): mismo mecanismo MDG logit de tilde a_S/tilde a_K (2 ramas:
                # Cooperate/Collude), usando a_F^{1,*} (recien calculado arriba) como intencion
                # latente y M_{tau=1}^F (temperatura propia de la Familia, MDG_MULT_FAMILY --
                # multiplicadores 1.0, numericamente igual a la generica) como temperatura.
                _num_coop_1 = np.exp(1.0 / _mt_1_F_tab3) if _a_F_star1 == "Cooperate" else np.exp(0.0)
                _num_col_1 = np.exp(1.0 / _mt_1_F_tab3) if _a_F_star1 == "Collude" else np.exp(0.0)
                _denom_f_1 = _num_coop_1 + _num_col_1
                _p_coop_1 = float(_num_coop_1 / _denom_f_1)
                _p_col_1 = float(_num_col_1 / _denom_f_1)
                _u_f_1 = float(np.random.default_rng().random())
                _act_f_tau1 = "Cooperate" if _u_f_1 <= _p_coop_1 else "Collude"

                # ── Secuestrador (tau=1), tipo verdadero: problema dinamico dado (alpha_1*,
                # gamma_1*) ya elegidos por el Estado. Vive en su propio archivo
                # (train_captor_true_type_net.py), con su PROPIA red entrenada
                # (captor_true_type_value_net_T10.pt) -- consistente de punta a punta con el
                # peso por tipo verdadero Pr(m,d|theta_true) en los 10 niveles de entrenamiento,
                # a diferencia de la red del Estado (captor_value_net_T10.pt), que usa el peso
                # marginal mu-mezclado en todos sus niveles. p_cap se refina con las
                # probabilidades REALES del sorteo de tilde a_S (p_rescue_1/p_nego_1).
                # Fallback: si la red propia aun no existe (entrenamiento pendiente), usa la
                # red del Estado como aproximacion temporal, marcado explicitamente abajo.
                _using_fallback_net_tab3 = _net_true_type_tab3 is None
                _net_for_captor_tab3 = _net_true_type_tab3 if _net_true_type_tab3 is not None else _net_tab3
                _T_for_captor_tab3 = (
                    _T_true_type_trained_tab3 if _T_true_type_trained_tab3 is not None else _T_trained_tab3
                )

                def _v_next_fn_scalar_tab3(theta, mu, tau_next, p):
                    if tau_next > _T_for_captor_tab3:
                        return 0.0
                    x = cvn.encode_input(theta, mu, tau_next, _T_for_captor_tab3, p)
                    with torch.no_grad():
                        v = _net_for_captor_tab3(torch.from_numpy(x[None, :])).numpy()
                    return float(v[0])

                _branch_true1, _U_rel_true1, _U_kill_true1, _V_cont_true1, _k_extras1 = (
                    cttn.solve_captor_true_type_continuation(
                        cov_perp, _a_star, _g_star, _mu1_tab3, 1, _p_opt_tab3,
                        _v_next_fn_scalar_tab3, _p_rescue_1, _p_nego_1,
                    )
                )
                _branch_to_action_tab3 = {"rel": "Release", "kill": "Kill", "cont": "Continue"}
                _a_K_star1 = _branch_to_action_tab3[_branch_true1]

                # tilde a_K (tau=1): logit PURO de 3 ramas, `eq:logit-hybrid` (Working_paper_
                # eng.tex Sec. 4.2.1, "Stochastic Implementation: MDG Process") == `eq:LI-atilde`
                # (Bernal_H.tex) -- MISMO mecanismo, sin mezcla, que tilde a_S/tilde a_F. Remark
                # 4.2 (axiomatic justification) prueba que este logit es la UNICA ley sobre A^i
                # que satisface (A1) full support, (A2) dominance, (A3) maxima entropia -- por
                # eso la mezcla de inercia usada antes (P_inercia+lambda*P_racional) se elimino:
                # no es una ley de esa forma y viola la caracterizacion de unicidad del Remark.
                # T_t = M_{tau=1}^K (`eq:temperatura-piso`/`eq:m-t`, literal: real H(mu_1)/
                # H(mu_0), MDG_MULT_CAPTOR -- temperatura PROPIA del Secuestrador, extension
                # aprobada) ya garantiza full support + dominance sobre a_K^{1,*}.
                _num_cont_k1 = np.exp(1.0 / _mt_1_K_tab3) if _a_K_star1 == "Continue" else np.exp(0.0)
                _num_rel_k1 = np.exp(1.0 / _mt_1_K_tab3) if _a_K_star1 == "Release" else np.exp(0.0)
                _num_kill_k1 = np.exp(1.0 / _mt_1_K_tab3) if _a_K_star1 == "Kill" else np.exp(0.0)
                _denom_k1 = _num_cont_k1 + _num_rel_k1 + _num_kill_k1
                _p_cont_k1 = float(_num_cont_k1 / _denom_k1)
                _p_rel_k1 = float(_num_rel_k1 / _denom_k1)
                _p_kill_k1 = float(_num_kill_k1 / _denom_k1)
                _u_k_1 = float(np.random.default_rng().random())
                if _u_k_1 <= _p_cont_k1:
                    _act_k_tau1 = "Continue"
                elif _u_k_1 <= _p_cont_k1 + _p_rel_k1:
                    _act_k_tau1 = "Release"
                else:
                    _act_k_tau1 = "Kill"

                # ── Voz (tau=1,2,...): sorteo genuino del canal de comunicacion, siguiendo el
                # mismo mecanismo generativo YA VALIDADO en app.py/rational_behavior.py
                # (eq:voz-descomp, eq:Lvoz, eq:LC) -- Bernoulli(pi_call) para V_tau, seguido de
                # un draw Gaussiano de x_tau^obs (4 rasgos) si V_tau=1. Se acumula en
                # session_state["voice_path"] (una entrada por click, extensible a tau=2,3,...).
                # La pi_call REALIZADA (Beta, kappa=30, anclada al prior PI_CALL) se sortea UNA
                # sola vez -- rasgo propio del incidente, no de cada periodo -- y se reutiliza en
                # clicks futuros. Ilustrativo/prospectivo: NO realimenta mu_1 (ya calculado con
                # la senal fija de tau=0, Block F).
                _voice_available_tab3 = _RB_AVAILABLE and st.session_state.get("include_voice_posterior_checkbox", True)
                if _voice_available_tab3:
                    if st.session_state["voice_pi_call_realized"] is None:
                        st.session_state["voice_pi_call_realized"] = sample_incident_pi_call_realized(
                            PI_CALL, kappa=VOZ_KAPPA_REALIZED,
                        )
                    _pi_realized_tab3 = st.session_state["voice_pi_call_realized"]
                    _tau_new_voice = len(st.session_state["voice_path"]) + 1
                    _rng_voice = np.random.default_rng()
                    _V_tau_voice = draw_voice_indicator(_pi_realized_tab3[cov_perp], _rng_voice)
                    _x_tau_obs_voice = (
                        sample_voice_observation(cov_perp, VOZ_PARAMS_DEFAULT, _rng_voice)
                        if _V_tau_voice == 1 else None
                    )
                    _L_C_tau_voice, _L_voz_tau_voice = communication_likelihood_LC(
                        cov_perp, V_t=_V_tau_voice, omega_voz=float(omega_voz),
                        pi_call=_pi_realized_tab3, x_obs=_x_tau_obs_voice,
                        voz_params_by_theta=VOZ_PARAMS_DEFAULT,
                    )
                    st.session_state["voice_path"].append({
                        "tau": _tau_new_voice, "V_t": int(_V_tau_voice),
                        "x_obs": _x_tau_obs_voice.tolist() if _x_tau_obs_voice is not None else None,
                        "L_voz": float(_L_voz_tau_voice) if _V_tau_voice == 1 else None,
                        "L_C": float(_L_C_tau_voice),
                    })

                # ── m (tau=1,2,...): sorteo genuino del resultado fisico (bloque de riesgos
                # competitivos), reusando cvn.outcome_probs_grid SIN CAMBIOS (eq:hj/eq:pCont/
                # eq:xi, Working_paper_eng.tex; eq:LH-compacta, Bernal_H.tex) -- evaluado en
                # (alpha_1*,gamma_1*) YA optimos del Estado y M_{tau=1}, para el tipo verdadero
                # (mismo criterio que a_K/voz). Transformada inversa: mismo orden G_t que el
                # expander de Block C en tau=0 (Cont<pay<kill<res<rel). Se acumula en
                # session_state["m_path"] (extensible a tau=2,3,...). Ilustrativo: NO realimenta
                # mu_1 (ya calculado con el m de tau=0). Si m_1 != Cont, el episodio cierra en
                # tau=1 por definicion (eq:stopping-time, Bernal_H.tex) -- se marca explicitamente.
                _tau_new_m = len(st.session_state["m_path"]) + 1
                _probs_m_true, _pdet_m_true = cvn.outcome_probs_grid(
                    _a_star, _g_star, cov_perp, _p_opt_tab3, _mt_1_tab3,
                )
                _v_draw_m = float(np.random.default_rng().random())
                _bound_cont_m = _probs_m_true["Cont"]
                _bound_1_m = _bound_cont_m + _probs_m_true["1"]
                _bound_2_m = _bound_1_m + _probs_m_true["2"]
                _bound_3_m = _bound_2_m + _probs_m_true["3"]
                if _v_draw_m <= _bound_cont_m:
                    _m_outcome_key1 = "Cont"
                elif _v_draw_m <= _bound_1_m:
                    _m_outcome_key1 = "1"
                elif _v_draw_m <= _bound_2_m:
                    _m_outcome_key1 = "2"
                elif _v_draw_m <= _bound_3_m:
                    _m_outcome_key1 = "3"
                else:
                    _m_outcome_key1 = "4"
                _m_label_map_tab3 = {
                    "Cont": "Continue Captivity (cont)", "1": "Ransom Paid (j=1)",
                    "2": "Victim Deceased (j=2)", "3": "Tactical Rescue (j=3)",
                    "4": "Exogenous Release (j=4)",
                }
                _m_outcome_tau1 = _m_label_map_tab3[_m_outcome_key1]
                _m_closes_episode1 = _m_outcome_key1 != "Cont"
                st.session_state["m_path"].append({
                    "tau": _tau_new_m, "v": _v_draw_m, "outcome_key": _m_outcome_key1,
                    "outcome": _m_outcome_tau1, "probs": dict(_probs_m_true),
                    "closes_episode": _m_closes_episode1,
                })

                # ── kappa_h(theta_K,1) and -sgn per type: MISMA formula `eq:hj` (Working_paper_
                # eng.tex) que _outcome_probs_tab3 usa para tau=0 (linea ~1519 arriba) --
                # incluye los mismos terminos phi_F/phi_K/zeta_R (indicadores de las acciones
                # EJECUTADAS a_tilde^F/K/S, "material state" C_t) -- pero reevaluada en
                # (alpha_1*,gamma_1*), M_{tau=1} (literal eq:m-t, real H(mu_1)/H(mu_0)), y las acciones REALIZADAS de
                # tau=1 (act_f_tau1/act_k_tau1/act_s_tau1, ya sorteadas arriba en este mismo
                # click) en vez de session_state["act_f"/"act_k"/"act_s"] (las de tau=0).
                def _lam123_tau1(tipo_kh1: str):
                    beta_K_1_kh1 = BETAS_K[tipo_kh1]["Pago"]
                    beta_K_2_kh1 = BETAS_K[tipo_kh1]["Muerte"]
                    beta_K_3_kh1 = BETAS_K[tipo_kh1]["Rescate"]
                    beta_z_kh1 = BETAS_Z[cov_zone]
                    beta_F_1_kh1 = 0.80 if cov_wealth == "High" else 0.00
                    beta_V_1_kh1 = 1.36 if cov_vict == "Public sector" else 0.00
                    beta_S_kh1 = 0.50 if cov_state == "Lax" else 0.00
                    # CORREGIDO (aprobado por el usuario, fundamentado en eq:hj/C_t(theta_K)
                    # de ambos papers -- el "material state" debe reflejar la accion
                    # REALMENTE ejecutada de ese periodo): el espacio de la Familia en
                    # tau>=1 es "Cooperate"/"Collude", nunca "Pay" ("Pay" solo existe en el
                    # simulador exogeno de tau=0) -- el check anterior =="Pay" nunca disparaba,
                    # dejando phi_F_1/2/3 siempre en 0.00 silenciosamente.
                    phi_F_1_kh1 = 3.20 if _act_f_tau1 == "Cooperate" else 0.00
                    phi_K_1_kh1 = -1.15 if _act_k_tau1 == "Continue" else 0.00
                    phi_F_2_kh1 = -1.50 if _act_f_tau1 == "Cooperate" else 0.00
                    phi_K_kill_2_kh1 = 4.00 if _act_k_tau1 == "Kill" else 0.00
                    phi_K_cont_2_kh1 = 0.50 if _act_k_tau1 == "Continue" else 0.00
                    zeta_R_3_kh1 = 2.50 if _act_s_tau1 == "Rescue" else 0.00
                    phi_F_3_kh1 = -1.00 if _act_f_tau1 == "Cooperate" else 0.00
                    phi_K_3_kh1 = 0.50 if _act_k_tau1 == "Continue" else 0.00
                    za_kh1 = ZETAS_POLITICA[tipo_kh1]["zeta_alpha"]
                    zg_kh1 = ZETAS_POLITICA[tipo_kh1]["zeta_gamma"]
                    eta_0_kh1 = ETA_0_PDET[tipo_kh1]
                    u_det_kh1 = eta_0_kh1 + ETA_1_PDET * _a_star + ETA_2_PDET * _g_star
                    p_det_kh1 = 1.0 / (1.0 + np.exp(-u_det_kh1))
                    idx_pago_kh1 = (
                        beta_K_1_kh1 + beta_z_kh1 + beta_F_1_kh1 - beta_V_1_kh1 + beta_S_kh1
                        - za_kh1 * _a_star - zg_kh1 * _g_star - zd * p_det_kh1 + phi_F_1_kh1 + phi_K_1_kh1
                    )
                    idx_muerte_kh1 = (
                        beta_K_2_kh1 + beta_z_kh1 + beta_S_kh1
                        + za_kh1 * _a_star + zg_kh1 * _g_star - zd * p_det_kh1
                        - phi_F_2_kh1 + phi_K_kill_2_kh1 + phi_K_cont_2_kh1
                    )
                    idx_rescate_kh1 = (
                        -beta_S_kh1 + beta_K_3_kh1 + beta_z_kh1
                        + za_kh1 * _a_star + zg_kh1 * _g_star + zd * p_det_kh1
                        + zeta_R_3_kh1 - phi_F_3_kh1 + phi_K_3_kh1
                    )
                    _lam_pay_kh1 = _mt_1_tab3 * LAMBDAS_0["Pago"] * np.exp(idx_pago_kh1)
                    _lam_kill_kh1 = _mt_1_tab3 * LAMBDAS_0["Muerte"] * np.exp(idx_muerte_kh1)
                    _lam_res_kh1 = _mt_1_tab3 * LAMBDAS_0["Rescate"] * np.exp(idx_rescate_kh1)
                    return _lam_pay_kh1, _lam_kill_kh1, _lam_res_kh1, zg_kh1

                _kappa_h_tau1_by_type: dict[str, float] = {}
                _neg_sign_kappa_h_tau1_by_type: dict[str, int] = {}
                for _th_kh1 in ["DC", "PAR", "ELN", "FARC"]:
                    _lam1_kh1, _lam2_kh1, _lam3_kh1, _zg_kh1_v = _lam123_tau1(_th_kh1)
                    _kappa_h_tau1_by_type[_th_kh1] = float(_zg_kh1_v * (_lam2_kh1 + _lam3_kh1 - _lam1_kh1))
                    _neg_sign_kappa_h_tau1_by_type[_th_kh1] = -int(np.sign(_kappa_h_tau1_by_type[_th_kh1]))

                # ── d (tau=1,2,...): sorteo genuino de deteccion (eq:detection), NUEVO --
                # generaliza el boton de Block D (solo tau=0) a cada periodo, necesario para
                # que mu pueda encadenarse tau->tau+1 (eq:bayes-update requiere m_tau Y d_tau).
                # Reusa rp.draw_d (run_period.py) -- misma formula, sin duplicar.
                _d_tau1_result = rp.draw_d(_a_star, _g_star, cov_perp, _p_opt_tab3, np.random.default_rng())

                # ── mu_2: primer paso de la cadena mu_tau -> mu_{tau+1} (eq:bayes-update),
                # GENERICO (rp.bayes_update_mu, cvn.outcome_probs_grid/p_cap_eff_grid
                # tau-parametrizados) -- NO la closure especifica de tau=0
                # (_outcome_probs_tab3, que lee session_state["act_f"] fijo en tau=0).
                # Ilustrativo/preparatorio: mu_1 (ya reportado arriba) NO se recalcula ni
                # cambia -- mu_2 es el punto de partida para el ciclo tau=2...T_max de abajo.
                _voice_L_C_by_type_tau1 = {th: 1.0 for th in cvn.TIPOS}
                if _voice_available_tab3:
                    for _th_vlc in cvn.TIPOS:
                        _l_c_th, _ = communication_likelihood_LC(
                            _th_vlc, V_t=_V_tau_voice, omega_voz=float(omega_voz),
                            pi_call=_pi_realized_tab3, x_obs=_x_tau_obs_voice,
                            voz_params_by_theta=VOZ_PARAMS_DEFAULT,
                        )
                        _voice_L_C_by_type_tau1[_th_vlc] = float(_l_c_th)
                _mu2_seed_tab3 = rp.bayes_update_mu(
                    _mu1_tab3, _a_star, _g_star, _mt_1_tab3, _p_opt_tab3,
                    _m_outcome_key1, _d_tau1_result["d"], _act_k_tau1, _voice_L_C_by_type_tau1,
                )

                st.session_state["tau1_state_opt_result"] = {
                    "alpha": _a_star, "gamma": _g_star, "a_S": _aS_star,
                    "V_by_type": _v_by_type, "feasible": _feasible,
                    "H_mu": _extra.get("H_mu", 0.0), "delta_H": _extra.get("delta_H", 0.0),
                    "ir_k_true_gap": _extra.get("ir_k_true_gap", 0.0), "T_trained": _T_trained_tab3,
                    "T_tau1": _mt_1_tab3, "floor_selected": _extra.get("floor_selected", 0.0),
                    "T_tau1_S": _mt_1_S_tab3, "T_tau1_F": _mt_1_F_tab3, "T_tau1_K": _mt_1_K_tab3,
                    "act_s_tau1": _act_s_tau1, "u_s_tau1": _u_s_1,
                    "p_rescue_tau1": _p_rescue_1, "p_nego_tau1": _p_nego_1,
                    "a_F_star1": _a_F_star1, "U_coop_f1": _U_coop_f1, "U_col_f1": _U_col_f1,
                    "ir_f_gap1": _ir_f_gap1, "fam_extras1": _fam_extras1,
                    "act_f_tau1": _act_f_tau1, "u_f_tau1": _u_f_1,
                    "p_coop_tau1": _p_coop_1, "p_col_tau1": _p_col_1,
                    "a_K_star1": _a_K_star1, "branch_true1": _branch_true1,
                    "U_rel_true1": _U_rel_true1, "U_kill_true1": _U_kill_true1,
                    "V_cont_true1": _V_cont_true1, "k_extras1": _k_extras1,
                    "act_k_tau1": _act_k_tau1, "u_k_tau1": _u_k_1,
                    "p_cont_k_tau1": _p_cont_k1, "p_rel_k_tau1": _p_rel_k1, "p_kill_k_tau1": _p_kill_k1,
                    "using_fallback_net": _using_fallback_net_tab3,
                    "voice_available": _voice_available_tab3,
                    "voice_tau": (st.session_state["voice_path"][-1]["tau"] if _voice_available_tab3 else None),
                    "voice_V": (st.session_state["voice_path"][-1]["V_t"] if _voice_available_tab3 else None),
                    "voice_x_obs": (st.session_state["voice_path"][-1]["x_obs"] if _voice_available_tab3 else None),
                    "voice_L_voz": (st.session_state["voice_path"][-1]["L_voz"] if _voice_available_tab3 else None),
                    "voice_L_C": (st.session_state["voice_path"][-1]["L_C"] if _voice_available_tab3 else None),
                    "voice_pi_call_realized_true": (
                        _pi_realized_tab3[cov_perp] if _voice_available_tab3 else None
                    ),
                    "m_tau": _tau_new_m, "m_v": _v_draw_m, "m_outcome_key": _m_outcome_key1,
                    "m_outcome": _m_outcome_tau1, "m_probs": dict(_probs_m_true),
                    "m_closes_episode": _m_closes_episode1,
                    "kappa_h_tau1": dict(_kappa_h_tau1_by_type),
                    "neg_sign_kappa_h_tau1": dict(_neg_sign_kappa_h_tau1_by_type),
                    "d_tau1": _d_tau1_result["d"], "d_u_tau1": _d_tau1_result["u"],
                    "p_det_tau1": _d_tau1_result["p_det"], "mu2_seed": dict(_mu2_seed_tab3),
                    "mu_tau": dict(_mu1_tab3),
                    "benchmarks_tau1": _extra.get("benchmarks_by_type", {}),
                    "alpha_R_mu": _extra.get("alpha_R_mu", 0.5),
                    "gamma_R_mu": _extra.get("gamma_R_mu", 0.5),
                    "alpha_N_mu": _extra.get("alpha_N_mu", 0.5),
                    "gamma_N_mu": _extra.get("gamma_N_mu", 0.5),
                }

                # ── ciclo tau=2...T_max (extension APPROVED por el usuario): reusa
                # run_period.run_one_period (misma formula que tau=1 arriba, generalizada),
                # sembrado con mu2_seed. Se detiene en el primer cierre (m_tau!=Cont,
                # eq:stopping-time) salvo que se marque la extension contrafactual.
                st.session_state["tau_history"] = {1: dict(st.session_state["tau1_state_opt_result"])}
                _t_max_tab3 = int(st.session_state.get("t_max_input", 1))
                if _t_max_tab3 > 1:
                    _mu_running_tab3 = dict(_mu2_seed_tab3)
                    _p_rescue_run_tab3, _p_nego_run_tab3 = _p_rescue_1, _p_nego_1
                    _rng_loop_tab3 = np.random.default_rng()
                    _closed_at_tab3 = None
                    _counterfactual_tab3 = bool(st.session_state.get("counterfactual_ext_input", True))
                    _progress_tab3 = st.progress(0.0, text=f"Simulating τ=2…{_t_max_tab3}")
                    for _tau_loop in range(2, _t_max_tab3 + 1):
                        if _closed_at_tab3 is not None and not _counterfactual_tab3:
                            break
                        _r_loop = rp.run_one_period(
                            _tau_loop, _mu_running_tab3, _H_mu0_tab3,
                            _p_rescue_run_tab3, _p_nego_run_tab3, cov_perp, _theta_f_opt_tab3,
                            _p_opt_tab3, _net_tab3, _net_for_captor_tab3,
                            (_pi_realized_tab3 if _voice_available_tab3 else None),
                            VOZ_PARAMS_DEFAULT, float(omega_voz), _rng_loop_tab3,
                            T0_S=float(T0_S), T0_F=float(T0_F), T0_K=float(T0_K),
                        )
                        st.session_state["tau_history"][_tau_loop] = _r_loop
                        if _r_loop["m"]["closes_episode"] and _closed_at_tab3 is None:
                            _closed_at_tab3 = _tau_loop
                        _mu_running_tab3 = _r_loop["mu_next"]
                        _p_rescue_run_tab3, _p_nego_run_tab3 = _r_loop["p_rescue_next"], _r_loop["p_nego_next"]
                        _progress_tab3.progress(
                            (_tau_loop - 1) / max(1, _t_max_tab3 - 1),
                            text=f"τ={_tau_loop}/{_t_max_tab3}" + (f" — closed at τ={_closed_at_tab3}" if _closed_at_tab3 else ""),
                        )
                    _progress_tab3.empty()
                    st.session_state["tau_display_max"] = max(st.session_state["tau_history"].keys())
                    st.session_state["tau_closed_at"] = _closed_at_tab3
                else:
                    st.session_state["tau_display_max"] = 1
                    st.session_state["tau_closed_at"] = None
            _voice_msg = ""
            if _voice_available_tab3:
                _voice_msg = (
                    f", V(voice,τ={_tau_new_voice})={'call' if _V_tau_voice == 1 else 'silence'} "
                    f"(L_C={_L_C_tau_voice:.4f})"
                )
            st.success(
                f"Done: a_S*={st.session_state['tau1_state_opt_result']['a_S']}, "
                f"α₁*={_a_star:.4f}, γ₁*={_g_star:.4f}, Γ₁(μ₁) factible={_feasible}, "
                f"ã_S(τ=1)={_act_s_tau1} (u={_u_s_1:.4f}), "
                f"a_F¹*={_a_F_star1}, ã_F(τ=1)={_act_f_tau1}, "
                f"a_K¹*({cov_perp})={_a_K_star1}, ã_K(τ=1)={_act_k_tau1}"
                f"{_voice_msg}, m(τ={_tau_new_m})={_m_outcome_tau1} (v={_v_draw_m:.4f})"
            )
            if _using_fallback_net_tab3:
                st.warning(
                    "`captor_true_type_value_net_T10.pt` does not exist yet — the Captor block "
                    "(true type) temporarily used the State's network "
                    "(`captor_value_net_T10.pt`, μ-mixed marginal weight) for continuation, "
                    "instead of its own network (true-type weight). Retrain and restart the "
                    "server to use the correct network."
                )
            if not _voice_available_tab3:
                if not _RB_AVAILABLE:
                    st.warning(
                        "`rational_behavior.py` could not be imported — the genuine voice draw "
                        "(τ≥1) was not generated; the $V$(voice) row of Table 5.2 only shows τ=0 "
                        "(Block F's static scalar mechanism)."
                    )
                else:
                    st.info("Voice evidence excluded from posterior updates by user setting (L_C = 1.0).")
            if _m_closes_episode1:
                st.warning(
                    f"$m_1=${_m_outcome_tau1} ≠ Cont ⇒ by `eq:stopping-time` (Bernal_H.tex), "
                    f"the episode **closes at τ=1**. Any extension to τ=2+ would be "
                    "counterfactual under an absorbing state (the same treatment the paper "
                    "itself uses in its calibration), not a real trajectory."
                )

    _grid_ag = np.linspace(0.0, 1.0, 101)
    _G_alpha, _G_gamma = np.meshgrid(_grid_ag, _grid_ag)

    # omega_p, omega_k rescaled (from the 15 / 200,000 pair used elsewhere) so that the
    # ransom-transfer term and the hazard term are comparable in magnitude to C_ops/
    # C_maint — otherwise one term mechanically dominates and forces a corner solution
    # regardless of the cost-function shape. R enters here in millions of COP (the
    # slider's native unit) rather than raw pesos, for the same reason.
    _omega_p_tab3 = 0.15
    _omega_k_tab3 = 200.0

    _p_opt_tab3 = cvn.Params(
        R_millions=float(ransom_R_millions),
        beta_tilde=dict(BETA_TILDE_TAB1),
        cov_zone=cov_zone, cov_wealth=cov_wealth, cov_vict=cov_vict, cov_state=cov_state,
        T_mad=float(T_mad), T0=float(T_0), eta_cal=float(eta_cal), c_bar=float(c_bar),
        H_ratio=float(H_ratio), lambda_4=float(lambda_4), eta_1=float(eta_1),
        eta_2=float(eta_2), beta_R=float(beta_R),
    )
    _mt_tab3 = cvn.m_t(0, _p_opt_tab3)
    _v_next_fn_local = _v_next_fn_tab3 if (_run_state_opt_clicked and '_v_next_fn_tab3' in locals()) else None
    _benchmarks_tab3: dict[str, dict[str, float]] = {}
    for _th_bm in ["DC", "PAR", "ELN", "FARC"]:
        # Rescue branch: p_surv does not depend on (alpha,gamma) (Block E formula), so
        # argmin V_0^R reduces to argmin C_ops(gamma,alpha;theta_K).
        _V_R_grid = _c_ops_tab3(_G_gamma, _G_alpha, _th_bm)

        # Negotiation branch: genuine (alpha,gamma) trade-off via the death hazard and R.
        _probs_0, _pdet_0 = cvn.outcome_probs_grid(_G_alpha, _G_gamma, _th_bm, _p_opt_tab3, _mt_tab3)
        _h2_0 = _probs_0["2"]
        _V_N_grid = (
            _omega_p_tab3 * float(ransom_R_millions) * (1.0 - _G_alpha)
            + _omega_k_tab3 * _h2_0
            + _c_maint_tab3(_G_gamma, _G_alpha, _th_bm)
        )

        # Evaluate Feasibility under degenerate belief mu_theta = 1.0 (for this type)
        # Restricción del Secuestrador (IR^K)
        _U_rel_0 = -cvn.KAPPA_REL[_th_bm]
        _p_cap_0 = cvn.p_cap_eff_grid(_G_alpha, _G_gamma, _th_bm, _p_opt_tab3, 0.5, 0.5)
        _U_kill_0 = (1 - _p_cap_0) * cvn.ETA_REP[_th_bm] - _p_cap_0 * cvn.F_CAP[_th_bm]
        _C_tau_0 = cvn.PHI_COST[_th_bm] * np.exp(cvn.KAPPA_C[_th_bm] * _G_gamma) + cvn.NU_COST[_th_bm]
        _p_pay_0 = cvn.p_pay_eff_grid(_G_alpha, _G_gamma, _th_bm, _p_opt_tab3, _pdet_0, _mt_tab3, 0.5, 0.5)
        
        if _v_next_fn_local is not None:
            _mu_degenerate_0 = {th: np.ones_like(_G_alpha) if th == _th_bm else np.zeros_like(_G_alpha) for th in cvn.TIPOS}
            _v_next_acc_0 = _v_next_fn_local(_mu_degenerate_0, 1, _p_opt_tab3)
            _V_cont_next_0 = _v_next_acc_0[_th_bm]
        else:
            _V_cont_next_0 = np.zeros_like(_G_alpha)

        _V_cont_0 = (
            _p_pay_0 * float(ransom_R_millions) * (1.0 - _G_alpha) - _C_tau_0 - _p_cap_0 * cvn.F_CAP[_th_bm]
            + _p_opt_tab3.beta_tilde[_th_bm] * (1.0 - _p_cap_0) * _V_cont_next_0
        )
        _ir_k_feasible_0 = (_U_rel_0 - np.maximum(_V_cont_0, _U_kill_0)) >= -1e-9

        # Restricción de la Familia (IR^F)
        _p_surv_0 = cvn.p_surv_raw(_th_bm, _th_bm, _p_opt_tab3)
        _wealth_key_0 = "High wealth" if cov_wealth == "High" else "Low wealth"
        _e_tau_f_0 = cvn.PHI_F[_wealth_key_0] * np.exp(cvn.KAPPA_F[_wealth_key_0] * _G_gamma) + cvn.NU_F[_wealth_key_0]
        _U_coop_f_0 = _p_surv_0 * cvn.V_L_FAMILY - _e_tau_f_0
        _U_col_f_0 = _probs_0["4"] * cvn.V_L_FAMILY - float(ransom_R_millions) - _pdet_0 * cvn.F_COL
        _ir_f_feasible_0 = _U_coop_f_0 >= _U_col_f_0

        _feasible_mask_0 = _ir_k_feasible_0 & _ir_f_feasible_0

        if np.any(_feasible_mask_0):
            _V_R_masked = np.where(_feasible_mask_0, _V_R_grid, np.inf)
            _V_N_masked = np.where(_feasible_mask_0, _V_N_grid, np.inf)
            _idx_R = np.unravel_index(np.argmin(_V_R_masked), _V_R_masked.shape)
            _idx_N = np.unravel_index(np.argmin(_V_N_masked), _V_N_masked.shape)
        else:
            _idx_R = np.unravel_index(np.argmin(_V_R_grid), _V_R_grid.shape)
            _idx_N = np.unravel_index(np.argmin(_V_N_grid), _V_N_grid.shape)

        _alpha_R_star = float(_G_alpha[_idx_R])
        _gamma_R_star = float(_G_gamma[_idx_R])
        _alpha_N_star = float(_G_alpha[_idx_N])
        _gamma_N_star = float(_G_gamma[_idx_N])

        _benchmarks_tab3[_th_bm] = {
            "alpha_R": _alpha_R_star, "gamma_R": _gamma_R_star,
            "alpha_N": _alpha_N_star, "gamma_N": _gamma_N_star,
            "V_R": float(_V_R_grid[_idx_R]), "V_N": float(_V_N_grid[_idx_N]),
        }

    _alpha_R_mu_tab3 = sum(_mu0_tab3[th] * _benchmarks_tab3[th]["alpha_R"] for th in _mu0_tab3)
    _gamma_R_mu_tab3 = sum(_mu0_tab3[th] * _benchmarks_tab3[th]["gamma_R"] for th in _mu0_tab3)
    _alpha_N_mu_tab3 = sum(_mu0_tab3[th] * _benchmarks_tab3[th]["alpha_N"] for th in _mu0_tab3)
    _gamma_N_mu_tab3 = sum(_mu0_tab3[th] * _benchmarks_tab3[th]["gamma_N"] for th in _mu0_tab3)

    # tau=1 per-type perfect-info benchmarks (gamma_R^theta,*, alpha_R^theta,*, gamma_N^theta,*,
    # alpha_N^theta,*): genuinely recomputed argmins of V_1^R and V_1^N at tau=1.
    # Replaced redundant recalculation here, as these are now computed dynamically inside
    # cvn.solve_state_problem and populated directly in st.session_state["tau1_state_opt_result"]
    # upon execution of Tab 3.
    pass

    st.markdown("---")

    def _adapt_period_to_display(_r: dict) -> dict:
        """Traduce las claves de run_period.run_one_period() a las mismas claves que ya
        usa tau1_state_opt_result (§Tabla 5.2/popover), para reusar el renderizado
        existente sin reescribirlo por cada tau. Benchmarks per-tipo, kappa_h y los
        alpha/gamma ponderados por creencia ya se generalizan a cualquier tau (ver
        run_period.compute_benchmarks_tau/compute_kappa_h_tau/compute_belief_weighted);
        solo la fila de voz sigue ausente si `rational_behavior.py` no está disponible."""
        return {
            "alpha": _r["alpha"], "gamma": _r["gamma"], "a_S": _r["a_S"], "feasible": _r["feasible"],
            "H_mu": _r["H_mu"], "delta_H": _r["delta_H"], "ir_k_true_gap": _r["ir_k_true_gap"],
            "T_trained": rp.T_TRAINED, "T_tau1": _r["T_generic"], "T_tau1_S": _r["T_S"],
            "T_tau1_F": _r["T_F"], "T_tau1_K": _r["T_K"], "floor_selected": float("nan"),
            "act_s_tau1": _r["act_s"], "u_s_tau1": _r["u_s"],
            "p_rescue_tau1": _r["p_rescue"], "p_nego_tau1": _r["p_nego"],
            "a_F_star1": _r["a_F_star"], "U_coop_f1": _r["U_coop_f"], "U_col_f1": _r["U_col_f"],
            "ir_f_gap1": _r["ir_f_gap"], "act_f_tau1": _r["act_f"], "u_f_tau1": _r["u_f"],
            "p_coop_tau1": _r["p_coop"], "p_col_tau1": _r["p_col"],
            "a_K_star1": _r["a_K_star"], "branch_true1": _r["branch_true"],
            "U_rel_true1": _r["U_rel_true"], "U_kill_true1": _r["U_kill_true"],
            "V_cont_true1": _r["V_cont_true"], "act_k_tau1": _r["act_k"], "u_k_tau1": _r["u_k"],
            "p_cont_k_tau1": _r["p_cont_k"], "p_rel_k_tau1": _r["p_rel_k"], "p_kill_k_tau1": _r["p_kill_k"],
            "m_tau": _r["m"]["v"], "m_v": _r["m"]["v"], "m_outcome_key": _r["m"]["outcome_key"],
            "m_outcome": _r["m"]["outcome"], "m_probs": _r["m"]["probs"],
            "m_closes_episode": _r["m"]["closes_episode"],
            "d_tau1": _r["d"]["d"], "d_u_tau1": _r["d"]["u"], "p_det_tau1": _r["d"]["p_det"],
            "voice_available": _r["voice"] is not None,
            "voice_tau": _r["tau"] if _r["voice"] else None,
            "voice_V": _r["voice"]["V_t"] if _r["voice"] else None,
            "voice_x_obs": _r["voice"]["x_obs"] if _r["voice"] else None,
            "voice_L_voz": _r["voice"]["L_voz"] if _r["voice"] else None,
            "voice_L_C": _r["voice"]["L_C"] if _r["voice"] else None,
            "voice_pi_call_realized_true": (
                st.session_state.get("voice_pi_call_realized", {}).get(cov_perp)
                if _r["voice"] else None
            ),
            "using_fallback_net": False, "mu_tau": dict(_r["mu_tau"]),
            "benchmarks_tau1": _r.get("benchmarks_by_type", {}),
            "neg_sign_kappa_h_tau1": _r.get("neg_sign_kappa_h"),
            "alpha_R_mu": _r.get("alpha_R_mu", 0.5), "gamma_R_mu": _r.get("gamma_R_mu", 0.5),
            "alpha_N_mu": _r.get("alpha_N_mu", 0.5), "gamma_N_mu": _r.get("gamma_N_mu", 0.5),
        }

    _tau_display_max_tab3 = int(st.session_state.get("tau_display_max", 1))
    _tau_history_tab3 = st.session_state.get("tau_history", {})
    if _tau_display_max_tab3 > 1 and _tau_history_tab3:
        _sel_col1, _sel_col2 = st.columns([1, 3])
        with _sel_col1:
            _tau_view_sel = st.number_input(
                "View τ =", min_value=1, max_value=_tau_display_max_tab3, value=1, step=1,
                key="tau_view_selector",
            )
        with _sel_col2:
            _closed_note = st.session_state.get("tau_closed_at")
            if _closed_note:
                st.caption(f"⚠ Episode closed at τ={_closed_note} (m≠Cont, `eq:stopping-time`) — "
                            f"later periods are a counterfactual extension under an absorbing state.")
            st.caption(f"Simulated: τ=1…{_tau_display_max_tab3}. The “τ=1” column below shows the τ chosen above.")
    else:
        _tau_view_sel = 1

    if _tau_history_tab3:
        st.markdown(
            f"#### Table 5.2 — summary of all simulated τ "
            f"(τ=1…{_tau_display_max_tab3}, per chosen T_max)"
        )

        def _row_from_normalized_tab3(_tau_i: int, _nd: dict) -> dict:
            """Una fila por τ con TODAS las variables de la Tabla 5.2 detallada (misma
            fuente/fórmula que esa tabla -- ver _adapt_period_to_display y
            tau1_state_opt_result, de donde _nd sale). Se transpone (τ como columnas)
            más abajo con .T antes de mostrarla."""
            _mu_d = _nd.get("mu_tau") or {}
            _bench = _nd.get("benchmarks_tau1") or {}
            _kh = _nd.get("neg_sign_kappa_h_tau1") or {}
            _row = {
                "τ": _tau_i,
                "a_F*": _nd.get("a_F_star1"),
                "ã_F": _nd.get("act_f_tau1"),
                f"a_K*({cov_perp})": _nd.get("a_K_star1"),
                f"ã_K({cov_perp})": _nd.get("act_k_tau1"),
                "a_S*": _nd.get("a_S"),
                "ã_S": _nd.get("act_s_tau1"),
                "α_t*": _nd.get("alpha"),
                "γ_t*": _nd.get("gamma"),
            }
            # Labels sin llaves de LaTeX crudas ("gamma_R(DC)*" en vez de "gamma_R^{DC},*") --
            # st.dataframe no renderiza LaTeX, así que la sintaxis ^{...} se veía literal.
            for _f, _lbl in [("gamma_R", "γ_R"), ("alpha_R", "α_R"), ("gamma_N", "γ_N"), ("alpha_N", "α_N")]:
                for _th_row in ["DC", "PAR", "ELN", "FARC"]:
                    _b = _bench.get(_th_row)
                    _row[f"{_lbl}({_th_row})*"] = _b[_f] if _b else None
            _row["H(μ_τ)"] = _nd.get("H_mu")
            _row["ΔH"] = _nd.get("delta_H")
            _row["Γ_τ(μ_τ) feasible"] = bool(_nd.get("feasible")) if _nd.get("feasible") is not None else None
            _gap = _nd.get("ir_k_true_gap")
            _row[f"IR^K({cov_perp}) OK"] = ("Yes" if _gap >= -1e-9 else "No") if _gap is not None else None
            for _th_row in ["DC", "PAR", "ELN", "FARC"]:
                _row[f"-sgn(κ_h({_th_row}))"] = _kh.get(_th_row)
            _row["m outcome"] = _nd.get("m_outcome")
            _row["v (m draw)"] = _nd.get("m_v")
            _row["closes episode"] = bool(_nd.get("m_closes_episode")) if _nd.get("m_closes_episode") is not None else None
            for _th_row in ["DC", "PAR", "ELN", "FARC"]:
                _row[f"μ_{_th_row}"] = _mu_d.get(_th_row)
            _row["α_R^μ"] = _nd.get("alpha_R_mu")
            _row["γ_R^μ"] = _nd.get("gamma_R_mu")
            _row["α_N^μ"] = _nd.get("alpha_N_mu")
            _row["γ_N^μ"] = _nd.get("gamma_N_mu")
            _row["L_C(voice)"] = _nd.get("voice_L_C") if st.session_state.get("include_voice_posterior_checkbox", True) else 1.0
            _vv = _nd.get("voice_V")
            _row["V_τ (signal)"] = ("Call" if _vv == 1 else "Silence") if (_vv is not None and st.session_state.get("include_voice_posterior_checkbox", True)) else ("Ignored" if not st.session_state.get("include_voice_posterior_checkbox", True) else None)
            _row["d (detection)"] = (bool(_nd.get("d_tau1")) if _nd.get("d_tau1") is not None else None)
            _row["p_det"] = _nd.get("p_det_tau1")
            _row["ι = max_θ μ_τ(θ)"] = (max(_mu_d.values()) if _mu_d else None)
            _row["M_t"] = _nd.get("T_tau1")
            _row["M_t^S"] = _nd.get("T_tau1_S")
            _row["M_t^F"] = _nd.get("T_tau1_F")
            _row["M_t^K"] = _nd.get("T_tau1_K")
            return _row

        _m_tau0_outcome_tab3 = st.session_state.get("m_tau0_outcome")
        _nd0_full_tab3 = {
            "a_F_star1": a_F_star, "act_f_tau1": st.session_state.get("act_f"),
            "a_K_star1": a_K_star, "act_k_tau1": st.session_state.get("act_k"),
            "a_S": a_S_star, "act_s_tau1": st.session_state.get("act_s"),
            "alpha": float(alpha_val), "gamma": float(gamma_val),
            "benchmarks_tau1": _benchmarks_tab3,
            "H_mu": _H_mu0_tab3, "delta_H": _delta_H_tab3,
            "feasible": _gamma_t_ev_tab3, "ir_k_true_gap": _ir_k_true_gap_tab3,
            "neg_sign_kappa_h_tau1": _neg_sign_kappa_h_by_type_tab3,
            "m_outcome": _m_tau0_outcome_tab3, "m_v": st.session_state.get("m_tau0_draw"),
            "m_closes_episode": (
                (_m_tau0_outcome_tab3 not in (None, "Continue Captivity (cont)"))
                if _m_tau0_outcome_tab3 is not None else None
            ),
            "mu_tau": _mu0_tab3,
            "alpha_R_mu": _alpha_R_mu_tab3, "gamma_R_mu": _gamma_R_mu_tab3,
            "alpha_N_mu": _alpha_N_mu_tab3, "gamma_N_mu": _gamma_N_mu_tab3,
            "voice_L_C": float(L_voz), "voice_V": (1 if voice_emitted else 0),
            "d_tau1": st.session_state.get("d_tau0_realized"),
            "p_det_tau1": st.session_state.get("d_tau0_pdet"),
            # T_tau1 (M_{tau=0}): CORRECTED to M(0)=min(1,(0/T_mad)^2)=0 identically, for any
            # T_mad>0 (eq:hj) -- no Params object needed. Previously reused the generic
            # Block-A T_t (temperature formula), the same bug already fixed for tau>=1
            # elsewhere in this file. _p_opt_tab3 is NOT in scope at this point (defined in
            # a different conditional branch), hence the hardcoded 0.0 rather than a call
            # to cvn.m_t(0, _p_opt_tab3).
            "T_tau1": 0.0, "T_tau1_S": float(T_t_S), "T_tau1_F": float(T_t_F), "T_tau1_K": float(T_t_K),
        }
        _row0_summary_tab3 = _row_from_normalized_tab3(0, _nd0_full_tab3)

        _rows_all_tau_tab3 = [_row0_summary_tab3]
        for _tau_i in sorted(_tau_history_tab3.keys()):
            _raw_i = _tau_history_tab3[_tau_i]
            _nd_i = (
                st.session_state["tau1_state_opt_result"]
                if _tau_i == 1
                else _adapt_period_to_display(_raw_i)
            )
            _rows_all_tau_tab3.append(_row_from_normalized_tab3(_tau_i, _nd_i))
        _df_all_tau_tab3 = pd.DataFrame(_rows_all_tau_tab3).set_index("τ").T

        def _fmt4_tab3(_v):
            if isinstance(_v, bool) or _v is None:
                return _v
            if isinstance(_v, (int, float)):
                return _v if _v != _v else f"{_v:.4f}"  # deja NaN tal cual (celda vacía)
            return _v

        st.dataframe(_df_all_tau_tab3.map(_fmt4_tab3), use_container_width=True)
        st.caption(
            "Transposed: rows = variables from the detailed Table 5.2 below, columns = τ "
            "(τ=0 exogenous + τ=1…T_max chosen). Same formulas/sources as that table (see "
            "descriptions there); empty cells (`None`) in the per-type benchmark rows "
            "(γ_R^θ, α_R^θ, γ_N^θ, α_N^θ), α^μ/γ^μ and κ_h(θ,τ) for τ≥2 indicate that those "
            "rows are only computed for τ=0 and τ=1 (not generalized to τ≥2 — a known scope "
            "limitation, not an error)."
        )

        st.markdown("##### MDG draws by τ — $\\tilde a_S,\\tilde a_F,\\tilde a_K$ and $m$")

        def _sorteos_row_from_normalized_tab3(_tau_i: int, _nd: dict) -> dict:
            return {
                "τ": _tau_i,
                "u_S": _nd.get("u_s_tau1"), "P(Rescue)": _nd.get("p_rescue_tau1"),
                "P(Negotiate)": _nd.get("p_nego_tau1"), "ã_S": _nd.get("act_s_tau1"),
                "u_F": _nd.get("u_f_tau1"), "P(Cooperate)": _nd.get("p_coop_tau1"),
                "P(Collude)": _nd.get("p_col_tau1"), "ã_F": _nd.get("act_f_tau1"),
                "u_K": _nd.get("u_k_tau1"), "P(Continue)": _nd.get("p_cont_k_tau1"),
                "P(Release)": _nd.get("p_rel_k_tau1"), "P(Kill)": _nd.get("p_kill_k_tau1"),
                f"ã_K({cov_perp})": _nd.get("act_k_tau1"),
                "v (m draw)": _nd.get("m_v"), "m outcome": _nd.get("m_outcome"),
                "u_d (d draw)": _nd.get("d_u_tau1"), "p_det": _nd.get("p_det_tau1"),
                "d realized": (bool(_nd.get("d_tau1")) if _nd.get("d_tau1") is not None else None),
            }

        _row0_sorteos_tab3 = {
            "τ": 0,
            "u_S": st.session_state.get("u_s"), "P(Rescue)": st.session_state.get("p_rescue_draw"),
            "P(Negotiate)": st.session_state.get("p_nego_draw"), "ã_S": st.session_state.get("act_s"),
            "u_F": st.session_state.get("u_f"), "P(Cooperate)": st.session_state.get("p_coop_draw"),
            "P(Collude)": st.session_state.get("p_col_draw"), "ã_F": st.session_state.get("act_f"),
            "u_K": st.session_state.get("u_k"), "P(Continue)": st.session_state.get("p_cont_draw"),
            "P(Release)": st.session_state.get("p_rel_draw"), "P(Kill)": st.session_state.get("p_kill_draw"),
            f"ã_K({cov_perp})": st.session_state.get("act_k"),
            "v (m draw)": st.session_state.get("m_tau0_draw"), "m outcome": _m_tau0_outcome_tab3,
            "u_d (d draw)": st.session_state.get("d_tau0_draw"), "p_det": st.session_state.get("d_tau0_pdet"),
            "d realized": (
                bool(st.session_state.get("d_tau0_realized"))
                if st.session_state.get("d_tau0_realized") is not None else None
            ),
        }
        _rows_sorteos_tab3 = [_row0_sorteos_tab3]
        _nd_by_tau_sorteos_tab3 = {0: _nd0_full_tab3}
        for _tau_i in sorted(_tau_history_tab3.keys()):
            _raw_i = _tau_history_tab3[_tau_i]
            _nd_i = (
                st.session_state["tau1_state_opt_result"]
                if _tau_i == 1
                else _adapt_period_to_display(_raw_i)
            )
            _nd_by_tau_sorteos_tab3[_tau_i] = _nd_i
            _rows_sorteos_tab3.append(_sorteos_row_from_normalized_tab3(_tau_i, _nd_i))
        # Persisted for the "4. Graphs" tab (reads this, does not rebuild anything on its own).
        st.session_state["tau_history_normalized"] = _nd_by_tau_sorteos_tab3
        _df_sorteos_tab3 = pd.DataFrame(_rows_sorteos_tab3).set_index("τ")
        st.dataframe(_df_sorteos_tab3, use_container_width=True)
        st.caption(
            "Uniform draw $u\\sim U(0,1)$ and the corresponding logit probability that produced "
            "each executed MDG action ($\\tilde a_S,\\tilde a_F,\\tilde a_K$), plus the physical "
            "outcome draw $m$ (`eq:LH-compacta`) and detection draw $d$ (`eq:detection`), for "
            "each τ (τ=0 exogenous from Tab 1 + τ=1…T_max simulated). ã_F at τ=0 uses the "
            "Pay/Cooperate labels from Tab 1 (Block Live Simulator, different from the "
            "Cooperate/Collude used at τ≥1); not modified, this is that section's existing "
            "behavior."
        )

        # Ejemplo trabajado: como leer una fila, usando datos REALES de la fila mas informativa
        # disponible (prefiere tau=1 si ya se corrio, si no cae a tau=0).
        _ej_tau_sorteos = 1 if 1 in _nd_by_tau_sorteos_tab3 else 0
        _ej_nd_sorteos = _nd_by_tau_sorteos_tab3[_ej_tau_sorteos]
        _ej_row_sorteos = next(r for r in _rows_sorteos_tab3 if r["τ"] == _ej_tau_sorteos)

        def _f4(_v):
            return f"{_v:.4f}" if isinstance(_v, (int, float)) and _v == _v else "—"

        _ej_u_s, _ej_p_r, _ej_p_n = _ej_row_sorteos["u_S"], _ej_row_sorteos["P(Rescue)"], _ej_row_sorteos["P(Negotiate)"]
        _ej_u_f, _ej_p_coop, _ej_p_col = _ej_row_sorteos["u_F"], _ej_row_sorteos["P(Cooperate)"], _ej_row_sorteos["P(Collude)"]
        _ej_u_k = _ej_row_sorteos["u_K"]
        _ej_p_cont, _ej_p_rel, _ej_p_kill = _ej_row_sorteos["P(Continue)"], _ej_row_sorteos["P(Release)"], _ej_row_sorteos["P(Kill)"]
        _ej_v_m = _ej_row_sorteos["v (m draw)"]
        _ej_probs_m = _ej_nd_sorteos.get("m_probs") or {}
        with st.expander(f"❓ How to read this table — example with row τ={_ej_tau_sorteos}", expanded=False):
            st.markdown(
                f"""
Each block (S, F, K, $m$) follows the **same pattern**: the logit probability of each option is
computed (`P(...)` columns), accumulated into $[0,1)$ intervals in the order they appear, and
the uniform draw `u` is compared against those intervals — the option whose interval contains
`u` is the one executed ($\\tilde a$).

**State ($\\tilde a_S$), row τ={_ej_tau_sorteos}:**
$P(\\text{{Rescue}})={_f4(_ej_p_r)}$, $P(\\text{{Negotiate}})={_f4(_ej_p_n)}$ → intervals
$[0,\\ {_f4(_ej_p_r)})=$ Rescue, $[{_f4(_ej_p_r)},\\ 1)=$ Negotiate.
Draw $u_S={_f4(_ej_u_s)}$ → falls in **{_ej_row_sorteos['ã_S']}**.

**Family ($\\tilde a_F$), row τ={_ej_tau_sorteos}:**
$P(\\text{{Cooperate}})={_f4(_ej_p_coop)}$, $P(\\text{{Collude}})={_f4(_ej_p_col)}$ → intervals
$[0,\\ {_f4(_ej_p_coop)})=$ Cooperate, $[{_f4(_ej_p_coop)},\\ 1)=$ Collude.
Draw $u_F={_f4(_ej_u_f)}$ → falls in **{_ej_row_sorteos['ã_F']}**.

**Captor ($\\tilde a_K$), row τ={_ej_tau_sorteos}** (3 options, same mechanism with 3 intervals):
$P(\\text{{Continue}})={_f4(_ej_p_cont)}$, $P(\\text{{Release}})={_f4(_ej_p_rel)}$, $P(\\text{{Kill}})={_f4(_ej_p_kill)}$ →
intervals $[0,\\ {_f4(_ej_p_cont)})=$ Continue, $[{_f4(_ej_p_cont)},\\ {_f4(_ej_p_cont+_ej_p_rel) if isinstance(_ej_p_cont,(int,float)) and isinstance(_ej_p_rel,(int,float)) else "—"})=$ Release,
remainder $=$ Kill. Draw $u_K={_f4(_ej_u_k)}$ → falls in **{_ej_row_sorteos[f'ã_K({cov_perp})']}**.

**Physical outcome $m$** (same mechanism, 5 intervals: Cont/1/2/3/4 — probabilities not shown as a
column in this table but computed: {", ".join(f"{k}={_f4(v)}" for k, v in _ej_probs_m.items()) if _ej_probs_m else "n/a"}).
Draw $v={_f4(_ej_v_m)}$ → falls in **{_ej_row_sorteos['m outcome']}**.

In short: **`u`/`v` is the random number; `P(...)` are the logit probabilities of each branch
(they depend on $T_\\tau^i$ — the lower the temperature, the narrower the interval for the
alternatives and the wider the one for the intended action); the result `ã`/`m outcome` is
simply which interval the draw fell into.**
                """
            )

        st.markdown("---")

    # _t1opt: usado por el popover "MDG draws" de abajo (la tabla Variable/Description/
    # valores tau=0/tau=1 que vivia aqui se movio a la pestaña "5. Description" -- ver
    # tambien "Table 5.2 -- summary of all simulated tau" mas arriba, que ya muestra estos
    # mismos valores para TODOS los tau, no solo tau=0/tau=1).
    if _tau_view_sel == 1:
        _t1opt = st.session_state.get("tau1_state_opt_result")
    else:
        _t1opt = _adapt_period_to_display(_tau_history_tab3[int(_tau_view_sel)])

    # ── detalle de los sorteos MDG (Estado/Familia/Secuestrador), unificado en un solo
    # popover con selector de τ. Popover nativo de Streamlit (clic, no hover) -- un
    # componente HTML/iframe no puede escuchar eventos de mouse sobre una tabla renderizada
    # por st.markdown; portar toda la Tabla 5.2 a HTML/JS para lograr hover real fue evaluado
    # y descartado por su riesgo de regresión visual sobre las ~40 filas existentes.
    def _render_mdg_draw_section(intent, probs, u_draw, atilde, eu=None, eu_label="\\mathbb{E}[U]"):
        argmax_pop = max(probs, key=probs.get)
        st.markdown(f"$\\mathbb{{P}}_I(\\tilde a \\mid a^{{*}} = \\text{{{intent}}})$")
        pi_md = "| Action | Prob. |  |\n|---|---|---|\n"
        for _act, _p in probs.items():
            _star = "⭐" if _act == intent else ""
            pi_md += f"| {_act} | {_p:.4f} | {_star} |\n"
        st.markdown(pi_md)
        int_md = "| Interval | Action | Prob. |  |\n|---|---|---|---|\n"
        curr = 0.0
        acts_list = list(probs.keys())
        for _i, (_act, _p) in enumerate(probs.items()):
            lo, hi = curr, curr + _p
            is_last = (_i == len(acts_list) - 1)
            hi_str = "1.0000" if is_last else f"{hi:.4f}"
            in_iv = (lo <= u_draw < hi) or (is_last and u_draw >= lo)
            mark = "🎯" if in_iv else ""
            int_md += f"| [{lo:.4f}, {hi_str}) | {_act} | {_p:.4f} | {mark} |\n"
            curr += _p
        st.markdown(int_md)
        st.success(f"$u = {u_draw:.4f} \\to \\tilde a = \\text{{{atilde}}}$ 🎯")
        st.caption(f"argmax $\\mathbb{{P}}_I$: **{argmax_pop}** ⭐" + (f" &ensp; ${eu_label} = {eu:.4f}$" if eu is not None else ""))

    _tau1_has_draw = bool(_t1opt and "act_s_tau1" in _t1opt)
    with st.popover("ℹ️ MDG draws — State / Family / Captor / Voice / m"):
        _tau_sel_options = [0, int(_tau_view_sel)] if _tau1_has_draw else [0]
        _tau_sel = st.radio(
            "τ", _tau_sel_options, horizontal=True, format_func=lambda t: f"τ={t}", key="_mdg_tau_selector_tab3",
        )
        if _tau_sel != 0:
            st.caption(f"Showing τ={int(_tau_view_sel)} (chosen above, in “View τ =”).")
        if _tau_sel == 0:
            st.markdown("##### State ($\\tilde a_S$)")
            _render_mdg_draw_section(
                a_S_star,
                {"Rescue": st.session_state["p_rescue_draw"], "Negotiate": st.session_state["p_nego_draw"]},
                st.session_state["u_s"], st.session_state["act_s"],
            )
            st.markdown("##### Family ($\\tilde a_F$)")
            _render_mdg_draw_section(
                a_F_star,
                {"Cooperate": st.session_state["p_coop_draw"], "Pay": st.session_state["p_col_draw"]},
                st.session_state["u_f"], st.session_state["act_f"],
            )
            st.markdown("##### Captor ($\\tilde a_K$)")
            _render_mdg_draw_section(
                a_K_star,
                {
                    "Continue": st.session_state["p_cont_draw"], "Release": st.session_state["p_rel_draw"],
                    "Kill": st.session_state["p_kill_draw"],
                },
                st.session_state["u_k"], st.session_state["act_k"],
            )
        else:
            _iota_1_tab3 = float(max(_mu1_tab3.values())) if _mu1_tab3 else float("nan")
            st.caption(f"$\\iota_1 = {_iota_1_tab3:.4f}$ — each player uses its own $T_t^i$ (approved extension, see below).")

            st.markdown("##### State ($\\tilde a_S$)")
            if "T_tau1_S" in _t1opt:
                st.caption(f"$T_t^S=M_{{\\tau=1}}^S={_t1opt['T_tau1_S']:.4f}$ ($T_0^S={float(T0_S):.2f}$ slider, `MDG_MULT_STATE eta_cal/c_bar`: 1.2/14.0)")
            _render_mdg_draw_section(
                _t1opt["a_S"], {"Rescue": _t1opt["p_rescue_tau1"], "Negotiate": _t1opt["p_nego_tau1"]},
                _t1opt["u_s_tau1"], _t1opt["act_s_tau1"],
                eu=-float(_t1opt["floor_selected"]), eu_label="\\mathbb{E}[U_S]",
            )

            st.markdown("##### Family ($\\tilde a_F$)")
            if "act_f_tau1" in _t1opt:
                if "T_tau1_F" in _t1opt:
                    st.caption(f"$T_t^F=M_{{\\tau=1}}^F={_t1opt['T_tau1_F']:.4f}$ ($T_0^F={float(T0_F):.2f}$ slider, `MDG_MULT_FAMILY eta_cal/c_bar`: 1.0/18.0)")
                _eu_f_pop = _t1opt["U_coop_f1"] if _t1opt["a_F_star1"] == "Cooperate" else _t1opt["U_col_f1"]
                _render_mdg_draw_section(
                    _t1opt["a_F_star1"], {"Cooperate": _t1opt["p_coop_tau1"], "Collude": _t1opt["p_col_tau1"]},
                    _t1opt["u_f_tau1"], _t1opt["act_f_tau1"],
                    eu=_eu_f_pop, eu_label="\\mathbb{E}[U_F]",
                )
            else:
                st.caption("Pending (press “Run State Optimization” again to generate it).")

            st.markdown("##### Captor ($\\tilde a_K$)")
            st.caption(
                "Pure logit `eq:logit-hybrid` (Working_paper_eng.tex Sec. 4.2.1) ≡ `eq:LI-atilde` "
                "(Bernal_H.tex) — same mechanism as ã_S/ã_F, centered on $a_K^{1,*}$, "
                f"$T_t^K=M_{{\\tau=1}}^K={_t1opt.get('T_tau1_K', _t1opt['T_tau1']):.4f}$ "
                f"($T_0^K={float(T0_K):.2f}$ slider, `MDG_MULT_CAPTOR eta_cal/c_bar`: 0.8/22.0 — "
                "literal `eq:m-t`, real $H(\\mu_1)/H(\\mu_0)$, Captor's own temperature)."
            )
            _eu_k_pop = max(_t1opt["U_rel_true1"], _t1opt["U_kill_true1"], _t1opt["V_cont_true1"])
            _render_mdg_draw_section(
                _t1opt["a_K_star1"],
                {
                    "Continue": _t1opt["p_cont_k_tau1"], "Release": _t1opt["p_rel_k_tau1"],
                    "Kill": _t1opt["p_kill_k_tau1"],
                },
                _t1opt["u_k_tau1"], _t1opt["act_k_tau1"],
                eu=_eu_k_pop, eu_label="\\mathbb{E}[U_K]",
            )

            st.markdown("##### Voice ($V_\\tau$, $x_\\tau^{obs}$)")
            if _t1opt.get("voice_available"):
                _pi_r_true = _t1opt["voice_pi_call_realized_true"]
                st.markdown(
                    f"$\\tilde\\pi_{{call}}({cov_perp})={_pi_r_true:.4f}$ "
                    f"(realized, drawn once; prior $\\pi_{{call}}({cov_perp})="
                    f"{PI_CALL[cov_perp]:.4f}$, $\\kappa={VOZ_KAPPA_REALIZED:.0f}$)"
                )
                if _t1opt["voice_V"] == 1:
                    st.success(
                        f"$V_{{\\tau={_t1opt['voice_tau']}}}=1$ (call) 🎯 — "
                        f"$x^{{obs}}=[{', '.join(f'{v:.2f}' for v in _t1opt['voice_x_obs'])}]$"
                    )
                    st.caption(
                        f"$\\mathcal L_{{voice,{_t1opt['voice_tau']}}}({cov_perp})={_t1opt['voice_L_voz']:.4f}$ "
                        f"&ensp; $\\mathcal L_{{C,{_t1opt['voice_tau']}}}({cov_perp})={_t1opt['voice_L_C']:.4f}$"
                    )
                else:
                    st.info(f"$V_{{\\tau={_t1opt['voice_tau']}}}=0$ (silence) — no $x^{{obs}}$")
                    st.caption(f"$\\mathcal L_{{C,{_t1opt['voice_tau']}}}({cov_perp})={_t1opt['voice_L_C']:.4f}$")
                st.caption(
                    "Genuine draw (`rational_behavior.py`, same mechanism as app.py): "
                    "$V_\\tau\\sim\\text{Bern}(\\tilde\\pi_{call})$, then $x_\\tau^{obs}=\\bar x(\\theta_K)+"
                    "\\varepsilon_L+\\varepsilon_S$ if $V_\\tau=1$ (`eq:voz-descomp`). Illustrative — "
                    "does not feed back into $\\mu_1$ (already computed with τ=0's fixed signal, Block F)."
                )
            else:
                st.caption("`rational_behavior.py` not available — voice draw not generated.")

            st.markdown("##### $m_\\tau$ (physical outcome — competing risks)")
            if "m_outcome" in _t1opt:
                _m_probs1 = _t1opt["m_probs"]
                _m_key_label = {
                    "Cont": "Cont", "1": "pay (j=1)", "2": "kill (j=2)",
                    "3": "res (j=3)", "4": "rel (j=4)",
                }
                _m_prob_md = "| Cause | Prob. |  |\n|---|---|---|\n"
                for _k_pop in ["Cont", "1", "2", "3", "4"]:
                    _star_m = "🎯" if _k_pop == _t1opt["m_outcome_key"] else ""
                    _m_prob_md += f"| {_m_key_label[_k_pop]} | {_m_probs1[_k_pop]:.4f} | {_star_m} |\n"
                st.markdown(_m_prob_md)
                _m_int_md = "| Interval | Cause | Prob. |  |\n|---|---|---|---|\n"
                _curr_m = 0.0
                for _i_pop, _k_pop in enumerate(["Cont", "1", "2", "3", "4"]):
                    _lo_pop, _hi_pop = _curr_m, _curr_m + _m_probs1[_k_pop]
                    _is_last_pop = _i_pop == 4
                    _hi_str_pop = "1.0000" if _is_last_pop else f"{_hi_pop:.4f}"
                    _in_iv_pop = (_lo_pop <= _t1opt["m_v"] < _hi_pop) or (_is_last_pop and _t1opt["m_v"] >= _lo_pop)
                    _mark_pop = "🎯" if _in_iv_pop else ""
                    _m_int_md += f"| [{_lo_pop:.4f}, {_hi_str_pop}) | {_m_key_label[_k_pop]} | {_m_probs1[_k_pop]:.4f} | {_mark_pop} |\n"
                    _curr_m = _hi_pop
                st.markdown(_m_int_md)
                st.success(f"$v_1 = {_t1opt['m_v']:.4f} \\to m_1 = \\text{{{_t1opt['m_outcome']}}}$ 🎯")
                if _t1opt["m_closes_episode"]:
                    st.warning(
                        r"$m_1\neq\text{cont}\Rightarrow\tau=1$ by `eq:stopping-time` "
                        "(Bernal_H.tex) — the episode closes at this period; any τ=2+ "
                        "would be counterfactual under an absorbing state."
                    )
                st.caption(
                    "Reuses `cvn.outcome_probs_grid` unchanged, evaluated at "
                    f"$(\\alpha_1^*,\\gamma_1^*,M_{{\\tau=1}})$ for {cov_perp} "
                    "(`eq:hj`/`eq:pCont`/`eq:xi`, Working_paper_eng.tex ≡ `eq:LH-compacta`, "
                    "Bernal_H.tex). Illustrative — does not feed back into $\\mu_1$."
                )
            else:
                st.caption("Pending (press “Run State Optimization” again to generate it).")

with tab4:
    st.markdown("## 📈 Graphs — Trajectory across τ")

    # --- PERSISTENT STORAGE SLOT MANAGER ---
    with st.expander("💾 Save & Load Simulation Runs (8 slots available)", expanded=True):
        # Display message toast if present
        if "saved_run_message" in st.session_state:
            st.toast(st.session_state["saved_run_message"])
            st.session_state.pop("saved_run_message", None)
            
        runs = load_saved_runs()
        col_left, col_right = st.columns(2)
        
        for slot_idx in range(1, 9):
            target_col = col_left if slot_idx <= 4 else col_right
            with target_col:
                slot_key = str(slot_idx)
                st.markdown(f"##### Slot {slot_idx}")
                
                if slot_key in runs:
                    run_info = runs[slot_key]
                    st.caption(f"🏷️ **{run_info['name']}**")
                    st.caption(f"🕒 {run_info['timestamp']}")
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.button(
                            f"▶️ Load", 
                            key=f"load_slot_{slot_idx}", 
                            use_container_width=True,
                            on_click=load_run_from_slot,
                            args=(slot_idx,)
                        )
                    with b_col2:
                        st.button(
                            f"🗑️ Clear", 
                            key=f"clear_slot_{slot_idx}", 
                            use_container_width=True,
                            on_click=delete_run_from_slot,
                            args=(slot_idx,)
                        )
                else:
                    st.caption("*[Empty]*")
                    name_input = st.text_input(
                        "Label (optional):", 
                        value="", 
                        placeholder="Label/Name (optional)", 
                        key=f"name_input_{slot_idx}", 
                        label_visibility="collapsed"
                    )
                    
                    active_run_exists = "tau_history" in st.session_state and bool(st.session_state["tau_history"])
                    st.button(
                        f"💾 Save Current", 
                        key=f"save_slot_{slot_idx}", 
                        disabled=not active_run_exists, 
                        use_container_width=True,
                        help="Saves the currently active simulation run to this slot" if active_run_exists else "No active simulation run to save",
                        on_click=save_current_run_to_slot,
                        args=(slot_idx,)
                    )
                
                st.markdown("---")
    _tau_hist_norm_tab4 = st.session_state.get("tau_history_normalized", {})
    if not _tau_hist_norm_tab4:
        st.info(
            "No data available yet — click “▶️ Run State Optimization” in the "
            "**3. Results** tab (with T_max ≥ 1) to generate the trajectory τ=0…T_max that "
            "feeds these graphs."
        )
    else:
        _taus_sorted_tab4 = sorted(_tau_hist_norm_tab4.keys())
        _tau_closed_tab4 = st.session_state.get("tau_closed_at")
        _mu_colors_tab4 = {"DC": "#4F46E5", "PAR": "#F59E0B", "ELN": "#10B981", "FARC": "#E11D48"}

        def _add_closure_band_tab4(fig):
            # Franja angosta SOLO sobre el periodo tau_closed_at (el primer tau con m!=Cont),
            # no una region que se extienda hasta el final -- solo marca ese punto puntual.
            if _tau_closed_tab4 is not None and _taus_sorted_tab4:
                fig.add_vrect(
                    x0=_tau_closed_tab4 - 0.5, x1=_tau_closed_tab4 + 0.5,
                    fillcolor="lightgray", opacity=0.45, layer="below", line_width=0,
                    annotation_text=f"τ={_tau_closed_tab4}: first m≠Cont",
                    annotation_position="top",
                )
            return fig

        def _auto_yrange_tab4(*series, pad_frac=0.12, clamp01=False):
            """Rango de eje Y ajustado a los datos reales (con margen), en vez de un rango
            fijo [0,1] que aplana visualmente variaciones pequeñas. clamp01 respeta los
            límites naturales [0,1] de alpha/gamma (nunca se sale de ese rango, pero SÍ
            se acerca a los datos si estos ocupan solo una franja angosta de [0,1])."""
            _vals = [v for s in series for v in s if isinstance(v, (int, float)) and v == v]
            if not _vals:
                return [0, 1] if clamp01 else None
            _lo, _hi = min(_vals), max(_vals)
            if _lo == _hi:
                _pad = 0.05 if clamp01 else max(abs(_lo) * 0.1, 1e-6)
            else:
                _pad = (_hi - _lo) * pad_frac
            _lo, _hi = _lo - _pad, _hi + _pad
            if clamp01:
                _lo, _hi = max(0.0, _lo), min(1.0, _hi)
            return [_lo, _hi]

        # ---- Grafica 1: mu(theta), 4 tipos en una figura, tipo verdadero con asterisco ----
        st.markdown("### μ(θ) — Posterior belief evolution")
        fig_mu_tab4 = go.Figure()
        for _th_tab4 in cvn.TIPOS:
            _y_mu_tab4 = [_tau_hist_norm_tab4[t].get("mu_tau", {}).get(_th_tab4) for t in _taus_sorted_tab4]
            _is_true_tab4 = (_th_tab4 == cov_perp)
            fig_mu_tab4.add_trace(go.Scatter(
                x=_taus_sorted_tab4, y=_y_mu_tab4, mode="lines+markers",
                name=f"{_th_tab4}*" if _is_true_tab4 else _th_tab4,
                line=dict(color=_mu_colors_tab4[_th_tab4], width=3.5 if _is_true_tab4 else 1.5),
            ))
        _add_closure_band_tab4(fig_mu_tab4)
        fig_mu_tab4.update_layout(
            xaxis_title="τ", yaxis_title="μ_τ(θ)", yaxis=dict(range=[0, 1]),
            height=420, legend_title_text="Type (* = true type)",
        )
        st.plotly_chart(fig_mu_tab4, use_container_width=True)

        # ---- Graficas 2/3: instrumento optimo del Estado vs. el teorico de la RAMA
        # REALMENTE sorteada (a_tilde_S: Rescue->rama R, Negotiate->rama N), solo tipo
        # verdadero -- 2 lineas, no 9.
        def _plot_instrument_tab4(field_key, avg_r_key, avg_n_key, ytitle):
            fig = go.Figure()
            _y_star = [_tau_hist_norm_tab4[t].get(field_key) for t in _taus_sorted_tab4]
            fig.add_trace(go.Scatter(
                x=_taus_sorted_tab4, y=_y_star, mode="lines+markers",
                name=f"{ytitle}*_τ (instrument used by the State)",
                line=dict(color="black", width=3.5),
            ))
            
            # Rescue average (alpha_R_mu or gamma_R_mu)
            _y_avg_R = [_tau_hist_norm_tab4[t].get(avg_r_key) for t in _taus_sorted_tab4]
            fig.add_trace(go.Scatter(
                x=_taus_sorted_tab4, y=_y_avg_R, mode="lines+markers",
                name=f"{ytitle}_R^μ (Rescue branch average)",
                line=dict(color="#1f77b4", width=2.5, dash="dash"),
            ))
            
            # Negotiation average (alpha_N_mu or gamma_N_mu)
            _y_avg_N = [_tau_hist_norm_tab4[t].get(avg_n_key) for t in _taus_sorted_tab4]
            fig.add_trace(go.Scatter(
                x=_taus_sorted_tab4, y=_y_avg_N, mode="lines+markers",
                name=f"{ytitle}_N^μ (Negotiation branch average)",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            ))
            
            _add_closure_band_tab4(fig)
            fig.update_layout(
                xaxis_title="τ", yaxis_title=ytitle,
                yaxis=dict(range=_auto_yrange_tab4(_y_star, _y_avg_R, _y_avg_N, clamp01=True)),
                height=420, legend=dict(font=dict(size=10)),
            )
            return fig

        st.markdown(f"### γ — State instrument vs. R and N branch averages ({cov_perp}*)")
        st.plotly_chart(_plot_instrument_tab4("gamma", "gamma_R_mu", "gamma_N_mu", "γ"), use_container_width=True)

        st.markdown(f"### α — State instrument vs. R and N branch averages ({cov_perp}*)")
        st.plotly_chart(_plot_instrument_tab4("alpha", "alpha_R_mu", "alpha_N_mu", "α"), use_container_width=True)

        # ---- Graficas trayectoria (instrumento*, iota) -- 20 flechas INDEPENDIENTES
        # (no encadenadas: cada flecha va de tau_i a su propio tau_i+1 inmediato, no al
        # siguiente punto muestreado), equidistantes dentro de [0, T_max]. El ultimo tau
        # disponible se marca con un circulo distintivo en vez de una flecha (no tiene
        # "siguiente" al que apuntar).
        def _plot_trajectory_arrows_tab4(field_key, xtitle):
            _x_all = {t: _tau_hist_norm_tab4[t].get(field_key) for t in _taus_sorted_tab4}
            _y_all = {
                t: (max((_tau_hist_norm_tab4[t].get("mu_tau") or {}).values())
                    if _tau_hist_norm_tab4[t].get("mu_tau") else None)
                for t in _taus_sorted_tab4
            }
            _valid_taus = [t for t in _taus_sorted_tab4 if _x_all[t] is not None and _y_all[t] is not None]
            fig = go.Figure()
            if len(_valid_taus) < 1:
                fig.update_layout(xaxis_title=xtitle, yaxis_title="ι = max_θ μ_τ(θ)", height=440)
                return fig

            # Detect the convergence point (where coordinates stop changing significantly)
            _t0_traj = _valid_taus[0]
            _last_tau = _valid_taus[-1]
            _last_x = _x_all[_last_tau]
            _last_y = _y_all[_last_tau]
            _epsilon_conv = 1e-4

            _t_conv = _last_tau
            for _t in reversed(_valid_taus):
                _dist = ((_x_all[_t] - _last_x)**2 + (_y_all[_t] - _last_y)**2)**0.5
                if _dist > _epsilon_conv:
                    # Found the last point with significant movement.
                    # The convergence point is the one immediately after.
                    _idx_t = _valid_taus.index(_t)
                    if _idx_t + 1 < len(_valid_taus):
                        _t_conv = _valid_taus[_idx_t + 1]
                    else:
                        _t_conv = _last_tau
                    break

            # Sample 20 arrows exclusively within the active phase of the simulation
            # (from _t0_traj to _t_conv). This prevents all arrows from collapsing
            # onto the convergence point when T_max is large but convergence is early.
            _valid_taus_active = [t for t in _valid_taus if t <= _t_conv]
            _step_traj = max(1.0, (_t_conv - _t0_traj) / 20.0)
            _target_positions = [_t0_traj + _k * _step_traj for _k in range(20)]
            _sample_taus = []
            for _pos in _target_positions:
                _closest = min(_valid_taus_active, key=lambda t: abs(t - _pos))
                if _closest not in _sample_taus:
                    _sample_taus.append(_closest)
            if _last_tau not in _sample_taus:
                _sample_taus.append(_last_tau)
            _sample_taus = sorted(_sample_taus)
            _arrow_taus = [t for t in _sample_taus if t != _last_tau]

            # Visible axis ranges to compute proper screen-space arrow lengths
            _xs_all_sampled = [_x_all[t] for t in _sample_taus]
            _ys_all_sampled = [_y_all[t] for t in _sample_taus]
            _x_range = _auto_yrange_tab4(_xs_all_sampled, clamp01=True)
            _y_range = _auto_yrange_tab4(_ys_all_sampled, clamp01=True)
            _x_axis_len = max(_x_range[1] - _x_range[0], 1e-6)
            _y_axis_len = max(_y_range[1] - _y_range[0], 1e-6)

            _has_arrows = False
            _active_pts = []
            for _t in _valid_taus_active:
                _p = (_x_all[_t], _y_all[_t])
                if not _active_pts or ((_p[0] - _active_pts[-1][0])**2 + (_p[1] - _active_pts[-1][1])**2)**0.5 > 1e-7:
                    _active_pts.append(_p)

            if len(_active_pts) >= 2:
                _norm_pts = [(_p[0] / _x_axis_len, _p[1] / _y_axis_len) for _p in _active_pts]
                _cum_dist = [0.0]
                for _idx in range(len(_norm_pts) - 1):
                    _d = ((_norm_pts[_idx+1][0] - _norm_pts[_idx][0])**2 + (_norm_pts[_idx+1][1] - _norm_pts[_idx][1])**2)**0.5
                    _cum_dist.append(_cum_dist[-1] + _d)
                
                _total_L = _cum_dist[-1]
                if _total_L > 1e-7:
                    _has_arrows = True
                    _S_pts = []
                    _directions = []
                    
                    for _k in range(21):
                        _s = _k * _total_L / 20.0
                        _seg_idx = 0
                        for _j in range(len(_cum_dist) - 1):
                            if _cum_dist[_j] <= _s <= _cum_dist[_j+1]:
                                _seg_idx = _j
                                break
                        else:
                            _seg_idx = len(_cum_dist) - 2
                        
                        _seg_len = _cum_dist[_seg_idx+1] - _cum_dist[_seg_idx]
                        _frac = (_s - _cum_dist[_seg_idx]) / _seg_len if _seg_len > 1e-7 else 0.0
                        
                        _p_start = _active_pts[_seg_idx]
                        _p_end = _active_pts[_seg_idx+1]
                        _S_x = _p_start[0] + _frac * (_p_end[0] - _p_start[0])
                        _S_y = _p_start[1] + _frac * (_p_end[1] - _p_start[1])
                        _S_pts.append((_S_x, _S_y))
                        
                        _dx = _p_end[0] - _p_start[0]
                        _dy = _p_end[1] - _p_start[1]
                        _dx_norm = _dx / _x_axis_len
                        _dy_norm = _dy / _y_axis_len
                        _d_norm = (_dx_norm**2 + _dy_norm**2)**0.5
                        if _d_norm > 1e-7:
                            _directions.append((_dx / _d_norm, _dy / _d_norm))
                        else:
                            _directions.append((0.0, 0.0))

                    # We want small, elegant arrows (1.5% of axis range).
                    # If segments are very close, scale them down to 60% of distance.
                    _arrow_len = min(0.015, (_total_L / 20.0) * 0.6)
                    
                    for _k in range(20):
                        _S_x, _S_y = _S_pts[_k]
                        _ux, _uy = _directions[_k]
                        if abs(_ux) < 1e-7 and abs(_uy) < 1e-7:
                            continue
                        _dx_draw = _ux * _arrow_len
                        _dy_draw = _uy * _arrow_len
                        
                        fig.add_annotation(
                            x=_S_x + _dx_draw, y=_S_y + _dy_draw, ax=_S_x, ay=_S_y,
                            xref="x", yref="y", axref="x", ayref="y",
                            text="",
                            showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.3,
                            arrowcolor="#334155", opacity=0.85,
                        )
                    
                    # Add trace for the 20 sampled points S_0...S_19
                    fig.add_trace(go.Scatter(
                        x=[_pt[0] for _pt in _S_pts[:-1]], y=[_pt[1] for _pt in _S_pts[:-1]],
                        mode="markers", marker=dict(size=4.0, color="#475569"),
                        text=[f"S_{_k}" for _k in range(20)], hoverinfo="text", name="sampled τ",
                    ))

            if not _has_arrows:
                fig.add_trace(go.Scatter(
                    x=[_x_all[t] for t in _arrow_taus], y=[_y_all[t] for t in _arrow_taus],
                    mode="markers", marker=dict(size=4.5, color="#475569"),
                    text=[f"τ={t}" for t in _arrow_taus], hoverinfo="text", name="sampled τ",
                ))

            # Always plot the final point as a circle at the last tau
            fig.add_trace(go.Scatter(
                x=[_x_all[_last_tau]], y=[_y_all[_last_tau]],
                mode="markers",
                marker=dict(size=14, color="#0f172a", symbol="circle-open", line=dict(width=2.5)),
                text=[f"τ={_last_tau} (last)"], hoverinfo="text", name=f"τ={_last_tau} (last)",
            ))
            fig.update_layout(
                xaxis_title=xtitle, yaxis_title="ι = max_θ μ_τ(θ)",
                yaxis=dict(range=_y_range),
                xaxis=dict(range=_x_range),
                height=440,
            )
            # No closure band aqui -- el eje X es gamma*/alpha*, no tau, asi que la franja
            # de "tau_closed_at" (pensada para ejes-X en tau) no aplica a este tipo de grafica.
            return fig

        st.markdown("### γ* vs. ι — 20 independent arrows (each points to its own next τ), last τ marked with a circle")
        st.plotly_chart(_plot_trajectory_arrows_tab4("gamma", "γ*_τ"), use_container_width=True)

        st.markdown("### α* vs. ι — 20 independent arrows (each points to its own next τ), last τ marked with a circle")
        st.plotly_chart(_plot_trajectory_arrows_tab4("alpha", "α*_τ"), use_container_width=True)

        # ---- Grafica 4: Delta H, en barras ----
        st.markdown("### ΔH — Expected entropy reduction")
        fig_dh_tab4 = go.Figure()
        _y_dh_tab4 = [_tau_hist_norm_tab4[t].get("delta_H") for t in _taus_sorted_tab4]
        fig_dh_tab4.add_trace(go.Bar(
            x=_taus_sorted_tab4, y=_y_dh_tab4, name="ΔH_τ", marker_color="#0284C7",
        ))
        _add_closure_band_tab4(fig_dh_tab4)
        fig_dh_tab4.update_layout(
            xaxis_title="τ", yaxis_title="ΔH_τ",
            yaxis=dict(range=_auto_yrange_tab4(_y_dh_tab4)),
            height=380,
        )
        st.plotly_chart(fig_dh_tab4, use_container_width=True)

        st.caption(
            "Data retrieved from `st.session_state['tau_history_normalized']` (exogenous τ=0 + "
            "simulated τ=1…T_max, generated by clicking “Run State Optimization” in "
            "tab 3). The gray band marks the first τ where m≠Cont (`eq:stopping-time`) "
            "— everything after is a counterfactual extension under an absorbing state if "
            "you selected that option when running the simulation."
        )

with tab5:
    st.markdown("## Description — how each variable of Table 5.2 is calculated")
    st.markdown(
        "Reference for every row of **“Table 5.2 — summary of all simulated τ”** (Results tab). "
        "τ=0 values are exogenous Tab 1 inputs; τ≥1 values are recomputed **at every simulated "
        "period** (τ=1…T_max) using that period's own solved instruments $(\\alpha_\\tau^*,"
        "\\gamma_\\tau^*)$, belief $\\mu_\\tau$, and MDG draws — not just once at τ=1."
    )

    _desc_rows: list[tuple[str, str]] = []
    _desc_rows.append((
        r"$a_F^{*}$",
        "Family — τ=0: intended (exogenous, Tab 1 selectbox); τ≥1: $\\arg\\max\\{\\mathcal U_\\tau^F(coop),"
        "\\mathcal U_\\tau^F(col)\\}$ (`eq:f-coop`/`eq:f-col`, exact paper formula, $\\mu_\\tau$-weighted "
        "over $\\theta_K$), resolved fresh every period.",
    ))
    _desc_rows.append((
        r"$\tilde a_F$",
        "Family — MDG-executed action; τ≥1: draw generated every period using $a_F^{\\tau,*}$ and "
        "$T_\\tau^F$ (Family's own temperature — approved extension, `MDG_MULT_FAMILY`, same logit "
        "mechanism as $\\tilde a_S/\\tilde a_K$).",
    ))
    _desc_rows.append((
        r"$a_K^{*}(\theta_K)$",
        "Captor — τ=0: intended (exogenous, Tab 1 \"Perpetrator\" selector); τ≥1: "
        "$\\arg\\max\\{U_{rel},U_{kill},V_{cont,\\tau}\\}$ for the true type (`eq:k-bellman`), "
        "$\\tilde p_{cap,\\tau}$ refined with the real draw of $\\tilde a_S$, continuation weighted "
        "by $\\Pr(m,d\\mid\\theta_K^{true})$ — resolved fresh every period.",
    ))
    _desc_rows.append((
        r"$\tilde a_K(\theta_K)$",
        "Captor — MDG-executed action; τ≥1: pure logit `eq:logit-hybrid` (Working_paper_eng.tex "
        "Sec. 4.2.1) ≡ `eq:LI-atilde` (Bernal_H.tex), centered on $a_K^{\\tau,*}(\\theta_K^{true})$ "
        "with $T_\\tau^K$ (Captor's own temperature — approved extension, `MDG_MULT_CAPTOR`) — "
        "same logit mechanism as $\\tilde a_S/\\tilde a_F$, each with its own $T_\\tau^i$.",
    ))
    _desc_rows.append((
        r"$a_S^{*}$ optimal",
        "State — τ=0: intended (exogenous slider); τ≥1: $\\arg\\min$ over $\\Gamma_\\tau(\\mu_\\tau)$-"
        "restricted branch floors, using the trained value net ($T=10$, clamped for τ>10) — "
        "re-solved every period.",
    ))
    _desc_rows.append((
        r"$\tilde a_S$",
        "State — MDG-executed action; τ≥1: draw generated every period using $a_S^{\\tau,*}$ and "
        "$T_\\tau^S$ (State's own temperature — approved extension, `MDG_MULT_STATE`, literal "
        "`eq:m-t`, real $H(\\mu_\\tau)/H(\\mu_0)$).",
    ))
    _desc_rows.append((
        r"$\alpha_t^{*}$ State",
        "State instrument — financial blocking; τ=0: exogenous slider; τ≥1: $\\arg\\min$ under "
        "$\\Gamma_\\tau(\\mu_\\tau)$ (grid search, trained value net), re-solved every period.",
    ))
    _desc_rows.append((
        r"$\gamma_t^{*}$ State",
        "State instrument — operational pressure; τ=0: exogenous slider; τ≥1: $\\arg\\min$ under "
        "$\\Gamma_\\tau(\\mu_\\tau)$ (grid search, trained value net), re-solved every period.",
    ))
    for _th_row in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$\gamma_R({_th_row})^*$",
            f"Perfect-info benchmark, rescue branch — argmin $C_{{ops}}$ ({_th_row}); recomputed "
            f"every τ via $\\omega_K(1-p_{{surv}})+C_{{ops}}$ using `cvn.p_surv_raw` (constant "
            f"shift; $C_{{ops}}$ itself has no τ-dependence).",
        ))
    for _th_row in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$\alpha_R({_th_row})^*$",
            f"Perfect-info benchmark, rescue branch — argmin $C_{{ops}}$ ({_th_row}); recomputed "
            f"every τ via $\\omega_K(1-p_{{surv}})+C_{{ops}}$ using `cvn.p_surv_raw` (constant "
            f"shift; $C_{{ops}}$ itself has no τ-dependence).",
        ))
    for _th_row in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$\gamma_N({_th_row})^*$",
            f"Perfect-info benchmark, negotiation branch — argmin $V_0^N$ ({_th_row}); recomputed "
            f"every τ with $M_\\tau$ (literal `eq:m-t`, real $H(\\mu_\\tau)/H(\\mu_0)$, not the "
            f"$H_{{ratio}}$ slider proxy).",
        ))
    for _th_row in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$\alpha_N({_th_row})^*$",
            f"Perfect-info benchmark, negotiation branch — argmin $V_0^N$ ({_th_row}); recomputed "
            f"every τ with $M_\\tau$ (literal `eq:m-t`, real $H(\\mu_\\tau)/H(\\mu_0)$, not the "
            f"$H_{{ratio}}$ slider proxy).",
        ))
    _desc_rows.append((
        r"$H(\mu)$",
        "Belief entropy; τ=0: Block G priors; τ≥1: at $\\mu_\\tau$, recomputed every period from "
        "the evolving belief.",
    ))
    _desc_rows.append((
        r"$\Delta H$ State",
        "Expected entropy reduction; τ=0: minimal-record Bayes over 10 $(m,d)$ pairs at exogenous "
        "$(\\alpha_0,\\gamma_0)$; τ≥1: at the solved $(\\alpha_\\tau^*,\\gamma_\\tau^*)$, recomputed "
        "every period. Shows as empty/NaN when $\\Gamma_\\tau(\\mu_\\tau)$ has no feasible "
        "$(\\alpha,\\gamma)$ point for that period's belief — not a missing-data bug, the model's "
        "own infeasibility fallback.",
    ))
    _desc_rows.append((
        r"$\Gamma_t(\mu_t)$ under EV",
        "$IR^K_{EV}\\wedge IC^K_{EV}\\wedge IR^F$; τ=0 myopic; τ≥1: solved with the trained value "
        "net, at $(\\alpha_\\tau^*,\\gamma_\\tau^*)$, re-checked every period.",
    ))
    _desc_rows.append((
        r"$IR^K(\theta_K)$ true type",
        "Captor's true-type participation check; τ=0: myopic; τ≥1: same check using the trained "
        "value net's continuation value, re-checked every period.",
    ))
    for _th_tab3 in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$-\operatorname{{sgn}}(\kappa_h(\mathrm{{{_th_tab3}}},t))$",
            f"Sign of $\\partial\\mathbb{{E}}[\\tau\\mid\\theta_K]/\\partial\\gamma_t^*$ ({_th_tab3}), "
            "`eq:hj`/`eq:kappa-c`; τ=0: at $(\\alpha_0,\\gamma_0)$ and the realized "
            "$(\\tilde a^F,\\tilde a^K,\\tilde a^S)$ of that period; τ≥1: same formula, recomputed "
            "every period at $(\\alpha_\\tau^*,\\gamma_\\tau^*,M_\\tau)$ and that period's own "
            "realized draws.",
        ))
    _desc_rows.append((
        r"$m$ outcome / $v$ (m draw)",
        "Realized outcome, $\\mathcal{G}_t$ inverse-transform draw; τ=0: Tab 1 Block C expander "
        "(exogenous $\\alpha_0,\\gamma_0$); τ≥1: genuine redraw every period at "
        "$(\\alpha_\\tau^*,\\gamma_\\tau^*,M_\\tau)$ via `cvn.outcome_probs_grid` "
        "(`eq:hj`/`eq:pCont`/`eq:xi`≡`eq:LH-compacta`).",
    ))
    _desc_rows.append((
        r"closes episode",
        "True iff $m_\\tau\\neq$ cont — by `eq:stopping-time` the episode closes at that τ; later "
        "τ (if simulated) are a counterfactual extension under an absorbing state.",
    ))
    for _th_tab3 in ["DC", "PAR", "ELN", "FARC"]:
        _desc_rows.append((
            rf"$\mu(\mathrm{{{_th_tab3}}})$",
            "Posterior belief mass; τ=0: prior (Block G); τ≥1: full Bayes update "
            "(`eq:bayes-update`), chained period by period from $\\mu_1$ onward using that "
            "period's own realized $m,d$ and voice likelihood.",
        ))
    _desc_rows.append((
        r"$\alpha_R^{\mu}$",
        "Belief-weighted rescue-branch instrument, $\\sum_\\theta \\mu_\\tau(\\theta)\\alpha_R^{\\theta,*}$, "
        "recomputed every τ with that period's own $\\mu_\\tau$.",
    ))
    _desc_rows.append((
        r"$\gamma_R^{\mu}$",
        "Belief-weighted rescue-branch instrument, $\\sum_\\theta \\mu_\\tau(\\theta)\\gamma_R^{\\theta,*}$, "
        "recomputed every τ with that period's own $\\mu_\\tau$.",
    ))
    _desc_rows.append((
        r"$\alpha_N^{\mu}$",
        "Belief-weighted negotiation-branch instrument, $\\sum_\\theta \\mu_\\tau(\\theta)\\alpha_N^{\\theta,*}$, "
        "recomputed every τ with that period's own $\\mu_\\tau$.",
    ))
    _desc_rows.append((
        r"$\gamma_N^{\mu}$",
        "Belief-weighted negotiation-branch instrument, $\\sum_\\theta \\mu_\\tau(\\theta)\\gamma_N^{\\theta,*}$, "
        "recomputed every τ with that period's own $\\mu_\\tau$.",
    ))
    _desc_rows.append((
        r"$L_C$ (voice)",
        "Voice/acoustic evidence likelihood; τ=0: Block F static scalar formula ($L_{voz}$, fixed "
        "Tab 1 slider, illustrative only); τ≥1: genuine draw $V_\\tau\\sim\\text{Bern}(\\tilde\\pi_{call}"
        "(\\theta_K^{true}))$, then $x_\\tau^{obs}$ (4 features, `eq:voz-descomp`) if $V_\\tau=1$ — "
        "reuses `rational_behavior.py` (same mechanism as app.py). $\\mathcal L_{C,\\tau}$ feeds "
        "into `eq:bayes-update` every period (unlike the τ=0 $L_{voz}$ scalar, which is diagnostic "
        "display only).",
    ))
    _desc_rows.append((
        r"$V_\tau$ (signal)",
        "Voice-emission indicator ($V_\\tau\\in\\{0,1\\}$, `eq:voz-descomp`/`eq:LC`); τ=0: static, "
        "based on Block F's checkbox; τ≥1: genuine Bernoulli draw every period, "
        "$V_\\tau\\sim\\text{Bern}(\\tilde\\pi_{call}(\\theta_K^{true}))$.",
    ))
    _desc_rows.append((
        r"$d$ (detection) / $p_{det}$",
        "Collusion detection probability (`eq:detection`); τ=0: at exogenous $(\\alpha_0,\\gamma_0)$ "
        "(Block D); τ≥1: same formula, recomputed every period at $(\\alpha_\\tau^*,\\gamma_\\tau^*)$, "
        "with a genuine Bernoulli draw of $d_\\tau$ feeding the belief update.",
    ))
    _desc_rows.append((
        r"$\iota$",
        "Informational precision, $\\max_\\theta \\mu_\\tau(\\theta)$; τ=0: Block G priors; τ≥1: at "
        "$\\mu_\\tau$ (full Bayes update, `eq:bayes-update`), recomputed every period.",
    ))
    _desc_rows.append((
        r"$M_t$ (generic / $m$)",
        "MDG noise temperature, system-level (feeds the $m$ draw and its Bayes-update "
        "likelihood); τ=0: `eq:m-t` with the $H_{ratio}$ slider proxy for $H(\\mu_t)/H(\\mu_0)$ "
        "(Block A, $t$ fixed at 0); τ≥1: `eq:m-t` **literal** — real $H(\\mu_\\tau)/H(\\mu_0)$ "
        "ratio, recomputed every period, not the $H_{ratio}$ slider.",
    ))
    _desc_rows.append((
        r"$M_t^S,\ M_t^F,\ M_t^K$",
        "Per-player MDG temperatures ($T_\\tau^S,T_\\tau^F,T_\\tau^K$) — approved extension, NOT "
        "literal in Bernal_H.tex/Working_paper_eng.tex (which define one shared system "
        "temperature). Same functional form as $M_t$, each using its own $T_0^i$ slider (Tab 1 "
        "Block A) and `MDG_MULT_STATE/FAMILY/CAPTOR` multipliers; recomputed every period and "
        "used only for that player's own $\\tilde a^i$ draw (not for $m$).",
    ))

    _desc_table_md = "| Variable | Description |\n|---|---|\n" + "\n".join(
        r"| {} | {} |".format(*row) for row in _desc_rows
    )
    st.markdown(_desc_table_md)

# End of file
