import os
import json
import pandas as pd
from pathlib import Path

# Paths
CUR_DIR = Path(__file__).resolve().parent
RUNS_FILE = CUR_DIR / "saved_runs.json"
FIG_DIR = CUR_DIR.parent / "Figures"

# Create output folder if it doesn't exist
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(RUNS_FILE, "r") as f:
    runs = json.load(f)

# Slot configurations
SLOT_CONFIGS = {
    "1": {"true_type": "DC", "include_voice": True, "filename": "DC.csv"},
    "6": {"true_type": "PAR", "include_voice": True, "filename": "PAR.csv"},
    "2": {"true_type": "ELN", "include_voice": True, "filename": "ELN.csv"},
    "5": {"true_type": "FARC", "include_voice": True, "filename": "FARC.csv"},
    "8": {"true_type": "DC", "include_voice": False, "filename": "DC_no_voice.csv"},
    "4": {"true_type": "PAR", "include_voice": False, "filename": "PAR_no_voice.csv"},
    "7": {"true_type": "ELN", "include_voice": False, "filename": "ELN_no_voice.csv"},
    "3": {"true_type": "FARC", "include_voice": False, "filename": "FARC_no_voice.csv"}
}

def rebuild_row(tau_i, nd, cov_perp, include_voice):
    mu_d = nd.get("mu_tau") or {}
    bench = nd.get("benchmarks_tau1") or {}
    kh = nd.get("neg_sign_kappa_h_tau1") or {}
    
    row = {
        "τ": tau_i,
        "a_F*": nd.get("a_F_star1"),
        "ã_F": nd.get("act_f_tau1"),
        f"a_K*({cov_perp})": nd.get("a_K_star1"),
        f"ã_K({cov_perp})": nd.get("act_k_tau1"),
        "a_S*": nd.get("a_S"),
        "ã_S": nd.get("act_s_tau1"),
        "α_t*": nd.get("alpha"),
        "γ_t*": nd.get("gamma"),
    }
    
    for f, lbl in [("gamma_R", "γ_R"), ("alpha_R", "α_R"), ("gamma_N", "γ_N"), ("alpha_N", "α_N")]:
        for th_row in ["DC", "PAR", "ELN", "FARC"]:
            b = bench.get(th_row)
            row[f"{lbl}({th_row})*"] = b[f] if b else None
            
    row["H(μ_τ)"] = nd.get("H_mu")
    row["ΔH"] = nd.get("delta_H")
    row["Γ_τ(μ_τ) feasible"] = bool(nd.get("feasible")) if nd.get("feasible") is not None else None
    
    gap = nd.get("ir_k_true_gap")
    row[f"IR^K({cov_perp}) OK"] = ("Yes" if gap >= -1e-9 else "No") if gap is not None else None
    
    for th_row in ["DC", "PAR", "ELN", "FARC"]:
        row[f"-sgn(κ_h({th_row}))"] = kh.get(th_row)
        
    row["m outcome"] = nd.get("m_outcome")
    row["v (m draw)"] = nd.get("m_v")
    row["closes episode"] = bool(nd.get("m_closes_episode")) if nd.get("m_closes_episode") is not None else None
    
    for th_row in ["DC", "PAR", "ELN", "FARC"]:
        row[f"μ_{th_row}"] = mu_d.get(th_row)
        
    row["α_R^μ"] = nd.get("alpha_R_mu")
    row["γ_R^μ"] = nd.get("gamma_R_mu")
    row["α_N^μ"] = nd.get("alpha_N_mu")
    row["γ_N^μ"] = nd.get("gamma_N_mu")
    
    row["L_C(voice)"] = nd.get("voice_L_C") if include_voice else 1.0
    
    vv = nd.get("voice_V")
    row["V_τ (signal)"] = ("Call" if vv == 1 else "Silence") if (vv is not None and include_voice) else ("Ignored" if not include_voice else None)
    
    row["d (detection)"] = (bool(nd.get("d_tau1")) if nd.get("d_tau1") is not None else None)
    row["p_det"] = nd.get("p_det_tau1")
    row["ι = max_θ μ_τ(θ)"] = (max(mu_d.values()) if mu_d else None)
    
    row["M_t"] = nd.get("T_tau1")
    row["M_t^S"] = nd.get("T_tau1_S")
    row["M_t^F"] = nd.get("T_tau1_F")
    row["M_t^K"] = nd.get("T_tau1_K")
    
    return row

def generate_csv(slot_key, config):
    print(f"Generating {config['filename']} from Slot {slot_key}...")
    run_data = runs[slot_key]["data"]
    tau_hist = run_data["tau_history_normalized"]
    
    ts = sorted([int(k) for k in tau_hist.keys()])
    
    rows = []
    for t in ts:
        raw_nd = tau_hist[str(t)]
        row = rebuild_row(t, raw_nd, config["true_type"], config["include_voice"])
        rows.append(row)
        
    # Build dataframe and transpose
    df = pd.DataFrame(rows).set_index("τ").T
    
    # Export to Figures/
    out_path = FIG_DIR / config["filename"]
    df.to_csv(out_path)
    print(f"Successfully saved to {out_path}")

def main():
    for slot, config in SLOT_CONFIGS.items():
        generate_csv(slot, config)
        
    # Sync generated CSVs to main/Figures and main_esp/Figures
    import shutil
    dest_dirs = [
        CUR_DIR.parent / "main" / "Figures",
        CUR_DIR.parent / "main_esp" / "Figures"
    ]
    print("Syncing CSVs to main/Figures and main_esp/Figures...")
    for filename in os.listdir(FIG_DIR):
        if filename.endswith(".csv"):
            src_file = FIG_DIR / filename
            for dest_dir in dest_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dest_dir / filename)
    print("All CSVs successfully exported and synced!")

if __name__ == "__main__":
    main()
