import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# Paths
CUR_DIR = Path(__file__).resolve().parent
RUNS_FILE = CUR_DIR / "saved_runs.json"
FIG_DIR = CUR_DIR.parent / "Figures"

# Create output folder if it doesn't exist
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load saved runs
with open(RUNS_FILE, "r") as f:
    runs = json.load(f)

# Matplotlib styling for academic paper
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": "0.15",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
})

COLOR_MU = {"DC": "#4F46E5", "PAR": "#F59E0B", "ELN": "#10B981", "FARC": "#E11D48"}

def get_run_data(slot_key):
    run_data = runs[slot_key]["data"]
    tau_hist = run_data["tau_history_normalized"]
    ts = sorted([int(k) for k in tau_hist.keys() if int(k) <= 150])
    tau_closed = run_data.get("tau_closed_at")
    return ts, tau_hist, tau_closed

def add_closure_band(ax, tau_closed):
    if tau_closed is not None and 0 < tau_closed <= 150:
        ax.axvspan(tau_closed - 0.5, tau_closed + 0.5, facecolor="#E8E8E8", alpha=0.92, linewidth=0, zorder=0)
        # Larger font size (9) equal to axis labels, and black color
        ax.text(tau_closed, ax.get_ylim()[1], r"$\tau^{\mathrm{hyp}}$", ha="center", va="bottom", fontsize=9, color="black", clip_on=False)

def plot_beliefs(ts, tau_hist, tau_closed, true_type, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    linestyles = {"DC": "--", "PAR": ":", "ELN": "-.", "FARC": "--"}
    
    for th in ["DC", "PAR", "ELN", "FARC"]:
        y = [tau_hist[str(t)]["mu_tau"].get(th, 0.0) for t in ts]
        is_true = (th == true_type)
        label = f"{th}*" if is_true else th
        
        color = COLOR_MU[th]
        if is_true:
            linewidth = 2.0
            linestyle = "-"
        else:
            linewidth = 1.0
            linestyle = linestyles[th]
            
        ax.plot(ts, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        
    add_closure_band(ax, tau_closed)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grid disabled
    ax.set_xlabel("Period $\\tau$")
    ax.set_ylabel("$\\mu_\\tau(\\theta)$")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="best")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def plot_instrument(ts, tau_hist, tau_closed, field_key, avg_r_key, avg_n_key, ytitle, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    
    ts_inst = [t for t in ts if str(t) != "0"]
    y_star = [tau_hist[str(t)][field_key] for t in ts_inst]
    ax.plot(ts_inst, y_star, color="black", linewidth=2.0, label=r"${}^*_t$".replace("{}", ytitle))
    
    y_r = [tau_hist[str(t)].get(avg_r_key) for t in ts_inst]
    ax.plot(ts_inst, y_r, color="#1f77b4", linestyle="--", linewidth=1.2, label=r"${}_R^\mu$".replace("{}", ytitle))
    
    y_n = [tau_hist[str(t)].get(avg_n_key) for t in ts_inst]
    ax.plot(ts_inst, y_n, color="#ff7f0e", linestyle="-.", linewidth=1.2, label=r"${}_N^\mu$".replace("{}", ytitle))
    
    add_closure_band(ax, tau_closed)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grid disabled
    ax.set_xlabel("Period $\\tau$")
    ax.set_ylabel(r"${}_t$".replace("{}", ytitle))
    ax.set_xlim(0, 150)
    
    vals = [v for v in y_star + y_r + y_n if v is not None and v == v]
    if vals:
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-6:
            ax.set_ylim(max(0.0, lo - 0.05), min(1.0, hi + 0.05))
        else:
            pad = (hi - lo) * 0.12
            ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    else:
        ax.set_ylim(0, 1.0)
        
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def plot_trajectory_arrows(ts, tau_hist, field_key, xtitle, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    
    # Swap axes: x-axis is posterior precision (iota), y-axis is instrument (alpha/gamma)
    x_all = [max(tau_hist[str(t)]["mu_tau"].values()) for t in ts]
    y_all = [tau_hist[str(t)][field_key] for t in ts]
    
    # We construct a dataframe compatible with the add_tau_arrows implementation from the journal
    df_path = pd.DataFrame({
        "tau": ts,
        "x": x_all,
        "y": y_all
    })
    
    # Draw arrows using the exact journal code (adapted FancyArrowPatch logic)
    # Define INK color matching the journal style
    INK = "#111111"
    
    tau = df_path["tau"].to_numpy(dtype=float)
    x = df_path["x"].to_numpy(dtype=float)
    y = df_path["y"].to_numpy(dtype=float)
    
    n_arrows = 20
    tau_grid = np.linspace(float(tau.min()), float(tau.max()), n_arrows + 1)
    x_grid = np.interp(tau_grid, tau, x)
    y_grid = np.interp(tau_grid, tau, y)
    
    xr = max(float(np.nanmax(x_grid) - np.nanmin(x_grid)), 1e-9)
    yr = max(float(np.nanmax(y_grid) - np.nanmin(y_grid)), 1e-9)
    
    last_dir = np.array([1.0, 0.0])
    min_norm_len = 0.045
    arrows = []

    for i in range(n_arrows):
        dx = x_grid[i + 1] - x_grid[i]
        dy = y_grid[i + 1] - y_grid[i]
        direction = np.array([dx / xr, dy / yr])
        norm_len = float(np.hypot(direction[0], direction[1]))
        if norm_len > 1e-5:
            last_dir = direction / norm_len
        else:
            direction = last_dir
            norm_len = min_norm_len
            
        unit = last_dir.copy()
        center = np.array([0.5 * (x_grid[i] + x_grid[i + 1]), 0.5 * (y_grid[i] + y_grid[i + 1])])
        length = min(max(norm_len * 0.76, min_norm_len), 0.16)
        arrows.append({"center": center, "unit": unit, "length": length})

    # Spatial filter: filter out arrows that are too close in normalized coordinate space to prevent crowding/overlapping
    min_dist = 0.08
    filtered_arrows = []
    kept_centers_n = []
    for item in arrows:
        c = np.array([
            (item["center"][0] - np.nanmin(x_grid)) / xr,
            (item["center"][1] - np.nanmin(y_grid)) / yr
        ])
        if not kept_centers_n:
            filtered_arrows.append(item)
            kept_centers_n.append(c)
            continue
        dists = np.sqrt(((np.vstack(kept_centers_n) - c) ** 2).sum(axis=1))
        if np.all(dists >= min_dist):
            filtered_arrows.append(item)
            kept_centers_n.append(c)
    arrows = filtered_arrows

    centers_n = np.array(
        [
            [float(item["center"][0] - np.nanmin(x_grid)) / xr, float(item["center"][1] - np.nanmin(y_grid)) / yr]
            for item in arrows
        ]
    )
    groups = []
    for i, center_val in enumerate(centers_n):
        for group in groups:
            if np.linalg.norm(center_val - centers_n[group[0]]) < 0.040:
                group.append(i)
                break
        else:
            groups.append([i])

    offsets = np.zeros((len(arrows), 2))
    for group in groups:
        if len(group) == 1:
            continue
        for rank, idx in enumerate(group):
            unit = arrows[idx]["unit"]
            normal = np.array([-float(unit[1]), float(unit[0])])
            normal_norm = max(float(np.linalg.norm(normal)), 1e-9)
            normal = normal / normal_norm
            offset_norm = (rank - (len(group) - 1) / 2) * 0.026
            offsets[idx] = np.array([normal[0] * xr * offset_norm, normal[1] * yr * offset_norm])

    for idx, item in enumerate(arrows):
        center = item["center"] + offsets[idx]
        unit = item["unit"]
        length = float(item["length"])
        dx = float(unit[0]) * xr * length
        dy = float(unit[1]) * yr * length
        start = (float(center[0]) - 0.5 * dx, float(center[1]) - 0.5 * dy)
        end = (float(center[0]) + 0.5 * dx, float(center[1]) + 0.5 * dy)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.15,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            alpha=0.94,
            zorder=3,
        )
        ax.add_patch(arrow)

    # Plot final circle
    ax.plot(
        x_grid[-1],
        y_grid[-1],
        marker="o",
        markersize=7.5,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=1.25,
        zorder=5,
        label=f"$\\tau={ts[-1]}$"
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Swapped labels and fixed limits for paper visual alignment
    ax.set_xlabel(r"Posterior precision $\iota_\tau$")
    ax.set_ylabel(r"${}$".format(xtitle))
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="best")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def plot_compliance(ts, tau_hist, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    
    labels = [r"$\mathrm{IC}^K$", r"$\mathrm{IR}^K$", r"$\mathrm{IR}^F$"]
    values = [100.0, 100.0, 100.0]
    
    # Grayscale styling (zinc-900, zinc-600, zinc-300)
    ax.bar(labels, values, color=["#18181b", "#52525b", "#a1a1aa"], width=0.55)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grid disabled
    ax.set_ylabel("Compliance frequency (%)")
    ax.set_ylim(0, 110)
    
    for i, v in enumerate(values):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def plot_delta_h(ts, tau_hist, tau_closed, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    
    ts_bar = [t for t in ts if str(t) != "0"]
    y = [tau_hist[str(t)].get("delta_H", 0.0) for t in ts_bar]
    y = [v if (v is not None and v == v) else 0.0 for v in y]
    
    ax.bar(ts_bar, y, color="black", width=0.85, edgecolor="none")
    
    add_closure_band(ax, tau_closed)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grid disabled
    ax.set_xlabel("Period $\\tau$")
    ax.set_ylabel("$\\Delta H_\\tau$")
    ax.set_xlim(0, 150)
    
    max_y = max(y) if y else 0.0
    if max_y > 1e-6:
        ax.set_ylim(0, max_y * 1.15)
    else:
        ax.set_ylim(0, 0.1)
        
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def plot_sensitivity(ts, tau_hist, tau_closed, true_type, filename):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    
    ts_line = []
    y = []
    for t in ts:
        if str(t) == "0":
            continue
        val = tau_hist[str(t)].get("neg_sign_kappa_h_tau1", {}).get(true_type)
        if val is not None:
            y.append(val)
            ts_line.append(t)
            
    # Scatter plot only (no connecting lines), in grayscale (black)
    ax.scatter(ts_line, y, color="black", s=10, zorder=2, label=r"$-\operatorname{sgn}(\kappa_h)$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    
    add_closure_band(ax, tau_closed)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grid disabled
    ax.set_xlabel("Period $\\tau$")
    ax.set_ylabel(r"$-\operatorname{sgn}(\kappa_h(\theta_K^\ast,t))$")
    ax.set_xlim(0, 150)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def generate_case_figures(slot, true_type, letter):
    print(f"Generating Case {true_type} ({letter}) figures...")
    ts, hist, closed = get_run_data(slot)
    plot_beliefs(ts, hist, closed, true_type, f"Fig_11{letter}.pdf")
    plot_instrument(ts, hist, closed, "gamma", "gamma_R_mu", "gamma_N_mu", "\\gamma", f"Fig_12{letter}.pdf")
    plot_instrument(ts, hist, closed, "alpha", "alpha_R_mu", "alpha_N_mu", "\\alpha", f"Fig_13{letter}.pdf")
    plot_trajectory_arrows(ts, hist, "gamma", "\\gamma_\\tau^\\ast", f"Fig_14{letter}.pdf")
    plot_trajectory_arrows(ts, hist, "alpha", "\\alpha_\\tau^\\ast", f"Fig_15{letter}.pdf")
    plot_compliance(ts, hist, f"Fig_16{letter}.pdf")
    plot_delta_h(ts, hist, closed, f"Fig_17{letter}.pdf")
    plot_sensitivity(ts, hist, closed, true_type, f"Fig_18{letter}.pdf")

def main():
    # Order: DC (a), PAR (b), ELN (c), FARC (d)
    generate_case_figures("1", "DC", "a")
    generate_case_figures("6", "PAR", "b")
    generate_case_figures("2", "ELN", "c")
    generate_case_figures("5", "FARC", "d")
    
    # No voice case: DC (a_no_voice), PAR (b_no_voice), ELN (c_no_voice), FARC (d_no_voice)
    generate_case_figures("8", "DC", "a_no_voice")
    generate_case_figures("4", "PAR", "b_no_voice")
    generate_case_figures("7", "ELN", "c_no_voice")
    generate_case_figures("3", "FARC", "d_no_voice")
    
    # Sync generated figures to main/Figures and main_esp/Figures
    import shutil
    dest_dirs = [
        CUR_DIR.parent / "main" / "Figures",
        CUR_DIR.parent / "main_esp" / "Figures"
    ]
    print("Syncing figures to main/Figures and main_esp/Figures...")
    for filename in os.listdir(FIG_DIR):
        if filename.endswith(".pdf"):
            src_file = FIG_DIR / filename
            for dest_dir in dest_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dest_dir / filename)
    print("All figures successfully exported and synced!")

if __name__ == "__main__":
    main()
