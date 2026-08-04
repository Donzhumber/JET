import numpy as np
import torch
from train_captor_value_net import (
    CaptorValueNet, Params, TIPOS, solve_state_problem, encode_input_grid, GA,
)

ckpt = torch.load("captor_value_net_T10.pt", map_location="cpu", weights_only=False)
net = CaptorValueNet()
net.load_state_dict(ckpt["state_dict"])
net.eval()
print("Cargado: T =", ckpt["T"], " N =", ckpt["N"])

# Escenario de referencia = defaults actuales de Tab 1
p_ref = Params(
    R_millions=20.0,
    beta_tilde={"DC": 0.92, "PAR": 0.92, "ELN": 0.92, "FARC": 0.92},
    cov_zone="Metropolis", cov_wealth="Standard", cov_vict="Private", cov_state="Strict",
    T_mad=5.0, T0=0.30, eta_cal=0.075, c_bar=0.05, H_ratio=0.80,
    lambda_4=0.0005, eta_1=1.0, eta_2=1.0, beta_R=7.0,
)
mu1 = {"DC": 0.0605, "PAR": 0.8364, "ELN": 0.0338, "FARC": 0.0693}  # ejemplo ya usado (m=Muerte,d=1)


def v_next_fn(mu2, tau_next, p, T=10):
    if tau_next > T:
        return {th: np.zeros_like(GA) for th in TIPOS}
    out = {}
    with torch.no_grad():
        for th in TIPOS:
            x = encode_input_grid(th, mu2, tau_next, T, p)
            v = net(torch.from_numpy(x)).numpy()
            out[th] = v.reshape(GA.shape)
    return out


print("\n=== 1. Monotonicidad de V_tau(theta|mu fijo) a través de tau ===")
mu_fixed = {"DC": 0.25, "PAR": 0.25, "ELN": 0.25, "FARC": 0.25}
for th in TIPOS:
    x = encode_input_grid(th, {t: np.full_like(GA, mu_fixed[t]) for t in TIPOS}, 1, 10, p_ref)
    vals = []
    for tau in range(1, 11):
        x_tau = x.copy()
        x_tau[:, 4] = tau / 10.0  # columna tau_norm
        with torch.no_grad():
            v = net(torch.from_numpy(x_tau[:1])).item()
        vals.append(v)
    print(f"  {th}: " + " -> ".join(f"{v:.3f}" for v in vals) + f"  (tau=1..10)")

print("\n=== 2. Comparacion contra caso miope (V_next=0, es decir tau=T) ===")
tipo_real = max(mu1, key=mu1.get)
theta_f = "Low wealth"
a1_myopic, g1_myopic, aS_myopic, v_myopic, feas_myopic, extra_myopic = solve_state_problem(
    mu1, 10, 10, p_ref, None, tipo_real, theta_f  # tau=T=10 -> sin red, exactamente el caso miope
)
print(f"  Miope (tau=T, V_next=0): a_S*={aS_myopic}, alpha*={a1_myopic:.3f}, gamma*={g1_myopic:.3f}, factible={feas_myopic}")
print(f"  V_tau por tipo (miope): {[f'{th}={v_myopic[th]:.3f}' for th in TIPOS]}")

print("\n=== 3. Resultado final para tau=1, mu_1 real, con la red entrenada (dinamico completo) ===")
a1_star, g1_star, aS1_star, v1_by_type, feas1, extra1 = solve_state_problem(
    mu1, 1, 10, p_ref, lambda m2, tn, pp: v_next_fn(m2, tn, pp), tipo_real, theta_f
)
print(f"  Dinamico (tau=1, horizonte T=10): a_S*={aS1_star}, alpha_1*={a1_star:.4f}, gamma_1*={g1_star:.4f}, factible={feas1}")
print(f"  V_1 por tipo: {[f'{th}={v1_by_type[th]:.4f}' for th in TIPOS]}")

print("\n=== 4. Diferencia dinamico vs. miope (mismo mu_1, mismo escenario) ===")
print(f"  alpha: {g1_star - g1_myopic:+.4f}  gamma: {a1_star - a1_myopic:+.4f}")
for th in TIPOS:
    print(f"  V_{th}: miope={v_myopic[th]:.4f}  dinamico={v1_by_type[th]:.4f}  diff={v1_by_type[th]-v_myopic[th]:+.4f}")
