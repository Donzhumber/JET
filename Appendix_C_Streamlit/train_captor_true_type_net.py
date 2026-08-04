"""
Backward induction (T periodos) entrenando una red de valor PROPIA del Secuestrador,
CaptorValueNet (misma arquitectura, checkpoint distinto: captor_true_type_value_net_T10.pt),
consistente de punta a punta con la ponderacion por TIPO VERDADERO -- a diferencia de
train_captor_value_net.py, cuya red se entrena ponderando las 10 ramas (m,d) por la marginal
mu_tau-mezclada (apropiada para el Estado, que no conoce theta_K).

En cada punto de entrenamiento:
  1. solve_state_problem(...) -- usa la red YA ENTRENADA y CONGELADA del Estado
     (captor_value_net_T10.pt) para obtener (alpha_tau*, gamma_tau*). No se reentrena aqui;
     el Secuestrador solo REACCIONA a lo que el Estado racionalmente haria.
  2. p_rescue/p_nego (mezcla de p_cap, eq:p-cap): protocolo de consistencia temporal --
     en produccion (app_DL.py) se ancla al desenlace YA EJECUTADO del periodo anterior
     (degenerado, (1,0) o (0,1)); aqui, sobre trayectorias SINTETICAS sin un "periodo
     anterior real", se deja el default neutro (0.5,0.5) de solve_state_problem/
     solve_captor_true_type_continuation.
  3. solve_captor_true_type_continuation(...) -- para cada tipo verdadero theta_K, calcula el
     valor Bellman V^K_tau(theta_K) en el punto (alpha_tau*, gamma_tau*) ya elegido, ponderando
     las 10 ramas (m,d) por Pr(m,d|theta_K) (riesgo propio del tipo, no el pronostico
     mu-marginal del Estado). Ese valor es el target de regresion para la red de ESTE archivo.

Uso:
    python3 train_captor_true_type_net.py --T 10 --N 2000 \
        --state-net captor_value_net_T10.pt --out captor_true_type_value_net_T10.pt
"""
import argparse
import json
import time

import numpy as np
import torch

from train_captor_value_net import (
    TIPOS, Params, CaptorValueNet, KAPPA_REL, ETA_REP, F_CAP, PHI_COST, KAPPA_C, NU_COST,
    GA, m_t, outcome_probs_grid, p_cap_eff_grid, p_pay_eff_grid, encode_input, encode_input_grid,
    solve_state_problem, simulate_trajectories,
)

# NOTA: p_cap_eff_grid y C_tau (PHI_COST/NU_COST) son COMPARTIDOS con
# train_captor_value_net.py -- misma formula exacta (eq:p-cap/eq:cost-function-kidnapper),
# mismos parametros, usados tanto por el Estado (dentro de solve_state_problem, para
# IC^K/IR^K) como por el Secuestrador (aqui) -- consistentes entre si por construccion.
# Un intento anterior de esta sesion le agrego a p_cap un termino dependiente de la
# precision informacional (iota_tau) y recalibro C_tau a la baja, buscando que "Continuar"
# fuera viable con identificacion baja; se revirtio por completo para que las ecuaciones
# coincidan exactamente con Bernal_H.tex/Working_paper_eng.tex (sin iota_tau en p_cap).


def solve_captor_true_type_continuation(
    theta_true: str, alpha_star: float, gamma_star: float, mu_tau: dict, tau: int,
    p: Params, v_next_fn_scalar, p_rescue: float, p_nego: float,
    probs_by_type=None, pdet_by_type=None,
):
    """eq:k-bellman/eq:kidnapper-cont evaluados en el TIPO VERDADERO theta_true, en el punto
    (alpha_star, gamma_star) YA elegido por el Estado (no es una busqueda nueva). A diferencia
    de solve_state_problem (que pesa las 10 ramas (m,d) por la marginal mu_tau-mezclada, igual
    para los 4 tipos), aqui la expectativa de continuacion se pesa por Pr(m,d|theta_true) --
    la propia funcion de riesgo del tipo verdadero, no el pronostico promedio del Estado --
    mientras que mu_2(m,d) en si sigue siendo la actualizacion bayesiana completa (los 4
    tipos), porque esa es la creencia que regira las acciones FUTURAS del Estado.
    p_rescue/p_nego: protocolo de consistencia temporal (ver docstring de solve_state_problem
    en train_captor_value_net.py) -- degenerado (act_s de tau-1) en produccion, 0.5/0.5 en
    entrenamiento (ver main()).
    p_coop/p_col: se quedan en 0.5/0.5 -- no existe sorteo de tilde a_F en tau=1.
    p_cap: eq:p-cap EXACTA (sin termino de identificacion -- un intento anterior de esta
    sesion agrego uno, revertido para que las ecuaciones coincidan con Bernal_H.tex/
    Working_paper_eng.tex). C_tau usa PHI_COST/NU_COST originales (sin recalibrar).
    probs_by_type/pdet_by_type: si se pasan (precomputados, misma (alpha,gamma,tau,p) para
    los 4 tipos), evita recomputar outcome_probs_grid 4 veces cuando se llama en un bucle
    sobre TIPOS (como hace main() de este archivo) -- optimizacion de entrenamiento, sin
    cambiar el resultado. Si no se pasan, se calculan aqui (uso standalone, p.ej. en app_DL.py).
    v_next_fn_scalar(theta, mu, tau_next, p) -> float debe manejar tau_next>T devolviendo 0.0.
    Devuelve (branch, U_rel, U_kill, V_cont, extras-dict)."""
    mt = m_t(tau, p)
    p_coop, p_col = 0.5, 0.5

    if probs_by_type is None or pdet_by_type is None:
        probs_by_type, pdet_by_type = {}, {}
        for th in TIPOS:
            probs_by_type[th], pdet_by_type[th] = outcome_probs_grid(alpha_star, gamma_star, th, p, mt)
    probs_true, pdet_true = probs_by_type[theta_true], pdet_by_type[theta_true]

    p_cap_true = float(p_cap_eff_grid(alpha_star, gamma_star, theta_true, p, p_rescue, p_nego))
    p_pay_true = float(p_pay_eff_grid(alpha_star, gamma_star, theta_true, p, pdet_true, mt, p_coop, p_col))

    U_rel = -KAPPA_REL[theta_true]
    U_kill = (1.0 - p_cap_true) * ETA_REP[theta_true] - p_cap_true * F_CAP[theta_true]
    C_tau = PHI_COST[theta_true] * np.exp(KAPPA_C[theta_true] * gamma_star) + NU_COST[theta_true]

    v_next_acc = 0.0
    if v_next_fn_scalar is not None:
        for m_key in ["Cont", "1", "2", "3", "4"]:
            for d in (0, 1):
                weight = probs_true[m_key] * (pdet_true if d == 1 else (1.0 - pdet_true))
                if weight < 1e-12:
                    continue
                w = {
                    th: mu_tau[th] * probs_by_type[th][m_key]
                    * (pdet_by_type[th] if d == 1 else (1.0 - pdet_by_type[th]))
                    for th in TIPOS
                }
                z = sum(w.values())
                mu2 = (
                    {th: w[th] / z for th in TIPOS} if z > 1e-15
                    else {th: 1.0 / len(TIPOS) for th in TIPOS}
                )
                v_next_acc += weight * v_next_fn_scalar(theta_true, mu2, tau + 1, p)

    V_cont = (
        p_pay_true * p.R_millions * (1.0 - alpha_star) - C_tau
        - p_cap_true * F_CAP[theta_true]
        + p.beta_tilde[theta_true] * (1.0 - p_cap_true) * v_next_acc
    )

    # Orden de desempate (extension APPROVED by the user): bajo empate EXACTO (V_cont==U_rel),
    # max() de Python nunca reemplaza el primer maximo por uno igual -- "cont" primero implica
    # que gana Continuar en ese caso, no Liberar. No es un cambio de formula (U_rel/U_kill/
    # V_cont se calculan identico); solo determina el ganador en el caso limite de empate,
    # que es lo que hace posible el punto de indiferencia (Route 3): el Estado ve el mismo
    # empate como IR^K satisfecho ("equivalente a Liberar", via el -1e-9 nativo de
    # solve_state_problem, sin ensanchar nada), mientras el Secuestrador ve "Continuar".
    branch_vals = {"cont": V_cont, "kill": U_kill, "rel": U_rel}
    branch = max(branch_vals, key=branch_vals.get)
    extras = {"p_cap_true": p_cap_true, "p_pay_true": p_pay_true, "C_tau": C_tau, "v_next_acc": v_next_acc}
    return branch, float(U_rel), float(U_kill), float(V_cont), extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--state-net", type=str, default="captor_value_net_T10.pt")
    ap.add_argument("--out", type=str, default="captor_true_type_value_net_T10.pt")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    T, N = args.T, args.N

    print(f"[{time.strftime('%H:%M:%S')}] Cargando red del Estado (congelada): {args.state_net}", flush=True)
    state_ckpt = torch.load(args.state_net, map_location="cpu", weights_only=False)
    state_net = CaptorValueNet()
    state_net.load_state_dict(state_ckpt["state_dict"])
    state_net.eval()
    T_state = int(state_ckpt["T"])

    def state_v_next_fn(mu2: dict, tau_next: int, p: Params):
        if tau_next > T_state:
            return {th: np.zeros_like(GA) for th in TIPOS}
        out = {}
        with torch.no_grad():
            for th in TIPOS:
                x = encode_input_grid(th, mu2, tau_next, T_state, p)
                v = state_net(torch.from_numpy(x)).numpy()
                out[th] = v.reshape(GA.shape)
        return out

    print(f"[{time.strftime('%H:%M:%S')}] Simulando {N} trayectorias Monte Carlo, T={T}...", flush=True)
    t0 = time.time()
    trajs = simulate_trajectories(N, T, seed=42)
    print(f"[{time.strftime('%H:%M:%S')}] Simulacion lista en {time.time()-t0:.1f}s", flush=True)

    net = CaptorValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def captor_v_next_fn_factory(net):
        def fn(theta: str, mu: dict, tau_next: int, p: Params):
            if tau_next > T:
                return 0.0
            x = encode_input(theta, mu, tau_next, T, p)
            with torch.no_grad():
                v = net(torch.from_numpy(x[None, :])).numpy()
            return float(v[0])
        return fn

    loss_history = []
    for tau in range(T, 0, -1):
        t_level0 = time.time()
        captor_v_next_fn = None if tau == T else captor_v_next_fn_factory(net)
        X_list, y_list = [], []
        oracle_time = 0.0
        for i, (p, path) in enumerate(trajs):
            mu_tau = path[tau - 1]
            theta_f = "High wealth" if p.cov_wealth == "High" else "Low wealth"
            ot0 = time.time()

            # 1. Estado: (alpha*, gamma*) con la red del Estado, congelada. p_rescue_prev/
            # p_nego_prev se dejan en su default (0.5,0.5): no hay un "periodo anterior real"
            # en una trayectoria sintetica de Monte Carlo (protocolo de consistencia temporal,
            # ver docstring de solve_state_problem).
            a_star, g_star, _, _, _, _ = solve_state_problem(
                mu_tau, tau, T_state, p, state_v_next_fn, max(mu_tau, key=mu_tau.get), theta_f
            )

            # 2. p_rescue/p_nego: neutro fijo (mismo protocolo que el paso 1 -- sin registro
            # real de un periodo anterior en las trayectorias sinteticas de entrenamiento).
            mt_tau = m_t(tau, p)
            p_rescue, p_nego = 0.5, 0.5

            # 3. Secuestrador, por tipo verdadero: probs_by_type/pdet_by_type se calculan UNA
            # sola vez por punto (alpha*,gamma*,tau,p) y se reusan para los 4 tipos -- evita
            # recomputar outcome_probs_grid 4x por punto.
            probs_by_type, pdet_by_type = {}, {}
            for th in TIPOS:
                probs_by_type[th], pdet_by_type[th] = outcome_probs_grid(a_star, g_star, th, p, mt_tau)

            for th in TIPOS:
                _, U_rel, U_kill, V_cont, _ = solve_captor_true_type_continuation(
                    th, a_star, g_star, mu_tau, tau, p, captor_v_next_fn, p_rescue, p_nego,
                    probs_by_type=probs_by_type, pdet_by_type=pdet_by_type,
                )
                X_list.append(encode_input(th, mu_tau, tau, T, p))
                y_list.append(max(U_rel, U_kill, V_cont))

            oracle_time += time.time() - ot0
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
