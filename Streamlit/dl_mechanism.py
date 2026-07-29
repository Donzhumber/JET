import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List

# =====================================================================
# 1. PARÁMETROS ESTRUCTURALES Y ENTORNOS (PASO 1)
# =====================================================================
TIPOS_SECUESTRADOR = ["DC", "PAR", "ELN", "FARC"]

PARAMS_ORGANIZACION = {
    "DC": {
        "lambda_pay": 0.08,
        "lambda_kill": 0.005,
        "lambda_res": 0.01,
        "c_alpha": 1.2,
        "c_gamma": 0.5,
        "eta_0": -0.8,
        "alpha_leth": 1.5,
        "u_rel_loss": 0.3,
        "k_rel": 0.2
    },
    "PAR": {
        "lambda_pay": 0.04,
        "lambda_kill": 0.02,
        "lambda_res": 0.03,
        "c_alpha": 2.5,
        "c_gamma": 1.8,
        "eta_0": -1.5,
        "alpha_leth": 2.8,
        "u_rel_loss": 0.8,
        "k_rel": 0.6
    },
    "ELN": {
        "lambda_pay": 0.02,
        "lambda_kill": 0.015,
        "lambda_res": 0.04,
        "c_alpha": 3.0,
        "c_gamma": 2.2,
        "eta_0": -2.0,
        "alpha_leth": 3.2,
        "u_rel_loss": 1.2,
        "k_rel": 0.8
    },
    "FARC": {
        "lambda_pay": 0.01,
        "lambda_kill": 0.025,
        "lambda_res": 0.05,
        "c_alpha": 3.5,
        "c_gamma": 2.5,
        "eta_0": -2.5,
        "alpha_leth": 3.5,
        "u_rel_loss": 1.5,
        "k_rel": 1.0
    }
}

class KidnappingEnvironment:
    """
    Representa el entorno estructural con los riesgos competitivos de Cox
    y la actualización bayesiana de creencias de acuerdo a Bernal_H.tex.
    """
    def __init__(self, T_mad: float = 5.0, lambda_4: float = 0.0005):
        self.T_mad = T_mad
        self.lambda_4 = lambda_4
        self.TIPOS = TIPOS_SECUESTRADOR
        
    def get_maturation_filter(self, t: float) -> float:
        # Eq. (1721): M(t) = min(1, (t/T_mad)^2)
        return float(min(1.0, (t / max(1e-9, self.T_mad)) ** 2))
        
    def compute_cox_hazards(self, alpha: float, gamma: float, theta_K: str, t: float) -> Dict[str, float]:
        """
        Calcula las intensidades específicas de causa de Cox ajustadas por madurez (Eq. 1711, 1721).
        """
        p = PARAMS_ORGANIZACION[theta_K]
        M_t = self.get_maturation_filter(t)
        
        # Intensidades basales multiplicadas por exponencial del instrumento
        lambda_pay = p["lambda_pay"] * (1.0 - alpha) * M_t
        lambda_kill = p["lambda_kill"] * (1.0 + gamma) * M_t
        lambda_res = p["lambda_res"] * (1.0 + gamma) * M_t
        lambda_release = self.lambda_4  # Salida exógena inalterada por madurez
        
        return {
            "Ransom Payment": lambda_pay,
            "Death": lambda_kill,
            "Tactical Rescue": lambda_res,
            "Release": lambda_release
        }

# =====================================================================
# 2. POLÍTICA DEL ESTADO COMO RED NEURONAL (PASO 2)
# =====================================================================
class StateNetwork(nn.Module):
    """
    Red Neuronal del Planificador Social (RegretNet).
    Inputs: [mu_DC, mu_PAR, mu_ELN, mu_FARC, theta_F, theta_V, t] (Tamaño: 7)
    Outputs: [alpha_t, gamma_t] en [0, 1]^2 (Sigmoid activation)
    """
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        super(StateNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # Fuerza el rango de salida a [0, 1]^2
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# =====================================================================
# 3. CAPA PROBABILÍSTICA CON TEMPERATURA DE ENTROPÍA MDG (PASO 3)
# =====================================================================
class MDGLayer:
    """
    Actualiza la temperatura MDG según la entropía de Shannon (Eq. 550) y
    aplica la activación Softmax stochástica con fricción institucional.
    """
    def __init__(self, T0: float = 0.30, c_bar: float = 0.05, eta_cal: float = 0.075):
        self.T0 = T0
        self.c_bar = c_bar
        self.eta_cal = eta_cal
        
    def compute_temperature(self, mu: Dict[str, float], mu_0: Dict[str, float], t: int) -> float:
        # H(mu_t) = -sum(mu * ln(mu))
        H_t = -sum(v * np.log(v) for v in mu.values() if v > 1e-12)
        H_0 = -sum(v * np.log(v) for v in mu_0.values() if v > 1e-12)
        H_0 = max(H_0, 1e-12)
        
        # Eq. (550): T_t = T_0 * max{ (H(mu_t)/H(mu_0)) * e^(-eta * t), c }
        return float(self.T0 * max((H_t / H_0) * np.exp(-self.eta_cal * t), self.c_bar))
        
    def sample_action(self, intent: str, alternatives: List[str], temp: float, rng: np.random.Generator) -> str:
        # Eq. (280): Distribución logit de Mano de Dios - Guadalupe
        probs = np.array([np.exp(1.0 / temp) if alt == intent else np.exp(0.0) for alt in alternatives])
        probs /= np.sum(probs)
        return rng.choice(alternatives, p=probs)

# =====================================================================
# 4. SOLVEDOR ENTRENADOR REGRETNET (PASOS 5 Y 6)
# =====================================================================
class RegretNetTrainer:
    """
    Optimiza la red de política del Estado utilizando el costo social dual del
    modelo Bernal_H y el método del Lagrangiano Aumentado para forzar restricciones IC.
    """
    def __init__(self, net: StateNetwork, lr: float = 0.005, rho: float = 1.0):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.rho = rho  # Coeficiente de penalización cuadrática
        
        # Inicializar multiplicadores de Lagrange para las restricciones IC de los 4 tipos
        # IC_theta: Regret_theta <= 0
        self.lambdas = {th: torch.zeros(1, requires_grad=False) for th in TIPOS_SECUESTRADOR}
        
    def compute_captor_utility(self, alpha: torch.Tensor, gamma: torch.Tensor, theta_K: str) -> torch.Tensor:
        """
        Calcula la utilidad esperada del secuestrador (Eq. 384).
        U^K = lambda_pay * R * (1 - alpha) - c_gamma * gamma
        """
        p = PARAMS_ORGANIZACION[theta_K]
        # Escala R_base = 100.0
        R_base = 100.0
        return p["lambda_pay"] * R_base * (1.0 - alpha) - p["c_gamma"] * gamma

    def train_step(self, mu_t: Dict[str, float], theta_F: float, theta_V: float, t: int, psi_H: float) -> Tuple[float, Dict[str, float]]:
        self.optimizer.zero_grad()
        
        # Convertir datos del estado a tensores de PyTorch
        mu_tensor = torch.tensor([mu_t[th] for th in TIPOS_SECUESTRADOR], dtype=torch.float32)
        state_in = torch.cat([mu_tensor, torch.tensor([theta_F, theta_V, t / 150.0], dtype=torch.float32)])
        
        # Evaluar la red para el reporte honesto
        policy_honest = self.net(state_in)
        alpha_h, gamma_h = policy_honest[0], policy_honest[1]
        
        # 1. Pérdida social esperada honesta (Eq. 2182 + 2221 + 2223)
        R_base = 100.0
        omega_p = 0.5
        omega_k = 2.0
        
        social_loss = torch.tensor(0.0, dtype=torch.float32)
        
        # Calcular pérdidas por cada tipo ponderado por su creencia mu_t
        for th in TIPOS_SECUESTRADOR:
            p = PARAMS_ORGANIZACION[th]
            weight = mu_tensor[TIPOS_SECUESTRADOR.index(th)]
            
            # C_maint cuadrático de Eq. (2279)
            c_maint = p["c_alpha"] * (alpha_h ** 2) + p["c_gamma"] * (gamma_h ** 2) + 0.2 * (p["c_alpha"] + p["c_gamma"]) * alpha_h * gamma_h
            
            # Pérdida en rama de negociación: costo de transferencia + probabilidad de muerte + mantenimiento
            loss_th = omega_p * R_base * (1.0 - alpha_h) + omega_k * p["lambda_kill"] * (1.0 + gamma_h) + c_maint
            social_loss += weight * loss_th
            
        # Subsido de exploración de entropía
        entropy_t = -torch.sum(mu_tensor * torch.log(mu_tensor + 1e-12))
        # Para el gradiente dinámico, penalizamos la pérdida con la entropía actual para incentivar la reducción
        social_loss += psi_H * entropy_t
        
        # 2. Cálculo del Arrepentimiento Empírico (Regret) para restricciones IC
        regrets = {}
        for true_th in TIPOS_SECUESTRADOR:
            # Utilidad honesta del secuestrador
            u_honest = self.compute_captor_utility(alpha_h, gamma_h, true_th)
            
            # Buscar el desvío óptimo simulando reportes alternativos
            max_u_lying = torch.tensor(-1e9, dtype=torch.float32)
            
            for reported_th in TIPOS_SECUESTRADOR:
                if reported_th == true_th:
                    continue
                # Evaluar política del Estado si el secuestrador reportara mentirosamente reported_th
                mu_lying = torch.zeros(4, dtype=torch.float32)
                mu_lying[TIPOS_SECUESTRADOR.index(reported_th)] = 1.0
                state_lying = torch.cat([mu_lying, torch.tensor([theta_F, theta_V, t / 150.0], dtype=torch.float32)])
                
                policy_lying = self.net(state_lying)
                alpha_lying, gamma_lying = policy_lying[0], policy_lying[1]
                
                u_lying = self.compute_captor_utility(alpha_lying, gamma_lying, true_th)
                if u_lying > max_u_lying:
                    max_u_lying = u_lying
            
            # Regret = max(0, U_lying - U_honest)
            regrets[true_th] = torch.clamp(max_u_lying - u_honest, min=0.0)
            
        # 3. Formular la función de pérdida del Lagrangiano Aumentado
        lagrangian_terms = torch.tensor(0.0, dtype=torch.float32)
        penalty_terms = torch.tensor(0.0, dtype=torch.float32)
        
        for th in TIPOS_SECUESTRADOR:
            lagrangian_terms += self.lambdas[th] * regrets[th]
            penalty_terms += (self.rho / 2.0) * (regrets[th] ** 2)
            
        total_loss = social_loss + lagrangian_terms + penalty_terms
        
        # Backpropagation
        total_loss.backward()
        self.optimizer.step()
        
        # 4. Actualizar multiplicadores de Lagrange (Dual Ascent step)
        for th in TIPOS_SECUESTRADOR:
            self.lambdas[th].data += self.rho * regrets[th].data
            self.lambdas[th].data = torch.clamp(self.lambdas[th].data, min=0.0)
            
        return float(social_loss.item()), {th: float(regrets[th].item()) for th in TIPOS_SECUESTRADOR}
