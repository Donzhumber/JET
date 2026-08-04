# Deep Learning App (`app_DL.py`) — Documentación Técnica Detallada

Este documento describe, paso a paso y a nivel de código, cómo `app_DL.py` transforma los
inputs del usuario en la Tabla 5.2 y en las visualizaciones de diagnóstico. Sirve como
referencia para auditar o replicar el procedimiento.

## 1. Archivos del módulo

| Archivo | Rol | ¿Se usa en `app_DL.py`? |
|---|---|---|
| `app_DL.py` | Interfaz Streamlit + loop de simulación período a período | — (es el orquestador) |
| `dl_mechanism.py` | Parámetros estructurales, solver analítico `solve_state_quadratic_program`, clase `MechanismPolicyMLP` | Solo se usa `solve_state_quadratic_program`; `policy_mlp` se importa pero **nunca se invoca** |
| `dl_engine.py` | `BackwardInductionMLP`, `run_backward_induction_dl` | **No se importa** desde `app_DL.py` — módulo huérfano |

> ⚠️ **Nota importante sobre el nombre "Deep Learning":** la rama que el radio de la barra
> lateral llama *"Deep Learning Solver (Emulator + O(1) Boundary)"* ejecuta
> `solve_state_quadratic_program` (ver §4), que es un **solver analítico cerrado** (condiciones
> de primer orden + evaluación de esquinas/bordes de `[0,1]²`), **no una red neuronal entrenada**.
> Las clases `MechanismPolicyMLP` (en `dl_mechanism.py`) y `BackwardInductionMLP` (en
> `dl_engine.py`) tienen pesos **aleatorios sin entrenar** (`rng.normal(...)` con semilla fija) y
> ninguna de las dos participa en el cálculo que produce la Tabla 5.2. El "Deep Learning" aquí es
> conceptual/etiqueta del artículo (evocando "inferencia en O(1)" en vez de grid search), no una
> ejecución real de una NN entrenada con datos.

---

## 2. Entrada de datos (Sidebar → estado inicial)

Todo el "dato" es sintético: no se carga ningún CSV/fuente externa en este módulo (a diferencia
de `app.py` en la raíz de `Streamlit/`, que sí usa `Data_CMH.csv`). Los inputs del usuario
parametrizan un generador aleatorio con semilla fija.

| Control (sidebar) | Variable | Efecto |
|---|---|---|
| `escenario` (selectbox) | `tipo_real`, `priors_init`, `family_wealth`, `victim_profile`, `region`, `psi_H_default`, `c_bar_default` | Fija el tipo verdadero de secuestrador oculto y la creencia previa `μ₀` (líneas 91–122) |
| `T_horizon` (slider 50–200) | `T_horizon` | Número de períodos `t` que corre el loop |
| `psi_H` (slider 0–1) | `psi_H` | Peso de la ganancia de información / exploración activa en la función de pérdida del Estado |
| `c_bar` (slider 0–0.2) | `c_bar` | Piso inferior de la temperatura MDG (ruido mínimo, aun con creencias muy concentradas) |
| `metodologia` (radio) | `metodologia` | Selecciona la rama del solver: analítico "Deep Learning" vs. grid search 11×11 |
| botón `Start Simulation` | `iniciar_sim` | Dispara el bloque `if iniciar_sim:` que corre todo el loop |

Estado inicial (líneas 178–196):

```python
p_init = priors_init / sum(priors_init)          # normaliza a distribución de probabilidad
mu_t = {tipo: p_init[i] for tipo, i in ...}       # creencia inicial μ_0 sobre {DC, PAR, ELN, FARC}
rng = np.random.default_rng(1007)                 # semilla fija → reproducible
R_base, omega_p, omega_k = 100.0, 0.5, 2.0        # escala de rescate y ponderadores de costo
```

---

## 3. El loop principal `for t in range(T_horizon)` (líneas 198–355)

Cada iteración representa un período `t` de negociación y produce **una fila** del DataFrame
final. Los diez pasos, en orden:

### Paso 1–2 — Entropía y precisión de la creencia
```python
entropy_t = -Σ μ_t(θ) ln μ_t(θ)      # Entropía de Shannon H(μ_t)
entropy_0 = -Σ μ_0(θ) ln μ_0(θ)      # Entropía inicial (normalizador)
iota_t    = max(μ_t.values())        # precisión: probabilidad del tipo más creído
hat_theta_t = argmax μ_t              # tipo más probable según el Estado
```
`iota_t` mide qué tan seguro está el Estado de haber identificado al secuestrador real.

### Paso 3 — Temperatura MDG (Mano de Dios – Guadalupe)
```python
temp_t = max( (entropy_t/entropy_0) * exp(-0.01 * t), c_bar )
```
La temperatura decae geométricamente con `t` (a medida que el juego avanza, el "ruido" del
mecanismo se reduce), pero nunca baja del piso `c_bar`. Esta `temp_t` alimenta el sorteo MDG
(paso 8) — cuanto más alta, más aleatoria la acción "realizada" vs. la "óptima".

### Paso 4 — Coeficientes cuadráticos de la pérdida del Estado
```python
q_alpha       = Σ μ_t(θ) · c_alpha(θ)          # costo esperado de interdicción financiera
q_gamma       = Σ μ_t(θ) · c_gamma(θ)          # costo esperado de presión operativa
q_gamma_alpha = 0.2 · (q_alpha + q_gamma)       # término de complementariedad entre instrumentos
b_alpha = -omega_p · R_base
b_gamma = 0.0
const   =  omega_p · R_base
```
Estos coeficientes definen la forma cuadrática de la pérdida que el Estado minimiza sobre
`(α, γ) ∈ [0,1]²`, ponderando el costo por tipo con la creencia actual `μ_t` (esto es lo que hace
al problema "bayesiano": el Estado no conoce `θ_K` real, solo su distribución posterior).

### Paso 5 — Invocación del solver (bifurcación por metodología)

**Rama "Deep Learning" → `solve_state_quadratic_program`** (`dl_mechanism.py:95-169`):
1. Calcula el mínimo interior sin restricciones vía condiciones de primer orden:
   `det = 4·q_γ·q_α − q_γα²`; si `|det|` no es ~0, resuelve el sistema lineal 2×2 y acepta el
   punto solo si cae dentro de `[0,1]²`.
2. Agrega como candidatos las 4 esquinas `(0,0), (0,1), (1,0), (1,1)`.
3. Agrega los mínimos condicionales en cada uno de los 4 bordes del cuadrado (fijando `γ=0`,
   `γ=1`, `α=0`, `α=1` y resolviendo la derivada de la variable libre).
4. Evalúa la pérdida cuadrática en **todos** los candidatos, restando un término de "ganancia de
   información" si `psi_H > 0`:
   ```python
   entropy_gain = psi_H * (a*0.15 + g*0.1) * (1 - t/150)
   loss -= entropy_gain
   ```
   (la ganancia decae linealmente con `t`, capturando que explorar es más valioso al principio).
5. Devuelve `(α*, γ*, valor_óptimo)` — el candidato de menor pérdida.

Es un solver **O(1)** porque evalúa un número fijo y pequeño de candidatos (a lo sumo 9) en vez
de una grilla, aprovechando que el óptimo de un problema cuadrático restringido a un cuadrado
siempre está en el interior, una arista o una esquina.

**Rama "Classical Solver" (líneas 229–241):** grid search burdo de 11×11 puntos en
`[0,1]²`, evaluando la misma expresión cuadrática (sin el término de entropía) y quedándose con
el mínimo. Es el benchmark de fuerza bruta contra el que se compara la velocidad del solver
analítico (de ahí las métricas de "Speedup Factor" en la Tabla 5.2 / Tab 2).

### Paso 6 — Acción óptima del Estado: Rescatar vs. Negociar
```python
val_rescue = omega_k * (1 - (0.5 + 0.4·iota_t si hat_theta_t==tipo_real, si no 0.3)) + 1.5·γ*²
if val_rescue < val_opt:  a_S* = "Rescue"; α*=0, γ*=0.9   # rescate anula interdicción, sube presión
else:                     a_S* = "Negotiate"
```

### Paso 7 — Mejor respuesta del secuestrador y la familia
Secuestrador real (usa los parámetros de `tipo_real`, no la creencia del Estado):
```python
u_cont = λ_pay · R_base · (1-α*) - c_gamma · γ*
u_kill = -alpha_leth
u_rel  = -k_rel
a_K* = argmax{u_cont, u_kill, u_rel}   # Continuar / Matar / Liberar
```
Familia:
```python
u_coop = 0.8·R_base - 0.5·γ*
u_col  = 0.5·R_base - 2.0·α*
a_F* = "Cooperate" si u_coop > u_col, si no "Collude"
```

### Paso 8 — Sorteo MDG: de la acción óptima a la acción "realizada"
```python
def draw_mdg(actions, intent, temp):
    probs = [exp(1/temp) si act==intent, si no exp(0)=1]
    probs /= sum(probs)
    return rng.choice(actions, p=probs)
```
Esto implementa la ley de implementación MDG del marco teórico (Tab 1, fórmula del logit con
temperatura `T_t`): la acción "óptima" (`a_S*`, `a_K*`, `a_F*`) se ejecuta con alta probabilidad,
pero con ruido `temp_t` puede desviarse — de ahí las columnas `atilde_S`, `atilde_K`, `atilde_F`
en la tabla (acción *realizada* vs. *intencionada*).

### Paso 9 — Riesgos en competencia (competing risks) y absorción
Si el caso sigue "Continue", se calculan 4 tasas de salida (pago, muerte, rescate táctico,
liberación) en función de `(α*, γ*)` y los parámetros del tipo real:
```python
p_cont = exp(-Σ tasas)
if rng.random() > p_cont:
    outcome_realized = rng.choice(["Ransom Payment","Death","Tactical Rescue","Release"], p=tasas normalizadas)
    absorb_step = t
```
Una vez que `outcome_realized != "Continue"`, permanece fijo en las iteraciones siguientes (el
caso ya "absorbió" / terminó).

### Paso 10 — Actualización bayesiana de creencias
Para cada tipo candidato `θ`, se recalcula la verosimilitud del outcome observado bajo los
parámetros de ese tipo (mismas fórmulas de tasas que el paso 9, pero evaluadas con los
parámetros de cada `θ`, no solo el real):
```python
lik(θ) = p_cont(θ)                  si outcome_realized == "Continue"
       = tasa_outcome(θ)            en otro caso
mu_t(θ) ∝ mu_t(θ) · lik(θ)          # regla de Bayes, luego normalizar
```
Si el denominador colapsa numéricamente (`< 1e-12`), se reinicia a uniforme `0.25` cada tipo
(salvaguarda numérica).

Finalmente se evalúan restricciones ex-post `IC_K` (siempre `True` en esta implementación —
placeholder), `IR_K` (`u_cont ≥ -1`) e `IR_F` (`u_coop ≥ u_col`), y se agrega la fila al registro
`t_records` con las 20 columnas (creencias, entropía, precisión, instrumentos, acciones óptimas y
realizadas, outcome, cumplimiento de restricciones, tiempo de cómputo del paso en ms).

---

## 4. Generación de la Tabla 5.2 (Tab 2, líneas 361–373)

1. `df_sim = pd.DataFrame(t_records)` — un DataFrame con una fila por período simulado.
2. Métricas de desempeño (tarjetas superiores): tiempo total, tiempo promedio por paso, factor de
   *speedup* frente al límite de 2 minutos asumido como referencia de usuario.
3. `df_t52 = df_sim[[...12 columnas...]]` selecciona el subconjunto de columnas que conforman la
   tabla del artículo (creencias `μ`, instrumentos `α*/γ*`, acciones óptimas/realizadas del Estado
   y el secuestrador, outcome).
4. `st.dataframe(df_t52.head(15), ...)` — se muestran únicamente los **primeros 15 períodos**.

La tabla **no es estática ni pre-calculada**: se reconstruye por completo cada vez que se pulsa
"Start Simulation", pero al usar `rng = np.random.default_rng(1007)` con semilla fija, es
**determinística** para una combinación dada de escenario/sliders (correr dos veces con los
mismos inputs produce exactamente la misma tabla).

---

## 5. Visualizaciones (Tab 3, líneas 376–478)

| Gráfico | Datos | Propósito |
|---|---|---|
| Convergencia de creencias | `μ_t(θ)` para los 4 tipos vs. `t`, línea vertical en `absorb_step` | Ver si el Estado converge a identificar `tipo_real` antes de la absorción |
| Trayectoria de instrumentos | `α*_t`, `γ*_t` vs. `t` | Ver cómo evolucionan interdicción y presión óptimas |
| Incertidumbre y precisión | `H(μ_t)`, `ι_t` vs. `t` | Evolución de entropía (incertidumbre) vs. precisión (confianza) |
| Factibilidad de restricciones | `IC_K`, `IR_F` (1/0) vs. `t` | Verificación ex-post de que las restricciones de incentivos se cumplen período a período |

---

## 6. Resumen del flujo completo

```
Sidebar inputs (escenario, T, ψ_H, c̄, metodología)
        │
        ▼
μ_0 normalizado + semilla RNG fija (1007)
        │
        ▼
┌── for t in range(T_horizon): ──────────────────────────────┐
│ 1. H(μ_t), ι_t                                              │
│ 2. Temperatura MDG T_t = max((H_t/H_0)·e^{-0.01t}, c̄)      │
│ 3. Coeficientes cuadráticos ponderados por μ_t              │
│ 4. Solver (analítico O(1) o grid 11×11) → (α*, γ*)          │
│ 5. Rescatar vs Negociar                                     │
│ 6. Mejor respuesta secuestrador real / familia              │
│ 7. Sorteo MDG (ruido logit con temperatura T_t)             │
│ 8. Riesgos en competencia → outcome / absorción             │
│ 9. Actualización bayesiana de μ_t                           │
│10. Registrar fila                                           │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
df_sim (T_horizon filas) → Tabla 5.2 = primeras 15 filas, columnas seleccionadas
        │
        ▼
Gráficos de diagnóstico (Tab 3)
```
