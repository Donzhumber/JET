# Tabla 5.2 — Cálculo de la columna τ=1, variable por variable

Documento generado a partir de la implementación real en `app_DL.py` (bloque "Run State
Optimization", pestaña 3 "Results"). Todas las fórmulas citadas (`eq:...`) están tomadas de
`Bernal_H.tex` y/o `Working_paper_eng.tex`. Salvo que se indique lo contrario, todos los
valores de τ=1 se calculan **dentro del mismo click** del botón "Run State Optimization",
en el orden en que aparecen abajo (cada bloque puede depender de los anteriores).

Convención: `_t1opt` = `st.session_state["tau1_state_opt_result"]`, el diccionario que
guarda todos los resultados de τ=1 tras presionar el botón.

---

## 1. $a_F^{*}$ — acción óptima de la Familia

**Fórmula**: $a_F^{1,*}=\arg\max\{\mathcal U_1^F(\text{coop}),\mathcal U_1^F(\text{col})\}$
(`eq:f-coop`/`eq:f-col`/`eq:ir-family`, fórmula exacta del paper).

**Cómo se calculó (τ=1)**: llamando a `famopt.family_utilities(mu1, theta_f, alpha_1*, gamma_1*, 1, p)`
(`family_optimization.py`), que usa $\mu_1$ (ya actualizado por Bayes), $(\alpha_1^*,\gamma_1^*)$
(recién resueltos por el Estado, ver §5), y las fórmulas $V_L$/$F_{col}$/$p_{surv}$/$p_{rel}$
ponderadas por $\mu_1$ sobre los 4 tipos de $\theta_K$. Es la MISMA fórmula que usa internamente
`solve_state_problem` como restricción $IR^F$ del Estado — no hay una versión simplificada en paralelo.

---

## 2. $\tilde a_F$ — acción de la Familia ejecutada vía MDG

**Fórmula**: mecanismo logit de 2 ramas (`eq:LI-atilde`), centrado en $a_F^{1,*}$ (§1), con
temperatura $M_{\tau=1}$.

**Cómo se calculó**: $\mathbb P(\text{Cooperate})=\dfrac{e^{1/M_{\tau=1}\cdot\mathbb 1\{a_F^{1,*}=\text{Cooperate}\}}}{\sum_a e^{1/M_{\tau=1}\cdot\mathbb 1\{a=a_F^{1,*}\}}}$
(y análogo para Collude), con $M_{\tau=1}=$`cvn.m_t(1,p)`. Sorteo $u\sim\text{Unif}(0,1)$
fresco (`np.random.default_rng().random()`) y mapeo por intervalo — exactamente el mismo
mecanismo que ya se usaba para $\tilde a_S/\tilde a_K$.

---

## 3. $a_K^{*}(\theta_K)$ — acción óptima del Secuestrador, tipo verdadero

**Fórmula**: $a_K^{1,*}(\theta_K^{true})=\arg\max\{U_{rel},U_{kill},V_{cont,1}\}$ (`eq:k-bellman`,
`eq:k-rel`, `eq:k-kill`, `eq:kidnapper-cont`).

**Cómo se calculó**: `cttn.solve_captor_true_type_continuation(cov_perp, alpha_1*, gamma_1*, mu1, 1, p, v_next_fn, p_rescue_1, p_nego_1)`
(`train_captor_true_type_net.py`), usando la red **propia** del Secuestrador
(`captor_true_type_value_net_T10.pt`, entrenada con peso $\Pr(m,d\mid\theta_K^{true})$ —
distinta de la red del Estado, que usa peso marginal μ-mezclado). $\tilde p_{cap,1}$ se refina
con las probabilidades REALES del sorteo de $\tilde a_S$ (§2 arriba, en el sentido de que usa
$p_{rescue,1}/p_{nego,1}$ del sorteo de $\tilde a_S$, no 0.5/0.5 neutro). Si la red propia no
existe aún, cae de vuelta (fallback, con warning explícito) a la red del Estado.

---

## 4. $\tilde a_K(\theta_K)$ — acción del Secuestrador ejecutada vía MDG

**Fórmula**: mecanismo logit de 3 ramas, pero MEZCLADO con inercia (extensión no dictada
literalmente por el paper, aprobada explícitamente por el usuario):
$$\mathbb P_I(\tilde a\mid\tau)=(1-\lambda(\tau))P^{inercia}+\lambda(\tau)P^{racional}_\tau,\qquad \lambda(\tau)=1-M_\tau/M_1$$

**Cómo se calculó**: probabilidades racionales del logit de 3 ramas centrado en $a_K^{1,*}$
(§3) con $M_{\tau=1}$; luego `cvn.mdg_inertia_mix_probs({...}, 1, p)` mezcla con
$P^{inercia}(\text{Continue})=0.9$ (resto repartido proporcional al racional). En $\tau=1$,
$\lambda(1)=0$ exacto ⇒ 100% inercia (Continue≈0.9 sin importar cuál sea $a_K^{1,*}$). Sorteo
$u\sim\text{Unif}(0,1)$ fresco sobre las probabilidades YA mezcladas.

---

## 5. $a_S^{*}$ óptima — acción/instrumentos óptimos del Estado

**Fórmula**: $\arg\min$ sobre los "floors" de rama restringidos por $\Gamma_1(\mu_1)$
($IR^K_{EV}\wedge IC^K_{EV}\wedge IR^F$), usando la red entrenada del Estado ($T=10$).

**Cómo se calculó**: `cvn.solve_state_problem(mu1, 1, T_trained, p, v_next_fn, cov_perp, theta_f, p_rescue_prev=_p_rescue_prev_tab3, p_nego_prev=_p_nego_prev_tab3)`
— grid search $101\times101$ sobre $(\alpha,\gamma)$ con máscaras booleanas IC/IR. El par
$(p_{rescue\_prev},p_{nego\_prev})$ se ancla al desenlace YA EJECUTADO por el Estado en τ=0
(`act_s` de sesión): $(1,0)$ si Rescue, $(0,1)$ si Negotiate — protocolo de consistencia
temporal (extensión aprobada, no literal del paper, para el uso interno de `eq:p-cap`).

---

## 6. $\tilde a_S$ — acción del Estado ejecutada vía MDG

**Cómo se calculó**: mismo mecanismo logit de 2 ramas que §2, centrado en $a_S^{1,*}$ (§5) con
$M_{\tau=1}$. Sorteo fresco $u\sim\text{Unif}(0,1)$.

---

## 7–8. $\alpha_t^{*}$, $\gamma_t^{*}$ Estado — instrumentos continuos

**Cómo se calculó**: son directamente `_t1opt["alpha"]`/`_t1opt["gamma"]`, el resultado del
mismo `solve_state_problem` de §5 (no hay cálculo adicional — son el mismo argmin).

---

## 9–12. Benchmarks de información perfecta por tipo — $\gamma_R^{\theta,*},\alpha_R^{\theta,*},\gamma_N^{\theta,*},\alpha_N^{\theta,*}$ (×4 tipos)

**Rama rescate** ($\gamma_R,\alpha_R$): $\arg\min\ \omega_K(1-p_{surv})+C_{ops}(\gamma,\alpha;\theta)$,
recalculado con `cvn.p_surv_raw(theta,theta,p)` (desplazamiento constante — $C_{ops}$ no
depende de $\tau$, así que el argmin no puede moverse numéricamente, pero ya no es una copia
estática de τ=0).

**Rama negociación** ($\gamma_N,\alpha_N$): $\arg\min\ \omega_P R(1-\alpha)+\omega_K h_2(\gamma,\alpha;\theta)+C_{maint}(\gamma,\alpha;\theta)$,
con $h_2$ el hazard de muerte **genuinamente reevaluado** vía `cvn.outcome_probs_grid(...,M_{\tau=1})`
en vez del $M_t$ fijo del slider de Tab 1.

**Cómo se calculó**: grid search $101\times101$ por tipo, solo si el botón fue presionado y
$\mu_1$ está lista; guardado en `_t1opt["benchmarks_tau1"]`.

---

## 13. $H(\mu)$ — entropía de la creencia

**Fórmula**: $H(\mu_\tau)=-\sum_\theta\mu_\tau(\theta)\log\mu_\tau(\theta)$.

**Cómo se calculó**: evaluada en $\mu_1$ (ya actualizado por Bayes), devuelta directamente por
`solve_state_problem` como `extra["H_mu"]`.

---

## 14. $\Delta H$ Estado — reducción esperada de entropía

**Cómo se calculó**: en τ=1, se evalúa "un paso hacia adelante" en el mismo punto
$(\alpha_1^*,\gamma_1^*)$ recién resuelto — Bayes "minimal record" (solo $m,d$, sin voz/
implementación) sobre las 10 combinaciones $(m,d)$, ponderadas por su propia probabilidad
predictiva. Devuelto por `solve_state_problem` como `extra["delta_H"]`.

---

## 15. $\Gamma_t(\mu_t)$ bajo EV — factibilidad

**Cómo se calculó**: `_t1opt["feasible"]`, el booleano de factibilidad ($IR^K_{EV}\wedge IC^K_{EV}\wedge IR^F$)
que ya determina el propio grid search de §5 — no es un chequeo aparte, es el resultado del
argmin restringido.

---

## 16. $IR^K(\theta_K)$ tipo verdadero

**Cómo se calculó**: `_t1opt["ir_k_true_gap"]` = $V_{cont,1}-\max(U_{rel},U_{kill})$ (o el gap
análogo) para $\theta_K^{true}$, devuelto por `solve_state_problem`. "Yes" si el gap $\ge -10^{-9}$.

---

## 17. $-\text{sgn}(\kappa_h(\theta_K,t))$ (×4 tipos)

**Fórmula**: $\kappa_h(\theta,\tau)=\zeta_\gamma(\theta)\cdot(\tilde\lambda_2+\tilde\lambda_3-\tilde\lambda_1)$
(signo de $\partial\mathbb E[\tau\mid\theta_K]/\partial\gamma_t^*$), vía `eq:hj`
(Working_paper_eng.tex).

**Cómo se calculó**: misma fórmula que τ=0 (función `_outcome_probs_tab3`), pero:
- $(\alpha,\gamma)\to(\alpha_1^*,\gamma_1^*)$ (§7–8),
- $M_t\to M_{\tau=1}$,
- acciones ejecutadas $(\tilde a^F,\tilde a^K,\tilde a^S)$ (que entran vía los términos
  $\phi_F,\phi_K,\zeta_R$ del "material state" $\mathcal C_t$) $\to$ las REALIZADAS de τ=1
  (§2, §4, §6) en vez de las de τ=0.

Implementado como una función local nueva (`_lam123_tau1`), duplicando la estructura exacta de
`_outcome_probs_tab3` con esas 3 sustituciones — verificado numéricamente idéntica a la fórmula
de τ=0 cuando se le dan los mismos inputs.

---

## 18. $m$ — resultado físico realizado (riesgos competitivos)

**Fórmula**: multinomial de 5 celdas (`eq:LH-compacta`, Bernal_H.tex ≡ `eq:hj`/`eq:pCont`/
`eq:xi`, Working_paper_eng.tex), transformada inversa (`eq:m-law-inverse`/mismo orden
$\mathcal G_t$ que el expander de Block C).

**Cómo se calculó**: `cvn.outcome_probs_grid(alpha_1*, gamma_1*, cov_perp, p, M_{\tau=1})`
(sin cambios, misma función ya usada en §9–12) evaluada para el **tipo verdadero**. Sorteo
$v_1\sim\text{Unif}(0,1)$ fresco, mapeado por intervalos $[0,p_{Cont,1})$, $[p_{Cont,1},
p_{Cont,1}+\bar h_1)$, etc. (Cont<pay<kill<res<rel). Si $m_1\neq$ Cont, se marca
explícitamente que el episodio cierra en τ=1 por `eq:stopping-time` (Bernal_H.tex). Es
**ilustrativo**: no realimenta $\mu_1$ (ya calculado con el $m$ de τ=0).

---

## 19. $\mu(\theta)$ (×4 tipos) — creencia posterior

**Fórmula**: `eq:bayes-update` — actualización Bayes completa usando **todos** los tipos como
peso (a diferencia de la continuación del Secuestrador, que pesa solo por el tipo verdadero).

**Cómo se calculó**: $\mu_1(\theta)\propto\mu_0(\theta)\cdot\mathcal L_{I,K}(\theta)\cdot\Pr(m\mid\theta)\cdot\Pr(d\mid\theta)\cdot\mathcal L_{C}(\theta)$,
usando las señales YA REALIZADAS de τ=0 ($m$ de Tab 1 Block C, $d$ de Block D, $\tilde a_K$ de
Block A, $V_t$/voz de Block F) — se calcula en cuanto $m$ y $d$ están sorteados, **no depende**
de presionar el botón "Run State Optimization" (a diferencia de casi todas las demás filas).

---

## 20–23. $\alpha_R^{\mu},\gamma_R^{\mu},\alpha_N^{\mu},\gamma_N^{\mu}$ — instrumentos ponderados por creencia

**Fórmula**: $\sum_\theta\mu_\tau(\theta)\cdot(\text{benchmark}^{\theta,*})$.

**Cómo se calculó**: promedio ponderado por $\mu_1$ (§19) de los 4 benchmarks por tipo de
§9–12 (`_t1opt["benchmarks_tau1"]`), calculado justo después del grid search de benchmarks.

---

## 24. $V$ (voz) — verosimilitud de evidencia de voz

**Fórmula**: `eq:LC`/`eq:Lvoz-diag`: $\mathcal L_{C,\tau}(\theta_K)=[\mathcal L_{voz,\tau}(\theta_K)\pi_{call}(\theta_K)]^{\omega_{voz}}$
si $V_\tau=1$, o $[1-\pi_{call}(\theta_K)]^{\omega_{voz}}$ si $V_\tau=0$; `eq:voz-descomp`:
$x_\tau^{obs}=\bar x(\theta_K)+\varepsilon_L+\varepsilon_S$.

**Cómo se calculó**: sorteo GENUINO (no reusa el slider fijo de τ=0), reutilizando
`rational_behavior.py` (mismo mecanismo ya validado en app.py) sin modificarlo:
1. $\tilde\pi_{call}(\theta)$ **realizado** se sortea UNA sola vez (primer click) vía
   `sample_incident_pi_call_realized` — Beta($\kappa\pi,\kappa(1-\pi)$), $\kappa=30$, anclado
   al prior `PI_CALL` — y se reutiliza en clicks futuros (rasgo del incidente, no del período).
2. $V_1\sim\text{Bern}(\tilde\pi_{call}(\theta_K^{true}))$ vía `draw_voice_indicator`.
3. Si $V_1=1$: $x_1^{obs}$ (vector de 4 rasgos acústicos) vía `sample_voice_observation`,
   usando `VOZ_PARAMS_DEFAULT` (calibración copiada verbatim de `app.py`).
4. $\mathcal L_{C,1}/\mathcal L_{voz,1}$ vía `communication_likelihood_LC`.

Ilustrativo: no realimenta $\mu_1$ (ya calculado con la señal fija de τ=0, Block F).

---

## 25. $V_\tau$ (signal) — indicador de llamada/silencio

**Cómo se calculó**: el mismo $V_1$ sorteado en el paso 2 de §24, mostrado como "Call"/"Silence"
en vez del valor numérico de verosimilitud.

---

## 26. $d$ (det.) — probabilidad de detección de colusión

**Fórmula**: `eq:detection`: $p_{det,\tau}(\theta_K)=\Lambda(\eta_0(\theta_K)+\eta_1\alpha_\tau^*+\eta_2\gamma_\tau^*)$.

**Cómo se calculó**: misma fórmula exacta que τ=0 (Block D), recalculada en
$(\alpha_1^*,\gamma_1^*)$ (§7–8) en vez de los sliders exógenos de Tab 1 — mismos $\eta_0(\theta_K^{true})$,
$\eta_1$, $\eta_2$ de sesión.

---

## 27. $\iota$ — precisión informacional

**Fórmula**: $\iota_\tau=\max_\theta\mu_\tau(\theta)$.

**Cómo se calculó**: `max(_mu1_tab3.values())` — no depende del botón, disponible en cuanto
$\mu_1$ (§19) está lista.

---

## 28. $M_t$ — temperatura de ruido MDG

**Fórmula**: `eq:m-t`: $M_\tau=T_0\max\{H_{ratio}e^{-\eta_{cal}\tau},\underline c\}$.

**Cómo se calculó**: `cvn.m_t(1, p)` — reevaluación genuina en $\tau=1$ (no el $M_t$ fijo del
slider "día" de Tab 1, que corresponde a τ=0). Es el mismo valor usado como temperatura en
todos los sorteos MDG de τ=1 (§2, §4, §6) y en los benchmarks (§9–12, §17, §18).

---

## Resumen de dependencias (orden de cómputo dentro del click)

```
solve_state_problem(mu1, ...)          →  a_S*, alpha_1*, gamma_1* (§5,7,8)
  → tilde a_S (§6)
  → family_utilities(...)              →  a_F* (§1)
    → tilde a_F (§2)
  → solve_captor_true_type_continuation → a_K*(theta_true) (§3)
    → tilde a_K (§4, con mezcla de inercia)
  → sorteo de voz (§24, §25)
  → sorteo de m (§18)
  → kappa_h(theta,1) (§17)
  → benchmarks tau=1 + alpha/gamma^mu (§9-12, §20-23)   [fuera del click principal,
                                                            pero requiere "_run_state_opt_clicked"]
mu_1 (§19), iota (§27)                  →  NO dependen del botón, solo de (m,d) de tau=0
d(det.) (§26), M_t (§28)                →  recalculo directo, sin dependencias nuevas
```

**Filas ilustrativas (no realimentan $\mu_1$)**: $m$ (§18) y $V$/$V_\tau$ (§24, §25) — ambas
son sorteos genuinos para τ=1, pero su función es prospectiva (insumo de un futuro $\mu_2$ si
se implementan ciclos τ=2+), no retroactiva sobre $\mu_1$, que ya quedó fijado con las señales
de τ=0.
