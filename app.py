import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random

from structured_note_sim import run_structured_note, load_prices, run_backtest

# ==========================================================
# CACHED WRAPPERS
# ==========================================================

@st.cache_data
def cached_load_prices():
    return load_prices()

@st.cache_data
def cached_backtest(coupon_rate, floor_level, call_steepness):
    return run_backtest(coupon_rate=coupon_rate, floor_level=floor_level,
                        call_steepness=call_steepness)

# ==========================================================
# SESSION STATE
# ==========================================================

if "results" not in st.session_state:
    st.session_state["results"] = None

if "path_num" not in st.session_state:
    st.session_state["path_num"] = 0

# ==========================================================
# PAGE CONFIG + THEME
# ==========================================================

st.set_page_config(
    page_title="Structured Note Simulator",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    /* Main background and text */
    .stApp { background-color: #ffffff; color: #1a3a1a; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f0f7f0; }
    [data-testid="stSidebar"] * { color: #1a3a1a !important; }

    /* Headers */
    h1, h2, h3, h4 { color: #1a6b1a !important; }

    /* Metric values */
    [data-testid="stMetricValue"] { color: #1a6b1a !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #4a7a4a !important; }

    /* Buttons */
    .stButton > button {
        background-color: #1a6b1a !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #145214 !important;
    }

    /* Sidebar run button */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1a6b1a !important;
        color: white !important;
        width: 100%;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #1a6b1a; }
    .stTabs [data-baseweb="tab"] { color: #4a7a4a !important; }
    .stTabs [aria-selected="true"] {
        color: #1a6b1a !important;
        border-bottom: 2px solid #1a6b1a !important;
        font-weight: 600;
    }

    /* Sliders */
    [data-testid="stSlider"] > div > div > div > div { background-color: #1a6b1a !important; }

    /* Expander */
    [data-testid="stExpander"] { border: 1px solid #c8e6c8 !important; border-radius: 6px; }
    [data-testid="stExpander"] summary { color: #1a6b1a !important; font-weight: 600; }

    /* Success / info / warning boxes */
    [data-testid="stAlert"] { border-radius: 6px; }

    /* Dataframe header */
    [data-testid="stDataFrame"] th { background-color: #e8f5e8 !important; color: #1a3a1a !important; }

    /* Divider */
    hr { border-color: #c8e6c8; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LANGUAGE TOGGLE
# ==========================================================

lang = st.sidebar.radio("🌐 Language / Idioma", ["English", "Español"], horizontal=True)
ES = (lang == "Español")

def t(en, es):
    """Return es if Spanish mode, en otherwise."""
    return es if ES else en

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(t("Note Parameters", "Parámetros de la Nota"))

coupon = st.sidebar.slider(
    t("Coupon (% p.a.)", "Cupón (% anual)"),
    min_value=0, max_value=20, value=10,
    help=t("Annual coupon paid if the issuer calls the note early.",
           "Cupón anual pagado si el emisor ejerce el call anticipadamente.")
)

floor = st.sidebar.slider(
    t("Piso de Capital (%)", "Piso de Capital (%)"),
    min_value=50, max_value=100, value=95,
    help=t("Redención mínima como % del nocional al vencimiento.",
           "Redención mínima como % del nocional al vencimiento.")
)

st.sidebar.header(t("Simulation Parameters", "Parámetros de Simulación"))

n_paths = st.sidebar.slider(
    t("Monte Carlo Paths", "Trayectorias de Monte Carlo"),
    min_value=1000, max_value=50000, value=10000, step=1000,
)

call_steepness = st.sidebar.slider(
    t("Issuer Call Decisiveness", "Decisión de Ejercicio del Emisor"),
    min_value=5, max_value=50, value=20,
    help=t(
        "Controls how aggressively the issuer exercises the call. "
        "Low (~5): significant discretion around the strike. "
        "High (~50): nearly automatic when in-the-money.",
        "Controla qué tan agresivamente el emisor ejerce el call. "
        "Bajo (~5): discreción significativa cerca del strike. "
        "Alto (~50): casi automático cuando está in-the-money."
    )
)

seed = st.sidebar.number_input(t("Random Seed", "Semilla Aleatoria"), value=42)
run_button = st.sidebar.button(t("🚀 Run Simulation", "🚀 Ejecutar Simulación"))

# ==========================================================
# TITLE
# ==========================================================

st.title(t("📈 Multi-Asset Structured Note Simulator",
           "📈 Simulador de Notas Estructuradas Multi-Activo"))
st.markdown(t(
    "Historical Heston calibration + Monte Carlo simulation of a "
    "**worst-of callable note** on SPX / SX5E / SMI.",
    "Calibración histórica de Heston + simulación de Monte Carlo de una "
    "**callable note worst-of** sobre SPX / SX5E / SMI."
))

# ==========================================================
# HOW THIS NOTE WORKS
# ==========================================================

floor_level = floor / 100
fl = floor_level
maturity_label = t("12-month", "12 meses")

with st.expander(t("📖 How This Note Works", "📖 Cómo Funciona Esta Nota"), expanded=False):

    if not ES:
        st.markdown(f"""
### Structure at a Glance

This is a **{maturity_label} Callable Note with {floor}% Capital Protection**
on a worst-of basket of three global equity indices: **S&P 500 (SPX)**, **Euro Stoxx 50 (SX5E)**, and **Swiss Market Index (SMI)**.

---

#### 1 · Worst-of Basket
Instead of tracking a single index, the note tracks the **worst-performing** of the three.
This is the key source of risk: even if two indices perform well, a single underperformer
drags the payout down. In exchange for taking on this correlation risk, the investor receives
a higher coupon than a single-index product would offer.

#### 2 · Capital Floor ({floor}%)
If the note is not called and the worst-of basket finishes below the call strike at maturity,
the investor receives **{floor}% of their initial investment** back. This floor is a contractual
obligation of the issuer — it is not a capital guarantee, and investors remain exposed to the
riesgo crediticio del emisor.

#### 3 · Issuer Call (Quarterly)
At **3M, 6M, and 9M**, the issuer has the **right** (but not the obligation) to redeem the note early.
The issuer will generally call when the worst-of basket is above the {floor}% strike and it is
economically advantageous for them to do so. The further above the strike the basket is, the more
likely they are to call. This is modelled as a probabilistic decision rather than an automatic trigger.

#### 4 · Coupon ({coupon}% p.a.)
If the issuer calls at observation date $t$ (expressed in years), the investor receives their
principal back plus a pro-rata coupon:

$$\\text{{Payout}} = 100\\% + {coupon}\\% \\times t$$

| Observation | Time (t) | Coupon Received |
|-------------|----------|----------------|
| 3M          | 0.25     | {coupon * 0.25:.1f}% |
| 6M          | 0.50     | {coupon * 0.50:.1f}% |
| 9M          | 0.75     | {coupon * 0.75:.1f}% |

#### 5 · Maturity Payoff (if not called)
At maturity, the payout depends on where the worst-of basket finishes relative to its initial level:

**If worst-of ≥ {floor}%:** the investor receives the floor back plus full one-for-one participation
in any performance above it — there is no cap on the upside.

**If worst-of < {floor}%:** the investor receives exactly {floor}%, regardless of how far below
the strike the basket has fallen.

In both cases the payout can be written as:

$$\\text{{Payout}} = {floor}\\% + \\max(0,\\ \\text{{Worst-of Final Performance}} - {floor}\\%)$$

---
> **Modelling note:** The issuer call is modelled as discretionary, not automatic.
> The "Issuer Call Decisiveness" slider controls how sharply the call probability
> rises above the {floor}% strike — at maximum (50) it approaches an automatic call,
> at minimum (5) the issuer exercises significant discretion even well above the strike.
""")
    else:
        st.markdown(f"""
### Estructura General

Esta es una **Callable Note de {maturity_label} con {floor}% de Protección de Capital**
sobre una cesta worst-of de tres índices de renta variable globales: **S&P 500 (SPX)**, **Euro Stoxx 50 (SX5E)** y **Swiss Market Index (SMI)**.

---

#### 1 · Cesta Worst-of
En lugar de seguir un solo índice, la nota sigue el **peor desempeño** de los tres.
Esta es la principal fuente de riesgo: aunque dos índices tengan buen rendimiento, un solo
rezagado arrastra el payout hacia abajo. A cambio de asumir este riesgo de correlación, el
inversor recibe un cupón más alto que el de un producto sobre un solo índice.

#### 2 · Capital Floor ({floor}%)
Si la nota no es llamada y la cesta worst-of cierra por debajo del strike al vencimiento,
el inversor recibe de vuelta **el {floor}% de su inversión inicial**. Este piso es una
obligación contractual del emisor — no es una garantía de capital, y los inversores mantienen
exposición al riesgo de crédito del emisor.

#### 3 · Issuer Call (Trimestral)
En **3M, 6M y 9M**, el emisor tiene el **derecho** (pero no la obligación) de rescatar la nota anticipadamente.
El emisor generalmente ejercerá el call cuando la cesta worst-of esté por encima del strike de {floor}%
y sea económicamente conveniente hacerlo. Cuanto más por encima del strike esté la cesta, más probable
es que llame. Esto se modela como una decisión probabilística, no como un trigger automático.

#### 4 · Cupón ({coupon}% anual)
Si el emisor ejerce el call en la fecha de observación $t$ (expresada en años), el inversor
recibe su principal más un cupón pro-rata:

$$\\text{{Payout}} = 100\\% + {coupon}\\% \\times t$$

| Observación | Tiempo (t) | Cupón Recibido |
|-------------|------------|---------------|
| 3M          | 0.25       | {coupon * 0.25:.1f}% |
| 6M          | 0.50       | {coupon * 0.50:.1f}% |
| 9M          | 0.75       | {coupon * 0.75:.1f}% |

#### 5 · Payout al Vencimiento (si no fue llamada)
Al vencimiento, el payout depende de dónde cierre la cesta worst-of respecto a su nivel inicial:

**Si worst-of ≥ {floor}%:** el inversor recibe el floor más participación uno a uno en cualquier
rendimiento por encima — no hay límite al alza.

**Si worst-of < {floor}%:** el inversor recibe exactamente {floor}%, sin importar cuánto por debajo
del strike haya caído la cesta.

En ambos casos el payout se puede escribir como:

$$\\text{{Payout}} = {floor}\\% + \\max(0,\\ \\text{{Rendimiento Final Worst-of}} - {floor}\\%)$$

---
> **Nota de modelización:** El issuer call se modela como discrecional, no automático.
> El slider "Decisión de Ejercicio del Emisor" controla qué tan pronunciada es la probabilidad de call
> por encima del strike de {floor}% — en máximo (50) se aproxima a un call automático,
> en mínimo (5) el emisor ejerce discreción significativa incluso bien por encima del strike.
""")

    st.markdown(t("#### Issuer Call Probability Curve", "#### Curva de Probabilidad de Ejercicio del Emisor"))
    st.caption(t(
        "Probability the issuer calls at a given worst-of performance level, for the current decisiveness setting.",
        "Probabilidad de que el emisor ejerza el call dado un nivel de rendimiento worst-of, con la configuración actual."
    ))

    perf_range = np.linspace(0.80, 1.20, 300)
    call_probs = 1.0 / (1.0 + np.exp(-call_steepness * (perf_range - fl)))
    call_probs[perf_range < fl] = 0.0

    fig_sig = px.line(
        pd.DataFrame({
            t("Worst-of Performance", "Rendimiento Worst-of"): perf_range,
            t("P(issuer calls)", "P(emisor ejerce)"): call_probs
        }),
        x=t("Worst-of Performance", "Rendimiento Worst-of"),
        y=t("P(issuer calls)", "P(emisor ejerce)"),
    )
    fig_sig.add_vline(
        x=fl, line_dash="dash", line_color="#1a6b1a",
        annotation_text=f"Call Strike ({floor}%)", annotation_position="top right",
    )
    fig_sig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        xaxis=dict(tickformat=".0%"),
        height=280, margin=dict(t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_sig, use_container_width=True)

# ==========================================================
# RUN SIMULATION
# ==========================================================

if run_button:
    with st.spinner(t("Running Heston calibration and Monte Carlo simulation...",
                      "Ejecutando calibración Heston y simulación de Monte Carlo...")):
        st.session_state["results"] = run_structured_note(
            coupon_rate=coupon / 100,
            floor_level=fl,
            n_paths=n_paths,
            seed=seed,
            call_steepness=float(call_steepness),
        )
        st.session_state["path_num"] = 0

# ==========================================================
# MONTE CARLO RESULTS
# ==========================================================

if st.session_state["results"] is not None:

    results = st.session_state["results"]
    asset_names = results["asset_names"]

    st.success(t("Simulation complete.", "Simulación completada."))

    st.header(t("Summary Statistics", "Estadísticas Resumidas"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t("Expected IRR", "TIR Esperada"),            f"{results['expected_irr']:.2%}")
    with c2:
        st.metric(t("Expected Return", "Rendimiento Esperado"),     f"{results['expected_total_return']:.2%}")
    with c3:
        st.metric(t("Maturity Probability", "Prob. Vencimiento"), f"{results['prob_maturity']:.2%}")
    with c4:
        st.metric(t("Capital Floor Probability", "Prob. Piso de Capital"), f"{results['prob_floor']:.2%}")

    st.header(t("Issuer Call Probabilities", "Probabilidades de Ejercicio del Emisor"))
    st.caption(t(
        "Probability the issuer exercises the call at each quarterly observation date, "
        "given the simulated paths and the current decisiveness setting.",
        "Probabilidad de que el emisor ejerza el call en cada fecha de observación trimestral, "
        "dados los paths simulados y la configuración actual de decisión."
    ))
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric(t("3M Call", "Ejercicio 3M"),          f"{results['prob_q1']:.2%}")
    with q2:
        st.metric(t("6M Call", "Ejercicio 6M"),          f"{results['prob_q2']:.2%}")
    with q3:
        st.metric(t("9M Call", "Ejercicio 9M"),          f"{results['prob_q3']:.2%}")
    with q4:
        st.metric(t("Reaches Maturity", "Llega al Vencimiento"), f"{results['prob_maturity']:.2%}")

    tab1, tab2, tab3, tab4 = st.tabs([
        t("📊 Payoff & Distribution", "📊 Rendimiento y Distribución"),
        t("📈 Price Path Fan Chart",   "📈 Abanico de Trayectorias de Precio"),
        t("🔍 Path Explorer",          "🔍 Explorador de Trayectorias"),
        t("🔗 Correlation Diagnostics","🔗 Diagnóstico de Correlaciones"),
    ])

    # ----------------------------------------------------------
    # TAB 1
    # ----------------------------------------------------------

    with tab1:

        st.subheader(t("Maturity Payoff Profile vs Simulated Outcomes",
                       "Perfil de Rendimiento al Vencimiento vs Resultados Simulados"))
        st.markdown(t(
            "The line shows the **contractual payoff** at maturity. "
            "The histogram shows the **distribution of simulated terminal worst-of values** "
            "for paths that reached maturity (not called early).",
            "La línea muestra el **payoff contractual** al vencimiento. "
            "El histograma muestra la **distribución de los valores terminales worst-of simulados** "
            "para las trayectorias que llegaron al vencimiento (no llamados anticipadamente)."
        ))

        worst_perf_grid = np.linspace(0.50, 1.50, 500)
        payoff_grid = np.where(worst_perf_grid < fl, fl, worst_perf_grid)

        autocall_events = results["autocall_events"]
        worst_of_paths  = results["worst_of_paths"]
        N = worst_of_paths.shape[1] - 1
        terminal_worst = worst_of_paths[autocall_events == 0, N]

        fig = go.Figure()
        if len(terminal_worst) > 0:
            fig.add_trace(go.Histogram(
                x=terminal_worst, nbinsx=60,
                name=t("Simulated terminal worst-of", "Worst-of terminal simulado"),
                yaxis="y2", opacity=0.4,
                marker_color="#1a6b1a", histnorm="probability",
            ))
        fig.add_trace(go.Scatter(
            x=worst_perf_grid, y=payoff_grid,
            mode="lines", name=t("Contractual payoff", "Rendimiento contractual"),
            line=dict(color="#145214", width=2.5),
        ))
        fig.add_vline(
            x=fl, line_dash="dash", line_color="#4a7a4a",
            annotation_text=t(f"Floor / Call Strike ({floor}%)", f"Floor / Call Strike ({floor}%)"),
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title=t("Worst-of Final Performance", "Rendimiento Final Worst-of"), tickformat=".0%"),
            yaxis=dict(title=t("Note Payoff", "Rendimiento de la Nota"), tickformat=".0%"),
            yaxis2=dict(title=t("Probability (maturity paths)", "Probabilidad (trayectorias al vencimiento)"),
                        overlaying="y", side="right", showgrid=False, tickformat=".1%"),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        prob_floor = results["prob_floor"]
        if prob_floor > 0:
            st.info(t(
                f"**{prob_floor:.1%}** of simulated paths reach maturity with the "
                f"worst-of below {floor}%, triggering the capital floor.",
                f"**{prob_floor:.1%}** de los paths simulados llegan al vencimiento con el "
                f"worst-of por debajo de {floor}%, activando el piso de capital."
            ))

    # ----------------------------------------------------------
    # TAB 2 — Fan Chart
    # ----------------------------------------------------------

    with tab2:

        st.subheader(t("Simulated Price Path Fan Chart",
                       "Abanico de Trayectorias de Precio Simulados"))
        st.markdown(t(
            "Percentile bands across all simulated paths. The **median** (50th) shows the "
            "central tendency, the shaded bands show the spread. The upward drift from the "
            "calibrated μ is visible in the median rising over time.",
            "Bandas de percentiles sobre todos los paths simulados. La **mediana** (percentil 50) muestra la "
            "tendencia central; las bandas sombreadas muestran la dispersión. El drift alcista del "
            "μ calibrado se aprecia en la mediana creciente a lo largo del tiempo."
        ))

        sim_prices = results["sim_prices"]
        N_steps = sim_prices.shape[1] - 1
        t_grid  = np.linspace(0, 1.0, N_steps + 1)
        pcts    = [5, 25, 50, 75, 95]

        for asset_idx, asset_name in enumerate(asset_names):

            paths = sim_prices[:, :, asset_idx]
            S0_i  = paths[:, 0].mean()
            bands = np.percentile(paths, pcts, axis=0)

            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([t_grid, t_grid[::-1]]),
                y=np.concatenate([bands[4], bands[0][::-1]]),
                fill="toself", fillcolor="rgba(26,107,26,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name=t("5th–95th pct", "Pct 5–95"),
            ))
            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([t_grid, t_grid[::-1]]),
                y=np.concatenate([bands[3], bands[1][::-1]]),
                fill="toself", fillcolor="rgba(26,107,26,0.20)",
                line=dict(color="rgba(0,0,0,0)"),
                name=t("25th–75th pct", "Pct 25–75"),
            ))
            fig_fan.add_trace(go.Scatter(
                x=t_grid, y=bands[2],
                mode="lines", name=t("Median", "Mediana"),
                line=dict(color="#1a6b1a", width=2),
            ))
            fig_fan.add_hline(
                y=S0_i, line_dash="dash", line_color="#888",
                annotation_text="S₀", annotation_position="right",
            )
            for q, label in [(0.25, "3M"), (0.50, "6M"), (0.75, "9M")]:
                fig_fan.add_vline(x=q, line_dash="dot", line_color="#aaa",
                                  annotation_text=label, annotation_position="top")
            fig_fan.update_layout(
                title=f"{asset_name} — " + t("Simulated Price Distribution",
                                              "Distribución de Precios Simulados"),
                xaxis=dict(title=t("Time (years)", "Tiempo (años)"), tickformat=".2f"),
                yaxis=dict(title=t("Price", "Precio")),
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        st.markdown(t("### Worst-of Performance Fan Chart",
                      "### Abanico de Rendimiento Worst-of"))
        st.caption(t(
            "Percentile bands for the worst-of basket performance. "
            "The dashed line marks the 95% floor / call strike.",
            "Bandas de percentiles para el rendimiento de la cesta worst-of. "
            "La línea punteada marca el floor / call strike del 95%."
        ))

        worst_of_paths = results["worst_of_paths"]
        bands_w = np.percentile(worst_of_paths, pcts, axis=0)

        fig_wof = go.Figure()
        fig_wof.add_trace(go.Scatter(
            x=np.concatenate([t_grid, t_grid[::-1]]),
            y=np.concatenate([bands_w[4], bands_w[0][::-1]]),
            fill="toself", fillcolor="rgba(26,107,26,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name=t("5th–95th pct", "Pct 5–95"),
        ))
        fig_wof.add_trace(go.Scatter(
            x=np.concatenate([t_grid, t_grid[::-1]]),
            y=np.concatenate([bands_w[3], bands_w[1][::-1]]),
            fill="toself", fillcolor="rgba(26,107,26,0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            name=t("25th–75th pct", "Pct 25–75"),
        ))
        fig_wof.add_trace(go.Scatter(
            x=t_grid, y=bands_w[2],
            mode="lines", name=t("Median", "Mediana"),
            line=dict(color="#1a6b1a", width=2),
        ))
        fig_wof.add_hline(
            y=fl, line_dash="dash", line_color="#c0392b",
            annotation_text=f"Floor / Call Strike ({floor}%)",
            annotation_position="bottom right",
        )
        for q, label in [(0.25, "3M"), (0.50, "6M"), (0.75, "9M")]:
            fig_wof.add_vline(x=q, line_dash="dot", line_color="#aaa",
                              annotation_text=label, annotation_position="top")
        fig_wof.update_layout(
            xaxis=dict(title=t("Time (years)", "Tiempo (años)"), tickformat=".2f"),
            yaxis=dict(title=t("Performance vs Initial", "Rendimiento vs Inicial"), tickformat=".0%"),
            hovermode="x unified",
            legend=dict(x=0.01, y=0.01),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_wof, use_container_width=True)

    # ----------------------------------------------------------
    # TAB 3 — Path Explorer
    # ----------------------------------------------------------

    with tab3:

        st.subheader(t("Single Path Explorer", "Explorador de Trayectoria Individual"))
        st.markdown(t(
            "Step through individual Monte Carlo paths. Vertical dotted lines mark "
            "each quarterly observation date. Green star = issuer called here.",
            "Navega por paths individuales de Monte Carlo. Las líneas punteadas marcan "
            "cada fecha de observación trimestral. Estrella verde = emisor ejerció el call aquí."
        ))

        sim_prices = results["sim_prices"]
        N = sim_prices.shape[1] - 1
        steps_per_quarter = N // 4
        obs_steps  = [steps_per_quarter, steps_per_quarter * 2, steps_per_quarter * 3]
        obs_labels = ["3M", "6M", "9M"]
        max_path   = sim_prices.shape[0] - 1

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(t("🎲 Random Path", "🎲 Trayectoria Aleatoria")):
                st.session_state["path_num"] = random.randint(0, max_path)
        with c2:
            if st.button(t("⬅ Previous", "⬅ Anterior")):
                st.session_state["path_num"] = max(0, st.session_state["path_num"] - 1)
        with c3:
            if st.button(t("Next ➡", "Siguiente ➡")):
                st.session_state["path_num"] = min(max_path, st.session_state["path_num"] + 1)

        path_num = st.session_state["path_num"]
        st.caption(t(f"Path #{path_num} of {max_path}", f"Path #{path_num} de {max_path}"))

        path_df = pd.DataFrame({
            name: sim_prices[path_num, :, i]
            for i, name in enumerate(asset_names)
        })
        fig_prices = px.line(
            path_df,
            title=t(f"Asset Price Paths — Path #{path_num}",
                    f"Paths de Precios — Path #{path_num}"),
            labels={"value": t("Price", "Precio"), "index": t("Time Step", "Paso de Tiempo")},
            color_discrete_sequence=["#1a6b1a", "#2ecc71", "#145214"],
        )
        for step, label in zip(obs_steps, obs_labels):
            fig_prices.add_vline(x=step, line_dash="dot", line_color="#aaa",
                                 annotation_text=label, annotation_position="top")
        fig_prices.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_prices, use_container_width=True)

        worst_path  = results["worst_of_paths"][path_num]
        autocall_q  = int(results["autocall_events"][path_num])

        fig_worst = go.Figure()
        fig_worst.add_trace(go.Scatter(
            y=worst_path, mode="lines",
            name=t("Worst-of Performance", "Rendimiento Worst-of"),
            line=dict(color="#1a6b1a", width=2),
        ))
        fig_worst.add_hline(
            y=fl, line_dash="dash", line_color="#c0392b",
            annotation_text=f"Call Strike / Floor ({floor}%)",
            annotation_position="bottom right",
        )
        for i, (step, label) in enumerate(zip(obs_steps, obs_labels)):
            called_here   = (autocall_q == i + 1)
            marker_color  = "#1a6b1a" if called_here else "#888"
            marker_symbol = "star"    if called_here else "circle"
            fig_worst.add_trace(go.Scatter(
                x=[step], y=[worst_path[step]],
                mode="markers",
                marker=dict(size=12, color=marker_color, symbol=marker_symbol),
                name=f"{label} {t('← Called', '← Llamado') if called_here else t('(continued)', '(continúa)')}",
            ))
            fig_worst.add_vline(x=step, line_dash="dot", line_color="#aaa",
                                annotation_text=label, annotation_position="top")
        fig_worst.update_layout(
            title=t(f"Worst-of Performance Path #{path_num}",
                    f"Rendimiento Worst-of Path #{path_num}"),
            yaxis=dict(title=t("Performance vs Initial", "Rendimiento vs Inicial"), tickformat=".0%"),
            xaxis=dict(title=t("Time Step", "Paso de Tiempo")),
            hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_worst, use_container_width=True)

        nominal_payout = results["nominal_payoffs"][path_num]
        irr            = results["annualized_returns"][path_num]
        final_worst    = worst_path[-1]

        if autocall_q > 0:
            t_called = autocall_q * 0.25
            outcome_label  = t(f"✅ Issuer called at {obs_labels[autocall_q - 1]}",
                               f"✅ Emisor ejerció call en {obs_labels[autocall_q - 1]}")
            outcome_detail = t(
                f"Worst-of was **{worst_path[obs_steps[autocall_q-1]]:.1%}** "
                f"at observation. Note redeemed after **{t_called:.2g} years**.",
                f"El worst-of era **{worst_path[obs_steps[autocall_q-1]]:.1%}** "
                f"en la observación. Nota rescatada tras **{t_called:.2g} años**."
            )
        elif final_worst >= fl:
            outcome_label  = t("📈 Reached maturity — upside participation",
                               "📈 Llegó al vencimiento — participación al alza")
            outcome_detail = t(
                f"Worst-of finished at **{final_worst:.1%}**, above the floor.",
                f"El worst-of cerró en **{final_worst:.1%}**, por encima del floor."
            )
        else:
            outcome_label  = t("🛡️ Reached maturity — capital floor applied",
                               "🛡️ Llegó al vencimiento — piso de capital aplicado")
            outcome_detail = t(
                f"Worst-of finished at **{final_worst:.1%}**, below the {floor}% floor.",
                f"El worst-of cerró en **{final_worst:.1%}**, por debajo del piso de {floor}%."
            )

        st.markdown(f"### {outcome_label}")
        st.markdown(outcome_detail)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("Nominal Payout", "Rendimiento Nominal"),        f"{nominal_payout:.2%}")
        with col2:
            st.metric(t("Annualised Return", "Rendimiento Anualizado"),  f"{irr:.2%}")

    # ----------------------------------------------------------
    # TAB 4 — Correlation Diagnostics
    # ----------------------------------------------------------

    with tab4:

        st.subheader(t("Correlation Diagnostics", "Diagnóstico de Correlaciones"))
        st.markdown(t(
            "**Input** correlations are estimated from historical data by the Heston calibrator. "
            "**Realized** correlations are computed from the simulated paths — they should be "
            "close to the inputs, validating the Cholesky correlation structure.",
            "Las correlaciones de **input** se estiman de datos históricos mediante el calibrador Heston. "
            "Las correlaciones **realizadas** se calculan de los paths simulados — deben estar "
            "cerca de los inputs, validando la estructura de correlación de Cholesky."
        ))

        corr_SS       = results["corr_SS"]
        realized_corr = results["sim_results"]["realized_corr"]
        diff          = realized_corr - corr_SS

        def corr_heatmap(matrix, title, zmin=-1, zmax=1):
            df = pd.DataFrame(matrix, index=asset_names, columns=asset_names)
            fig = px.imshow(df, text_auto=".3f",
                            color_continuous_scale=[[0,"#c0392b"],[0.5,"white"],[1,"#1a6b1a"]],
                            zmin=zmin, zmax=zmax, title=title, aspect="auto")
            fig.update_layout(coloraxis_showscale=False, paper_bgcolor="white")
            return fig

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(corr_heatmap(corr_SS,
                t("Input (Calibrated)", "Input (Calibrado)")),
                use_container_width=True)
        with col2:
            st.plotly_chart(corr_heatmap(realized_corr,
                t("Realized (Simulated)", "Realizada (Simulada)")),
                use_container_width=True)
        with col3:
            st.plotly_chart(corr_heatmap(diff,
                t("Difference (Realized − Input)", "Diferencia (Realizada − Input)"),
                zmin=-0.1, zmax=0.1),
                use_container_width=True)

        max_err = np.max(np.abs(diff - np.diag(np.diag(diff))))
        if max_err < 0.05:
            st.success(t(
                f"Max off-diagonal error: **{max_err:.4f}** — correlation structure well reproduced.",
                f"Error máximo fuera de la diagonal: **{max_err:.4f}** — estructura de correlación bien reproducida."
            ))
        else:
            st.warning(t(
                f"Max off-diagonal error: **{max_err:.4f}** — consider increasing n_paths.",
                f"Error máximo fuera de la diagonal: **{max_err:.4f}** — considere aumentar los paths de Monte Carlo."
            ))

        st.markdown("---")
        st.subheader(t("Calibrated Heston Parameters", "Parámetros Heston Calibrados"))
        st.caption(t("Estimated from historical price data via Method of Moments.",
                     "Estimados de datos históricos de precios mediante el Método de Momentos."))

        param_rows = []
        for p in results["params"]:
            ok, margin = p.feller_condition()
            param_rows.append({
                t("Asset", "Activo"):          p.name,
                "S₀":                          f"{p.S0:.1f}",
                t("μ (drift p.a.)", "μ (tendencia anual)"): f"{p.mu*100:.1f}%",
                t("V₀ (σ)", "V₀ (σ)"):         f"{np.sqrt(p.V0)*100:.1f}%",
                t("θ (long-run σ)", "θ (σ largo plazo)"): f"{np.sqrt(p.theta)*100:.1f}%",
                t("κ (mean rev.)", "κ (rev. media)"): f"{p.kappa:.3f}",
                "ξ (vol-of-vol)":               f"{p.xi:.3f}",
                "ρ (leverage)":                 f"{p.rho:.3f}",
                "Feller ✓":                    "✅" if ok else "⚠️",
            })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

        t_dof_val = results.get("t_dof", "N/A")
        st.info(t(
            f"**Student-t Copula:** ν = {t_dof_val} degrees of freedom, "
            f"fitted from the historical return distribution of each asset. "
            f"This captures joint tail dependence — the tendency for all three indices "
            f"to fall simultaneously in stress scenarios, which the Gaussian copula underestimates.",
            f"**Cópula Student-t:** ν = {t_dof_val} grados de libertad, "
            f"ajustados a partir de la distribución histórica de retornos de cada activo. "
            f"Esto captura la dependencia de colas conjunta — la tendencia de los tres índices "
            f"a caer simultáneamente en escenarios de estrés, que la cópula Gaussiana subestima."
        ))

        st.caption(t(
            "**Note on ρ (leverage effect):** Textbook equity models assume ρ ≈ −0.65 for SPX "
            "(volatility spikes when markets fall). The calibration here shows ρ near zero or slightly "
            "positive because the 2021–2026 calibration window was dominated by a bull market regime "
            "where returns and volatility were largely uncorrelated. This is what the data shows — "
            "not a bug. A risk-neutral calibration from the options surface would typically recover "
            "the expected negative ρ.",
            "**Nota sobre ρ (efecto leverage):** Los modelos canónicos de renta variable asumen ρ ≈ −0.65 para SPX "
            "(la volatilidad sube cuando los mercados caen). La calibración aquí muestra ρ cercano a cero o ligeramente "
            "positivo porque la ventana de calibración 2021–2026 estuvo dominada por un régimen alcista "
            "donde los retornos y la volatilidad estuvieron en gran medida incorrelacionados. Esto es lo que muestran "
            "los datos — no es un error. Una calibración risk-neutral desde la superficie de opciones "
            "típicamente recuperaría el ρ negativo esperado."
        ))

else:
    st.info(t(
        "Configure parameters in the sidebar and click **🚀 Run Simulation** to begin.",
        "Configure los parámetros en la barra lateral y haga clic en **🚀 Ejecutar Simulación** para comenzar."
    ))

# ==========================================================
# HISTORICAL BACKTEST
# ==========================================================

st.markdown("---")
st.header(t("📅 Historical Backtest", "📅 Backtest Histórico"))
st.markdown(t(
    "Evaluates how this note would have performed if issued on every available "
    "date between **June 2022** and **June 2025**, using actual realized index prices. "
    "No simulation — just the real historical path. "
    "The issuer call uses the same probabilistic model as the forward simulation: "
    "the issuer calls with increasing probability as the worst-of basket rises above "
    "the call strike, controlled by the Issuer Call Decisiveness slider. "
    "Results update automatically when you change the coupon, floor, or decisiveness sliders.",
    "Evalúa cómo habría funcionado esta nota si se hubiera emitido en cada fecha disponible "
    "entre **junio 2022** y **junio 2025**, usando los precios reales de los índices. "
    "Sin simulación — solo el path histórico real. "
    "El issuer call usa el mismo modelo probabilístico que la simulación forward: "
    "el emisor ejerce el call con probabilidad creciente a medida que la cesta worst-of sube "
    "por encima del call strike, controlado por el slider de Decisión de Call. "
    "Los resultados se actualizan automáticamente al cambiar los sliders de cupón, floor o decisión."
))

with st.spinner(t("Running historical backtest...", "Ejecutando backtest histórico...")):
    try:
        bt, bt_summary = cached_backtest(
            coupon_rate=coupon / 100,
            floor_level=fl,
            call_steepness=float(call_steepness),
        )
    except Exception as e:
        st.error(t(f"Backtest failed: {e}", f"Backtest falló: {e}"))
        bt, bt_summary = pd.DataFrame(), {}

if bt.empty:
    st.warning(t(
        "No backtest results — check that SPX.csv, SX5E.csv, SMI.csv are present.",
        "Sin resultados de backtest — verifique que SPX.csv, SX5E.csv, SMI.csv estén presentes."
    ))
else:
    outcome_map = {
        0: t("Maturity", "Vencimiento"),
        1: t("Called at 3M", "Ejercicio en 3M"),
        2: t("Called at 6M", "Ejercicio en 6M"),
        3: t("Called at 9M", "Ejercicio en 9M"),
    }
    bt["Outcome"] = bt["Call Quarter"].map(outcome_map)
    floor_label = t("Floor Applied", "Piso Aplicado")
    bt.loc[(bt["Call Quarter"] == 0) & (bt["Worst Final Perf"] < fl), "Outcome"] = floor_label

    color_map = {
        t("Called at 3M", "Ejercicio en 3M"): "#2ecc71",
        t("Called at 6M", "Ejercicio en 6M"): "#27ae60",
        t("Called at 9M", "Ejercicio en 9M"): "#1a8a4a",
        t("Maturity", "Vencimiento"):    "#3498db",
        t("Floor Applied", "Piso Aplicado"): "#e74c3c",
    }

    st.markdown(t("### Across All Historical Issue Dates",
                  "### Sobre Todas las Fechas de Emisión Históricas"))
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        st.metric(t("Issue Dates Tested", "Fechas de Emisión"),   f"{bt_summary['n_issues']}")
    with b2:
        st.metric(t("Mean IRR", "TIR Promedio"),                   f"{bt_summary['mean_irr']:.2%}")
    with b3:
        st.metric(t("Median IRR", "TIR Mediana"),                  f"{bt_summary['median_irr']:.2%}")
    with b4:
        st.metric(t("Floor Triggered", "Piso Activado"),          f"{bt_summary['prob_floor']:.1%}")
    with b5:
        st.metric(t("Called Early", "Ejercicio Anticipado"),            f"{bt_summary['prob_called']:.1%}")

    bt_col1, bt_col2 = st.columns(2)

    with bt_col1:
        outcome_counts = bt["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", t("Count", "Cantidad")]
        fig_outcomes = px.bar(
            outcome_counts,
            x="Outcome", y=t("Count", "Cantidad"),
            color="Outcome", color_discrete_map=color_map,
            title=t("Outcome Distribution", "Distribución de Resultados"),
            text=t("Count", "Cantidad"),
        )
        fig_outcomes.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_outcomes, use_container_width=True)

    with bt_col2:
        maturity_bt = bt[bt["Call Quarter"] == 0]
        if not maturity_bt.empty:
            worst_counts = maturity_bt["Worst Asset"].value_counts().reset_index()
            worst_counts.columns = [t("Asset", "Activo"), t("Count", "Cantidad")]
            fig_drag = px.pie(
                worst_counts,
                names=t("Asset", "Activo"), values=t("Count", "Cantidad"),
                title=t("Worst-of Asset at Maturity (uncalled notes)",
                        "Activo Worst-of al Vencimiento (notas no ejercidas)"),
                hole=0.4,
                color_discrete_sequence=["#1a6b1a", "#2ecc71", "#145214"],
            )
            fig_drag.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig_drag, use_container_width=True)

    st.markdown(t("### Annualised Return by Issue Date",
                  "### Rendimiento Anualizado por Fecha de Emisión"))
    st.caption(t(
        "Each point is one historical issue date. Color shows the note outcome.",
        "Cada punto es una fecha de emisión histórica. El color muestra el resultado de la nota."
    ))
    fig_irr = px.scatter(
        bt, x="Issue Date", y="IRR", color="Outcome",
        color_discrete_map=color_map,
        hover_data=["Payout", "Worst Asset", "Worst Final Perf"],
        title=t("Realised IRR by Issue Date", "TIR Realizada por Fecha de Emisión"),
    )
    fig_irr.add_hline(
        y=0, line_dash="dash", line_color="#888",
        annotation_text=t("Break-even", "Break-even"), annotation_position="right"
    )
    fig_irr.update_layout(
        yaxis=dict(tickformat=".1%"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_irr, use_container_width=True)

    # ---- Historical price paths ----
    st.markdown(t("### Historical Price Paths", "### Trayectorias Históricas de Precio"))
    st.caption(t(
        "Actual index levels over the full data history. "
        "Vertical lines mark the start and end of the valid backtest window.",
        "Niveles reales de los índices durante todo el historial de datos. "
        "Las líneas verticales marcan el inicio y fin de la ventana de backtest válida."
    ))

    try:
        hist_prices = cached_load_prices()
        fig_hist = go.Figure()
        colors_hist = {"SPX": "#1a6b1a", "SX5E": "#2ecc71", "SMI": "#145214"}
        for col in hist_prices.columns:
            # Normalise to 100 at start for comparability
            normed = hist_prices[col] / hist_prices[col].iloc[0] * 100
            fig_hist.add_trace(go.Scatter(
                x=hist_prices.index, y=normed,
                mode="lines", name=col,
                line=dict(color=colors_hist.get(col, "#888"), width=1.5),
            ))
        # Backtest window markers
        bt_start = pd.Timestamp(str(bt["Issue Date"].min()))
        bt_end   = pd.Timestamp(str(bt["Issue Date"].max()))
        fig_hist.add_vline(x=bt_start, line_dash="dot", line_color="#888",
                           annotation_text=t("Backtest start", "Inicio backtest"),
                           annotation_position="top right")
        fig_hist.add_vline(x=bt_end, line_dash="dot", line_color="#888",
                           annotation_text=t("Backtest end", "Fin backtest"),
                           annotation_position="top left")
        fig_hist.update_layout(
            xaxis=dict(title=t("Date", "Fecha")),
            yaxis=dict(title=t("Normalised Level (base=100)", "Nivel Normalizado (base=100)")),
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ---- Historical worst-of path ----
        st.markdown(t("### Historical Worst-of Performance",
                      "### Rendimiento Histórico Worst-of"))
        st.caption(t(
            "Worst-performing index at each date relative to a rolling 1-year initial level. "
            "Shows how often and how deeply the basket dipped below the floor historically.",
            "Índice de peor desempeño en cada fecha relativo a un nivel inicial móvil de 1 año. "
            "Muestra con qué frecuencia y profundidad la cesta cayó por debajo del piso históricamente."
        ))

        # Rolling worst-of: for each date, compare to price 252 days ago
        prices_arr = hist_prices.values
        dates      = hist_prices.index
        rolling_worst = []
        rolling_dates = []
        for i in range(252, len(prices_arr)):
            perf = prices_arr[i] / prices_arr[i - 252]
            rolling_worst.append(perf.min())
            rolling_dates.append(dates[i])

        fig_wof_hist = go.Figure()
        fig_wof_hist.add_trace(go.Scatter(
            x=rolling_dates, y=rolling_worst,
            mode="lines", name=t("Worst-of (1Y rolling)", "Worst-of (móvil 1A)"),
            line=dict(color="#1a6b1a", width=1.5),
            fill="tozeroy", fillcolor="rgba(26,107,26,0.08)",
        ))
        fig_wof_hist.add_hline(
            y=fl, line_dash="dash", line_color="#c0392b",
            annotation_text=f"Floor / Call Strike ({floor}%)",
            annotation_position="bottom right",
        )
        fig_wof_hist.add_hline(
            y=1.0, line_dash="dot", line_color="#888",
            annotation_text=t("No change", "Sin cambio"),
            annotation_position="right",
        )
        fig_wof_hist.update_layout(
            xaxis=dict(title=t("Date", "Fecha")),
            yaxis=dict(title=t("1Y Worst-of Performance", "Rendimiento Worst-of 1A"),
                       tickformat=".0%"),
            hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_wof_hist, use_container_width=True)

    except Exception as e:
        st.warning(t(f"Could not load historical prices: {e}",
                     f"No se pudieron cargar los precios históricos: {e}"))

    with st.expander(t("View full backtest results table",
                       "Ver tabla completa de resultados del backtest")):
        display_bt = bt.copy()
        for col in ["IRR", "Payout", "Worst Final Perf", "SPX Perf", "SX5E Perf", "SMI Perf"]:
            display_bt[col] = display_bt[col].map(lambda x: f"{x:.2%}")
        st.dataframe(display_bt, use_container_width=True, hide_index=True)