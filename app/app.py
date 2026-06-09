"""
app/app.py  —  Streamlit entry point.
Run with:  streamlit run app/app.py
"""

import random
import sys
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from core            import NoteTerms, price_note
from core.calibrator import HestonCalibrator
from core.simulator  import HestonMultiSimulator
from core.backtest   import run_backtest
from data.loader     import load_prices

from translations import Translator
from charts import (
    build_irr_distribution, build_fan_chart, build_wof_fan,
    build_path_price_chart, build_path_wof_chart, build_corr_heatmap,
    build_backtest_outcome_bar, build_worst_asset_pie,
    build_backtest_irr_scatter, build_historical_prices,
    build_historical_wof_path,
)

# ==========================================================================
# Page config
# ==========================================================================
st.set_page_config(
    page_title = "Structured Note Simulator",
    page_icon  = "📈",
    layout     = "wide",
)
st.markdown("""<style>
[data-testid="stMetricValue"]{font-weight:700}
[data-testid="stSidebar"] .stButton>button{width:100%}
.stTabs [data-baseweb="tab-list"]{border-bottom:2px solid #1a6b1a}
.stTabs [aria-selected="true"]{border-bottom:2px solid #1a6b1a!important;font-weight:600}
[data-testid="stExpander"]{border:1px solid #c8e6c8!important;border-radius:6px}
hr{border-color:#c8e6c8}
</style>""", unsafe_allow_html=True)

# ==========================================================================
# Available underlyings
# ==========================================================================
UNDERLYING_OPTIONS = {
    # ── Equity indices ───────────────────────────────────────────────────
    "SPX — S&P 500":            "^GSPC",
    "NDX — Nasdaq 100":         "^NDX",
    "RUT — Russell 2000":       "^RUT",
    "SX5E — Euro Stoxx 50":     "^STOXX50E",
    "DAX — DAX 40":             "^GDAXI",
    "FTSE — FTSE 100":          "^FTSE",
    "CAC — CAC 40":             "^FCHI",
    "SMI — Swiss Market":       "^SSMI",
    "NKY — Nikkei 225":         "^N225",
    "HSI — Hang Seng":          "^HSI",
    "KOSPI — Korea":            "^KS11",
    "ASX — Australia":          "^AXJO",
    "IBEX — Spain":             "^IBEX",
    "MIB — Italy":              "FTSEMIB.MI",
    # ── US Banks & Financials ────────────────────────────────────────────
    "GS — Goldman Sachs":       "GS",
    "JPM — J.P. Morgan":        "JPM",
    "MS — Morgan Stanley":      "MS",
    "BAC — Bank of America":    "BAC",
    "C — Citigroup":            "C",
    "WFC — Wells Fargo":        "WFC",
    "BLK — BlackRock":          "BLK",
    # ── US Tech ─────────────────────────────────────────────────────────
    "AAPL — Apple":             "AAPL",
    "MSFT — Microsoft":         "MSFT",
    "NVDA — NVIDIA":            "NVDA",
    "AMZN — Amazon":            "AMZN",
    "META — Meta":              "META",
    "GOOGL — Alphabet":         "GOOGL",
    "TSLA — Tesla":             "TSLA",
    "PLTR — Palantir":          "PLTR",
    "AMD — AMD":                "AMD",
    "INTC — Intel":             "INTC",
    "CRM — Salesforce":         "CRM",
    "NFLX — Netflix":           "NFLX",
    # ── European stocks ──────────────────────────────────────────────────
    "ASML — ASML":              "ASML",
    "SAP — SAP":                "SAP",
    "NESN — Nestlé":            "NESN.SW",
    "NOVN — Novartis":          "NOVN.SW",
    "ROG — Roche":              "ROG.SW",
    "MC — LVMH":                "MC.PA",
    "OR — L'Oréal":             "OR.PA",
    "SAN — Santander":          "SAN.MC",
    # ── Commodities & ETFs ───────────────────────────────────────────────
    "GLD — Gold ETF":           "GLD",
    "SLV — Silver ETF":         "SLV",
    "USO — Oil ETF":            "USO",
    "XLE — Energy ETF":         "XLE",
    "XLF — Financials ETF":     "XLF",
    "EEM — EM ETF":             "EEM",
}
UNDERLYING_LABELS  = list(UNDERLYING_OPTIONS.keys())
_DISPLAY_TO_LABEL  = {k.split(" — ")[0]: k for k in UNDERLYING_OPTIONS.keys()}
# Also map by yfinance symbol → label for JSON loading
_TICKER_TO_LABEL   = {v: k for k, v in UNDERLYING_OPTIONS.items()}

# ==========================================================================
# Session state defaults
# ==========================================================================
_DEFAULTS = {
    "page":             "setup",   # "setup" | "dashboard"
    "run_terms":        None,
    "selected_tickers": None,
    "n_paths":          10000,
    "seed":             42,
    "results":          None,
    "path_num":         0,
    "setup_ul_default": None,    # set when JSON is loaded to override multiselect
    "custom_tickers":   {},      # {symbol: display_name} for user-entered tickers
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================================
# Cached helpers
# ==========================================================================
@st.cache_data
def _load_prices(tickers_tuple):
    return load_prices(source="yfinance", tickers=dict(tickers_tuple))

@st.cache_data
def _run_backtest_cached(tickers_tuple, terms_json):
    prices = _load_prices(tickers_tuple)
    t = NoteTerms.from_json(terms_json)
    return run_backtest(prices, t)

# ==========================================================================
# Language toggle (always in sidebar)
# ==========================================================================
lang_choice = st.sidebar.radio("🌐 Language / Idioma", ["English", "Español"],
                                horizontal=True)
tr = Translator("es" if lang_choice == "Español" else "en")

# ==========================================================================
# ─────────────────────────────────────────────────────────────────────────
#  PAGE 1 — SETUP
# ─────────────────────────────────────────────────────────────────────────
# ==========================================================================
if st.session_state["page"] == "setup":

    st.title("📈 Structured Note Simulator")
    st.markdown("Configure the note below, then click **Confirm & Run** to load the dashboard.")
    st.divider()

    # ── JSON upload ───────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload note config (JSON) — optional",
                                 type=["json"], key="setup_upload")
    loaded_terms = None
    if uploaded is not None:
        try:
            raw = uploaded.read().decode()
            loaded_terms = NoteTerms.from_json(raw)
            # Resolve tickers from config to known labels (or keep as custom)
            if loaded_terms.tickers:
                resolved = []
                custom_from_json = {}
                for sym, disp in loaded_terms.tickers.items():
                    if sym in _TICKER_TO_LABEL:
                        resolved.append(_TICKER_TO_LABEL[sym])
                    else:
                        # Not in our list — add as custom ticker
                        custom_lbl = f"{disp} — {sym} (custom)"
                        custom_from_json[sym] = disp
                        resolved.append(custom_lbl)
                # Only update if different from current to avoid re-render loops
                if st.session_state["setup_ul_default"] != resolved:
                    st.session_state["setup_ul_default"] = resolved
                    st.session_state["custom_tickers"] = custom_from_json
                    st.rerun()
            st.success(f"Config loaded: **{loaded_terms.name}**")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    base = loaded_terms or NoteTerms()

    st.divider()

    # ── Underlyings ───────────────────────────────────────────────────────
    st.subheader("Underlyings")

    # Build full option list including any custom tickers from session state
    custom_tickers = st.session_state.get("custom_tickers", {})
    all_labels = UNDERLYING_LABELS + [
        f"{disp} — {sym} (custom)" for sym, disp in custom_tickers.items()
    ]

    # Default: session state override (from JSON) or fallback
    default_ul = st.session_state["setup_ul_default"] or                  ["SPX — S&P 500", "SX5E — Euro Stoxx 50", "SMI — Swiss Market"]
    # Filter to only valid options
    default_ul = [d for d in default_ul if d in all_labels]

    selected_labels = st.multiselect(
        "Select underlyings (2–5)", all_labels,
        default=default_ul,
        key="setup_underlyings",
    )

    # ── Custom ticker input ───────────────────────────────────────────────
    with st.expander("➕ Add a custom ticker (not in the list above)"):
        st.caption("Enter any valid yfinance symbol, e.g. UBER, 2222.SR, BTC-USD")
        cc1, cc2, cc3 = st.columns([2, 2, 1])
        with cc1:
            custom_sym = st.text_input("yfinance symbol", placeholder="e.g. UBER",
                                        key="custom_sym_input").strip().upper()
        with cc2:
            custom_name = st.text_input("Display name", placeholder="e.g. Uber",
                                         key="custom_name_input").strip()
        with cc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add", key="add_custom_btn"):
                if custom_sym and custom_name:
                    ct = dict(st.session_state.get("custom_tickers", {}))
                    ct[custom_sym] = custom_name
                    st.session_state["custom_tickers"] = ct
                    # Auto-select the new ticker
                    new_lbl = f"{custom_name} — {custom_sym} (custom)"
                    current = list(st.session_state.get("setup_underlyings", default_ul))
                    if new_lbl not in current:
                        current.append(new_lbl)
                    st.session_state["setup_ul_default"] = current
                    st.rerun()
                else:
                    st.warning("Enter both a symbol and a display name.")

    st.divider()

    # ── Note terms ────────────────────────────────────────────────────────
    st.subheader("Note Terms")

    col1, col2, col3 = st.columns(3)

    maturity_opts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    with col1:
        maturity = st.selectbox(
            "Maturity (years)", maturity_opts,
            index=maturity_opts.index(base.maturity)
                  if base.maturity in maturity_opts else 1,
        )
        n_obs = st.number_input("Observation periods", 2, 60,
                                 value=base.n_obs, step=1)
        autocall_start = st.number_input(
            "Autocall start period", 1, int(n_obs),
            value=min(base.autocall_start_period, int(n_obs)),
        )

    with col2:
        coupon_pct = st.number_input(
            "Coupon per period (%)", 0.0, 20.0,
            value=round(base.coupon_rate * 100, 4),
            step=0.1, format="%.4f",
        )
        coupon_bar_pct = st.slider("Coupon barrier (%)", 0, 100,
                                    int(base.coupon_barrier * 100))
        memory = st.toggle("Memory coupon", value=base.memory)

    with col3:
        autocall_bar_pct = st.slider("Autocall barrier (%)", 50, 120,
                                      int(base.autocall_barrier * 100))
        ki_bar_pct = st.slider("Knock-in barrier (%)", 0, 100,
                                int(base.knock_in_barrier * 100))

    st.divider()

    # ── Basket types ──────────────────────────────────────────────────────
    st.subheader("Basket Types")
    basket_opts = ["worst_of", "best_of", "average"]
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        coupon_basket = st.selectbox("Coupon barrier check", basket_opts,
                                      index=basket_opts.index(base.coupon_basket))
    with bc2:
        autocall_basket = st.selectbox("Autocall trigger check", basket_opts,
                                        index=basket_opts.index(base.autocall_basket))
    with bc3:
        final_basket = st.selectbox("Final redemption check", basket_opts,
                                     index=basket_opts.index(base.final_basket))

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.subheader("Simulation")
    sc1, sc2 = st.columns(2)
    with sc1:
        n_paths = st.slider("Monte Carlo paths", 1000, 50000,
                             st.session_state["n_paths"], step=1000)
    with sc2:
        seed = st.number_input("Random seed", value=st.session_state["seed"])

    st.divider()

    # ── Confirm ───────────────────────────────────────────────────────────
    if len(selected_labels) < 2:
        st.warning("Select at least 2 underlyings to continue.")
    else:
        if st.button("✅ Confirm & Load Dashboard", type="primary",
                     use_container_width=True):
            custom_tickers = st.session_state.get("custom_tickers", {})
            # Build reverse map including custom tickers
            all_options = dict(UNDERLYING_OPTIONS)
            all_options.update({sym: disp for sym, disp in custom_tickers.items()})
            _all_label_to_sym = {k: v for k, v in UNDERLYING_OPTIONS.items()}
            _all_label_to_sym.update({
                f"{disp} — {sym} (custom)": sym
                for sym, disp in custom_tickers.items()
            })
            selected_tickers = {}
            for lbl in selected_labels[:5]:
                sym  = _all_label_to_sym.get(lbl)
                disp = lbl.split(" — ")[0]
                if sym:
                    selected_tickers[sym] = disp
            terms = NoteTerms(
                name                  = base.name if loaded_terms else "Custom Note",
                maturity              = float(maturity),
                n_obs                 = int(n_obs),
                coupon_rate           = coupon_pct / 100.0,
                coupon_barrier        = coupon_bar_pct / 100.0,
                autocall_barrier      = autocall_bar_pct / 100.0,
                autocall_start_period = int(autocall_start),
                knock_in_barrier      = ki_bar_pct / 100.0,
                principal_protection  = 1.0,
                memory                = memory,
                coupon_basket         = coupon_basket,
                autocall_basket       = autocall_basket,
                final_basket          = final_basket,
                call_steepness        = 100.0,
                tickers               = selected_tickers,
            )
            st.session_state["run_terms"]        = terms
            st.session_state["selected_tickers"] = selected_tickers
            st.session_state["n_paths"]          = n_paths
            st.session_state["seed"]             = int(seed)
            st.session_state["results"]          = None
            st.session_state["page"]             = "dashboard"
            st.session_state["setup_ul_default"] = None   # clear for next reconfigure
            st.rerun()

# ==========================================================================
# ─────────────────────────────────────────────────────────────────────────
#  PAGE 2 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────
# ==========================================================================
elif st.session_state["page"] == "dashboard":

    terms            = st.session_state["run_terms"]
    selected_tickers = st.session_state["selected_tickers"]
    n_paths          = st.session_state["n_paths"]
    seed             = st.session_state["seed"]
    tickers_tuple    = tuple(selected_tickers.items())

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("📋 Note")
    st.sidebar.markdown(f"**{terms.name}**")
    st.sidebar.markdown(
        f"{', '.join(selected_tickers.values())}  \n"
        f"{int(terms.maturity*12)}M · {terms.n_obs} obs · "
        f"{terms.coupon_rate*100:.4g}%/period"
    )
    st.sidebar.download_button(
        "⬇ Download config (JSON)",
        data=terms.to_json(),
        file_name="note_config.json",
        mime="application/json",
    )
    st.sidebar.divider()
    if st.sidebar.button("⚙️ Reconfigure Note"):
        st.session_state["page"]    = "setup"
        st.session_state["results"] = None
        st.rerun()
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

    # ── Title ─────────────────────────────────────────────────────────────
    st.title("📈 Multi-Asset Structured Note Simulator")
    st.markdown(
        f"**{terms.name}** — "
        f"{', '.join(selected_tickers.values())} · "
        f"{int(terms.maturity*12)}M · {terms.n_obs} obs · "
        f"Coupon {terms.coupon_rate*100:.4g}%/period · "
        f"{'Memory' if terms.memory else 'No memory'} · "
        f"KI {terms.knock_in_barrier:.0%} · Autocall {terms.autocall_barrier:.0%}"
    )

    with st.expander("📖 Note Structure Summary", expanded=False):
        # Underlyings table
        st.markdown("**Underlyings**")
        ul_df = pd.DataFrame([
            {"Display Name": disp, "yfinance Symbol": sym}
            for sym, disp in selected_tickers.items()
        ])
        st.dataframe(ul_df, use_container_width=True, hide_index=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Maturity", f"{terms.maturity}Y")
        c1.metric("Observations", f"{terms.n_obs}")
        c1.metric("Frequency", f"Every {terms.maturity/terms.n_obs*12:.1f}M")
        c2.metric("Coupon / period", f"{terms.coupon_rate*100:.4g}%")
        c2.metric("Coupon p.a.", f"{terms.coupon_rate * terms.n_obs / terms.maturity:.2%}")
        c2.metric("Memory", "Yes" if terms.memory else "No")
        c3.metric("Coupon barrier", f"{terms.coupon_barrier:.0%}")
        c3.metric("Autocall barrier", f"{terms.autocall_barrier:.0%}")
        c3.metric("Knock-in barrier", f"{terms.knock_in_barrier:.0%}")
        obs_df = pd.DataFrame({
            "Period": range(1, terms.n_obs + 1),
            "Time (Y)": [f"{t:.4g}" for t in terms.obs_times()],
            "Autocall eligible": ["✅" if i + 1 >= terms.autocall_start_period else "❌ coupon only"
                                   for i in range(terms.n_obs)],
        })
        st.dataframe(obs_df, use_container_width=True, hide_index=True)

    # ── Price fetch loading screen ────────────────────────────────────────
    if st.session_state["results"] is None and not run_button:
        with st.spinner(f"Fetching market data for {', '.join(selected_tickers.values())}…"):
            try:
                _load_prices(tickers_tuple)
                st.info("Market data ready. Click **🚀 Run Simulation** in the sidebar.")
            except Exception as e:
                st.error(f"Failed to fetch prices: {e}")

    # ── Run simulation ────────────────────────────────────────────────────
    if run_button:
        with st.spinner("Running Heston calibration and Monte Carlo simulation…"):
            prices = _load_prices(tickers_tuple)
            cal    = HestonCalibrator(prices_df=prices)
            cal_result = cal.calibrate()

            N_steps = max(252, round(252 * terms.maturity))
            sim = HestonMultiSimulator(
                params  = cal_result.params,
                corr_SS = cal_result.corr_SS,
                corr_VV = cal_result.corr_VV,
                corr_SV = cal_result.corr_SV,
                T       = terms.maturity,
                N       = N_steps,
                n_paths = n_paths,
                seed    = seed,
                t_dof   = cal_result.t_dof,
            )
            sim_results = sim.run()

            n_assets   = len(cal_result.params)
            sim_prices = np.stack(sim_results["S_paths"], axis=2)
            S0_vec     = np.array([p.S0 for p in cal_result.params]).reshape(1, 1, n_assets)
            perf_paths = sim_prices / S0_vec
            wof_paths  = perf_paths.min(axis=2)

            note_results = price_note(perf_paths, terms, seed=seed + 1)

            # Store S0 values separately — HestonParams objects can cause
            # serialization issues in session state across reruns
            s0_values = [p.S0 for p in cal_result.params]

            st.session_state["results"] = {
                **note_results,
                "worst_of_paths": wof_paths,
                "sim_prices":     sim_prices,
                "asset_names":    list(selected_tickers.values()),
                "s0_values":      s0_values,
                "params":         cal_result.params,
                "corr_SS":        cal_result.corr_SS,
                "sim_results":    sim_results,
                "t_dof":          cal_result.t_dof,
                "terms_snapshot": terms.to_dict(),
            }
            # Also store asset_names separately as a safety fallback
            st.session_state["last_asset_names"] = list(selected_tickers.values())
            st.session_state["path_num"] = 0
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────
    if st.session_state["results"] is not None:
        R           = st.session_state["results"]
        # Fallback: if asset_names is empty after rerun use the separately stored copy
        asset_names = R.get("asset_names") or                       st.session_state.get("last_asset_names") or                       list(selected_tickers.values())
        wof_paths   = R["worst_of_paths"]
        sim_prices  = R["sim_prices"]
        N           = wof_paths.shape[1] - 1
        run_terms   = NoteTerms.from_dict(R["terms_snapshot"])
        t_grid      = np.linspace(0, run_terms.maturity, N + 1)
        obs_steps_i = run_terms.obs_steps(N)
        obs_times_l = run_terms.obs_times()
        obs_pairs   = [(f"P{i+1}", t) for i, t in enumerate(obs_times_l)]

        st.success("Simulation complete.")
        st.header("Summary Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Expected IRR p.a.",     f"{R['expected_irr']:.2%}")
        c2.metric("Expected Total Return", f"{R['expected_total_return']:.2%}")
        c3.metric("Expected Coupon",       f"{R['expected_coupon']:.2%}")
        c4.metric("P(Autocalled)",         f"{R['prob_autocall']:.2%}")
        c5.metric("P(Knock-in)",           f"{R['prob_knock_in_total']:.2%}")

        with st.expander("Autocall probability by period", expanded=False):
            prob_by_period = R["prob_autocall_by_period"]
            ac_df = pd.DataFrame({
                "Period":      range(1, run_terms.n_obs + 1),
                "Time (Y)":    [f"{t:.3g}" for t in obs_times_l],
                "P(autocall)": [f"{p:.2%}" for p in prob_by_period],
                "Eligible":    ["✅" if i + 1 >= run_terms.autocall_start_period else "❌"
                                 for i in range(run_terms.n_obs)],
            })
            st.dataframe(ac_df, use_container_width=True, hide_index=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Payoff & Distribution", "📈 Price Paths",
            "🔍 Path Explorer",         "🔗 Correlation Diagnostics",
        ])

        with tab1:
            st.subheader("IRR Distribution — All Simulated Paths")
            coupon_pa = run_terms.coupon_rate * run_terms.n_obs / run_terms.maturity
            st.plotly_chart(
                build_irr_distribution(
                    R["annualized_returns"],
                    R["autocall_events"],
                    R["expected_irr"],
                    coupon_pa,
                    tr,
                ),
                use_container_width=True,
            )
            if R["prob_knock_in_total"] > 0:
                st.info(f"**{R['prob_knock_in_total']:.1%}** of paths trigger the "
                        f"knock-in barrier ({run_terms.knock_in_barrier:.0%}) at maturity.")

        with tab2:
            st.subheader("Simulated Price Path Fan Charts")
            # Worst-of fan first (most relevant for note pricing)
            st.markdown("#### Worst-of Basket Performance")
            st.plotly_chart(
                build_wof_fan(wof_paths, t_grid, run_terms.knock_in_barrier, obs_pairs, tr),
                use_container_width=True,
            )
            st.markdown("#### Individual Underlying Paths")
            # Per-asset fan charts
            for i, name in enumerate(asset_names):
                st.plotly_chart(
                    build_fan_chart(sim_prices[:, :, i], name, t_grid, obs_pairs, tr),
                    use_container_width=True,
                )

        with tab3:
            st.subheader("Single Path Explorer")
            max_path = sim_prices.shape[0] - 1
            pc1, pc2, pc3 = st.columns(3)
            if pc1.button("🎲 Random"):
                st.session_state["path_num"] = random.randint(0, max_path)
            if pc2.button("⬅ Prev"):
                st.session_state["path_num"] = max(0, st.session_state["path_num"] - 1)
            if pc3.button("Next ➡"):
                st.session_state["path_num"] = min(max_path, st.session_state["path_num"] + 1)

            pn = st.session_state["path_num"]
            st.caption(f"Path #{pn} of {max_path}")
            obs_labels = [lbl for lbl, _ in obs_pairs]

            path_df = pd.DataFrame(
                {n: sim_prices[pn, :, i] for i, n in enumerate(asset_names)}
            )
            st.plotly_chart(
                build_path_price_chart(path_df, pn, obs_steps_i, obs_labels, tr),
                use_container_width=True,
            )
            autocall_q    = int(R["autocall_events"][pn])
            # asset_paths: (N+1, n_assets) performance fractions for this path
            # Use s0_values (serialization-safe) instead of params objects
            s0_arr        = np.array(R.get("s0_values") or [p.S0 for p in R["params"]])
            asset_perf_pn = sim_prices[pn] / s0_arr[np.newaxis, :]
            st.plotly_chart(
                build_path_wof_chart(
                    wof_paths[pn], autocall_q, obs_steps_i, obs_labels,
                    run_terms.knock_in_barrier, pn, tr,
                    asset_paths=asset_perf_pn, asset_names=asset_names,
                ),
                use_container_width=True,
            )

            principal = float(R["principal_payoffs"][pn])
            coupons   = float(R["coupon_payoffs"][pn])
            irr       = float(R["annualized_returns"][pn])
            worst_f   = float(wof_paths[pn, -1])
            ki        = bool(R["knock_in_triggered"][pn])

            if autocall_q > 0:
                t_q = obs_times_l[autocall_q - 1]
                st.markdown(f"### ✅ Autocalled at period {autocall_q} ({t_q:.3g}Y)")
            elif ki:
                st.markdown(f"### ⚠️ Maturity — Knock-in (worst-of: {worst_f:.1%})")
            else:
                st.markdown(f"### 📈 Maturity — No knock-in (worst-of: {worst_f:.1%})")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Principal", f"{principal:.2%}")
            mc2.metric("Coupons",   f"{coupons:.2%}")
            mc3.metric("IRR p.a.",  f"{irr:.2%}")

        with tab4:
            st.subheader("Correlation Diagnostics")
            corr_SS       = R["corr_SS"]
            realized_corr = R["sim_results"]["realized_corr"]
            diff          = realized_corr - corr_SS
            hm1, hm2, hm3 = st.columns(3)
            hm1.plotly_chart(build_corr_heatmap(corr_SS,       asset_names, "Input"),     use_container_width=True)
            hm2.plotly_chart(build_corr_heatmap(realized_corr, asset_names, "Realized"),  use_container_width=True)
            hm3.plotly_chart(build_corr_heatmap(diff, asset_names, "Difference",
                                                  zmin=-0.1, zmax=0.1),                   use_container_width=True)
            max_err = float(np.max(np.abs(diff - np.diag(np.diag(diff)))))
            (st.success if max_err < 0.05 else st.warning)(
                f"Max off-diagonal error: **{max_err:.4f}**"
            )
            st.markdown("---")
            st.subheader("Calibrated Heston Parameters")
            rows = []
            for p in R["params"]:
                ok, _ = p.feller_condition()
                rows.append({
                    "Asset": p.name, "S₀": f"{p.S0:.1f}",
                    "μ p.a.": f"{p.mu*100:.1f}%",
                    "V₀ σ":   f"{np.sqrt(p.V0)*100:.1f}%",
                    "θ σ LR": f"{np.sqrt(p.theta)*100:.1f}%",
                    "κ": f"{p.kappa:.3f}", "ξ": f"{p.xi:.3f}", "ρ": f"{p.rho:.3f}",
                    "Feller": "✅" if ok else "⚠️",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info(f"**Student-t Copula:** ν = {R.get('t_dof','N/A')} d.f.")

    # ── Historical Backtest ───────────────────────────────────────────────
    st.markdown("---")
    st.header("📅 Historical Backtest")
    st.markdown("Evaluates this note on every valid issue date using actual realized prices.")

    with st.spinner("Running historical backtest…"):
        try:
            bt, bt_summary = _run_backtest_cached(tickers_tuple, terms.to_json())
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            bt, bt_summary = pd.DataFrame(), {}

    if bt.empty:
        st.warning("No backtest results. Check underlyings have sufficient history.")
    else:
        bt["Outcome"] = bt["Call Quarter"].map(
            {0: "Maturity", **{i: f"Autocalled P{i}" for i in range(1, terms.n_obs + 1)}}
        )
        bt.loc[(bt["Call Quarter"] == 0) & (bt["Worst Final Perf"] < terms.knock_in_barrier),
               "Outcome"] = "Knock-in"
        color_map = {
            "Maturity": "#3498db", "Knock-in": "#e74c3c",
            **{f"Autocalled P{i}": f"hsl({120 - i*4},55%,38%)" for i in range(1, terms.n_obs + 1)},
        }
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Issue Dates",  str(bt_summary.get("n_issues", 0)))
        b2.metric("Mean IRR",     f"{bt_summary.get('mean_irr', 0):.2%}")
        b3.metric("Median IRR",   f"{bt_summary.get('median_irr', 0):.2%}")
        b4.metric("Knock-in %",   f"{bt_summary.get('prob_knock_in', 0):.1%}")
        b5.metric("Autocalled %", f"{bt_summary.get('prob_called', 0):.1%}")

        col1, col2 = st.columns(2)
        col1.plotly_chart(build_backtest_outcome_bar(bt, color_map, tr), use_container_width=True)
        col2.plotly_chart(build_worst_asset_pie(bt, tr),                 use_container_width=True)
        st.plotly_chart(build_backtest_irr_scatter(bt, color_map, tr),   use_container_width=True)

        try:
            hist_prices = _load_prices(tickers_tuple)
            bt_start    = pd.Timestamp(str(bt["Issue Date"].min()))
            bt_end      = pd.Timestamp(str(bt["Issue Date"].max()))
            st.plotly_chart(build_historical_prices(hist_prices, bt_start, bt_end, tr),
                            use_container_width=True)
        except Exception:
            pass

        # ── Historical worst-of path explorer ─────────────────────────
        st.subheader("📅 Historical Path Explorer")
        st.caption("Select any issue date from the backtest to see the actual "
                   "per-asset performance and worst-of path over the note's life.")

        issue_dates = sorted(bt["Issue Date"].unique())
        selected_issue = st.selectbox(
            "Issue date",
            issue_dates,
            index=0,
            format_func=lambda d: str(d),
            key="bt_issue_selector",
        )
        if selected_issue is not None:
            row = bt[bt["Issue Date"] == selected_issue].iloc[0]
            st.markdown(
                f"**Outcome:** {row['Outcome']} &nbsp;|&nbsp; "
                f"**IRR:** {row['IRR']:.2%} &nbsp;|&nbsp; "
                f"**Worst asset:** {row['Worst Asset']} "
                f"({row['Worst Final Perf']:.1%})"
            )
            try:
                hist_prices = _load_prices(tickers_tuple)
                maturity_days   = round(terms.maturity * 252)
                obs_day_offsets = [round(t / terms.maturity * maturity_days)
                                   for t in terms.obs_times()]
                st.plotly_chart(
                    build_historical_wof_path(
                        hist_prices,
                        issue_date        = pd.Timestamp(str(selected_issue)),
                        maturity_days     = maturity_days,
                        obs_day_offsets   = obs_day_offsets,
                        knock_in_barrier  = terms.knock_in_barrier,
                        autocall_barrier  = terms.autocall_barrier,
                        call_quarter      = int(row["Call Quarter"]),
                        tr                = tr,
                    ),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not build path: {e}")