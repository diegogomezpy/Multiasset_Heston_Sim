import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random

from structured_note_sim import run_structured_note

# ==========================================================
# SESSION STATE
# ==========================================================

if "results" not in st.session_state:
    st.session_state["results"] = None

if "path_num" not in st.session_state:
    st.session_state["path_num"] = 0

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Structured Note Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Asset Structured Note Simulator")

st.markdown(
    """
    Historical Heston calibration + Monte Carlo simulation
    of a worst-of autocallable structured note.
    """
)

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================

st.sidebar.header("Simulation Parameters")

coupon = st.sidebar.slider(
    "Coupon (%)",
    min_value=0,
    max_value=20,
    value=10,
)

floor = st.sidebar.slider(
    "Capital Floor (%)",
    min_value=50,
    max_value=100,
    value=95,
)

n_paths = st.sidebar.slider(
    "Monte Carlo Paths",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
)

seed = st.sidebar.number_input(
    "Random Seed",
    value=42
)

run_button = st.sidebar.button(
    "🚀 Run Simulation"
)

# ==========================================================
# RUN SIMULATION
# ==========================================================

if run_button:

    with st.spinner(
        "Running calibration and simulation..."
    ):

        st.session_state["results"] = (
            run_structured_note(
                coupon_rate=coupon / 100,
                floor_level=floor / 100,
                n_paths=n_paths,
                seed=seed,
            )
        )

        st.session_state["path_num"] = 0

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

if st.session_state["results"] is not None:

    results = st.session_state["results"]

    st.success("Simulation complete.")

    # ======================================================
    # SUMMARY METRICS
    # ======================================================

    st.header("Summary Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Expected IRR",
            f"{results['expected_irr']:.2%}"
        )

    with c2:
        st.metric(
            "Expected Return",
            f"{results['expected_total_return']:.2%}"
        )

    with c3:
        st.metric(
            "Maturity Probability",
            f"{results['prob_maturity']:.2%}"
        )

    with c4:
        st.metric(
            "Capital Floor Probability",
            f"{results['prob_floor']:.2%}"
        )

    # ======================================================
    # AUTOCALL PROBABILITIES
    # ======================================================

    st.header("Autocall Probabilities")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "3M Autocall",
            f"{results['prob_q1']:.2%}"
        )

    with q2:
        st.metric(
            "6M Autocall",
            f"{results['prob_q2']:.2%}"
        )

    with q3:
        st.metric(
            "9M Autocall",
            f"{results['prob_q3']:.2%}"
        )

    # ======================================================
    # TABS
    # ======================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "Payoff Profile",
            "Single Path Explorer",
            "Correlation Diagnostics",
        ]
    )

    # ======================================================
    # TAB 1
    # ======================================================

    with tab1:

        st.subheader(
            "Structured Note Payoff Function"
        )

        floor_level = floor / 100

        worst_perf = np.linspace(
            0.50,
            1.50,
            500
        )

        payoff = np.where(
            worst_perf < floor_level,
            floor_level,
            worst_perf
        )

        payoff_df = pd.DataFrame({
            "Worst-of Final Performance":
                worst_perf,
            "Note Payoff":
                payoff,
        })

        fig = px.line(
            payoff_df,
            x="Worst-of Final Performance",
            y="Note Payoff",
            title="Maturity Payoff Profile",
        )

        fig.add_vline(
            x=floor_level,
            line_dash="dash",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            f"""
            **Interpretation**

            - If the worst-performing asset finishes below
              **{floor_level:.0%}**
              the note redeems at the floor.

            - Above **{floor_level:.0%}**
              the investor participates
              one-for-one in upside.
            """
        )

    # ======================================================
    # TAB 2
    # ======================================================

    with tab2:

        st.subheader(
            "Single Path Explorer"
        )

        sim_prices = results["sim_prices"]
        asset_names = results["asset_names"]

        max_path = (
            sim_prices.shape[0] - 1
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if st.button(
                "🎲 Random Path"
            ):

                st.session_state["path_num"] = (
                    random.randint(
                        0,
                        max_path
                    )
                )

        with c2:

            if st.button(
                "⬅ Previous"
            ):

                st.session_state["path_num"] = max(
                    0,
                    st.session_state["path_num"] - 1
                )

        with c3:

            if st.button(
                "Next ➡"
            ):

                st.session_state["path_num"] = min(
                    max_path,
                    st.session_state["path_num"] + 1
                )

        path_num = st.session_state["path_num"]

        st.write(
            f"Currently Viewing Path #{path_num}"
        )

        # ==============================================
        # Asset Paths
        # ==============================================

        path_df = pd.DataFrame()

        for i, name in enumerate(
            asset_names
        ):

            path_df[name] = sim_prices[
                path_num,
                :,
                i
            ]

        fig = px.line(
            path_df,
            title=f"Simulation Path {path_num}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==============================================
        # Worst-of Path
        # ==============================================

        worst_path = results[
            "worst_of_paths"
        ][path_num]

        fig2 = px.line(
            y=worst_path,
            labels={
                "y":
                "Worst-of Performance"
            }
        )

        fig2.update_layout(
            title=
            "Worst-of Performance Path"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ==============================================
        # Scenario Summary
        # ==============================================

        final_perf = worst_path[-1]

        if final_perf < floor / 100:

            outcome = (
                "Capital Floor Applied"
            )

        else:

            outcome = (
                "Upside Participation"
            )

        st.info(
            f"""
            Scenario #{path_num}

            Final Worst-of Performance:
            {final_perf:.2%}

            Outcome:
            {outcome}
            """
        )

    # ======================================================
    # TAB 3
    # ======================================================

    with tab3:

        st.subheader(
            "Correlation Matrix"
        )

        corr_df = pd.DataFrame(
            results["corr_SS"],
            index=asset_names,
            columns=asset_names,
        )

        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "Choose parameters in the sidebar and click Run Simulation."
    )