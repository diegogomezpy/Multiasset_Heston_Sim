
import numpy as np

from heston_calibrator import HestonCalibrator
from heston_simulator import HestonMultiSimulator


def run_structured_note(
    coupon_rate=0.10,
    floor_level=0.95,
    maturity=1.0,
    n_paths=50000,
    N=252,
    seed=42,
    csv_files=None,
    rv_window=21,
    mle_refine=False,
):
    """
    Runs the full historical calibration, Heston simulation,
    and structured note evaluation.

    Returns a dictionary containing summary statistics,
    simulation paths, and diagnostics for use in Streamlit.
    """

    if csv_files is None:
        csv_files = {
            "SPX.csv": "SPX",
            "SX5E.csv": "SX5E",
            "SMI.csv": "SMI",
        }

    # ==========================================================
    # 1. CALIBRATION
    # ==========================================================

    cal = HestonCalibrator(
        csv_files=csv_files,
        rv_window=rv_window,
        mle_refine=mle_refine,
    )

    result = cal.calibrate()

    # ==========================================================
    # 2. HESTON SIMULATION
    # ==========================================================

    sim = HestonMultiSimulator(
        params=result.params,
        corr_SS=result.corr_SS,
        corr_VV=result.corr_VV,
        corr_SV=result.corr_SV,
        T=maturity,
        N=N,
        n_paths=n_paths,
        seed=seed,
    )

    sim_results = sim.run()

    # ==========================================================
    # 3. STRUCTURED NOTE ENGINE
    # ==========================================================

    asset_names = [p.name for p in result.params]
    n_assets = len(asset_names)

    sim_prices = np.stack(
        sim_results["S_paths"],
        axis=2
    )

    S0_vector = np.array(
        [p.S0 for p in result.params]
    ).reshape(1, 1, n_assets)

    perf_paths = sim_prices / S0_vector

    worst_of_paths = np.min(
        perf_paths,
        axis=2
    )

    steps_per_quarter = N // 4

    obs_steps = [
        steps_per_quarter,
        steps_per_quarter * 2,
        steps_per_quarter * 3,
    ]

    nominal_payoffs = np.zeros(n_paths)

    annualized_returns = np.zeros(n_paths)

    autocall_events = np.zeros(n_paths)

    for idx in range(n_paths):

        autocalled_early = False

        for q, step in enumerate(obs_steps):

            t_years = (q + 1) * 0.25

            worst_perf = worst_of_paths[idx, step]

            if worst_perf >= floor_level:

                payout = (
                    1.0
                    + coupon_rate * t_years
                )

                nominal_payoffs[idx] = payout

                annualized_returns[idx] = (
                    payout ** (1.0 / t_years)
                ) - 1.0

                autocall_events[idx] = q + 1

                autocalled_early = True

                break

        if not autocalled_early:

            worst_final = worst_of_paths[idx, N]

            if worst_final >= floor_level:

                payout = (
                    floor_level
                    + (worst_final - floor_level)
                )

            else:

                payout = floor_level

            nominal_payoffs[idx] = payout

            annualized_returns[idx] = payout - 1.0

    # ==========================================================
    # 4. SUMMARY STATISTICS
    # ==========================================================

    mean_nominal_payout = np.mean(
        nominal_payoffs
    )

    mean_annualized_return = np.mean(
        annualized_returns
    )

    prob_q1 = np.mean(
        autocall_events == 1
    )

    prob_q2 = np.mean(
        autocall_events == 2
    )

    prob_q3 = np.mean(
        autocall_events == 3
    )

    prob_mat = np.mean(
        autocall_events == 0
    )

    prob_floor = np.mean(
        (autocall_events == 0)
        & (worst_of_paths[:, N] < floor_level)
    )

    return {

        # Summary statistics
        "expected_nominal_payout":
            mean_nominal_payout,

        "expected_total_return":
            mean_nominal_payout - 1.0,

        "expected_irr":
            mean_annualized_return,

        "prob_q1":
            prob_q1,

        "prob_q2":
            prob_q2,

        "prob_q3":
            prob_q3,

        "prob_maturity":
            prob_mat,

        "prob_floor":
            prob_floor,

        # Useful for Streamlit plots
        "annualized_returns":
            annualized_returns,

        "nominal_payoffs":
            nominal_payoffs,

        "autocall_events":
            autocall_events,

        "worst_of_paths":
            worst_of_paths,

        "sim_prices":
            sim_prices,

        "asset_names":
            asset_names,

        # Calibration outputs
        "params":
            result.params,

        "corr_SS":
            result.corr_SS,

        "corr_VV":
            result.corr_VV,

        "corr_SV":
            result.corr_SV,

        # Raw simulator output
        "sim_results":
            sim_results,
    }
