import numpy as np
import pandas as pd

from heston_calibrator import HestonCalibrator
from heston_simulator import HestonMultiSimulator


# ==========================================================================
# Backtest helpers
# ==========================================================================

DEFAULT_CSV_FILES = {
    "SPX.csv":  "SPX",
    "SX5E.csv": "SX5E",
    "SMI.csv":  "SMI",
}


def load_prices(csv_files=None):
    """
    Load and align adjusted close prices from Yahoo Finance CSV files.

    Parameters
    ----------
    csv_files : dict[str, str] or None
        Mapping from file path to display name.
        Defaults to {"SPX.csv": "SPX", "SX5E.csv": "SX5E", "SMI.csv": "SMI"}.

    Returns
    -------
    pd.DataFrame
        Aligned daily closing prices, one column per asset, sorted by date.
    """
    if csv_files is None:
        csv_files = DEFAULT_CSV_FILES

    dfs = {}
    for fname, name in csv_files.items():
        df = pd.read_csv(fname, index_col="Date", parse_dates=True)
        df = df[pd.to_datetime(df.index, errors="coerce").notna()]
        df.index = pd.to_datetime(df.index)
        dfs[name] = pd.to_numeric(df["Adj Close"], errors="coerce")

    prices = pd.DataFrame(dfs).dropna()
    prices.sort_index(inplace=True)
    return prices


def run_backtest(
    coupon_rate=0.10,
    floor_level=0.95,
    call_steepness=20.0,
    issue_freq_weeks=2,
    seed=42,
    csv_files=None,
):
    """
    Historical backtest: evaluate the note payoff on every valid issue date
    in the price history using the actual realized index prices.

    No simulation or model — purely historical price paths.
    The issuer call uses the same sigmoid model as the forward simulation:
    the issuer calls with probability sigmoid(steepness * (worst_of - floor_level))
    whenever worst_of >= floor_level at a quarterly observation date.

    Parameters
    ----------
    coupon_rate      : float  Annual coupon rate (e.g. 0.10).
    floor_level      : float  Capital floor and call strike (e.g. 0.95).
    call_steepness   : float  Sigmoid steepness. Default 20.
    issue_freq_weeks : int    Spacing between sampled issue dates in weeks.
    seed             : int    RNG seed for the sigmoid call draws.
    csv_files        : dict   Passed to load_prices().

    Returns
    -------
    pd.DataFrame  One row per issue date.
    dict          Summary statistics including money-weighted IRR.
    """
    prices = load_prices(csv_files)
    rng    = np.random.default_rng(seed)

    step_q          = 63
    maturity_offset = 252
    obs_offsets     = [step_q, step_q * 2, step_q * 3]

    first_valid   = prices.index[maturity_offset]
    last_valid    = prices.index[-(maturity_offset + 1)]
    sampled_dates = pd.date_range(start=first_valid, end=last_valid,
                                  freq=f"{issue_freq_weeks}W")

    asset_names = list(prices.columns)
    records     = []

    for issue_date in sampled_dates:
        issue_idx = prices.index.searchsorted(issue_date)
        if issue_idx >= len(prices) - maturity_offset:
            continue

        issue_date = prices.index[issue_idx]
        S0         = prices.iloc[issue_idx]
        called     = False
        call_quarter = 0
        payout     = None
        t_held     = 1.0

        for q, offset in enumerate(obs_offsets):
            obs_idx = issue_idx + offset
            if obs_idx >= len(prices):
                break
            perf       = prices.iloc[obs_idx] / S0
            worst_perf = perf.min()

            if worst_perf >= floor_level:
                p_call = _issuer_call_prob(worst_perf, floor_level, call_steepness)
                if rng.random() < p_call:
                    t_held       = (q + 1) * 0.25
                    payout       = 1.0 + coupon_rate * t_held
                    call_quarter = q + 1
                    called       = True
                    break

        mat_idx  = issue_idx + maturity_offset
        perf_mat = prices.iloc[mat_idx] / S0

        if not called:
            worst_final  = perf_mat.min()
            payout       = worst_final if worst_final >= floor_level else floor_level
            call_quarter = 0
            t_held       = 1.0

        irr = payout ** (1.0 / t_held) - 1.0

        row = {
            "Issue Date":       issue_date.date(),
            "Call Quarter":     call_quarter,
            "Payout":           payout,
            "IRR":              irr,
            "Worst Asset":      perf_mat.idxmin(),
            "Worst Final Perf": perf_mat.min(),
        }
        for name in asset_names:
            row[f"{name} Perf"] = perf_mat.get(name, np.nan)
        records.append(row)

    bt = pd.DataFrame(records)

    if bt.empty:
        return bt, {}

    floor_mask = (bt["Call Quarter"] == 0) & (bt["Worst Final Perf"] < floor_level)
    summary = {
        "n_issues":      len(bt),
        "mean_irr":      float(bt["IRR"].mean()),
        "median_irr":    float(bt["IRR"].median()),
        "prob_floor":    float(floor_mask.mean()),
        "prob_called":   float((bt["Call Quarter"] > 0).mean()),
        "prob_q1":       float((bt["Call Quarter"] == 1).mean()),
        "prob_q2":       float((bt["Call Quarter"] == 2).mean()),
        "prob_q3":       float((bt["Call Quarter"] == 3).mean()),
        "prob_maturity": float((bt["Call Quarter"] == 0).mean()),
    }

    return bt, summary


def _issuer_call_prob(worst_perf: float, floor_level: float, steepness: float) -> float:
    """
    Sigmoid probability that the issuer exercises the call at an observation date.

    p = 1 / (1 + exp(-steepness * (worst_perf - floor_level)))

    Below floor_level : p → 0   (issuer never calls when out of the money)
    At floor_level    : p = 0.5 (indifferent at the strike)
    Deep in the money : p → 1   (issuer almost certainly calls)

    steepness controls how decisive the issuer is:
      ~10 : gradual, meaningful discretion around the strike
      ~50 : nearly automatic (approaches hard-trigger behavior)
    """
    return 1.0 / (1.0 + np.exp(-steepness * (worst_perf - floor_level)))


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
    call_steepness=20.0,
):
    """
    Runs the full historical calibration, Heston simulation,
    and structured note evaluation.

    The issuer call is modelled as discretionary: at each observation date,
    the issuer calls with probability given by a sigmoid centred on floor_level.
    call_steepness controls how decisive the issuer is (~20 = realistic discretion,
    ~50 = nearly automatic trigger).

    The Student-t copula degrees of freedom are estimated automatically from
    the historical return data by the calibrator (MLE fit of univariate t to
    each asset's returns, median across assets).

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
        t_dof=result.t_dof,   # calibrated from return tail behaviour
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

    # Antithetic variates double the paths — use the actual count from the simulator
    n_sim = worst_of_paths.shape[0]

    nominal_payoffs    = np.zeros(n_sim)
    annualized_returns = np.zeros(n_sim)
    autocall_events    = np.zeros(n_sim)

    rng_call = np.random.default_rng(seed + 1 if seed is not None else None)

    for idx in range(n_sim):

        autocalled_early = False

        for q, step in enumerate(obs_steps):

            t_years    = (q + 1) * 0.25
            worst_perf = worst_of_paths[idx, step]

            if worst_perf >= floor_level:

                p_call = _issuer_call_prob(worst_perf, floor_level, call_steepness)

                if rng_call.random() < p_call:

                    payout = 1.0 + coupon_rate * t_years

                    nominal_payoffs[idx]    = payout
                    annualized_returns[idx] = payout ** (1.0 / t_years) - 1.0
                    autocall_events[idx]    = q + 1
                    autocalled_early        = True
                    break

        if not autocalled_early:

            worst_final = worst_of_paths[idx, N]

            payout = worst_final if worst_final >= floor_level else floor_level

            nominal_payoffs[idx]    = payout
            annualized_returns[idx] = payout - 1.0  # 1-year maturity: annualised = total return

    # ==========================================================
    # 4. SUMMARY STATISTICS
    # ==========================================================

    mean_nominal_payout    = float(np.mean(nominal_payoffs))
    mean_annualized_return = float(np.mean(annualized_returns))

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
        "expected_nominal_payout": mean_nominal_payout,
        "expected_total_return":   mean_nominal_payout - 1.0,
        "expected_irr":            mean_annualized_return,

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

        # Call model
        "call_steepness":  call_steepness,
        "t_dof":           result.t_dof,
    }