# Multi-Asset Heston Simulator & Structured Note Engine

A Python framework for calibrating, simulating, and analyzing a **multi-asset Heston stochastic volatility model** using historical market data.

The project supports:

- Historical calibration of Heston parameters
- Multi-asset Monte Carlo simulation
- Correlated stochastic volatility dynamics
- Structured note payoff analysis
- Correlation diagnostics and validation
- Real-world forecasting under the physical measure

---

## Features

### Historical Heston Calibration

Calibrates the following Heston parameters directly from historical data:

- κ (mean reversion speed)
- θ (long-run variance)
- ξ (volatility of volatility)
- ρ (leverage effect)
- V₀ (current variance)

Supports:

- Yahoo Finance downloads
- Raw Yahoo Finance CSV files
- Preloaded pandas DataFrames

Optional maximum likelihood refinement is available after Method-of-Moments calibration.

---

### Multi-Asset Simulation

Simulates multiple correlated assets simultaneously under stochastic volatility.

Captures:

- Return-return correlations
- Variance-variance correlations
- Asset-volatility leverage effects

Uses:

- Euler-Maruyama discretization
- Full truncation scheme for variance positivity
- Higham nearest-PSD correction when required

---

### Diagnostics Dashboard

Automatically generates:

- Simulated price paths
- Volatility paths
- Terminal log-return distributions
- Input correlation heatmaps
- Realized correlation heatmaps
- Correlation error diagnostics

Example output:

![Diagnostics](heston_multi_diagnostics.png)

---

# Model Dynamics

For each asset:

Price process:

S(t + dt) = S(t) * exp(-0.5 * V(t) * dt + sqrt(V(t)) * dW_S)

Variance process:

dV = κ(θ − V)dt + ξ * sqrt(V) * dW_V

Leverage relationship:

Corr(dW_S, dW_V) = ρ

---

## Correlation Structure

The simulator builds a full block correlation matrix:

```text
        | Corr_SS   Corr_SV |
C   =   |                   |
        | Corr_SVᵀ Corr_VV |
```

Where:

- Corr_SS = return-return correlations
- Corr_VV = variance-variance correlations
- Corr_SV = return-volatility correlations

This allows simultaneous dependence between asset returns and volatility shocks.

---

# Project Structure

```text
.
├── heston_calibrator.py
├── heston_simulator.py
├── structured_note_sim.py
├── SPX.csv
├── SX5E.csv
├── SMI.csv
└── heston_multi_diagnostics.png
```

---

# Calibration Methodology

## Step 1 — Data Loading

Historical adjusted close prices are loaded from:

- Yahoo Finance
- CSV files
- pandas DataFrames

---

## Step 2 — Return Construction

### One-Day Log Returns

r₁(t) = ln(S(t) / S(t−1))

Used for:

- Realized variance estimation
- Leverage effect estimation

### Two-Day Overlapping Returns

r₂(t) = ln(S(t) / S(t−2))

Used for:

- Cross-asset correlation estimation

Using overlapping two-day returns helps reduce artificial decorrelation caused by asynchronous market close times between U.S. and European markets.

---

## Step 3 — Realized Variance Proxy

Rolling realized variance:

RV(t) = Var(r₁) × 252

where r₁ denotes one-day log returns.

---

## Step 4 — Method of Moments Calibration

Parameters are estimated directly from historical data:

| Parameter | Estimation Method |
|------------|------------|
| θ | Sample variance |
| V₀ | Latest rolling variance |
| κ | AR(1) mean reversion |
| ξ | Variance increment volatility |
| ρ | Return-volatility correlation |

---

## Step 5 — Optional MLE Refinement

The initial Method-of-Moments estimates can be refined using approximate maximum likelihood estimation.

---

## Step 6 — Correlation Estimation

The calibration produces:

- Corr_SS
- Corr_VV
- Corr_SV

which are passed directly into the simulator.

---

# Example Usage

## Calibration

```python
from heston_calibrator import HestonCalibrator

cal = HestonCalibrator(
    csv_files={
        "SPX.csv": "SPX",
        "SX5E.csv": "SX5E",
        "SMI.csv": "SMI",
    },
    rv_window=21,
    mle_refine=False,
)

result = cal.calibrate()
```

---

## Simulation

```python
from heston_simulator import HestonMultiSimulator

sim = HestonMultiSimulator(
    params=result.params,
    corr_SS=result.corr_SS,
    corr_VV=result.corr_VV,
    corr_SV=result.corr_SV,
    T=1.0,
    N=252,
    n_paths=20000,
    seed=42,
)

results = sim.run()
sim.plot()
```

---

# Structured Note Engine

The repository includes a practical structured-product application.

## Underlyings

- S&P 500 (SPX)
- Euro Stoxx 50 (SX5E)
- Swiss Market Index (SMI)

---

## Product Type

Worst-of autocallable note.

---

## Observation Dates

- 3 months
- 6 months
- 9 months

---

## Autocall Condition

The note automatically redeems if the worst-performing underlying remains above 95% of its initial value:

```text
min(S_i / S_i,0) ≥ 95%
```

---

## Coupon

If called early:

- Principal returned
- 10% annual coupon paid pro-rata

Examples:

| Observation | Coupon |
|------------|------------|
| 3M | 2.5% |
| 6M | 5.0% |
| 9M | 7.5% |

---

## Maturity Payoff

If the worst-performing asset finishes above 95%:

```text
Payoff = 95% + Participation Above 95%
```

Otherwise:

```text
Payoff = 95%
```

representing the hard capital floor.

---

# Example Output Metrics

The simulation engine reports:

- Expected payout
- Expected total return
- Expected annualized IRR
- Probability of autocall at 3M
- Probability of autocall at 6M
- Probability of autocall at 9M
- Probability of maturity
- Probability of capital floor scenario

Example:

```text
============================================================
REAL-WORLD PROFILE PERFORMANCE FORECAST (P-MEASURE)
============================================================

Expected Annualized Return: 7.83%

Probability of Automatic Call:

Q1: 28.4%
Q2: 17.1%
Q3: 10.6%

Probability of Maturity:
43.9%

Probability of Capital Floor:
15.2%

============================================================
```

---

# Correlation Diagnostics

The simulator verifies that realized correlations match calibration targets.

Example:

| Pair | Target | Realized |
|--------|--------|--------|
| SPX-SX5E | 0.63 | 0.59 |
| SPX-SMI | 0.54 | 0.52 |
| SX5E-SMI | 0.79 | 0.77 |

This provides a consistency check for the correlation structure used in the Monte Carlo simulation.

---

# Dependencies

Install required packages:

```bash
pip install numpy pandas scipy matplotlib yfinance
```

---

# Applications

This framework can be used for:

- Structured note valuation
- Exotic equity derivatives
- Basket options
- Monte Carlo forecasting
- Portfolio stress testing
- Market risk analysis
- Correlation modeling
- Quantitative finance research
- Heston model experimentation

---

# Future Improvements

Potential extensions include:

- Risk-neutral calibration from option surfaces
- Local-stochastic volatility models
- Jump-diffusion models
- Variance swap pricing
- Barrier options
- Basket option pricing
- GPU acceleration
- Sobol and Quasi-Monte Carlo methods

---

# Disclaimer

This project was developed for educational, research, and quantitative finance applications. It is not investment advice and should not be relied upon as the sole basis for investment decisions.
