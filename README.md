# Multi-Asset Heston Simulator & Structured Note Engine

A Python framework for calibrating, simulating, and analyzing a **multi-asset Heston stochastic volatility model** using historical market data. The project supports calibration directly from Yahoo Finance CSV downloads, Monte Carlo simulation of correlated equity indices, and valuation of structured products such as worst-of autocallable notes.

---

## Features

### Historical Heston Calibration
- Calibrates Heston parameters from historical daily price data:
  - Mean reversion speed (κ)
  - Long-run variance (θ)
  - Volatility of volatility (ξ)
  - Leverage effect (ρ)
  - Initial variance (V₀)
- Supports:
  - Yahoo Finance downloads
  - Preloaded pandas DataFrames
  - Raw CSV files
- Optional MLE refinement after Method-of-Moments calibration

### Multi-Asset Simulation
- Simulates multiple correlated assets simultaneously
- Full stochastic volatility dynamics
- Correlated:
  - Asset returns
  - Variance processes
  - Asset-volatility leverage effects
- Positive semi-definite correlation correction using Higham projection
- Euler-Maruyama discretization with full truncation

### Diagnostics Dashboard
Automatically generates:

- Price path simulations
- Volatility path simulations
- Terminal log-return distributions
- Correlation diagnostics
- Input vs realized correlation comparison

Example output:

![Diagnostics](heston_multi_diagnostics.png)

---

## Model Dynamics

For each asset:

\[
dS_t = S_t \sqrt{V_t} \, dW_S
\]

\[
dV_t = \kappa(\theta - V_t)dt + \xi\sqrt{V_t}dW_V
\]

with

\[
Corr(dW_S,dW_V)=\rho
\]

Cross-asset dependencies are modeled through a block correlation matrix:

\[
C=
\begin{bmatrix}
Corr_{SS} & Corr_{SV} \\
Corr_{SV}^T & Corr_{VV}
\end{bmatrix}
\]

allowing simultaneous correlation between:

- Asset returns
- Variance shocks
- Leverage effects

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

## Calibration Methodology

### Step 1 — Data Loading

Historical adjusted close prices are loaded from:

- Yahoo Finance
- CSV files
- DataFrames

### Step 2 — Return Construction

Two return series are created:

#### 1-Day Returns

\[
r_t^{(1)}=\ln\left(\frac{S_t}{S_{t-1}}\right)
\]

Used for:

- Realized variance estimation
- Leverage effect estimation

#### 2-Day Overlapping Returns

\[
r_t^{(2)}=\ln\left(\frac{S_t}{S_{t-2}}\right)
\]

Used for:

- Cross-asset correlation estimation

This mitigates asynchronous market closing times between US and European indices.

### Step 3 — Realized Variance Proxy

Rolling realized variance:

\[
RV_t = Var(r_t) \times 252
\]

### Step 4 — Method of Moments Calibration

Parameters are estimated directly from historical data:

| Parameter | Estimation |
|------------|------------|
| θ | Long-run variance |
| V₀ | Latest realized variance |
| κ | AR(1) mean reversion |
| ξ | Variance increment volatility |
| ρ | Return-volatility correlation |

### Step 5 — Optional MLE Refinement

Refines estimates using approximate conditional likelihood.

### Step 6 — Correlation Estimation

Produces:

- Return-return correlations
- Variance-variance correlations
- Leverage matrix

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

## Multi-Asset Simulation

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

The repository includes a practical application:

## Worst-of Autocallable Note

Underlying basket:

- S&P 500 (SPX)
- Euro Stoxx 50 (SX5E)
- Swiss Market Index (SMI)

### Observation Dates

- 3 months
- 6 months
- 9 months

### Autocall Condition

Triggered if:

\[
Worst\ Asset \ge 95\%
\]

of initial value.

### Coupon

10% per annum, prorated to observation date.

### Maturity Payoff

If:

\[
Worst\ Asset \ge 95\%
\]

Investor receives upside participation.

Otherwise:

\[
95\%
\]

capital floor applies.

---

# Example Output Metrics

The engine computes:

- Expected payout
- Expected return
- Annualized IRR
- Probability of autocall at:
  - 3 months
  - 6 months
  - 9 months
- Probability of maturity
- Probability of capital floor scenario

Example:

```text
============================================================
REAL-WORLD PROFILE PERFORMANCE FORECAST (P-MEASURE)
============================================================
Expected Annualized Return: 7.83%

Probability of Autocall:
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

| Pair | Input | Realized |
|--------|--------|--------|
| SPX-SX5E | 0.63 | 0.59 |
| SPX-SMI | 0.54 | 0.52 |
| SX5E-SMI | 0.79 | 0.77 |

This serves as a consistency check for the Cholesky-based dependency structure.

---

# Dependencies

Install required packages:

```bash
pip install numpy pandas scipy matplotlib yfinance
```

---

# Applications

This framework can be used for:

- Exotic equity derivatives
- Structured note valuation
- Market risk simulation
- Portfolio stress testing
- Monte Carlo forecasting
- Correlation analysis
- Quantitative finance research
- Heston model experimentation

---

# Future Improvements

Potential extensions include:

- Risk-neutral calibration from option surfaces
- Local-stochastic volatility models
- Jump diffusion extensions
- Variance swap pricing
- Barrier options
- Basket options
- Structured credit applications
- GPU acceleration
- Sobol / Quasi-Monte Carlo simulation

---

# Disclaimer

This project is intended for educational and research purposes. It is not investment advice and should not be used as the sole basis for financial decisions.
