# Stochastic Asset Pricing & Monte Carlo Simulation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

A production-grade, object-oriented Monte Carlo simulation engine for modeling financial asset prices using **Geometric Brownian Motion (GBM)**. Specifically calibrated to the S&P 500 Index, this engine bridges theoretical stochastic calculus with practical Python implementation.

### Key Features

- **Mathematically Rigorous** — Proper GBM implementation with Itô drift adjustment
- **Reproducible Results** — Seed-controlled random number generation using NumPy's modern `Generator` API
- **Immutable Results** — Frozen dataclasses prevent accidental state mutation
- **Production Logging** — Structured logging instead of print statements
- **Vectorized Monte Carlo** — Run 10,000+ simulations efficiently
- **Extensible Architecture** — Abstract base classes for custom distributions

---

## Mathematical Foundations

### 1. Geometric Brownian Motion (GBM)

The simulation implements the standard model for stock price evolution. The continuous-time **Stochastic Differential Equation (SDE)** is:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Where:
| Symbol | Description |
|--------|-------------|
| $S_t$ | Asset price at time $t$ |
| $\mu$ | Drift (expected annual return) |
| $\sigma$ | Volatility (annual standard deviation) |
| $W_t$ | Wiener process (standard Brownian motion) |

### 2. Discrete Log-Return Implementation

The engine uses the **exact solution** to the GBM SDE via log-returns:

$$S_{t+\Delta t} = S_t \cdot \exp\left[\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right]$$

Where $Z \sim \mathcal{N}(0, 1)$.

> **Why the $-\frac{1}{2}\sigma^2$ term?**  
> This is the **Itô correction** (Jensen's inequality adjustment). It ensures that:
> $$\mathbb{E}[S_T] = S_0 \cdot e^{\mu T}$$
> Without this term, the expected return would be biased upward.

### 3. Volatility Time-Scaling

The engine rigorously implements the **Square Root of Time** rule:

$$\sigma_{\text{monthly}} = \frac{\sigma_{\text{annual}}}{\sqrt{12}}$$

$$\text{drift}_{\text{monthly}} = \frac{\mu - \frac{1}{2}\sigma^2}{12}$$

### 4. Supported Probability Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| **Normal** | $\mu$, $\sigma$ | Baseline GBM, Black-Scholes framework |
| **Student's t** | $\nu$, $\mu$, $\sigma$ | Fat tails, extreme events, VaR modeling |
| **Skew-Normal** | $\mu$, $\sigma$, $\alpha$ | Asymmetric returns, negative skew in equities |

---

## Software Architecture

### Design Principles

```
┌─────────────────────────────────────────────────────────────┐
│                      SP500Index                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   simulate()    │  │ run_monte_carlo │                  │
│  │  (single path)  │  │ (vectorized)    │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────────┐               │
│  │         SimulationResult (frozen)        │               │
│  │  prices | returns | rois | metadata      │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              PRandomVariable (ABC)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │NormalRandom  │ │ TRandom      │ │NormalSkewedRandom  │  │
│  │Variable      │ │ Variable     │ │Variable            │  │
│  └──────────────┘ └──────────────┘ └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Implementation | Benefit |
|---------|----------------|---------|
| **Abstract Factory** | `PRandomVariable` ABC with `@abstractmethod` | Open/Closed Principle — add new distributions without modifying core logic |
| **Immutable Value Object** | `SimulationResult` frozen dataclass | Thread-safe, prevents accidental mutation |
| **Dependency Injection** | Distribution classes passed at runtime | A/B testing different market hypotheses |
| **Modern RNG** | `np.random.default_rng(seed)` | Reproducibility + better statistical properties |

---

## Installation

### Prerequisites

- Python 3.10+
- Dependencies: `numpy`, `matplotlib`, `seaborn`, `scipy`

```bash
pip install numpy matplotlib seaborn scipy
```

---

## Quick Start

### Basic Simulation

```python
from sp500_simulation import SP500Index

# Initialize with seed for reproducibility
sim = SP500Index(initial_price=4800.0, seed=42)

# Run 2-year monthly simulation
result = sim.simulate(
    n_periods=24,           # 24 months
    annual_return=0.10,     # 10% expected return
    annual_volatility=0.18  # 18% volatility
)

# Access results
print(f"Final Price: ${result.final_price:,.2f}")
print(f"Total Return: {result.final_return_pct:+.2f}%")

# Visualize
sim.view_results(result)
```

### Monte Carlo Analysis

```python
from sp500_simulation import SP500Index

sim = SP500Index(initial_price=4800.0, seed=123)

# Run 10,000 simulations
paths = sim.run_monte_carlo(
    n_simulations=10000,
    n_periods=24,
    annual_return=0.10,
    annual_volatility=0.18
)

# paths.shape = (10000, 25)  # 10k paths × 25 time points
```

### Using Alternative Distributions

```python
from sp500_simulation import (
    NormalRandomVariable,
    TRandomVariable,
    NormalSkewedRandomVariable
)

# Standard Normal
normal_dist = NormalRandomVariable(mean=0.008, std_dev=0.05)
samples = normal_dist.generate_sample(n=1000)

# Student's t (fat tails) - df=5 for heavy tails
t_dist = TRandomVariable(df=5, mean=0.008, std_dev=0.05)
samples = t_dist.generate_sample(n=1000)

# Skew-Normal (negative skew for crash modeling)
skew_dist = NormalSkewedRandomVariable(mean=0.008, std_dev=0.05, alpha=-2)
samples = skew_dist.generate_sample(n=1000)
```

---

## API Reference

### `SP500Index`

```python
class SP500Index:
    def __init__(
        self,
        initial_price: float = 4800.0,
        seed: int | None = None
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_price` | `float` | `4800.0` | Starting index level |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

#### Methods

##### `simulate()`

```python
def simulate(
    self,
    n_periods: int = 24,
    annual_return: float = 0.10,
    annual_volatility: float = 0.18,
    use_log_returns: bool = True
) -> SimulationResult
```

##### `run_monte_carlo()`

```python
def run_monte_carlo(
    self,
    n_simulations: int = 1000,
    n_periods: int = 24,
    annual_return: float = 0.10,
    annual_volatility: float = 0.18
) -> np.ndarray  # shape: (n_simulations, n_periods + 1)
```

##### `view_results()`

```python
def view_results(self, result: SimulationResult | None = None) -> None
```

---

### `SimulationResult`

Immutable container for simulation outputs.

```python
@dataclass(frozen=True)
class SimulationResult:
    prices: np.ndarray          # Price path [S_0, S_1, ..., S_T]
    returns: np.ndarray         # Period returns [r_1, r_2, ..., r_T]
    rois: np.ndarray            # Cumulative ROI [1.0, ROI_1, ..., ROI_T]
    initial_price: float
    annual_return: float
    annual_volatility: float
    n_periods: int
    
    # Properties
    final_price: float          # S_T
    final_return: float         # (S_T / S_0) - 1
    final_return_pct: float     # final_return × 100
```

---

### Distribution Classes

All distributions inherit from `PRandomVariable` and implement:

```python
def generate_sample(self, n: int = 1) -> np.ndarray
```

| Class | Constructor | Notes |
|-------|-------------|-------|
| `NormalRandomVariable` | `(mean, std_dev)` | Standard Gaussian |
| `TRandomVariable` | `(df, mean, std_dev)` | `df > 2` for defined variance |
| `NormalSkewedRandomVariable` | `(mean, std_dev, alpha)` | `alpha < 0` = left skew |

> **Important**: The `std_dev` parameter is the **standard deviation**, not variance. This matches `scipy.stats` conventions.

---

## Configuration

### Module Constants

```python
DEFAULT_INITIAL_PRICE = 4800.0      # 2024 S&P 500 level
DEFAULT_ANNUAL_RETURN = 0.10        # 10% historical average
DEFAULT_ANNUAL_VOLATILITY = 0.18    # 18% historical volatility
DEFAULT_PERIODS = 24                # 2 years monthly
TRADING_PERIODS_PER_YEAR = 12       # Monthly granularity
```

### Logging

The module uses Python's `logging` module. Configure verbosity:

```python
import logging
logging.getLogger("sp500_simulation").setLevel(logging.WARNING)  # Suppress INFO
```

---

## Example Output

```
INFO:sp500_simulation:Simulation Parameters:
INFO:sp500_simulation:  Initial Price: $4,800.00
INFO:sp500_simulation:  Annual Return Target: 10.0%
INFO:sp500_simulation:  Annual Volatility: 18.0%
INFO:sp500_simulation:  Monthly Drift: 0.6983%
INFO:sp500_simulation:  Monthly Volatility: 5.20%
INFO:sp500_simulation:  Periods: 24 months
INFO:sp500_simulation:  Expected final return: ~21.0%
INFO:sp500_simulation:  Random Seed: 42
```

---

## Backward Compatibility

The legacy `simulate_index()` method is preserved but deprecated:

```python
# DEPRECATED - emits DeprecationWarning
prices = sp500.simulate_index(
    var_class=NormalRandomVariable,  # Ignored in new implementation
    n_periods=60,
    annual_return=0.10,
    annual_volatility=0.18
)

# Recommended
result = sp500.simulate(n_periods=60, annual_return=0.10, annual_volatility=0.18)
```

---

## Roadmap

- [ ] Add daily/weekly frequency options
- [ ] Implement jump-diffusion models (Merton)
- [ ] Add stochastic volatility (Heston model)
- [ ] Export results to pandas DataFrame
- [ ] Add confidence interval plotting for Monte Carlo
- [ ] Support for multiple correlated assets (portfolio simulation)

---

## References

1. Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
2. Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
3. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.


---

*Author: Luis Rojas | Quantitative Developer & Researcher*
