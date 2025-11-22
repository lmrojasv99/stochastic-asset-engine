# Stochastic Asset Pricing & Monte Carlo Simulation Engine

## Overview

This repository hosts a robust, object-oriented simulation engine for modeling financial asset prices, specifically calibrated to the S&P 500 Index. Built with a focus on quantitative rigor and software engineering best practices, this project bridges the gap between theoretical stochastic calculus and practical Python implementation.

The core module demonstrates a flexible framework for Monte Carlo simulations, allowing researchers to interchange underlying probability distributions (Gaussian, Student's-t, Skewed) to better capture real-world market phenomena like fat tails and asymmetry.

## Academic Rigor: Quantitative Foundations

### 1. Probability & Statistical Distributions
Financial returns rarely follow a perfect Normal distribution. This engine implements a polymorphic design to model returns using various statistical profiles:

*   **Gaussian (Normal) Distribution**: The standard assumption in the Black-Scholes-Merton framework. It assumes returns are symmetric and mesokurtic.
    *   *Use case*: Baseline modeling and theoretical comparisons.
*   **Student's t-Distribution**: A leptokurtic distribution that captures "fat tails"—extreme market events that occur more frequently than a Normal distribution predicts.
    *   *Relevance*: Critical for risk management and stress testing (e.g., modeling Value at Risk).
*   **Skewed Normal Distribution**: Introduces a shape parameter $\alpha$ to model asymmetry in returns.
    *   *Relevance*: Equity markets often exhibit negative skew (small gains are frequent, large crashes are rare but devastating).

### 2. Stochastic Calculus & Geometric Brownian Motion
The simulation is grounded in the dynamics of **Geometric Brownian Motion (GBM)**, the standard model for stock price evolution. The continuous-time Stochastic Differential Equation (SDE) is given by:

$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$

Where:
*   $S_t$ is the asset price.
*   $\mu$ is the drift (expected return).
*   $\sigma$ is the volatility.
*   $W_t$ is a Wiener process (Brownian motion).

**Discrete Implementation**:
To simulate this computationally, we discretize the process. The simulation models the discrete percentage return $r_t$ over a time step $\Delta t$:

$$ S_{t+1} = S_t \times (1 + r_t) $$

Where $r_t$ is a random variable drawn from one of the supported distributions. The engine includes a drift adjustment term ($0.5\sigma^2$) to align the arithmetic simulation with the expected geometric growth of the asset.

### 3. Volatility Modeling
The engine rigorously handles time-scaling of volatility, a crucial concept in quantitative finance. It implements the "Square Root of Time" rule to convert annual parameters into period-specific inputs:

$$ \sigma_{monthly} = \frac{\sigma_{annual}}{\sqrt{12}} $$

This ensures that the risk metrics remain consistent regardless of the simulation frequency (monthly, daily, etc.).

## Software Architecture: Modular & Scalable

The codebase utilizes advanced Object-Oriented Programming (OOP) principles to ensure maintainability and extensibility.

### 1. Abstract Base Classes (ABC) & Polymorphism
The system uses an Abstract Base Class `PRandomVariable` to define the interface for all statistical distributions.
*   **Benefit**: This adheres to the **Open/Closed Principle** (SOLID). New distributions (e.g., Cauchy, Levy) can be added by simply inheriting from `PRandomVariable` without modifying the core simulation logic.

### 2. Dependency Injection
The `SP500Index` simulation class does not hardcode the distribution logic. Instead, it accepts a distribution class (`var_class`) as an argument at runtime.
*   **Benefit**: This allows for rapid A/B testing of different hypotheses (e.g., "How does the S&P 500 behave under heavy-tailed regimes vs. normal regimes?") without changing the simulation code.

### 3. Scalability
*   **Financial Instruments**: The `SP500Index` class can be easily refactored into a generic `Asset` class, allowing the system to simulate Commodities, FX pairs, or Crypto assets by simply changing the drift/volatility parameters.
*   **Vectorization**: The implementation leverages `numpy` for vectorized sampling (`rvs` methods), ensuring the simulation remains performant even when scaling to millions of iterations.

## Project Goals

This project is designed to serve as a foundational tool for quantitative research.

*   **Specific**: Develop a Python-based Monte Carlo engine that simulates S&P 500 price paths using Normal, Student's-t, and Skewed distributions to analyze tail risk.
*   **Measurable**: The system must output price vectors and cumulative ROI metrics, visualized through dual-axis plotting (Price vs. Time, ROI vs. Time) for a 5-year horizon.
*   **Achievable**: Implemented using `scipy.stats` for reliable statistical generation and `pandas`/`numpy` for efficient data handling.
*   **Relevant**: Directly addresses the limitations of the Random Walk Hypothesis by allowing for non-Gaussian return modeling, a key requirement in modern quantitative finance and derivatives pricing.
*   **Time-bound**: The current release (v1.0) is fully functional, with the simulation running typically in under 1 second for standard 60-month projections.

## Getting Started

### Prerequisites
*   Python 3.8+
*   `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

### Usage Example

```python
from sp500_simulation import SP500Index, NormalRandomVariable

# Initialize the asset
sp500 = SP500Index(initial_price=4800.0)

# Run simulation using Normal Distribution
prices = sp500.simulate_index(
    var_class=NormalRandomVariable,
    n_periods=60,            # 5 Years
    annual_return=0.10,      # 10% Drift
    annual_volatility=0.18   # 18% Volatility
)

# Visualize
sp500.view_results()
```

---
*Author: Luis Rojas | Quantitative Developer & Researcher*