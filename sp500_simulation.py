"""
S&P 500 Monte Carlo Simulation using Geometric Brownian Motion.

This module provides tools for simulating S&P 500 index behavior using
stochastic processes and various probability distributions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, skewnorm, t

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module constants
DEFAULT_INITIAL_PRICE = 4800.0
DEFAULT_ANNUAL_RETURN = 0.10
DEFAULT_ANNUAL_VOLATILITY = 0.18
DEFAULT_PERIODS = 24
TRADING_PERIODS_PER_YEAR = 12
FIGURE_SIZE = (20, 6)


@dataclass(frozen=True)
class SimulationResult:
    """Immutable container for simulation results."""
    
    prices: np.ndarray
    returns: np.ndarray
    rois: np.ndarray
    initial_price: float
    annual_return: float
    annual_volatility: float
    n_periods: int
    
    @property
    def final_price(self) -> float:
        """Get the final simulated price."""
        return float(self.prices[-1])
    
    @property
    def final_return(self) -> float:
        """Get the total return over the simulation period."""
        return float(self.rois[-1] - 1.0)
    
    @property
    def final_return_pct(self) -> float:
        """Get the total return as a percentage."""
        return self.final_return * 100


class PRandomVariable(ABC):
    """Abstract base class for probability distributions used in simulations."""
    
    def __init__(self) -> None:
        self.name: str | None = None
        self.distname: str = "Unknown"
        self.params: dict = {}
        self.mean_sample: float | None = None
        self.mean_theoretical: float | None = None
        self.variance_sample: float | None = None
        self.variance_theoretical: float | None = None
    
    @abstractmethod
    def generate_sample(self, n: int = 1) -> np.ndarray:
        """
        Generate n samples from the distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate (default: 1)
        
        Returns
        -------
        np.ndarray
            Array of n samples from the distribution
        """
        raise NotImplementedError("Subclasses must implement generate_sample")

class TRandomVariable(PRandomVariable):
    """
    Student's t-distribution random variable.
    
    Parameters
    ----------
    df : float
        Degrees of freedom (must be > 0)
    mean : float
        Location parameter (mean of the distribution)
    std_dev : float
        Scale parameter (standard deviation, NOT variance)
    
    Note
    ----
    The scale parameter in scipy.stats.t is the standard deviation,
    not the variance. This is now correctly named std_dev to avoid confusion.
    """
    
    def __init__(self, df: float, mean: float, std_dev: float) -> None:
        super().__init__()
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        self.params = {
            "degrees_of_freedom": df,
            "mean": mean,
            "std_dev": std_dev,
        }
        self.distname = "Student's-t"
        self.mean_theoretical = mean if df > 1 else None
        # Variance only defined for df > 2
        self.variance_theoretical = (std_dev**2 * df / (df - 2)) if df > 2 else None
    
    def generate_sample(self, n: int = 1) -> np.ndarray:
        """Generate n samples from the Student's t-distribution."""
        return t.rvs(
            df=self.params["degrees_of_freedom"],
            loc=self.params["mean"],
            scale=self.params["std_dev"],
            size=n,
        )

class NormalRandomVariable(PRandomVariable):
    """
    Normal (Gaussian) distribution random variable.
    
    Parameters
    ----------
    mean : float
        Mean (μ) of the distribution
    std_dev : float
        Standard deviation (σ) of the distribution (NOT variance)
    
    Note
    ----
    The scale parameter in scipy.stats.norm is the standard deviation (σ),
    not the variance (σ²). This parameter is correctly named std_dev.
    """
    
    def __init__(self, mean: float, std_dev: float) -> None:
        super().__init__()
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        self.params = {"mean": mean, "std_dev": std_dev}
        self.distname = "Normal"
        self.mean_theoretical = mean
        self.mean_sample = mean
        self.variance_theoretical = std_dev ** 2
        self.variance_sample = std_dev ** 2
    
    def generate_sample(self, n: int = 1) -> np.ndarray:
        """Generate n samples from the Normal distribution."""
        return norm.rvs(
            loc=self.params["mean"],
            scale=self.params["std_dev"],
            size=n,
        )

class NormalSkewedRandomVariable(PRandomVariable):
    """
    Skew-Normal distribution random variable.
    
    Parameters
    ----------
    mean : float
        Location parameter (loc) of the distribution
    std_dev : float
        Scale parameter (standard deviation, NOT variance)
    alpha : float
        Shape parameter controlling skewness.
        - alpha > 0: right-skewed (positive skew)
        - alpha < 0: left-skewed (negative skew)
        - alpha = 0: reduces to normal distribution
    
    Note
    ----
    The actual mean of a skew-normal is NOT the loc parameter.
    The theoretical mean depends on alpha and requires adjustment.
    """
    
    def __init__(self, mean: float, std_dev: float, alpha: float) -> None:
        super().__init__()
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        self.distname = "Skew-Normal"
        self.params = {"mean": mean, "std_dev": std_dev, "alpha": alpha}
        
        # Compute theoretical moments for skew-normal
        delta = alpha / np.sqrt(1 + alpha**2)
        self.mean_theoretical = mean + std_dev * delta * np.sqrt(2 / np.pi)
        self.variance_theoretical = (std_dev**2) * (1 - 2 * delta**2 / np.pi)
    
    def generate_sample(self, n: int = 1) -> np.ndarray:
        """Generate n samples from the Skew-Normal distribution."""
        return skewnorm.rvs(
            loc=self.params["mean"],
            scale=self.params["std_dev"],
            a=self.params["alpha"],
            size=n,
        )



class SP500Index:
    """
    S&P 500 Index Simulation using Geometric Brownian Motion.
    
    This simulator models the S&P 500 index using GBM, which assumes that
    log-returns follow a normal distribution with constant drift and volatility.
    
    Historical S&P 500 Statistics (approximate):
    - Average annual return: ~10% (arithmetic mean)
    - Annual volatility (std dev): ~18%
    - Distribution: Approximately log-normal for prices
    
    Mathematical Model:
    -------------------
    GBM: dS = μS dt + σS dW
    
    Discrete approximation using log-returns:
    S(t+dt) = S(t) * exp((μ - 0.5σ²)dt + σ√dt * Z)
    
    where Z ~ N(0,1)
    
    Parameters
    ----------
    initial_price : float
        Starting index level (default ~4800 matches 2024 levels)
    seed : int, optional
        Random seed for reproducibility. If None, results will vary.
    
    Examples
    --------
    >>> sim = SP500Index(initial_price=4800, seed=42)
    >>> result = sim.simulate()
    >>> print(f"Final return: {result.final_return_pct:.2f}%")
    """
    
    def __init__(
        self,
        initial_price: float = DEFAULT_INITIAL_PRICE,
        seed: int | None = None,
    ) -> None:
        """Initialize S&P 500 simulation with optional reproducibility seed."""
        # Input validation
        if initial_price <= 0:
            raise ValueError("Initial price must be positive")
        
        self.initial_price = initial_price
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Store last simulation result for backward compatibility
        self._last_result: SimulationResult | None = None
    
    def simulate(
        self,
        n_periods: int = DEFAULT_PERIODS,
        annual_return: float = DEFAULT_ANNUAL_RETURN,
        annual_volatility: float = DEFAULT_ANNUAL_VOLATILITY,
        use_log_returns: bool = True,
    ) -> SimulationResult:
        """
        Run S&P 500 Monte Carlo simulation using Geometric Brownian Motion.
        
        This method returns an immutable SimulationResult without mutating
        the simulator's internal state (except for caching the last result).
        
        Parameters
        ----------
        n_periods : int
            Number of monthly periods to simulate (default: 24 = 2 years)
        annual_return : float
            Expected annual return as decimal (default: 0.10 = 10%)
        annual_volatility : float
            Annual volatility/standard deviation (default: 0.18 = 18%)
        use_log_returns : bool
            If True, use proper GBM with log-returns (recommended).
            If False, use arithmetic returns (simpler but less accurate).
        
        Returns
        -------
        SimulationResult
            Immutable dataclass containing prices, returns, and ROIs
        
        Raises
        ------
        ValueError
            If parameters are outside valid ranges
        
        Notes
        -----
        The GBM model uses the risk-neutral drift adjustment:
        
        drift = (μ - 0.5σ²) * dt
        
        This ensures that E[S(T)] = S(0) * exp(μT), matching the expected
        arithmetic return over the simulation period.
        """
        # Input validation
        if n_periods <= 0:
            raise ValueError("n_periods must be a positive integer")
        if annual_volatility <= 0:
            raise ValueError("annual_volatility must be positive")
        if annual_return <= -1.0:
            raise ValueError("annual_return must be greater than -100%")
        
        # Convert annual parameters to monthly
        dt = 1.0 / TRADING_PERIODS_PER_YEAR
        monthly_volatility = annual_volatility * np.sqrt(dt)
        
        if use_log_returns:
            # Proper GBM: drift adjustment for log-returns
            # E[log(S_t/S_0)] = (μ - 0.5σ²)t, so drift in log-space is reduced
            drift = (annual_return - 0.5 * annual_volatility**2) * dt
        else:
            # Simple arithmetic returns (less accurate for large volatility)
            drift = annual_return * dt
        
        # Log simulation parameters
        expected_final_return = (1 + annual_return) ** (n_periods / TRADING_PERIODS_PER_YEAR) - 1
        logger.info("Simulation Parameters:")
        logger.info(f"  Initial Price: ${self.initial_price:,.2f}")
        logger.info(f"  Annual Return Target: {annual_return * 100:.1f}%")
        logger.info(f"  Annual Volatility: {annual_volatility * 100:.1f}%")
        logger.info(f"  Monthly Drift: {drift * 100:.4f}%")
        logger.info(f"  Monthly Volatility: {monthly_volatility * 100:.2f}%")
        logger.info(f"  Periods: {n_periods} months")
        logger.info(f"  Expected final return: ~{expected_final_return:.1%}")
        logger.info(f"  Random Seed: {self.seed}")
        
        # Generate random shocks
        z = self.rng.standard_normal(n_periods)
        
        if use_log_returns:
            # Vectorized GBM simulation using log-returns
            log_returns = drift + monthly_volatility * z
            cumulative_log_returns = np.cumsum(log_returns)
            prices = self.initial_price * np.exp(cumulative_log_returns)
            prices = np.insert(prices, 0, self.initial_price)
            returns = log_returns
        else:
            # Iterative arithmetic returns (for comparison)
            returns = drift + monthly_volatility * z
            prices = [self.initial_price]
            for r in returns:
                prices.append(prices[-1] * (1 + r))
            prices = np.array(prices)
        
        # Calculate ROIs
        rois = prices / self.initial_price
        
        # Create immutable result
        result = SimulationResult(
            prices=prices,
            returns=np.array(returns),
            rois=rois,
            initial_price=self.initial_price,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            n_periods=n_periods,
        )
        
        # Cache for backward compatibility with view_results()
        self._last_result = result
        
        return result
    
    # Backward compatibility alias
    def simulate_index(
        self,
        var_class: Type[PRandomVariable] | None = None,
        n_periods: int = DEFAULT_PERIODS,
        annual_return: float = DEFAULT_ANNUAL_RETURN,
        annual_volatility: float = DEFAULT_ANNUAL_VOLATILITY,
    ) -> list:
        """
        Legacy method for backward compatibility.
        
        .. deprecated::
            Use simulate() instead, which returns an immutable SimulationResult.
        
        Parameters
        ----------
        var_class : Type[PRandomVariable], optional
            Ignored in new implementation. Kept for API compatibility.
        n_periods : int
            Number of periods to simulate
        annual_return : float
            Expected annual return
        annual_volatility : float
            Annual volatility
        
        Returns
        -------
        list
            Simulated prices (for backward compatibility)
        """
        import warnings
        warnings.warn(
            "simulate_index() is deprecated. Use simulate() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        if var_class is not None:
            logger.warning(
                "var_class parameter is ignored in the new implementation. "
                "The simulator now uses proper GBM with log-normal returns."
            )
        
        result = self.simulate(
            n_periods=n_periods,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
        )
        
        # Backward compatibility: set instance attributes
        self.prices = list(result.prices)
        self.returns = list(result.returns)
        self.rois = list(result.rois)
        self.current_price = result.final_price
        self.roi = result.rois[-1]
        
        return self.prices
    
    def view_results(self, result: SimulationResult | None = None) -> None:
        """
        Visualize simulation results with two plots.
        
        Creates a figure with:
        1. Index level over time
        2. Cumulative return over time
        
        Parameters
        ----------
        result : SimulationResult, optional
            The simulation result to visualize. If None, uses the last
            simulation result cached in self._last_result.
        
        Raises
        ------
        ValueError
            If no result is provided and no simulation has been run.
        """
        if result is None:
            result = self._last_result
        
        if result is None:
            raise ValueError(
                "No simulation result available. "
                "Run simulate() first or pass a SimulationResult."
            )
        
        fig, ax = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        fig.suptitle(
            "S&P 500 Index Simulation (Geometric Brownian Motion)",
            fontsize=16,
            fontweight="bold",
        )
        
        T = np.arange(len(result.prices))
        
        # Plot 1: Price evolution
        ax[0].plot(
            T, result.prices,
            color="#0066CC",
            label="Simulated Index",
            linewidth=2.5,
        )
        ax[0].axhline(
            result.initial_price,
            color="#00CC66",
            linestyle="--",
            label=f"Initial: ${result.initial_price:,.0f}",
            linewidth=1.5,
        )
        ax[0].axhline(
            result.final_price,
            color="#CC0066",
            linestyle="--",
            label=f"Final: ${result.final_price:,.0f}",
            linewidth=1.5,
        )
        ax[0].set_title(
            f"Index Level Over Time\n"
            f"(Monthly periods, μ={result.annual_return*100:.0f}% ann., "
            f"σ={result.annual_volatility*100:.0f}% ann.)",
            fontsize=12,
        )
        ax[0].set_xlabel("Period (Months)", fontsize=11)
        ax[0].set_ylabel("Index Level", fontsize=11)
        ax[0].legend(loc="best", fontsize=10)
        ax[0].grid(True, alpha=0.3, linestyle=":")
        
        # Plot 2: ROI evolution
        returns_pct = (result.rois - 1) * 100
        ax[1].plot(
            T, returns_pct,
            color="#CC6600",
            label="Cumulative Return",
            linewidth=2.5,
        )
        ax[1].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax[1].fill_between(
            T, returns_pct, 0,
            where=returns_pct >= 0,
            color="#00CC66",
            alpha=0.3,
            label="Gains",
        )
        ax[1].fill_between(
            T, returns_pct, 0,
            where=returns_pct < 0,
            color="#CC0066",
            alpha=0.3,
            label="Losses",
        )
        ax[1].set_title(
            f"Cumulative Return ({result.final_return_pct:+.1f}% total)",
            fontsize=12,
        )
        ax[1].set_ylabel("Return (%)", fontsize=11)
        ax[1].set_xlabel("Period (Months)", fontsize=11)
        ax[1].legend(loc="best", fontsize=10)
        ax[1].grid(True, alpha=0.3, linestyle=":")
        
        plt.tight_layout()
        plt.show()
    
    def run_monte_carlo(
        self,
        n_simulations: int = 1000,
        n_periods: int = DEFAULT_PERIODS,
        annual_return: float = DEFAULT_ANNUAL_RETURN,
        annual_volatility: float = DEFAULT_ANNUAL_VOLATILITY,
    ) -> np.ndarray:
        """
        Run multiple simulations for Monte Carlo analysis.
        
        Parameters
        ----------
        n_simulations : int
            Number of simulation paths to generate (default: 1000)
        n_periods : int
            Number of periods per simulation
        annual_return : float
            Expected annual return
        annual_volatility : float
            Annual volatility
        
        Returns
        -------
        np.ndarray
            Array of shape (n_simulations, n_periods + 1) containing
            all simulated price paths.
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        dt = 1.0 / TRADING_PERIODS_PER_YEAR
        drift = (annual_return - 0.5 * annual_volatility**2) * dt
        monthly_vol = annual_volatility * np.sqrt(dt)
        
        # Vectorized generation of all paths
        z = self.rng.standard_normal((n_simulations, n_periods))
        log_returns = drift + monthly_vol * z
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        
        # Build price paths
        paths = self.initial_price * np.exp(cumulative_log_returns)
        paths = np.insert(paths, 0, self.initial_price, axis=1)
        
        logger.info(f"Monte Carlo complete. Final returns statistics:")
        final_returns = (paths[:, -1] / self.initial_price - 1) * 100
        logger.info(f"  Mean: {np.mean(final_returns):.2f}%")
        logger.info(f"  Std: {np.std(final_returns):.2f}%")
        logger.info(f"  5th percentile: {np.percentile(final_returns, 5):.2f}%")
        logger.info(f"  95th percentile: {np.percentile(final_returns, 95):.2f}%")
        
        return paths
