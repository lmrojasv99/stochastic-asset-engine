import numpy as np
import pandas as pd
from random import choice, choices, sample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, norm, skewnorm
from abc import ABC

class PRandomVariable(ABC):  
    def __init__(self):
        self.name: str = None
        self.params: dict = None
        self.mean_sample: float = None
        self.mean_theoretical: float = None
        self.variance_sample: float = None 
        self.variance_theoretical: float = None
    
    def generate_sample(self, n: int = 1) -> list:
        pass

class TRandomVariable(PRandomVariable):
    def __init__(self, df, mean, variance):
        super().__init__()
        self.params = dict(
            degrees_of_freedom=df,
            mean=mean,
            scale=variance
        )
        self.distname = "Student's-t"
    
    def generate_sample(self, n: int = 1) -> list:
        return t.rvs(
            df=self.params["degrees_of_freedom"],
            loc=self.params["mean"],
            scale=self.params["scale"],
            size=n,
        )

class NormalRandomVariable(PRandomVariable):
    def __init__(self, mean: float, variance: float):
        super().__init__()
        self.mean_sample = mean
        self.variance_sample = variance
        self.params = {"mean": mean, "variance": variance}
        self.distname = "Normal"
    
    def generate_sample(self, n: int = 1) -> list:
        return norm.rvs(
            loc=self.params["mean"],
            scale=self.params["variance"],
            size=n
        )

class NormalSkewedRandomVariable(PRandomVariable):
    def __init__(self, mean: float, variance: float, alpha: float):
        super().__init__()
        self.distname = "Normal Skewed"
        self.mean_sample = mean
        self.variance_sample = variance
        self.alpha = alpha
        self.params = {"mean": mean, "variance": variance, "alpha": alpha}
    
    def generate_sample(self, n: int = 1) -> list:
        return skewnorm.rvs(
            loc=self.params["mean"],
            scale=self.params["variance"],
            a=self.params["alpha"],
            size=n,
        )



class SP500Index:
    """
    S&P 500 Index Simulation using Geometric Brownian Motion
    
    Historical S&P 500 Statistics:
    - Average annual return: ~10%
    - Annual volatility (std dev): ~18%
    - Distribution: Approximately log-normal
    """
    
    def __init__(self, initial_price: float = 4800.0):
        """
        Initialize S&P 500 simulation
        
        Parameters:
        -----------
        initial_price : float
            Starting index level (default ~4800 matches 2024 levels)
        """
        self.return_dist = None
        self.initial_price = initial_price
        self.current_price = initial_price
        self.return_ = 1.0
        self.roi = 1.0

        self.rois = [] 
        self.prices = []
        self.returns = []
    
    def simulate_index(self, var_class, n_periods=24, annual_return=0.10, annual_volatility=0.18):
        """
        Simulate S&P 500 using percentage returns with volatility adjustment
        
        Parameters:
        -----------
        var_class : class
            Random variable class (NormalRandomVariable recommended)
        n_periods : int
            Number of periods to simulate (default 24 for 2 years monthly)
        annual_return : float
            Expected annual return (default 0.10 = 10%)
        annual_volatility : float
            Annual volatility/standard deviation (default 0.18 = 18%)
        
        Returns:
        --------
        list : simulated prices
        """
        
        monthly_return = annual_return / 12
        monthly_volatility = annual_volatility / np.sqrt(12)
        
        adjusted_monthly_return = monthly_return + 0.5 * (monthly_volatility ** 2)
        
        print(f"Simulation Parameters:")
        print(f"  Annual Return Target: {annual_return*100:.1f}%")
        print(f"  Annual Volatility: {annual_volatility*100:.1f}%")
        print(f"  Monthly Return (adjusted): {adjusted_monthly_return*100:.3f}%")
        print(f"  Monthly Volatility: {monthly_volatility*100:.2f}%")
        print(f"  Periods: {n_periods} months")
        print(f"  Expected final return: ~{(1 + annual_return)**(n_periods/12) - 1:.1%}\n")
        
        self.return_dist = var_class(adjusted_monthly_return, monthly_volatility)
        self.current_price = self.initial_price
        self.prices = [self.initial_price]
        self.rois = [1.0]
        self.returns = []
        
        for t in range(n_periods):
            
            period_return = self.return_dist.generate_sample(n=1)[0]
            self.returns.append(period_return)
            
            
            self.current_price = self.current_price * (1 + period_return)
            self.roi = self.current_price / self.initial_price 
            
            self.rois.append(self.roi)
            self.prices.append(self.current_price)
        
        return self.prices

    def view_results(self):
        """
        Visualize simulation results with two plots:
        1. Index level over time
        2. Cumulative return over time
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle("S&P 500 Index Simulation (Geometric Brownian Motion)", fontsize=16, fontweight='bold')
        T = np.arange(start=0, stop=len(self.prices), step=1)
        
        # Plot 1: Price evolution
        ax[0].plot(T, self.prices, color="#0066CC", label="Simulated Index", linewidth=2.5)
        ax[0].axhline(self.initial_price, color='#00CC66', linestyle='--', 
                     label=f'Initial: {self.initial_price:.0f}', linewidth=1.5)
        ax[0].axhline(self.current_price, color='#CC0066', linestyle='--', 
                     label=f'Final: {self.current_price:.0f}', linewidth=1.5)   
        ax[0].set_title(f"Index Level Over Time\n(Monthly periods, μ=10% ann., σ=18% ann.)", fontsize=12)
        ax[0].set_xlabel("Period (Months)", fontsize=11)   
        ax[0].set_ylabel("Index Level", fontsize=11)
        ax[0].legend(loc='best', fontsize=10)
        ax[0].grid(True, alpha=0.3, linestyle=':')

        # Plot 2: ROI evolution
        returns_pct = [(r-1)*100 for r in self.rois]
        ax[1].plot(T, returns_pct, color="#CC6600", label="Cumulative Return", linewidth=2.5)
        ax[1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax[1].fill_between(T, returns_pct, 0, where=[r >= 0 for r in returns_pct], 
                          color='#00CC66', alpha=0.3, label='Gains')
        ax[1].fill_between(T, returns_pct, 0, where=[r < 0 for r in returns_pct], 
                          color='#CC0066', alpha=0.3, label='Losses')
        ax[1].set_title(f"Cumulative Return", fontsize=12)
        ax[1].set_ylabel("Return (%)", fontsize=11)
        ax[1].set_xlabel("Period (Months)", fontsize=11)
        ax[1].legend(loc='best', fontsize=10)
        ax[1].grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.show()

