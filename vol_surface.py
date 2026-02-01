"""
Quantitative Volatility Analysis

Volatility Surface Modeling: Scrapes option chains, constructs an Implied Volatility (IV) 
surface using RBF interpolation, and visualizes the skew/term structure.

Author: ben-245dev

Dependencies: yfinance, pandas, numpy, matplotlib, scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

class VolatilitySurface:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.data = None
        self.spot_price = None

    def fetch_data(self, max_expiries=12):
        """
        Scrape option chains for multiple expiration dates.
        """
        # Get current spot price
        try:
            hist = self.ticker.history(period="1d")
            self.spot_price = hist['Close'].iloc[-1]
            print(f"Spot Price for {self.ticker_symbol}: ${self.spot_price:.2f}")
        except Exception as e:
            print(f"Error fetching spot price: {e}")
            return

        options_data = []
        expirations = self.ticker.options[:max_expiries]
        
        print(f"Fetching data for {len(expirations)} expirations...")

        for expiry in expirations:
            try:
                # Calculate years to expiration (T)
                days_to_expiry = (pd.to_datetime(expiry) - datetime.now()).days
                T = days_to_expiry / 365.0
                
                # Filter out expired or extremely near-term options (< 1 week)
                if T < 0.02: continue

                # Fetch Option Chain
                chain = self.ticker.option_chain(expiry)
                
                # Combine Calls and Puts (or focus on one). 
                calls = chain.calls
                puts = chain.puts
                
                # Simple filtering: Calls where Strike >= Spot, Puts where Strike <= Spot
                # This focuses on Out-The-Money (OTM) options which are more liquid/relevant for IV
                relevant_calls = calls[calls['strike'] >= self.spot_price]
                relevant_puts = puts[puts['strike'] <= self.spot_price]
                
                combined = pd.concat([relevant_calls, relevant_puts])
                
                # Filter noise: Volume > 0 or Bid > 0 implies liquidity
                liquidity_mask = (combined['bid'] > 0) & (combined['impliedVolatility'] > 0)
                clean_data = combined[liquidity_mask][['strike', 'impliedVolatility']].copy()
                clean_data['T'] = T
                
                options_data.append(clean_data)
            except Exception as e:
                print(f"Skipping expiry {expiry}: {e}")
                continue

        if not options_data:
            print("No valid options data found.")
            return

        self.data = pd.concat(options_data)
        
        # Filter extreme outliers in Strike (e.g., only keep 70% to 130% of spot)
        self.data = self.data[
            (self.data['strike'] > self.spot_price * 0.7) & 
            (self.data['strike'] < self.spot_price * 1.3)
        ]
        print(f"Data processing complete. {len(self.data)} data points ready.")

    def plot_surface(self):
        """
        Plots the 3D Volatility Surface using Radial Basis Function interpolation.
        """
        if self.data is None or self.data.empty:
            print("Run fetch_data() first.")
            return

        x = self.data['strike']
        y = self.data['T']
        z = self.data['impliedVolatility']

        # Create a meshgrid for the plot
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate Z values (IV) on the grid
        rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
        Z = rbf(X, Y)

        # Plotting
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
    
        ax.set_title(f"Implied Volatility Surface: {self.ticker_symbol}", fontsize=15)
        ax.set_xlabel('Strike Price ($)', fontsize=12)
        ax.set_ylabel('Time to Expiry (Years)', fontsize=12)
        ax.set_zlabel('Implied Volatility', fontsize=12)

        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Rotate
        ax.view_init(elev=30, azim=-120)
        
        plt.show()

# Test
if __name__ == "__main__":
    # Choose a liquid ticker (e.g., SPY, AAPL, TSLA, NVDA)
    ticker = "SPY" 
    
    vol_surface = VolatilitySurface(ticker)
    vol_surface.fetch_data()
    vol_surface.plot_surface()
