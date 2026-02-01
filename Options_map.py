"""
Quantitative Volatility Tool for Delta Hedging

Volatility Surface Modeling: Scrapes option chains, constructs an Implied Volatility (IV) 
surface using RBF interpolation, and prices options using the Black-Scholes-Merton model.

This tool automates the extraction of the risk-free rate from Treasury yields and 
provides a framework for strategy risk-profiling and delta-neutral analysis.

Author: ben-245dev
Dependencies: yfinance, pandas, numpy, matplotlib, scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Station:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        # Automated Risk-Free Rate Scraping
        self.r = self._fetch_risk_free_rate()
        self.spot_price = None
        self.data_options = None
        self.rbf_surface = None

    def _fetch_risk_free_rate(self):
        """Fetches the 13-week T-Bill yield as the risk-free rate proxy."""
        try:
            t_bill = yf.Ticker("^IRX")
            current_yield = t_bill.history(period="1d")['Close'].iloc[-1]
            rate = current_yield / 100
            logging.info(f"Market Risk-Free Rate (^IRX): {rate:.2%}")
            return rate
        except Exception as e:
            logging.warning(f"Using default 4% RFR. Error: {e}")
            return 0.04

    def black_scholes_price(self, S, K, T, sigma, option_type='call'):
        """Calculates theoretical price and Greeks."""
        if T <= 0.0001: return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            
        return price, delta

    def build_engine(self, max_expiries=10):
        """Scrapes and prepares the IV surface."""
        hist = self.ticker.history(period="1d")
        self.spot_price = hist['Close'].iloc[-1]
        
        options_list = []
        for expiry in self.ticker.options[:max_expiries]:
            try:
                T = (pd.to_datetime(expiry) - datetime.now()).days / 365.0
                if T < 0.02: continue
                
                chain = self.ticker.option_chain(expiry)
                df = pd.concat([chain.calls, chain.puts])
                df = df[(df['bid'] > 0) & (df['impliedVolatility'] > 0.01)].copy()
                df['T'] = T
                options_list.append(df)
            except: continue
            
        self.data_options = pd.concat(options_list)
        # Interpolation for the continuous surface
        self.rbf_surface = Rbf(self.data_options['strike'], self.data_options['T'], 
                               self.data_options['impliedVolatility'], function='multiquadric', smooth=0.1)
        logging.info(f"Surface Engine Ready for {self.ticker_symbol}")

    def get_market_estimate(self, target_K, target_T, option_type='put'):
        """Estimates price using the interpolated IV surface."""
        iv_est = float(self.rbf_surface(target_K, target_T))
        price, delta = self.black_scholes_price(self.spot_price, target_K, target_T, iv_est, option_type)
        return {"price": price, "delta": delta, "iv": iv_est}

    def plot_surface(self):
        """Interactive 3D Visualization."""
        x, y = self.data_options['strike'], self.data_options['T']
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        X, Y = np.meshgrid(xi, yi)
        Z = self.rbf_surface(X, Y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)
        ax.set_title(f"Vol Surface & Skew: {self.ticker_symbol}")
        ax.set_xlabel("Strike"); ax.set_ylabel("Time (Y)"); ax.set_zlabel("IV")
        plt.colorbar(surf)
        plt.show()

# --- RUN ---
if __name__ == "__main__":
    engine = Station("SPY")
    engine.build_engine()
    
    # Example: Estimate a 30-day (0.08Y) ATM Put
    estimate = engine.get_market_estimate(engine.spot_price, 0.08, 'call')

    print(f"\nAnalysis:")
    print(f"- Estimated Price: ${estimate['price']:.2f}")
    print(f"- Delta: {estimate['delta']:.3f}")
    print(f"- Implied Vol: {estimate['iv']:.3%}")
    
    engine.plot_surface()
