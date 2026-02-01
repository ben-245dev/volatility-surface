import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLabel, QLineEdit, QComboBox, QPushButton, 
                             QDateEdit, QHeaderView, QGroupBox, QSplitter, QMessageBox)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QColor, QFont

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1. CORE QUANT LOGIC (Mathematical Engine)
# =============================================================================
class QuantEngine:
    def __init__(self):
        self.r = 0.04 # Default Risk Free Rate
        self.cache = {}

    def get_market_data(self, ticker):
        """Fetches Spot + RFR + Option Chain (for IV Surface)"""
        try:
            # RFR (Risk-Free Rate)
            try:
                irx = yf.Ticker("^IRX").history(period="1d")
                if not irx.empty: self.r = irx['Close'].iloc[-1] / 100
            except: pass

            # Spot Price
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1d")
            if hist.empty: return None
            spot = hist['Close'].iloc[-1]
            return spot
        except Exception as e:
            logging.error(f"Error data: {e}")
            return None

    def calculate_greeks(self, S, K, T, r, sigma, option_type):
        """Calculates Delta and Gamma using Black-Scholes"""
        if T <= 0.001: T = 0.001
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'Call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        else:
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
        return delta, gamma

    def estimate_iv(self, ticker, T, K, spot):
        # Placeholder for IV estimation or scraping
        return 0.25 

# =============================================================================
# 2. GRAPHICAL USER INTERFACE (GUI)
# =============================================================================
class DeltaHedgerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Delta Hedger Pro | Quantitative Terminal")
        self.resize(1450, 900)
        self.engine = QuantEngine()
        self.positions = [] # List of dictionaries
        
        # Dark Mode Pro Style
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #dcdcdc; font-size: 12px; font-weight: bold; }
            QGroupBox { border: 1px solid #3e3e3e; margin-top: 10px; font-weight: bold; color: #00ff7f; }
            QTableWidget { background-color: #252526; color: #ffffff; gridline-color: #3e3e3e; border: none; }
            QHeaderView::section { background-color: #333333; color: #ffffff; padding: 4px; border: 1px solid #3e3e3e; }
            QLineEdit, QComboBox, QDateEdit { background-color: #333333; color: #ffffff; border: 1px solid #3e3e3e; padding: 5px; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton#btn_clear { background-color: #d62828; }
            QPushButton#btn_calc { background-color: #28a745; }
            QPushButton#btn_del { background-color: #ff4d4d; padding: 4px; border-radius: 3px; font-weight: bold; }
            QPushButton#btn_del:hover { background-color: #ff0000; }
        """)

        self.init_ui()

    def init_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- TOP BAR: INPUTS ---
        input_group = QGroupBox("Add Position")
        ig_layout = QHBoxLayout()
        
        self.in_ticker = QLineEdit(); self.in_ticker.setPlaceholderText("Ticker (e.g., SPY)")
        self.in_side = QComboBox(); self.in_side.addItems(["Long (Buy)", "Short (Sell)"])
        self.in_type = QComboBox(); self.in_type.addItems(["Call", "Put"])
        self.in_strike = QLineEdit(); self.in_strike.setPlaceholderText("Strike")
        self.in_qty = QLineEdit(); self.in_qty.setPlaceholderText("Quantity")
        self.in_expiry = QDateEdit(); self.in_expiry.setDate(QDate.currentDate().addDays(30)); self.in_expiry.setCalendarPopup(True)

        btn_add = QPushButton("Add Position"); btn_add.clicked.connect(self.add_position)

        ig_layout.addWidget(QLabel("Ticker:")); ig_layout.addWidget(self.in_ticker)
        ig_layout.addWidget(QLabel("Side:")); ig_layout.addWidget(self.in_side)
        ig_layout.addWidget(QLabel("Type:")); ig_layout.addWidget(self.in_type)
        ig_layout.addWidget(QLabel("Qty:")); ig_layout.addWidget(self.in_qty)
        ig_layout.addWidget(QLabel("Strike:")); ig_layout.addWidget(self.in_strike)
        ig_layout.addWidget(QLabel("Expiry:")); ig_layout.addWidget(self.in_expiry)
        ig_layout.addWidget(btn_add)
        input_group.setLayout(ig_layout)

        # --- MIDDLE: TABLE & CHARTS SPLITTER ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Positions Table
        self.table = QTableWidget()
        self.table.setColumnCount(13) # Increased for Action column
        self.table.setHorizontalHeaderLabels([
            "Ticker", "Side", "Type", "Qty", "Strike", "Expiry", "DTE", 
            "Spot", "Est. IV", "Unit Delta", "Pos Delta", "Unit Gamma", "Action"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Fix Action column size to be smaller
        self.table.horizontalHeader().setSectionResizeMode(12, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(12, 60)
        
        splitter.addWidget(self.table)

        # Aggregated Delta Panel
        self.agg_panel = QGroupBox("Net Exposure (Global Delta per Ticker)")
        self.agg_layout = QVBoxLayout()
        self.agg_label = QLabel("No active positions.")
        self.agg_label.setStyleSheet("font-size: 14px; color: cyan;")
        self.agg_layout.addWidget(self.agg_label)
        self.agg_panel.setLayout(self.agg_layout)
        splitter.addWidget(self.agg_panel)

        main_layout.addWidget(input_group)
        main_layout.addWidget(splitter)

        # --- BOTTOM BAR ---
        btn_layout = QHBoxLayout()
        self.btn_calc = QPushButton("Refresh Prices & Greeks"); self.btn_calc.setObjectName("btn_calc")
        self.btn_calc.clicked.connect(self.refresh_calculations)
        
        btn_clear = QPushButton("Reset Portfolio"); btn_clear.setObjectName("btn_clear")
        btn_clear.clicked.connect(self.clear_table)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_calc)
        btn_layout.addWidget(btn_clear)
        main_layout.addLayout(btn_layout)

    def add_position(self):
        try:
            ticker = self.in_ticker.text().upper()
            strike = float(self.in_strike.text())
            qty = int(self.in_qty.text())
            if not ticker or qty <= 0: raise ValueError
            
            pos = {
                'ticker': ticker,
                'side': self.in_side.currentText(),
                'type': self.in_type.currentText(),
                'strike': strike,
                'expiry': self.in_expiry.date().toPyDate(),
                'qty': qty
            }
            self.positions.append(pos)
            self.refresh_calculations() # Immediate calculation
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Check fields (Strike/Qty must be numeric).")

    def remove_position(self, index):
        """Removes the position at the given index and refreshes UI."""
        if 0 <= index < len(self.positions):
            del self.positions[index]
            self.refresh_calculations()

    def refresh_calculations(self):
        """Core Loop: Fetch Data -> Calc Greeks -> Update UI"""
        # Note: We do NOT return immediately if empty, to ensure the table clears if last item deleted.
        
        self.table.setRowCount(len(self.positions))
        net_deltas = {} # Ticker -> Net Delta

        for row, p in enumerate(self.positions):
            # 1. Fetch Market Data
            spot = self.engine.get_market_data(p['ticker'])
            if spot is None: 
                # Fallback if no internet or error, just to show row exists
                spot = 0.0

            # 2. Time Logic
            today = datetime.now().date()
            dte = (p['expiry'] - today).days
            T = dte / 365.0

            # 3. Quant Logic
            iv = self.engine.estimate_iv(p['ticker'], T, p['strike'], spot)
            
            if spot > 0:
                unit_delta, unit_gamma = self.engine.calculate_greeks(
                    spot, p['strike'], T, self.engine.r, iv, p['type']
                )
            else:
                unit_delta, unit_gamma = 0.0, 0.0

            # 4. Position Delta
            # Long Call (+), Short Call (-), Long Put (-), Short Put (+)
            sign = 1 if "Long" in p['side'] else -1
            pos_delta = unit_delta * p['qty'] * sign * 100 

            # Aggregation
            if p['ticker'] not in net_deltas: net_deltas[p['ticker']] = 0
            net_deltas[p['ticker']] += pos_delta

            # 5. UI Update
            self.table.setItem(row, 0, QTableWidgetItem(p['ticker']))
            self.table.setItem(row, 1, QTableWidgetItem(p['side']))
            self.table.setItem(row, 2, QTableWidgetItem(p['type']))
            self.table.setItem(row, 3, QTableWidgetItem(str(p['qty'])))
            self.table.setItem(row, 4, QTableWidgetItem(f"{p['strike']}"))
            self.table.setItem(row, 5, QTableWidgetItem(str(p['expiry'])))
            self.table.setItem(row, 6, QTableWidgetItem(f"{dte} d"))
            self.table.setItem(row, 7, QTableWidgetItem(f"${spot:.2f}"))
            self.table.setItem(row, 8, QTableWidgetItem(f"{iv:.1%}"))
            
            d_item = QTableWidgetItem(f"{unit_delta:.2f}")
            d_item.setForeground(QColor("#00ff7f") if unit_delta > 0 else QColor("#ff4d4d"))
            self.table.setItem(row, 9, d_item)

            pd_item = QTableWidgetItem(f"{pos_delta:.1f}")
            pd_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.table.setItem(row, 10, pd_item)

            self.table.setItem(row, 11, QTableWidgetItem(f"{unit_gamma:.4f}"))

            # 6. Delete Button (Column 12)
            btn_del = QPushButton("X")
            btn_del.setObjectName("btn_del")
            # We use a lambda with default argument `r=row` to capture the current row index loop variable
            btn_del.clicked.connect(lambda checked, r=row: self.remove_position(r))
            
            # Create a widget to center the button
            cell_widget = QWidget()
            layout_cell = QHBoxLayout(cell_widget)
            layout_cell.addWidget(btn_del)
            layout_cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout_cell.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 12, cell_widget)

        # Update Aggregated Panel
        if not net_deltas:
            self.agg_label.setText("No active positions.")
        else:
            txt = ""
            for t, d in net_deltas.items():
                action = "BUY" if d < 0 else "SELL"
                color = "#ff4d4d" if d < 0 else "#00ff7f"
                hedge_qty = int(abs(d))
                txt += f"{t}: Net Delta {d:.1f} | Required Hedge: <span style='color:{color}'>{action} {hedge_qty} shares</span><br>"
            
            self.agg_label.setText(txt)

    def clear_table(self):
        self.positions = []
        self.table.setRowCount(0)
        self.agg_label.setText("Portfolio empty.")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = DeltaHedgerApp()
    window.show()
    sys.exit(app.exec())
