
# Tạo app.py hoàn chỉnh với cải tiến data provider
app_py_final = '''"""
VN Stock AI - Professional Deep Analysis System v2.0
Features: Real-time VN data, Technical Analysis, Fundamental Analysis,
Interactive Charts, Professional AI Recommendations
"""

import os
import json
import logging
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== DATA PROVIDERS ======================
class VNStockDataProvider:
    """Lấy dữ liệu chứng khoán Việt Nam từ nhiều nguồn"""
    
    @staticmethod
    def get_data(symbol, period="1y", stock_type="stock"):
        """Lấy dữ liệu với fallback giữa các nguồn"""
        
        if stock_type == "forex":
            return VNStockDataProvider._get_forex_data(symbol, period)
        
        # Thử vnstock trước (TCBS - chính xác nhất cho VN)
        data = VNStockDataProvider._get_vnstock_data(symbol, period)
        if data is not None:
            return data
        
        # Fallback sang yfinance
        data = VNStockDataProvider._get_yfinance_data(symbol, period)
        if data is not None:
            return data
            
        return None
    
    @staticmethod
    def _get_vnstock_data(symbol, period="1y"):
        """Lấy dữ liệu từ TCBS qua vnstock"""
        try:
            # Tính start_date từ period
            days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = days_map.get(period, 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Thử import vnstock
            try:
                from vnstock import stock_historical_data, ticker_overview
                
                df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df is None or df.empty:
                    return None
                
                # Đổi tên cột cho chuẩn
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'tradingDate': 'Date'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                
                # Lấy thông tin cơ bản
                try:
                    overview = ticker_overview(symbol)
                    info = overview.to_dict('records')[0] if not overview.empty else {}
                except:
                    info = {}
                
                return {
                    "history": df,
                    "info": info,
                    "symbol": symbol,
                    "source": "vnstock"
                }
            except ImportError:
                logger.info("vnstock not installed, skipping")
                return None
                
        except Exception as e:
            logger.warning(f"vnstock error for {symbol}: {e}")
            return None
    
    @staticmethod
    def _get_yfinance_data(symbol, period="1y"):
        """Lấy dữ liệu từ Yahoo Finance"""
        try:
            import yfinance as yf
            
            # Thử với suffix .VN
            vn_symbol = f"{symbol}.VN"
            ticker = yf.Ticker(vn_symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if hist.empty:
                # Thử không có suffix (cho quỹ, forex...)
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
            
            if hist.empty:
                return None
            
            return {
                "history": hist,
                "info": info,
                "symbol": symbol,
                "source": "yfinance"
            }
            
        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return None
    
    @staticmethod
    def _get_forex_data(pair, period="6mo"):
        """Lấy dữ liệu forex"""
        try:
            import yfinance as yf
            
            # Format: EURUSD=X
            forex_symbol = pair.replace(".", "").replace("/", "") + "=X"
            ticker = yf.Ticker(forex_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                # Thử với dấu gạch ngang
                forex_symbol = pair.replace(".", "-").replace("/", "-") + "=X"
                ticker = yf.Ticker(forex_symbol)
                hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            return {
                "history": hist,
                "info": ticker.info,
                "symbol": pair,
                "source": "yfinance"
            }
            
        except Exception as e:
            logger.warning(f"Forex error for {pair}: {e}")
            return None

# ====================== TECHNICAL ANALYSIS ======================
class TechnicalAnalyzer:
    """Phân tích kỹ thuật chuyên sâu với đầy đủ indicators"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_sma(prices, period):
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices, period):
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_obv(close, volume):
        """On Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def calculate_fibonacci(high, low):
        diff = high - low
        return {
            "0%": round(high, 2),
            "23.6%": round(high - 0.236 * diff, 2),
            "38.2%": round(high - 0.382 * diff, 2),
            "50%": round(high - 0.5 * diff, 2),
            "61.8%": round(high - 0.618 * diff, 2),
            "78.6%": round(high - 0.786 * diff, 2),
            "100%": round(low, 2)
        }
    
    @classmethod
    def full_analysis(cls, df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(index=df.index, dtype=float))
        
        # Indicators
        rsi = cls.calculate_rsi(close)
        macd, macd_sig, macd_hist = cls.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(close)
        sma20 = cls.calculate_sma(close, 20)
        sma50 = cls.calculate_sma(close, 50)
        sma200 = cls.calculate_sma(close, 200)
        ema12 = cls.calculate_ema(close, 12)
        ema26 = cls.calculate_ema(close, 26)
        stoch_k, stoch_d = cls.calculate_stochastic(high, low, close)
        atr = cls.calculate_atr(high, low, close)
        adx, plus_di, minus_di = cls.calculate_adx(high, low, close)
        obv = cls.calculate_obv(close, volume)
        
        # Fibonacci (60 ngày gần nhất)
        recent_high = high.tail(60).max()
        recent_low = low.tail(60).min()
        fib = cls.calculate_fibonacci(recent_high, recent_low)
        
        # Signals
        golden_cross = sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]
        death_cross = sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]
        
        # Trend
        if close.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
            trend = "UPTREND"
        elif close.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        # Volume analysis
        avg_volume = volume.tail(20).mean()
        latest_volume = volume.iloc[-1]
        volume_signal = "High" if latest_volume > avg_volume * 1.5 else "Low" if latest_volume < avg_volume * 0.5 else "Normal"
        
        return {
            "indicators": {
                "RSI(14)": round(rsi.iloc[-1], 2),
                "RSI_signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral",
                "MACD": round(macd.iloc[-1], 4),
                "MACD_signal": round(macd_sig.iloc[-1], 4),
                "MACD_histogram": round(macd_hist.iloc[-1], 4),
                "BB_upper": round(bb_upper.iloc[-1], 2),
                "BB_middle": round(bb_middle.iloc[-1], 2),
                "BB_lower": round(bb_lower.iloc[-1], 2),
                "SMA20": round(sma20.iloc[-1], 2),
                "SMA50": round(sma50.iloc[-1], 2),
                "SMA200": round(sma200.iloc[-1], 2),
                "EMA12": round(ema12.iloc[-1], 2),
                "EMA26": round(ema26.iloc[-1], 2),
                "Stochastic_K": round(stoch_k.iloc[-1], 2),
                "Stochastic_D": round(stoch_d.iloc[-1], 2),
                "ATR(14)": round(atr.iloc[-1], 2),
                "ADX(14)": round(adx.iloc[-1], 2),
                "ADX_trend": "Strong" if adx.iloc[-1] > 25 else "Weak",
                "OBV": round(obv.iloc[-1], 0),
            },
            "trend": trend,
            "signals": {
                "golden_cross": bool(golden_cross),
                "death_cross": bool(death_cross),
                "macd_bullish": macd.iloc[-1] > macd_sig.iloc[-1],
                "price_vs_bb": "Upper" if close.iloc[-1] > bb_upper.iloc[-1] else "Lower" if close.iloc[-1] < bb_lower.iloc[-1] else "Middle",
                "rsi_signal": "Buy" if rsi.iloc[-1] < 30 else "Sell" if rsi.iloc[-1] > 70 else "Neutral",
                "volume_signal": volume_signal,
            },
            "fibonacci": fib,
            "support_resistance": {
                "support_1": round(recent_low, 2),
                "support_2": round(recent_low * 0.98, 2),
                "resistance_1": round(recent_high, 2),
                "resistance_2": round(recent_high * 1.02, 2),
            },
            "latest_price": round(close.iloc[-1], 2),
            "price_change_1d": round((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2),
            "price_change_1w": round((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100, 2) if len(close) >= 5 else 0,
            "price_change_1m": round((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100, 2) if len(close) >= 20 else 0,
            "price_change_3m": round((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60] * 100, 2) if len(close) >= 60 else 0,
            "volatility": round(atr.iloc[-1] / close.iloc[-1] * 100, 2),
            "volume_avg_20d": round(avg_volume, 0),
            "volume_latest": round(latest_volume, 0),
        }

# ====================== CHART GENERATOR ======================
class ChartGenerator:
    """Tạo biểu đồ tương tác chuyên nghiệp bằng Plotly"""
    
    @staticmethod
    def create_technical_chart(df, symbol, tech_data):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df.get('Volume', pd.Series(index=df.index, dtype=float))
            
            # Calculate indicators
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()
            bb_upper, bb_middle, bb_lower = TechnicalAnalyzer.calculate_bollinger_bands(close)
            rsi = TechnicalAnalyzer.calculate_rsi(close)
            macd, macd_sig, macd_hist = TechnicalAnalyzer.calculate_macd(close)
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=(f'{symbol} - Biểu Đồ Kỹ Thuật', 'RSI (14)', 'MACD', 'Volume')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=high, low=low, close=close,
                name='Giá', increasing_line_color='#00e676', decreasing_line_color='#ff5252'
            ), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper',
                                    line=dict(color='rgba(0,212,255,0.6)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower',
                                    line=dict(color='rgba(0,212,255,0.6)', width=1),
                                    fill='tonexty', fillcolor='rgba(0,212,255,0.05)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_middle, name='BB Middle',
                                    line=dict(color='rgba(240,192,64,0.8)', width=1, dash='dash')), row=1, col=1)
            
            # SMA
            fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA20',
                                    line=dict(color='#ffd740', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA50',
                                    line=dict(color='#f0c040', width=1.5)), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI',
                                    line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
            
            # MACD
            colors_macd = ['#00e676' if h >= 0 else '#ff5252' for h in macd_hist]
            fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD Hist',
                                marker_color=colors_macd, opacity=0.7), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD',
                                    line=dict(color='#00d4ff', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name='Signal',
                                    line=dict(color='#f0c040', width=1.5)), row=3, col=1)
            
            # Volume
            vol_colors = ['#00e676' if close.iloc[i] >= df['Open'].iloc[i] else '#ff5252'
                         for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=volume, name='Volume',
                                marker_color=vol_colors, opacity=0.4), row=4, col=1)
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(6,11,20,1)',
                plot_bgcolor='rgba(12,20,33,1)',
                font=dict(family='Inter, sans-serif', color='#e8f4fd', size=11),
                xaxis_rangeslider_visible=False,
                height=900,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                           bgcolor='rgba(6,11,20,0.8)', font=dict(size=10)),
                margin=dict(l=50, r=50, t=100, b=50),
                title=dict(
                    text=f'<b>{symbol}</b> | Xu hướng: {tech_data.get("trend", "N/A")} | RSI: {tech_data.get("indicators", {}).get("RSI(14)", "N/A")}',
                    font=dict(size=14, color='#00d4ff'),
                    x=0.5
                )
            )
            
            fig.update_yaxes(title_text="Giá", row=1, col=1, gridcolor='rgba(30,48,80,0.3)')
            fig.update_yaxes(title_text="RSI", row=2, col=1, gridcolor='rgba(30,48,80,0.3)')
            fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor='rgba(30,48,80,0.3)')
            fig.update_yaxes(title_text="Vol", row=4, col=1, gridcolor='rgba(30,48,80,0.3)')
            fig.update_xaxes(gridcolor='rgba(30,48,80,0.3)')
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Technical chart error: {e}")
            return None
    
    @staticmethod
    def create_fundamental_chart(info, symbol):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if not info:
                return None
            
            # Extract metrics
            metrics = {
                'P/E': info.get('trailingPE') or info.get('forwardPE') or 0,
                'P/B': info.get('priceToBook') or 0,
                'P/S': info.get('priceToSalesTrailing12Months') or 0,
                'EPS': info.get('trailingEps') or 0,
                'ROE (%)': (info.get('returnOnEquity') or 0) * 100,
                'ROA (%)': (info.get('returnOnAssets') or 0) * 100,
                'Debt/Equity': (info.get('debtToEquity') or 0) / 100 if info.get('debtToEquity') else 0,
                'Current Ratio': info.get('currentRatio') or 0,
                'Quick Ratio': info.get('quickRatio') or 0,
                'Beta': info.get('beta') or 0,
            }
            
            # Clean data
            metrics = {k: round(v, 2) if v and v != 0 else 0 for k, v in metrics.items()}
            metrics = {k: v for k, v in metrics.items() if v != 0}
            
            if not metrics:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📊 Định Giá', '💰 Hiệu Quả', '🏦 Cơ Cấu Tài Chính', '📈 Tổng Quan'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]],
                vertical_spacing=0.15
            )
            
            # Valuation
            val_keys = ['P/E', 'P/B', 'P/S']
            val_data = {k: metrics[k] for k in val_keys if k in metrics and metrics[k] > 0}
            if val_data:
                colors_val = ['#00d4ff', '#f0c040', '#00e676']
                fig.add_trace(go.Bar(
                    x=list(val_data.keys()), y=list(val_data.values()),
                    marker_color=colors_val[:len(val_data)],
                    text=list(val_data.values()), textposition='auto',
                    textfont=dict(color='white', size=12)
                ), row=1, col=1)
            
            # Efficiency
            eff_keys = ['ROE (%)', 'ROA (%)']
            eff_data = {k: metrics[k] for k in eff_keys if k in metrics and metrics[k] != 0}
            if eff_data:
                fig.add_trace(go.Bar(
                    x=list(eff_data.keys()), y=list(eff_data.values()),
                    marker_color=['#00e676', '#ffd740'],
                    text=[f"{v}%" for v in eff_data.values()], textposition='auto',
                    textfont=dict(color='white', size=12)
                ), row=1, col=2)
            
            # Financial structure
            struct_keys = ['Debt/Equity', 'Current Ratio', 'Quick Ratio']
            struct_data = {k: metrics[k] for k in struct_keys if k in metrics and metrics[k] > 0}
            if struct_data:
                colors_struct = ['#ff5252', '#00d4ff', '#ffd740']
                fig.add_trace(go.Bar(
                    x=list(struct_data.keys()), y=list(struct_data.values()),
                    marker_color=colors_struct[:len(struct_data)],
                    text=list(struct_data.values()), textposition='auto',
                    textfont=dict(color='white', size=12)
                ), row=2, col=1)
            
            # EPS/Beta indicator
            eps_val = metrics.get('EPS', 0)
            beta_val = metrics.get('Beta', 0)
            if eps_val or beta_val:
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=eps_val if eps_val else beta_val,
                    title={"text": "EPS" if eps_val else "Beta", "font": {"size": 14, "color": "#00d4ff"}},
                    number={"font": {"size": 28, "color": "#00e676"}},
                    domain={'row': 1, 'column': 1}
                ), row=2, col=2)
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(6,11,20,1)',
                plot_bgcolor='rgba(12,20,33,1)',
                font=dict(family='Inter, sans-serif', color='#e8f4fd', size=11),
                height=550,
                showlegend=False,
                title=dict(
                    text=f'<b>Phân Tích Cơ Bản - {symbol}</b>',
                    font=dict(size=16, color='#00d4ff'),
                    x=0.5
                ),
                margin=dict(l=50, r=50, t=80, b=40)
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Fundamental chart error: {e}")
            return None

# ====================== AI ANALYZER ======================
class DeepAnalyzer:
    def __init__(self):
        self.client = None
        self.available = False
        self._init_groq()
    
    def _init_groq(self):
        key = os.getenv("GROQ_API_KEY_STOCK")
        if key:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                self.available = True
                logger.info("✅ Groq AI initialized")
            except Exception as e:
                logger.error(f"Groq init error: {e}")
    
    def analyze(self, symbol, stock_type, tech_data, fund_data=None, news_data=None):
        if not self.available:
            return self._fallback_analysis(symbol, tech_data, fund_data)
        
        tech_summary = self._format_tech(tech_data)
        fund_summary = self._format_fund(fund_data)
        news_summary = self._format_news(news_data)
        
        system = """Bạn là Giám đốc Phân tích Đầu tư (CIO) tại công ty chứng khoán hàng đầu Việt Nam với 20 năm kinh nghiệm.

YÊU CẦU BÁO CÁO:
1. Viết bằng tiếng Việt chuyên nghiệp, có cấu trúc rõ ràng
2. Mỗi phần phải có số liệu cụ thể, không chung chung
3. Đưa ra khuyến nghị đầu tư rõ ràng: **[MUA / BÁN / GIỮ]**
4. Chỉ rõ: Giá mục tiêu ngắn hạn (1-3 tháng), trung hạn (3-6 tháng), điểm cắt lỗ
5. Phân tích dựa TRÊN DỮ LIỆU THỰC TẾ được cung cấp, không bịa đặt

CẤU TRÚC BÁO CÁO:
## 1. TỔNG QUAN THỊ TRƯỜNG & CỔ PHIẾU
## 2. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU (RSI, MACD, BB, SMA, Fibonacci...)
## 3. PHÂN TÍCH CƠ BẢN (P/E, P/B, ROE, EPS...)
## 4. ĐÁNH GIÁ RỦI RO (ATR, Beta, Volatility)
## 5. KHUYẾN NGHỊ ĐẦU TƯ (MUA/BÁN/GIỮ + giá mục tiêu + stop-loss)"""
        
        user = f"""Phân tích chuyên sâu {stock_type.upper()} **{symbol}**:

{tech_summary}

{fund_summary}

{news_summary}

Hãy viết báo cáo phân tích chuyên nghiệp, chi tiết với số liệu cụ thể."""
        
        models = ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]
        
        for model in models:
            try:
                r = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.15,
                    max_tokens=4000
                )
                analysis = r.choices[0].message.content
                rec = self._extract_rec(analysis)
                return {"analysis": analysis, "recommendation": rec, "model_used": model}
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        return self._fallback_analysis(symbol, tech_data, fund_data)
    
    def _format_tech(self, data):
        ind = data.get('indicators', {})
        sig = data.get('signals', {})
        return f"""=== DỮ LIỆU KỸ THUẬT ===
Xu hướng: {data.get('trend', 'N/A')}
Giá hiện tại: {data.get('latest_price', 'N/A')}
Thay đổi: 1D={data.get('price_change_1d', 'N/A')}% | 1W={data.get('price_change_1w', 'N/A')}% | 1M={data.get('price_change_1m', 'N/A')}% | 3M={data.get('price_change_3m', 'N/A')}%

Chỉ báo:
- RSI(14): {ind.get('RSI(14)', 'N/A')} ({ind.get('RSI_signal', 'N/A')})
- MACD: {ind.get('MACD', 'N/A')} | Signal: {ind.get('MACD_signal', 'N/A')} | Hist: {ind.get('MACD_histogram', 'N/A')}
- Bollinger: Upper={ind.get('BB_upper', 'N/A')} | Middle={ind.get('BB_middle', 'N/A')} | Lower={ind.get('BB_lower', 'N/A')}
- SMA: 20={ind.get('SMA20', 'N/A')} | 50={ind.get('SMA50', 'N/A')} | 200={ind.get('SMA200', 'N/A')}
- Stochastic: K={ind.get('Stochastic_K', 'N/A')} D={ind.get('Stochastic_D', 'N/A')}
- ADX(14): {ind.get('ADX(14)', 'N/A')} ({ind.get('ADX_trend', 'N/A')})
- ATR(14): {ind.get('ATR(14)', 'N/A')}
- OBV: {ind.get('OBV', 'N/A')}

Tín hiệu: GoldenCross={sig.get('golden_cross', False)} | DeathCross={sig.get('death_cross', False)} | MACD_Bull={sig.get('macd_bullish', False)} | BB_Pos={sig.get('price_vs_bb', 'N/A')} | Volume={sig.get('volume_signal', 'N/A')}

Hỗ trợ/Kháng cự: {data.get('support_resistance', {})}
Fibonacci: {data.get('fibonacci', {})}
Volatility: {data.get('volatility', 'N/A')}%"""
    
    def _format_fund(self, info):
        if not info:
            return "=== KHÔNG CÓ DỮ LIỆU CƠ BẢN ==="
        return f"""=== DỮ LIỆU CƠ BẢN ===
P/E: {info.get('trailingPE', 'N/A')} | Forward P/E: {info.get('forwardPE', 'N/A')}
P/B: {info.get('priceToBook', 'N/A')} | P/S: {info.get('priceToSalesTrailing12Months', 'N/A')}
EPS: {info.get('trailingEps', 'N/A')} | EPS growth: {info.get('earningsGrowth', 'N/A')}
ROE: {(info.get('returnOnEquity', 0) or 0)*100:.2f}% | ROA: {(info.get('returnOnAssets', 0) or 0)*100:.2f}%
Debt/Equity: {info.get('debtToEquity', 'N/A')} | Current Ratio: {info.get('currentRatio', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')} | Beta: {info.get('beta', 'N/A')}
Dividend Yield: {info.get('dividendYield', 'N/A')} | 52W Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}
Revenue Growth: {info.get('revenueGrowth', 'N/A')} | Profit Margin: {info.get('profitMargins', 'N/A')}"""
    
    def _format_news(self, news):
        if not news:
            return ""
        return "=== TIN TỨC GẦN ĐÂY ===\\n" + "\\n".join([f"- {n.get('title', '')}" for n in news[:5]])
    
    def _extract_rec(self, text):
        t = text.upper()
        if any(x in t for x in ["MUA", "BUY", "TĂNG TỶ TRỌNG"]):
            return "BUY"
        elif any(x in t for x in ["BÁN", "SELL", "GIẢM TỶ TRỌNG"]):
            return "SELL"
        elif any(x in t for x in ["GIỮ", "HOLD", "TRUNG LẬP"]):
            return "HOLD"
        return "WATCH"
    
    def _fallback_analysis(self, symbol, tech_data, fund_data):
        ind = tech_data.get('indicators', {})
        sig = tech_data.get('signals', {})
        trend = tech_data.get('trend', 'SIDEWAYS')
        rsi = ind.get('RSI(14)', 50)
        macd_bull = sig.get('macd_bullish', False)
        
        if rsi < 30 and macd_bull and trend == "UPTREND":
            rec = "BUY"
        elif rsi > 70 and not macd_bull and trend == "DOWNTREND":
            rec = "SELL"
        elif 40 <= rsi <= 60:
            rec = "HOLD"
        else:
            rec = "WATCH"
        
        analysis = f"""## Phân Tích Tự Động - {symbol}

### 1. Tổng Quan
- **Xu hướng**: {trend}
- **Giá hiện tại**: {tech_data.get('latest_price', 'N/A')}
- **Biến động**: {tech_data.get('volatility', 'N/A')}%

### 2. Phân Tích Kỹ Thuật
| Chỉ báo | Giá trị | Tín hiệu |
|---------|---------|----------|
| RSI(14) | {ind.get('RSI(14)', 'N/A')} | {ind.get('RSI_signal', 'N/A')} |
| MACD | {ind.get('MACD', 'N/A')} | {'Bullish' if macd_bull else 'Bearish'} |
| BB Position | {sig.get('price_vs_bb', 'N/A')} | - |
| SMA20 | {ind.get('SMA20', 'N/A')} | - |
| SMA50 | {ind.get('SMA50', 'N/A')} | - |
| SMA200 | {ind.get('SMA200', 'N/A')} | - |
| ADX | {ind.get('ADX(14)', 'N/A')} | {ind.get('ADX_trend', 'N/A')} |

### 3. Hỗ Trợ & Kháng Cự
- **Hỗ trợ 1**: {tech_data.get('support_resistance', {}).get('support_1', 'N/A')}
- **Kháng cự 1**: {tech_data.get('support_resistance', {}).get('resistance_1', 'N/A')}

### 4. Khuyến Nghị
**[{rec}]** 

> ⚠️ Đây là phân tích tự động. Kết nối API Groq để có báo cáo chuyên sâu bởi AI.
"""
        return {"analysis": analysis, "recommendation": rec, "model_used": "fallback"}

# ====================== NEWS AGENT ======================
class NewsAgent:
    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                results = list(d.text(f"{symbol} cổ phiếu tin tức phân tích 2026", max_results=8))
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.error(f"News error: {e}")
            return []

# ====================== ORCHESTRATOR ======================
class Orchestrator:
    def __init__(self):
        self.data = VNStockDataProvider()
        self.tech = TechnicalAnalyzer()
        self.charts = ChartGenerator()
        self.ai = DeepAnalyzer()
        self.news = NewsAgent()
    
    def analyze(self, symbol, stock_type="stock"):
        logger.info(f"Analyzing {symbol} ({stock_type})")
        
        # Lấy dữ liệu
        period = "6mo" if stock_type == "forex" else "1y"
        raw = self.data.get_data(symbol, period, stock_type)
        
        if not raw or raw['history'].empty:
            return {
                "mode": stock_type,
                "error": f"Không tìm thấy dữ liệu cho {symbol}. Vui lòng kiểm tra lại mã.",
                "data": None
            }
        
        df = raw['history']
        info = raw.get('info', {})
        
        # Phân tích kỹ thuật
        tech_data = self.tech.full_analysis(df)
        
        # Tạo biểu đồ
        tech_chart = self.charts.create_technical_chart(df, symbol, tech_data)
        fund_chart = self.charts.create_fundamental_chart(info, symbol) if info else None
        
        # Tin tức
        news_data = self.news.get_news(symbol) if stock_type != "forex" else []
        
        # AI phân tích
        ai_result = self.ai.analyze(symbol, stock_type, tech_data, info, news_data)
        
        return {
            "mode": stock_type,
            "data": {
                "symbol": symbol,
                "type": stock_type,
                "analysis": ai_result["analysis"],
                "recommendation": ai_result["recommendation"],
                "model_used": ai_result.get("model_used", "unknown"),
                "technical": tech_data,
                "fundamental": {
                    "pe": info.get('trailingPE') if info else None,
                    "forward_pe": info.get('forwardPE') if info else None,
                    "pb": info.get('priceToBook') if info else None,
                    "ps": info.get('priceToSalesTrailing12Months') if info else None,
                    "roe": round((info.get('returnOnEquity', 0) or 0) * 100, 2) if info else None,
                    "roa": round((info.get('returnOnAssets', 0) or 0) * 100, 2) if info else None,
                    "eps": info.get('trailingEps') if info else None,
                    "eps_growth": info.get('earningsGrowth') if info else None,
                    "market_cap": info.get('marketCap') if info else None,
                    "beta": info.get('beta') if info else None,
                    "dividend_yield": info.get('dividendYield') if info else None,
                    "debt_equity": info.get('debtToEquity') if info else None,
                    "current_ratio": info.get('currentRatio') if info else None,
                    "revenue_growth": info.get('revenueGrowth') if info else None,
                    "profit_margin": info.get('profitMargins') if info else None,
                    "52w_high": info.get('fiftyTwoWeekHigh') if info else None,
                    "52w_low": info.get('fiftyTwoWeekLow') if info else None,
                } if info else None,
                "news_count": len(news_data),
                "has_documents": False,
                "charts": {
                    "technical": tech_chart,
                    "fundamental": fund_chart
                },
                "data_source": raw.get('source', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
        }

orc = Orchestrator()

# ====================== ROUTES ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip().upper()
        stock_type = request.form.get("type", "stock")
        
        logger.info(f"Analyze: {symbol} ({stock_type})")
        
        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã chứng khoán"}), 400
        
        result = orc.analyze(symbol, stock_type)
        
        if result.get("error"):
            return jsonify(result), 404
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq": orc.ai.available,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0-professional"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''

with open('/mnt/agents/output/app.py', 'w', encoding='utf-8') as f:
    f.write(app_py_final)

print("✅ Đã tạo app.py hoàn chỉnh v2.0")
print(f"Kích thước: {len(app_py_final)} ký tự")
