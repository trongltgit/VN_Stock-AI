
# Tạo file app.py mới với phân tích chuyên sâu
app_py_content = '''"""
VN Stock AI - Professional Deep Analysis System
Features: Real-time data, Technical Analysis (RSI, MACD, BB), Fundamental Analysis,
Interactive Charts, Professional AI Recommendations
"""

import os
import json
import logging
import base64
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
class StockDataProvider:
    """Lấy dữ liệu chứng khoán Việt Nam từ nhiều nguồn"""
    
    @staticmethod
    def get_stock_data(symbol, period="1y"):
        """Lấy dữ liệu giá lịch sử và thông tin cơ bản"""
        try:
            import yfinance as yf
            # VN stocks suffix
            vn_symbol = f"{symbol}.VN"
            ticker = yf.Ticker(vn_symbol)
            
            # Lịch sử giá
            hist = ticker.history(period=period)
            if hist.empty:
                # Thử không có suffix
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
            
            if hist.empty:
                return None
                
            info = ticker.info
            
            return {
                "history": hist,
                "info": info,
                "symbol": symbol
            }
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return None
    
    @staticmethod
    def get_forex_data(pair, period="6mo"):
        """Lấy dữ liệu forex"""
        try:
            import yfinance as yf
            # Format: EURUSD=X
            forex_symbol = pair.replace(".", "") + "=X"
            ticker = yf.Ticker(forex_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
                
            return {
                "history": hist,
                "symbol": pair,
                "info": ticker.info
            }
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return None

# ====================== TECHNICAL ANALYSIS ======================
class TechnicalAnalyzer:
    """Phân tích kỹ thuật chuyên sâu"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Tính RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Tính MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Tính Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_sma(prices, period):
        """Tính Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices, period):
        """Tính Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """Tính Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Tính Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_fibonacci_retracement(high, low):
        """Tính Fibonacci Retracement levels"""
        diff = high - low
        levels = {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "78.6%": high - 0.786 * diff,
            "100%": low
        }
        return levels
    
    @classmethod
    def full_analysis(cls, df):
        """Phân tích kỹ thuật toàn diện"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Các chỉ báo
        rsi = cls.calculate_rsi(close)
        macd, macd_signal, macd_hist = cls.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(close)
        sma20 = cls.calculate_sma(close, 20)
        sma50 = cls.calculate_sma(close, 50)
        sma200 = cls.calculate_sma(close, 200)
        ema12 = cls.calculate_ema(close, 12)
        ema26 = cls.calculate_ema(close, 26)
        stoch_k, stoch_d = cls.calculate_stochastic(high, low, close)
        atr = cls.calculate_atr(high, low, close)
        
        # Fibonacci
        recent_high = high.tail(60).max()
        recent_low = low.tail(60).min()
        fib_levels = cls.calculate_fibonacci_retracement(recent_high, recent_low)
        
        # Tín hiệu giao cắt
        golden_cross = sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]
        death_cross = sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]
        
        # Xu hướng
        trend = "UPTREND" if close.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1] else \\
                "DOWNTREND" if close.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1] else "SIDEWAYS"
        
        return {
            "indicators": {
                "RSI": round(rsi.iloc[-1], 2),
                "RSI_signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral",
                "MACD": round(macd.iloc[-1], 4),
                "MACD_signal": round(macd_signal.iloc[-1], 4),
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
                "ATR": round(atr.iloc[-1], 2),
            },
            "trend": trend,
            "signals": {
                "golden_cross": bool(golden_cross),
                "death_cross": bool(death_cross),
                "macd_bullish": macd.iloc[-1] > macd_signal.iloc[-1],
                "price_vs_bb": "Upper" if close.iloc[-1] > bb_upper.iloc[-1] else \\
                              "Lower" if close.iloc[-1] < bb_lower.iloc[-1] else "Middle"
            },
            "fibonacci": {k: round(v, 2) for k, v in fib_levels.items()},
            "support_resistance": {
                "support": round(recent_low, 2),
                "resistance": round(recent_high, 2)
            },
            "latest_price": round(close.iloc[-1], 2),
            "price_change_1d": round((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2),
            "price_change_1w": round((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100, 2) if len(close) >= 5 else 0,
            "price_change_1m": round((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100, 2) if len(close) >= 20 else 0,
            "volatility": round(atr.iloc[-1] / close.iloc[-1] * 100, 2)
        }

# ====================== CHART GENERATOR ======================
class ChartGenerator:
    """Tạo biểu đồ tương tác chuyên nghiệp"""
    
    @staticmethod
    def create_candlestick_chart(df, symbol, tech_data):
        """Tạo biểu đồ nến với indicators"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Tính indicators
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()
            bb_upper, bb_middle, bb_lower = TechnicalAnalyzer.calculate_bollinger_bands(close)
            
            # Tạo figure với subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{symbol} - Biểu Đồ Kỹ Thuật', 'RSI', 'MACD')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=high,
                low=low,
                close=close,
                name='Giá',
                increasing_line_color='#00e676',
                decreasing_line_color='#ff5252'
            ), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper', 
                                    line=dict(color='rgba(0,212,255,0.5)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower',
                                    line=dict(color='rgba(0,212,255,0.5)', width=1),
                                    fill='tonexty', fillcolor='rgba(0,212,255,0.05)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_middle, name='BB Middle',
                                    line=dict(color='rgba(240,192,64,0.7)', width=1, dash='dash')), row=1, col=1)
            
            # SMA
            fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA20',
                                    line=dict(color='#ffd740', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA50',
                                    line=dict(color='#f0c040', width=1.5)), row=1, col=1)
            
            # Volume
            colors = ['#00e676' if close.iloc[i] >= df['Open'].iloc[i] else '#ff5252' 
                     for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                                marker_color=colors, opacity=0.3), row=1, col=1)
            
            # RSI
            rsi = TechnicalAnalyzer.calculate_rsi(close)
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI',
                                    line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            
            # MACD
            macd, macd_sig, macd_hist = TechnicalAnalyzer.calculate_macd(close)
            colors_macd = ['#00e676' if h >= 0 else '#ff5252' for h in macd_hist]
            fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD Hist',
                                marker_color=colors_macd), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD',
                                    line=dict(color='#00d4ff', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name='Signal',
                                    line=dict(color='#f0c040', width=1.5)), row=3, col=1)
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(6,11,20,1)',
                plot_bgcolor='rgba(12,20,33,1)',
                font=dict(family='Inter, sans-serif', color='#e8f4fd'),
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            fig.update_yaxes(title_text="Giá", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            
            # Convert to HTML
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return chart_html
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
    
    @staticmethod
    def create_fundamental_chart(info, symbol):
        """Tạo biểu đồ phân tích cơ bản"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            metrics = {
                'P/E': info.get('trailingPE', 0) or info.get('forwardPE', 0),
                'P/B': info.get('priceToBook', 0),
                'P/S': info.get('priceToSalesTrailing12Months', 0),
                'EPS': info.get('trailingEps', 0),
                'ROE (%)': (info.get('returnOnEquity', 0) or 0) * 100,
                'ROA (%)': (info.get('returnOnAssets', 0) or 0) * 100,
                'Debt/Equity': info.get('debtToEquity', 0),
                'Current Ratio': info.get('currentRatio', 0)
            }
            
            # Loại bỏ giá trị 0 hoặc None
            metrics = {k: round(v, 2) if v else 0 for k, v in metrics.items()}
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Định Giá (P/E, P/B, P/S)', 'Hiệu Quả (ROE, ROA)', 
                               'Cơ Cấu Tài Chính', 'Tổng Quan'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Valuation
            valuation = {k: metrics[k] for k in ['P/E', 'P/B', 'P/S'] if metrics[k] > 0}
            if valuation:
                fig.add_trace(go.Bar(
                    x=list(valuation.keys()),
                    y=list(valuation.values()),
                    marker_color=['#00d4ff', '#f0c040', '#00e676'],
                    text=list(valuation.values()),
                    textposition='auto'
                ), row=1, col=1)
            
            # Efficiency
            efficiency = {k: metrics[k] for k in ['ROE (%)', 'ROA (%)'] if metrics[k] != 0}
            if efficiency:
                fig.add_trace(go.Bar(
                    x=list(efficiency.keys()),
                    y=list(efficiency.values()),
                    marker_color=['#00e676', '#ffd740'],
                    text=[f"{v}%" for v in efficiency.values()],
                    textposition='auto'
                ), row=1, col=2)
            
            # Financial structure
            structure = {k: metrics[k] for k in ['Debt/Equity', 'Current Ratio'] if metrics[k] > 0}
            if structure:
                fig.add_trace(go.Bar(
                    x=list(structure.keys()),
                    y=list(structure.values()),
                    marker_color=['#ff5252', '#00d4ff'],
                    text=list(structure.values()),
                    textposition='auto'
                ), row=2, col=1)
            
            # EPS Gauge
            eps = metrics.get('EPS', 0)
            if eps:
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=eps,
                    title={"text": "EPS"},
                    domain={'row': 1, 'column': 1}
                ), row=2, col=2)
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(6,11,20,1)',
                plot_bgcolor='rgba(12,20,33,1)',
                font=dict(family='Inter, sans-serif', color='#e8f4fd'),
                height=600,
                showlegend=False,
                title_text=f"Phân Tích Cơ Bản - {symbol}",
                title_font_size=16,
                title_font_color='#00d4ff'
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Fundamental chart error: {e}")
            return None

# ====================== AI ANALYSIS ======================
class DeepAnalyzer:
    """AI phân tích chuyên sâu"""
    
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
        """Phân tích toàn diện bằng AI"""
        if not self.available:
            return self._fallback_analysis(symbol, tech_data, fund_data)
        
        # Chuẩn bị prompt chuyên nghiệp
        tech_summary = self._format_tech_data(tech_data)
        fund_summary = self._format_fund_data(fund_data) if fund_data else "Không có dữ liệu cơ bản"
        news_summary = self._format_news_data(news_data) if news_data else ""
        
        system_prompt = """Bạn là chuyên gia phân tích chứng khoán cấp cao tại công ty chứng khoán hàng đầu Việt Nam.
        Phân tích phải chuyên nghiệp, chi tiết, có số liệu cụ thể và khuyến nghị rõ ràng.
        
        Cấu trúc báo cáo:
        1. TỔNG QUAN THỊ TRƯỜNG & CỔ PHIẾU
        2. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU
        3. PHÂN TÍCH CƠ BẢN
        4. ĐÁNH GIÁ RỦI RO
        5. KHUYẾN NGHỊ ĐẦU TƯ (MUA/BÁN/GIỮ)
        
        Quy tắc:
        - Luôn đưa ra khuyến nghị: **[MUA / BÁN / GIỮ]**
        - Chỉ rõ giá mục tiêu ngắn hạn (1-3 tháng) và trung hạn (3-6 tháng)
        - Chỉ rõ điểm cắt lỗ (stop-loss)
        - Phân tích dựa trên dữ liệu thực tế được cung cấp
        - Trả lời bằng tiếng Việt chuyên nghiệp"""
        
        user_prompt = f"""Phân tích chi tiết {stock_type.upper()} **{symbol}** dựa trên dữ liệu sau:

=== DỮ LIỆU KỸ THUẬT ===
{tech_summary}

=== DỮ LIỆU CƠ BẢN ===
{fund_summary}

=== TIN TỨC ===
{news_summary}

Hãy đưa ra báo cáo phân tích chuyên sâu với các số liệu cụ thể."""
        
        models = ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"]
        
        for model in models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                analysis = response.choices[0].message.content
                
                # Extract recommendation
                rec = self._extract_recommendation(analysis)
                
                return {
                    "analysis": analysis,
                    "recommendation": rec,
                    "model_used": model
                }
                
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        return self._fallback_analysis(symbol, tech_data, fund_data)
    
    def _format_tech_data(self, data):
        ind = data.get('indicators', {})
        sig = data.get('signals', {})
        return f"""
        Xu hướng: {data.get('trend', 'N/A')}
        Giá hiện tại: {data.get('latest_price', 'N/A')}
        Thay đổi 1D: {data.get('price_change_1d', 'N/A')}%
        Thay đổi 1W: {data.get('price_change_1w', 'N/A')}%
        Thay đổi 1M: {data.get('price_change_1m', 'N/A')}%
        
        Chỉ báo:
        - RSI(14): {ind.get('RSI', 'N/A')} ({ind.get('RSI_signal', 'N/A')})
        - MACD: {ind.get('MACD', 'N/A')} | Signal: {ind.get('MACD_signal', 'N/A')} | Hist: {ind.get('MACD_histogram', 'N/A')}
        - Bollinger Bands: Upper {ind.get('BB_upper', 'N/A')} | Middle {ind.get('BB_middle', 'N/A')} | Lower {ind.get('BB_lower', 'N/A')}
        - SMA20: {ind.get('SMA20', 'N/A')} | SMA50: {ind.get('SMA50', 'N/A')} | SMA200: {ind.get('SMA200', 'N/A')}
        - Stochastic: K={ind.get('Stochastic_K', 'N/A')} D={ind.get('Stochastic_D', 'N/A')}
        - ATR(14): {ind.get('ATR', 'N/A')}
        
        Tín hiệu:
        - Golden Cross: {sig.get('golden_cross', False)}
        - Death Cross: {sig.get('death_cross', False)}
        - MACD Bullish: {sig.get('macd_bullish', False)}
        - Vị trí BB: {sig.get('price_vs_bb', 'N/A')}
        
        Hỗ trợ/Kháng cự: {data.get('support_resistance', {})}
        Fibonacci: {data.get('fibonacci', {})}
        """
    
    def _format_fund_data(self, info):
        if not info:
            return "Không có dữ liệu"
        return f"""
        P/E: {info.get('trailingPE', 'N/A')}
        Forward P/E: {info.get('forwardPE', 'N/A')}
        P/B: {info.get('priceToBook', 'N/A')}
        P/S: {info.get('priceToSalesTrailing12Months', 'N/A')}
        EPS: {info.get('trailingEps', 'N/A')}
        ROE: {(info.get('returnOnEquity', 0) or 0) * 100:.2f}%
        ROA: {(info.get('returnOnAssets', 0) or 0) * 100:.2f}%
        Debt/Equity: {info.get('debtToEquity', 'N/A')}
        Current Ratio: {info.get('currentRatio', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        Beta: {info.get('beta', 'N/A')}
        Dividend Yield: {info.get('dividendYield', 'N/A')}
        52W High: {info.get('fiftyTwoWeekHigh', 'N/A')}
        52W Low: {info.get('fiftyTwoWeekLow', 'N/A')}
        """
    
    def _format_news_data(self, news):
        if not news:
            return ""
        return "\\n".join([f"- {n.get('title', '')}" for n in news[:5]])
    
    def _extract_recommendation(self, text):
        text_upper = text.upper()
        if any(x in text_upper for x in ["MUA", "BUY", "TĂNG TỶ TRỌNG", "OVERWEIGHT"]):
            return "BUY"
        elif any(x in text_upper for x in ["BÁN", "SELL", "GIẢM TỶ TRỌNG", "UNDERWEIGHT"]):
            return "SELL"
        elif any(x in text_upper for x in ["GIỮ", "HOLD", "TRUNG LẬP", "NEUTRAL"]):
            return "HOLD"
        return "WATCH"
    
    def _fallback_analysis(self, symbol, tech_data, fund_data):
        """Phân tích dự phòng khi không có AI"""
        ind = tech_data.get('indicators', {})
        trend = tech_data.get('trend', 'SIDEWAYS')
        
        # Logic đơn giản để đưa ra khuyến nghị
        rsi = ind.get('RSI', 50)
        macd_bull = tech_data.get('signals', {}).get('macd_bullish', False)
        
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
Xu hướng hiện tại: **{trend}**
Giá hiện tại: **{tech_data.get('latest_price', 'N/A')}**

### 2. Phân Tích Kỹ Thuật
- **RSI(14)**: {ind.get('RSI', 'N/A')} - {'Quá bán' if rsi < 30 else 'Quá mua' if rsi > 70 else 'Trung tính'}
- **MACD**: {'Tín hiệu mua' if macd_bull else 'Tín hiệu bán'}
- **Bollinger Bands**: Giá đang ở vùng {tech_data.get('signals', {}).get('price_vs_bb', 'N/A')}
- **SMA**: SMA20={ind.get('SMA20', 'N/A')} | SMA50={ind.get('SMA50', 'N/A')} | SMA200={ind.get('SMA200', 'N/A')}

### 3. Khuyến Nghị
**[{rec}]** 

*Lưu ý: Đây là phân tích tự động. Vui lòng kết nối API Groq để có phân tích chuyên sâu bởi AI.*
"""
        
        return {"analysis": analysis, "recommendation": rec, "model_used": "fallback"}

# ====================== NEWS AGENT ======================
class NewsAgent:
    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                results = list(d.text(f"{symbol} cổ phiếu tin tức 2026", max_results=8))
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.error(f"News error: {e}")
            return []

# ====================== ORCHESTRATOR ======================
class Orchestrator:
    def __init__(self):
        self.data_provider = StockDataProvider()
        self.tech_analyzer = TechnicalAnalyzer()
        self.chart_gen = ChartGenerator()
        self.ai_analyzer = DeepAnalyzer()
        self.news_agent = NewsAgent()
    
    def analyze(self, symbol, stock_type="stock"):
        logger.info(f"Starting deep analysis for {symbol} ({stock_type})")
        
        # 1. Lấy dữ liệu
        if stock_type == "forex":
            raw_data = self.data_provider.get_forex_data(symbol)
            fund_data = None
        else:
            raw_data = self.data_provider.get_stock_data(symbol)
            fund_data = raw_data.get('info') if raw_data else None
        
        if not raw_data or raw_data['history'].empty:
            return {
                "mode": stock_type,
                "error": f"Không tìm thấy dữ liệu cho {symbol}. Vui lòng kiểm tra lại mã.",
                "data": None
            }
        
        df = raw_data['history']
        
        # 2. Phân tích kỹ thuật
        tech_data = self.tech_analyzer.full_analysis(df)
        
        # 3. Tạo biểu đồ
        tech_chart = self.chart_gen.create_candlestick_chart(df, symbol, tech_data)
        fund_chart = self.chart_gen.create_fundamental_chart(fund_data, symbol) if fund_data else None
        
        # 4. Lấy tin tức
        news_data = self.news_agent.get_news(symbol) if stock_type != "forex" else []
        
        # 5. Phân tích AI
        ai_result = self.ai_analyzer.analyze(symbol, stock_type, tech_data, fund_data, news_data)
        
        # 6. Tổng hợp
        result = {
            "mode": stock_type,
            "data": {
                "symbol": symbol,
                "type": stock_type,
                "analysis": ai_result["analysis"],
                "recommendation": ai_result["recommendation"],
                "model_used": ai_result.get("model_used", "unknown"),
                "technical": tech_data,
                "fundamental": {
                    "pe": fund_data.get('trailingPE') if fund_data else None,
                    "pb": fund_data.get('priceToBook') if fund_data else None,
                    "roe": (fund_data.get('returnOnEquity', 0) or 0) * 100 if fund_data else None,
                    "eps": fund_data.get('trailingEps') if fund_data else None,
                    "market_cap": fund_data.get('marketCap') if fund_data else None,
                } if fund_data else None,
                "news_count": len(news_data),
                "has_documents": False,
                "charts": {
                    "technical": tech_chart,
                    "fundamental": fund_chart
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result

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
        
        logger.info(f"Analyze request: symbol={symbol}, type={stock_type}")
        
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
        "groq": orc.ai_analyzer.available,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0-professional"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''

# Lưu file
with open('/mnt/agents/output/app.py', 'w', encoding='utf-8') as f:
    f.write(app_py_content)

print("✅ Đã tạo app.py mới với phân tích chuyên sâu")
print(f"Kích thước: {len(app_py_content)} ký tự")
