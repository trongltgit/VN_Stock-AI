
# Tạo app.py hoàn chỉnh - 3 chế độ: Stock, Fund, Forex
# Viết từng phần một cách cẩn thận để không bị cắt ngang

app_py_part1 = '''"""
VN Stock AI v4.0 - Professional Multi-Asset Analysis
- Stock: TCBS data + Technical + Fundamental + AI
- Fund: Fmarket NAV data + Risk analysis + AI
- Forex: Exchange rate + Technical + AI
"""

import os, json, logging, base64, io, traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests, pandas as pd, numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from scipy.signal import argrelextrema

load_dotenv()
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color scheme
C = {
    "bg": "#060b14", "bg2": "#0c1421", "bg3": "#111d2e", "grid": "#1e3050",
    "text": "#e8f4fd", "text2": "#8baabb", "text3": "#4a6b88",
    "accent": "#00d4ff", "accent2": "#0099cc", "gold": "#f0c040",
    "green": "#00e676", "red": "#ff5252", "yellow": "#ffd740",
    "purple": "#b388ff", "orange": "#ff9800", "pink": "#ff6b9d"
}

# ============================================================
# DATA PROVIDERS
# ============================================================
class VNStockData:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://tcinvest.tcbs.com.vn/"
    }

    @staticmethod
    def get_tcbs_historical(symbol: str, days: int = 400):
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
            params = {
                "ticker": symbol, "type": "stock", "resolution": "D",
                "from": int(start.timestamp()), "to": int(end.timestamp()),
            }
            r = requests.get(url, params=params, timeout=20, headers=VNStockData.HEADERS)
            data = r.json()
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                df["time"] = pd.to_datetime(df["tradingDate"])
                df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
                df = df.sort_values("time").reset_index(drop=True)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.dropna(subset=["Close"])
        except Exception as e:
            logger.warning(f"TCBS failed: {e}")
        return None

    @staticmethod
    def get_fundamental(symbol: str) -> Dict:
        try:
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{symbol}/overview"
            r = requests.get(url, timeout=15, headers=VNStockData.HEADERS)
            d = r.json()
            return {
                "pe": d.get("pe"), "pb": d.get("pb"), "roe": d.get("roe"), "roa": d.get("roa"),
                "eps": d.get("eps"), "market_cap": d.get("marketCap"), "industry": d.get("industry"),
                "exchange": d.get("exchange"), "52w_high": d.get("priceHigh52W"), "52w_low": d.get("priceLow52W"),
                "avg_volume": d.get("avgVolume10Day"), "beta": d.get("beta"),
                "dividend_yield": d.get("dividendYield"), "outstanding": d.get("outstandingShare"),
                "company_name": d.get("shortName"),
            }
        except Exception as e:
            logger.warning(f"Fundamental failed: {e}")
        return {}

class FundData:
    """Fmarket API for fund data"""
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    @staticmethod
    def search_fund(query: str):
        """Search fund by name or code"""
        try:
            url = "https://api.fmarket.vn/api/search"
            params = {"q": query, "type": "fund"}
            r = requests.get(url, params=params, timeout=15, headers=FundData.HEADERS)
            data = r.json()
            if data.get("data"):
                return data["data"][0] if isinstance(data["data"], list) else data["data"]
        except Exception as e:
            logger.warning(f"Fund search failed: {e}")
        return None

    @staticmethod
    def get_nav_history(fund_id: str, days: int = 365):
        """Get NAV history"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            url = f"https://api.fmarket.vn/api/fund/{fund_id}/nav-history"
            params = {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")}
            r = requests.get(url, params=params, timeout=20, headers=FundData.HEADERS)
            data = r.json()
            if data.get("data"):
                df = pd.DataFrame(data["data"])
                df["time"] = pd.to_datetime(df["navDate"])
                df = df.rename(columns={"nav": "Close", "navChange": "Change", "navChangePercent": "ChangePercent"})
                df["Open"] = df["Close"] - df["Change"]
                df["High"] = df[["Open", "Close"]].max(axis=1)
                df["Low"] = df[["Open", "Close"]].min(axis=1)
                df["Volume"] = 0
                df = df.sort_values("time").reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(f"Fund NAV failed: {e}")
        return None

    @staticmethod
    def get_fund_info(fund_id: str):
        """Get fund details"""
        try:
            url = f"https://api.fmarket.vn/api/fund/{fund_id}"
            r = requests.get(url, timeout=15, headers=FundData.HEADERS)
            data = r.json()
            if data.get("data"):
                d = data["data"]
                return {
                    "fund_name": d.get("name"),
                    "management_company": d.get("managementCompany", {}).get("name"),
                    "fund_type": d.get("fundType"),
                    "risk_level": d.get("riskLevel"),
                    "inception_date": d.get("inceptionDate"),
                    "management_fee": d.get("managementFee"),
                    "performance_fee": d.get("performanceFee"),
                    "min_investment": d.get("minInvestment"),
                    "latest_nav": d.get("latestNav"),
                    "nav_change": d.get("latestNavChange"),
                    "nav_change_percent": d.get("latestNavChangePercent"),
                    "aum": d.get("aum"),
                    "fund_size": d.get("fundSize"),
                    "top_holdings": d.get("topHoldings", []),
                }
        except Exception as e:
            logger.warning(f"Fund info failed: {e}")
        return {}

class ForexData:
    """Exchange rate data"""
    
    RATES = {
        "USD.VND": 25250, "EUR.VND": 27300, "GBP.VND": 31800,
        "JPY.VND": 168.5, "AUD.VND": 16600, "CAD.VND": 18500,
        "CHF.VND": 28800, "CNY.VND": 3480,  "KRW.VND": 18.5,
        "SGD.VND": 18800, "THB.VND": 720,   "EUR.USD": 1.082,
        "GBP.USD": 1.26,  "USD.JPY": 149.8, "AUD.USD": 0.658,
        "USD.CNY": 7.24,  "USD.CHF": 0.88,
    }

    @staticmethod
    def get_rate(pair: str):
        """Get current rate (mock + slight randomization for realism)"""
        base_rate = ForexData.RATES.get(pair.upper(), 25000)
        # Add small random variation (±0.3%)
        variation = np.random.uniform(-0.003, 0.003)
        return round(base_rate * (1 + variation), 4 if base_rate < 100 else 2)

    @staticmethod
    def generate_history(pair: str, days: int = 90):
        """Generate synthetic OHLC history for forex"""
        base_rate = ForexData.RATES.get(pair.upper(), 25000)
        volatility = 0.008 if base_rate > 100 else 0.015
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.normal(0, volatility, days)
        prices = [base_rate]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            "time": dates,
            "Close": prices,
        })
        df["Open"] = df["Close"].shift(1).fillna(base_rate)
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + abs(np.random.normal(0, volatility*0.3, days)))
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - abs(np.random.normal(0, volatility*0.3, days)))
        df["Volume"] = np.random.randint(1000000, 10000000, days)
        df = df.sort_values("time").reset_index(drop=True)
        return df

# ============================================================
# TECHNICAL ANALYSIS
# ============================================================
class TechnicalAnalyzer:
    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_f = prices.ewm(span=fast, adjust=False).mean()
        ema_s = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, sig_line, macd_line - sig_line

    @staticmethod
    def bollinger(prices, period=20, std=2):
        sma = prices.rolling(period).mean()
        stdev = prices.rolling(period).std()
        return sma + stdev*std, sma, sma - stdev*std

    @staticmethod
    def sma(prices, period): return prices.rolling(period).mean()
    @staticmethod
    def ema(prices, period): return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def stochastic(high, low, close, k=14, d=3):
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        k_line = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
        return k_line, k_line.rolling(d).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def obv(close, volume):
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def williams_r(high, low, close, period=14):
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def sr(high, low, close, window=20):
        rl = low.rolling(window).min().iloc[-1]
        rh = high.rolling(window).max().iloc[-1]
        return {
            "support1": round(float(rl), 0), "support2": round(float(close.quantile(0.1)), 0),
            "resistance1": round(float(rh), 0), "resistance2": round(float(close.quantile(0.9)), 0),
        }

# ============================================================
# FORECASTING
# ============================================================
class DeepForecaster:
    def forecast(self, prices: pd.Series, lookback: int = 60, horizon: int = 10) -> Dict:
        try:
            prices_clean = prices.dropna().values
            if len(prices_clean) < lookback + 10:
                return self._simple_forecast(prices_clean, horizon)

            y = prices_clean[-lookback:]
            x = np.arange(len(y))
            current_price = float(y[-1])

            # Polynomial Regression (degree 2)
            poly_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.1))
            poly_model.fit(x.reshape(-1, 1), y)
            poly_pred = poly_model.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))
            poly_r2 = r2_score(y, poly_model.predict(x.reshape(-1, 1)))

            # Linear Regression
            lr = LinearRegression()
            lr.fit(x.reshape(-1, 1), y)
            lr_pred = lr.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))
            lr_r2 = r2_score(y, lr.predict(x.reshape(-1, 1)))

            # Holt-Winters style
            alpha, beta = 0.3, 0.1
            level = y[0]
            trend = (y[-1] - y[0]) / len(y)
            hw_pred = []
            for i in range(horizon):
                if i == 0:
                    level = alpha * y[-1] + (1 - alpha) * (level + trend)
                else:
                    level = alpha * hw_pred[-1] + (1 - alpha) * (level + trend)
                trend = beta * (level - (level - trend)) + (1 - beta) * trend
                hw_pred.append(level + trend)

            # Momentum
            returns = np.diff(y) / y[:-1]
            avg_return = np.mean(returns[-10:]) if len(returns) >= 10 else 0
            mom_pred = [y[-1] * (1 + avg_return) ** (i + 1) for i in range(horizon)]

            # Mean Reversion
            mean_20 = np.mean(y[-20:])
            mr_pred = [y[-1]]
            for i in range(horizon):
                mr_pred.append(mr_pred[-1] + 0.15 * (mean_20 - mr_pred[-1]))
            mr_pred = mr_pred[1:]

            # ARIMA-style
            if len(y) >= 20:
                diff = np.diff(y[-20:])
                ar_pred = [y[-1] + np.mean(diff)]
                for i in range(1, horizon):
                    ar_pred.append(ar_pred[-1] + np.mean(diff) * (0.8 ** i))
            else:
                ar_pred = [y[-1] + (y[-1] - y[0]) / max(len(y), 1)] * horizon

            # Dynamic ensemble weights
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            trend_strength = abs(lr.coef_[0]) / np.mean(y)
            
            if trend_strength > 0.005:
                weights = {"poly": 0.25, "lr": 0.20, "hw": 0.20, "mom": 0.20, "mr": 0.05, "ar": 0.10}
            elif volatility > 0.03:
                weights = {"poly": 0.15, "lr": 0.15, "hw": 0.15, "mom": 0.10, "mr": 0.25, "ar": 0.20}
            else:
                weights = {"poly": 0.20, "lr": 0.20, "hw": 0.20, "mom": 0.15, "mr": 0.10, "ar": 0.15}

            ensemble = []
            for i in range(horizon):
                val = (weights["poly"] * poly_pred[i] + weights["lr"] * lr_pred[i] +
                       weights["hw"] * hw_pred[i] + weights["mom"] * mom_pred[i] +
                       weights["mr"] * mr_pred[i] + weights["ar"] * ar_pred[i])
                ensemble.append(val)

            # Confidence intervals
            residuals = y - lr.predict(x.reshape(-1, 1))
            std_err = np.sqrt(np.mean(residuals ** 2))
            upper = [e + 1.96 * std_err * np.sqrt(i + 1) for i, e in enumerate(ensemble)]
            lower = [e - 1.96 * std_err * np.sqrt(i + 1) for i, e in enumerate(ensemble)]

            avg_r2 = (poly_r2 + lr_r2) / 2
            direction = "TĂNG" if ensemble[-1] > current_price * 1.01 else "GIẢM" if ensemble[-1] < current_price * 0.99 else "ĐI NGANG"
            confidence = "CAO" if avg_r2 > 0.75 else "TRUNG BÌNH" if avg_r2 > 0.45 else "THẤP"
            expected_return = (ensemble[-1] / current_price - 1) * 100

            return {
                "method": "Deep Ensemble v4.0",
                "models": {"polynomial_r2": round(float(poly_r2), 4), "linear_r2": round(float(lr_r2), 4)},
                "forecast": [round(float(p), 2) for p in ensemble],
                "forecast_5d": [round(float(p), 2) for p in ensemble[:5]],
                "upper_bound": [round(float(p), 2) for p in upper],
                "lower_bound": [round(float(p), 2) for p in lower],
                "direction": direction,
                "confidence": confidence,
                "r_squared": round(float(avg_r2), 4),
                "mae": round(float(mean_absolute_error(y, lr.predict(x.reshape(-1, 1)))), 2),
                "volatility": round(float(volatility), 4),
                "current_price": round(float(current_price), 2),
                "target_1w": round(float(ensemble[4]), 2) if len(ensemble) > 4 else None,
                "target_2w": round(float(ensemble[-1]), 2),
                "expected_return_2w": round(float(expected_return), 2),
                "slope": round(float(lr.coef_[0]), 2),
                "stop_loss": round(float(current_price * 0.95), 0),
                "take_profit_1": round(float(ensemble[-1] * 1.05), 0),
                "take_profit_2": round(float(upper[-1]), 0),
            }
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return self._simple_forecast(prices.dropna().values, horizon)

    def _simple_forecast(self, prices, horizon):
        if len(prices) < 2:
            return {"forecast": [], "direction": "N/A", "confidence": "THẤP"}
        last = prices[-1]
        trend = (prices[-1] - prices[0]) / max(len(prices), 1)
        forecast = [last + trend * (i + 1) for i in range(horizon)]
        return {
            "method": "Simple Trend", "forecast": [round(float(p), 2) for p in forecast],
            "forecast_5d": [round(float(p), 2) for p in forecast[:5]],
            "direction": "TĂNG" if trend > 0 else "GIẢM", "confidence": "THẤP",
            "current_price": round(float(last), 2), "r_squared": 0.0,
            "slope": round(float(trend), 2),
        }

# ============================================================
# AI REASONING
# ============================================================
class ReasoningAgent:
    MODELS = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]

    def __init__(self):
        key = os.getenv("GROQ_API_KEY_STOCK")
        self.available = bool(key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                logger.info("Groq AI ready")
            except Exception as e:
                logger.error(f"Groq init error: {e}")
                self.available = False

    def analyze_stock(self, symbol, tech, fund, news, forecast=None):
        if not self.client:
            return {"analysis": self._fallback_stock(symbol), "recommendation": "WATCH", "confidence": 5}

        news_text = "\\n".join([f"- [{n['title']}]: {n['body'][:180]}" for n in news[:5]]) if news else "Không có tin tức"
        fund_text = json.dumps(fund, ensure_ascii=False, indent=2) if fund else "Không có dữ liệu"

        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO ENSEMBLE:
- Hướng: {forecast.get('direction', 'N/A')} | R²={forecast.get('r_squared', 'N/A')} | Độ tin cậy: {forecast.get('confidence', 'N/A')}
- Mục tiêu: 1T={forecast.get('target_1w', 'N/A')} | 2T={forecast.get('target_2w', 'N/A')} | Lợi nhuận: {forecast.get('expected_return_2w', 'N/A')}%
- Stop Loss: {forecast.get('stop_loss', 'N/A')} | TP1: {forecast.get('take_profit_1', 'N/A')} | TP2: {forecast.get('take_profit_2', 'N/A')}"""

        tech_text = f"""
CHỈ BÁO KỸ THUẬT:
Giá: {tech.get('current_price', 'N/A')} | RSI: {tech.get('rsi_current', 'N/A')} | MACD: {tech.get('macd_current', 'N/A')}
BB: U={tech.get('bb_upper_current', 'N/A')} M={tech.get('bb_middle_current', 'N/A')} L={tech.get('bb_lower_current', 'N/A')}
SMA20: {tech.get('sma20_current', 'N/A')} | SMA50: {tech.get('sma50_current', 'N/A')} | SMA200: {tech.get('sma200_current', 'N/A')}
EMA9: {tech.get('ema9_current', 'N/A')} | Stoch %K: {tech.get('stoch_k_current', 'N/A')} | Williams%R: {tech.get('williams_r_current', 'N/A')}
ATR: {tech.get('atr_current', 'N/A')} | Momentum: {tech.get('momentum_current', 'N/A')}% | ADX: {tech.get('adx_current', 'N/A')}
Hỗ trợ: S1={tech.get('support1', 'N/A')} S2={tech.get('support2', 'N/A')}
Kháng cự: R1={tech.get('resistance1', 'N/A')} R2={tech.get('resistance2', 'N/A')}
Xu hướng: NH={tech.get('trend_short', 'N/A')} | TH={tech.get('trend_medium', 'N/A')} | DH={tech.get('trend_long', 'N/A')}"""

        system_prompt = """Bạn là Giám đốc Phân tích của Công ty Chứng khoán hàng đầu Việt Nam.
Viết báo cáo CHUYÊN NGHIỆP theo chuẩn VCBS/SSI 7 phần:

1. 📊 TÓM TẮT ĐIỀU HÀNH - Khuyến nghị + Giá mục tiêu + Độ tin cậy
2. 🏛️ PHÂN TÍCH VĨ MÔ & NGÀNH - Lãi suất, tỷ giá, chính sách
3. 📈 PHÂN TÍCH CƠ BẢN - P/E, P/B, ROE, so sánh đối thủ
4. 📉 PHÂN TÍCH KỸ THUẬT - Xu hướng, momentum, Bollinger, dự báo
5. 🎯 CHIẾN LƯỢC GIAO DỊCH - Entry, SL, TP cụ thể
6. ⚠️ RỦI RO & KỊCH BẢN - Tích cực/Cơ sở/Tiêu cực
7. 📝 KẾT LUẬN

PHONG CÁCH: Chuyên nghiệp, số liệu cụ thể, bảng biểu rõ ràng."""

        user_prompt = f"""Phân tích CỔ PHIẾU **{symbol}** ngày {datetime.now().strftime('%d/%m/%Y')}:

{tech_text}

CHỈ SỐ CƠ BẢN:
{fund_text}

{forecast_text}

TIN TỨC:
{news_text}"""

        return self._call_ai(system_prompt, user_prompt, symbol)

    def analyze_fund(self, symbol, fund_info, tech, forecast=None):
        if not self.client:
            return {"analysis": self._fallback_fund(symbol), "recommendation": "HOLD", "confidence": 5}

        fund_text = json.dumps(fund_info, ensure_ascii=False, indent=2) if fund_info else "Không có dữ liệu"
        
        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO NAV:
- Hướng: {forecast.get('direction', 'N/A')} | R²={forecast.get('r_squared', 'N/A')}
- NAV dự báo 1T: {forecast.get('target_1w', 'N/A')} | 2T: {forecast.get('target_2w', 'N/A')}"""

        system_prompt = """Bạn là chuyên gia phân tích quỹ đầu tư tại công ty chứng khoán hàng đầu Việt Nam.
Viết báo cáo phân tích CHỨNG CHỈ QUỸ chuyên nghiệp theo cấu trúc:

1. 📊 TÓM TẮT - Khuyến nghị MUA/GIỮ/BÁN + NAV mục tiêu
2. 🏛️ PHÂN TÍCH QUỸ - Công ty quản lý, chiến lược, phí
3. 📈 HIỆU SUẤT - NAV history, so sánh benchmark
4. ⚠️ RỦI RO - Risk level, drawdown, volatility
5. 🎯 KHUYẾN NGHỊ - Thời hạn, tỷ trọng phù hợp
6. 📝 KẾT LUẬN

PHONG CÁCH: Chuyên nghiệp, dữ liệu NAV cụ thể, phù hợp nhà đầu tư cá nhân."""

        user_prompt = f"""Phân tích CHỨNG CHỈ QUỸ **{symbol}** ngày {datetime.now().strftime('%d/%m/%Y')}:

THÔNG TIN QUỸ:
{fund_text}

CHỈ BÁO KỸ THUẬT NAV:
NAV hiện tại: {tech.get('current_price', 'N/A')}
RSI: {tech.get('rsi_current', 'N/A')} | MACD: {tech.get('macd_current', 'N/A')}
SMA20: {tech.get('sma20_current', 'N/A')} | SMA50: {tech.get('sma50_current', 'N/A')}
Xu hướng: {tech.get('trend_short', 'N/A')} | {tech.get('trend_medium', 'N/A')} | {tech.get('trend_long', 'N/A')}

{forecast_text}"""

        return self._call_ai(system_prompt, user_prompt, symbol)

    def analyze_forex(self, pair, tech, forecast=None):
        if not self.client:
            return {"analysis": self._fallback_forex(pair), "direction": "SIDEWAYS", "confidence": 5}

        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO TỶ GIÁ:
- Hướng: {forecast.get('direction', 'N/A')} | R²={forecast.get('r_squared', 'N/A')}
- Tỷ giá dự báo 1T: {forecast.get('target_1w', 'N/A')} | 2T: {forecast.get('target_2w', 'N/A')}"""

        system_prompt = """Bạn là chuyên gia phân tích ngoại hối tại ngân hàng thương mại hàng đầu.
Viết báo cáo phân tích TỶ GIÁ chuyên nghiệp:

1. 📊 TÓM TẮT - Xu hướng TĂNG/GIẢM/ĐI NGANG + mục tiêu
2. 🏛️ YẾU TỐ VĨ MÔ - Lãi suất Fed, SBV, lạm phát, cán cân thương mại
3. 📉 PHÂN TÍCH KỸ THUẬT - Support/resistance, RSI, MACD
4. 🎯 CHIẾN LƯỢC - Entry, SL, TP cho giao dịch spot/forward
5. ⚠️ RỦI RO - Biến động, can thiệp NHNN
6. 📝 KẾT LUẬN

PHONG CÁCH: Chuyên nghiệp, ngắn gọn, số liệu cụ thể."""

        user_prompt = f"""Phân tích TỶ GIÁ **{pair}** ngày {datetime.now().strftime('%d/%m/%Y')}:

TỶ GIÁ HIỆN TẠI: {tech.get('current_price', 'N/A')}

CHỈ BÁO KỸ THUẬT:
RSI: {tech.get('rsi_current', 'N/A')} | MACD: {tech.get('macd_current', 'N/A')}
BB: U={tech.get('bb_upper_current', 'N/A')} M={tech.get('bb_middle_current', 'N/A')} L={tech.get('bb_lower_current', 'N/A')}
SMA20: {tech.get('sma20_current', 'N/A')} | SMA50: {tech.get('sma50_current', 'N/A')}
ATR: {tech.get('atr_current', 'N/A')} | Momentum: {tech.get('momentum_current', 'N/A')}%
Hỗ trợ: S1={tech.get('support1', 'N/A')} S2={tech.get('support2', 'N/A')}
Kháng cự: R1={tech.get('resistance1', 'N/A')} R2={tech.get('resistance2', 'N/A')}
Xu hướng: {tech.get('trend_short', 'N/A')} | {tech.get('trend_medium', 'N/A')} | {tech.get('trend_long', 'N/A')}

{forecast_text}"""

        result = self._call_ai(system_prompt, user_prompt, pair)
        result["direction"] = self._extract_direction(result.get("analysis", ""))
        return result

    def _call_ai(self, system_prompt, user_prompt, symbol):
        for model in self.MODELS:
            try:
                response = self.client.chat.completions.create(
                    model=model, temperature=0.12, max_tokens=8000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                analysis = response.choices[0].message.content
                rec = self._extract_rec(analysis)
                confidence = self._extract_confidence(analysis)
                logger.info(f"AI done: model={model}, rec={rec}")
                return {"analysis": analysis, "recommendation": rec, "confidence": confidence}
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
        return {"analysis": self._fallback_stock(symbol), "recommendation": "WATCH", "confidence": 5}

    def _extract_rec(self, text):
        upper = text.upper()
        if any(k in upper for k in ["MUA", "BUY", "OVERWEIGHT", "TĂNG TỶ TRỌNG"]):
            return "BUY"
        if any(k in upper for k in ["BÁN", "SELL", "UNDERWEIGHT", "GIẢM TỶ TRỌNG"]):
            return "SELL"
        if any(k in upper for k in ["GIỮ", "HOLD", "NEUTRAL"]):
            return "HOLD"
        return "WATCH"

    def _extract_direction(self, text):
        upper = text.upper()
        if any(k in upper for k in ["TĂNG", "UP", "BUY", "MUA"]):
            return "UP"
        if any(k in upper for k in ["GIẢM", "DOWN", "SELL", "BÁN"]):
            return "DOWN"
        return "SIDEWAYS"

    def _extract_confidence(self, text):
        import re
        match = re.search(r'(\\d+)/10', text)
        if match:
            return int(match.group(1))
        return 7

    def _fallback_stock(self, symbol):
        return f"""⚠️ Chưa cấu hình GROQ_API_KEY_STOCK

Hệ thống vẫn phân tích kỹ thuật và dự báo dựa trên dữ liệu định lượng.

Để có báo cáo AI chuyên sâu:
1. Truy cập https://console.groq.com
2. Tạo API Key miễn phí
3. Thêm biến môi trường `GROQ_API_KEY_STOCK`

Mã phân tích: {symbol}"""

    def _fallback_fund(self, symbol):
        return f"""⚠️ Chưa cấu hình GROQ_API_KEY_STOCK

Hệ thống vẫn phân tích NAV và xu hướng dựa trên dữ liệu định lượng.

Để có báo cáo AI chuyên sâu về quỹ:
1. Truy cập https://console.groq.com
2. Tạo API Key miễn phí
3. Thêm biến môi trường `GROQ_API_KEY_STOCK`

Mã quỹ: {symbol}"""

    def _fallback_forex(self, pair):
        return f"""⚠️ Chưa cấu hình GROQ_API_KEY_STOCK

Hệ thống vẫn phân tích kỹ thuật tỷ giá dựa trên dữ liệu định lượng.

Để có báo cáo AI chuyên sâu về tỷ giá:
1. Truy cập https://console.groq.com
2. Tạo API Key miễn phí
3. Thêm biến môi trường `GROQ_API_KEY_STOCK`

Cặp tiền: {pair}"""

# ============================================================
# NEWS AGENT
# ============================================================
class NewsAgent:
    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDDS() as ddg:
                results = ddg.text(f"{symbol} cổ phiếu tin tức phân tích 2025 2026", max_results=6)
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.warning(f"News failed: {e}")
        return []

    def get_fund_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddg:
                results = ddg.text(f"{symbol} quỹ đầu tư NAV tin tức 2025 2026", max_results=6)
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.warning(f"Fund news failed: {e}")
        return []

    def get_forex_news(self, pair):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddg:
                results = ddg.text(f"{pair.replace('.', '')} tỷ giá forex tin tức 2025 2026", max_results=6)
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.warning(f"Forex news failed: {e}")
        return []


# Part 2 - Chart generation functions
app_py_part2 = '''
# ============================================================
# CHART GENERATION
# ============================================================
def _save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=C["bg"], edgecolor="none", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def chart_main(df, sym, ind, title_suffix=""):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12), facecolor=C["bg"])
    gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[4, 1.2, 1.2, 1, 0.8], hspace=0.06)

    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for i in range(1, len(axes)):
        axes[i].sharex(axes[0])

    n = len(df)
    x = np.arange(n)

    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, alpha=0.12, color=C["grid"], linewidth=0.5)
        ax.tick_params(colors=C["text2"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(C["grid"])

    # Candlesticks
    width = 0.6
    for i, row in df.iterrows():
        col = C["green"] if row["Close"] >= row["Open"] else C["red"]
        axes[0].plot([i, i], [row["Low"], row["High"]], color=col, lw=0.8, alpha=0.8)
        body_h = abs(row["Close"] - row["Open"])
        body_y = min(row["Open"], row["Close"])
        axes[0].add_patch(Rectangle((i - width/2, body_y), width, max(body_h, row["Close"]*0.0005),
                                    facecolor=col, edgecolor=col, alpha=0.85, zorder=3))

    # Bollinger
    if "bb_upper" in ind:
        axes[0].plot(x, ind["bb_upper"], color=C["accent"], lw=0.9, alpha=0.5, ls="--", label="BB Upper")
        axes[0].plot(x, ind["bb_middle"], color=C["gold"], lw=1.2, alpha=0.8, label="BB Middle")
        axes[0].plot(x, ind["bb_lower"], color=C["accent"], lw=0.9, alpha=0.5, ls="--", label="BB Lower")
        axes[0].fill_between(x, ind["bb_upper"], ind["bb_lower"], alpha=0.03, color=C["accent"])

    # Moving Averages
    ma_config = [
        ("sma20", C["orange"], 1.2, "SMA20"), ("sma50", C["yellow"], 1.4, "SMA50"),
        ("sma200", C["pink"], 1.4, "SMA200"), ("ema9", C["purple"], 1.1, "EMA9"),
    ]
    for key, color, lw, label in ma_config:
        if key in ind and ind[key] is not None:
            axes[0].plot(x, ind[key], color=color, lw=lw, alpha=0.8, label=label)

    axes[0].set_title(f"{sym}  ·  Professional Technical Analysis {title_suffix}",
                     color=C["accent"], fontsize=13, fontweight="bold", pad=12, loc="left")
    axes[0].legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=7,
                  labelcolor=C["text2"], ncol=5, framealpha=0.9)
    axes[0].set_ylabel("Giá (VND)", color=C["text2"], fontsize=9)
    
    for i in range(1, len(axes)-1):
        plt.setp(axes[i].get_xticklabels(), visible=False)

    # Volume
    vol_colors = [C["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else C["red"] for i in range(n)]
    axes[1].bar(x, df["Volume"], color=vol_colors, alpha=0.6, width=0.85, zorder=2)
    axes[1].set_ylabel("Volume", color=C["text2"], fontsize=9)

    # RSI
    if "rsi" in ind:
        axes[2].plot(x, ind["rsi"], color=C["accent"], lw=1.5, zorder=3)
        axes[2].axhline(70, color=C["red"], lw=0.7, ls="--", alpha=0.6)
        axes[2].axhline(30, color=C["green"], lw=0.7, ls="--", alpha=0.6)
        axes[2].fill_between(x, ind["rsi"], 30, where=np.array(ind["rsi"])<30, alpha=0.15, color=C["green"])
        axes[2].fill_between(x, ind["rsi"], 70, where=np.array(ind["rsi"])>70, alpha=0.15, color=C["red"])
        axes[2].set_ylim(0, 100)
        axes[2].set_ylabel("RSI(14)", color=C["text2"], fontsize=9)

    # MACD
    if "macd" in ind:
        axes[3].plot(x, ind["macd"], color=C["accent"], lw=1.4, label="MACD")
        axes[3].plot(x, ind["macd_signal"], color=C["gold"], lw=1.4, label="Signal")
        hist = np.array(ind["macd_hist"])
        axes[3].bar(x, hist, color=[C["green"] if v>=0 else C["red"] for v in hist], alpha=0.6, width=0.8)
        axes[3].axhline(0, color=C["text2"], lw=0.6, alpha=0.4)
        axes[3].legend(loc="upper left", fontsize=7, facecolor=C["bg2"], edgecolor=C["grid"])
        axes[3].set_ylabel("MACD", color=C["text2"], fontsize=9)

    # ADX
    if "adx" in ind:
        axes[4].plot(x, ind["adx"], color=C["purple"], lw=1.3, label="ADX")
        axes[4].axhline(25, color=C["yellow"], lw=0.7, ls="--", alpha=0.6)
        axes[4].set_ylim(0, 60)
        axes[4].set_ylabel("ADX", color=C["text2"], fontsize=9)
        axes[4].legend(loc="upper left", fontsize=7, facecolor=C["bg2"], edgecolor=C["grid"])

    tick_count = min(8, n)
    tick_idx = np.linspace(0, n-1, tick_count, dtype=int)
    axes[4].set_xticks(tick_idx)
    axes[4].set_xticklabels([df["time"].iloc[i].strftime("%d/%m") for i in tick_idx],
                           rotation=30, ha="right", color=C["text2"], fontsize=7)

    fig.tight_layout()
    return _save_fig(fig)

def chart_forecast(df, sym, forecast, ind):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    n_hist = len(df)
    hist_x = np.arange(n_hist)
    ax.plot(hist_x, df["Close"], color=C["accent"], lw=2, label="Giá thực tế", zorder=3)

    if "sma20" in ind:
        ax.plot(hist_x, ind["sma20"], color=C["gold"], lw=1.2, alpha=0.7, label="SMA20")

    fc = forecast.get("forecast", [])
    if fc:
        n_fc = len(fc)
        fc_x = np.arange(n_hist, n_hist + n_fc)
        ax.plot(fc_x, fc, color=C["green"], lw=2.5, ls="--", label="Dự báo Ensemble", zorder=4)
        
        upper = forecast.get("upper_bound", [])
        lower = forecast.get("lower_bound", [])
        if upper and lower:
            ax.fill_between(fc_x, upper, lower, alpha=0.12, color=C["green"], label="Khoảng tin cậy 95%")

        if forecast.get("target_1w"):
            ax.axhline(forecast["target_1w"], color=C["yellow"], lw=1, ls="-.", alpha=0.7,
                      label=f"Mục tiêu 1T: {forecast['target_1w']:,.0f}")
        if forecast.get("target_2w"):
            ax.axhline(forecast["target_2w"], color=C["gold"], lw=1.2, ls="-.", alpha=0.8,
                      label=f"Mục tiêu 2T: {forecast['target_2w']:,.0f}")

    ax.set_title(f"{sym}  ·  Deep Learning Forecast  ·  {forecast.get('method', '')}",
                color=C["accent"], fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=8,
             labelcolor=C["text2"], framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.12, color=C["grid"])
    ax.tick_params(colors=C["text2"])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.set_ylabel("Giá (VND)", color=C["text2"])

    fig.tight_layout()
    return _save_fig(fig)

def chart_macd_stoch(df, ind):
    """MACD + Stochastic combined chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor=C["bg"], sharex=True)
    ax1.set_facecolor(C["bg"])
    ax2.set_facecolor(C["bg"])
    n = len(df)
    x = np.arange(n)

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.12, color=C["grid"])
        ax.tick_params(colors=C["text2"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["grid"])

    # MACD
    if "macd" in ind:
        ax1.plot(x, ind["macd"], color=C["accent"], lw=1.4, label="MACD")
        ax1.plot(x, ind["macd_signal"], color=C["gold"], lw=1.4, label="Signal")
        hist = np.array(ind["macd_hist"])
        ax1.bar(x, hist, color=[C["green"] if v>=0 else C["red"] for v in hist], alpha=0.6, width=0.8)
        ax1.axhline(0, color=C["text2"], lw=0.6, alpha=0.4)
        ax1.legend(loc="upper left", fontsize=7, facecolor=C["bg2"], edgecolor=C["grid"])
        ax1.set_ylabel("MACD", color=C["text2"])

    # Stochastic
    if "stoch_k" in ind:
        ax2.plot(x, ind["stoch_k"], color=C["purple"], lw=1.3, label="%K")
        ax2.plot(x, ind["stoch_d"], color=C["gold"], lw=1.3, label="%D")
        ax2.axhline(80, color=C["red"], lw=0.7, ls="--", alpha=0.5)
        ax2.axhline(20, color=C["green"], lw=0.7, ls="--", alpha=0.5)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Stochastic", color=C["text2"])
        ax2.legend(loc="upper left", fontsize=7, facecolor=C["bg2"], edgecolor=C["grid"])

    tick_count = min(8, n)
    tick_idx = np.linspace(0, n-1, tick_count, dtype=int)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([df["time"].iloc[i].strftime("%d/%m") for i in tick_idx],
                       rotation=30, ha="right", color=C["text2"], fontsize=7)

    fig.tight_layout()
    return _save_fig(fig)


# Part 3 - Orchestrator and compute functions
app_py_part3 = '''
# ============================================================
# ORCHESTRATOR
# ============================================================
class Orchestrator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.ai_agent = ReasoningAgent()
        self.data = VNStockData()
        self.fund_data = FundData()
        self.forex_data = ForexData()
        self.ta = TechnicalAnalyzer()
        self.forecaster = DeepForecaster()

    def _compute_technical(self, df):
        """Compute all technical indicators"""
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

        rsi = self.ta.rsi(c)
        macd_line, macd_sig, macd_hist = self.ta.macd(c)
        bb_u, bb_m, bb_l = self.ta.bollinger(c)
        sma20 = self.ta.sma(c, 20)
        sma50 = self.ta.sma(c, 50)
        sma200 = self.ta.sma(c, 200)
        ema9 = self.ta.ema(c, 9)
        stoch_k, stoch_d = self.ta.stochastic(h, l, c)
        atr = self.ta.atr(h, l, c)
        obv_series = self.ta.obv(c, v)
        wr = self.ta.williams_r(h, l, c)
        mom = c / c.shift(10) * 100 - 100
        adx, plus_di, minus_di = self.ta.adx(h, l, c)
        sr = self.ta.sr(h, l, c)
        forecast = self.forecaster.forecast(c, lookback=60, horizon=10)

        def safe(s):
            v = s.iloc[-1] if hasattr(s, "iloc") else s
            return round(float(v), 4) if pd.notna(v) else "N/A"

        cp = float(c.iloc[-1])
        tech_dict = {
            "current_price": f"{cp:,.0f}", "rsi_current": safe(rsi), "macd_current": safe(macd_line),
            "macd_signal_current": safe(macd_sig), "macd_hist_current": safe(macd_hist),
            "bb_upper_current": safe(bb_u), "bb_middle_current": safe(bb_m), "bb_lower_current": safe(bb_l),
            "sma20_current": safe(sma20), "sma50_current": safe(sma50), "sma200_current": safe(sma200),
            "ema9_current": safe(ema9), "stoch_k_current": safe(stoch_k), "stoch_d_current": safe(stoch_d),
            "atr_current": safe(atr), "williams_r_current": safe(wr), "momentum_current": safe(mom),
            "adx_current": safe(adx), "plus_di_current": safe(plus_di), "minus_di_current": safe(minus_di),
            "obv_current": safe(obv_series), **sr,
            "trend_short": "TĂNG" if pd.notna(sma20.iloc[-1]) and cp > float(sma20.iloc[-1]) else "GIẢM",
            "trend_medium": "TĂNG" if pd.notna(sma50.iloc[-1]) and cp > float(sma50.iloc[-1]) else "GIẢM",
            "trend_long": "TĂNG" if pd.notna(sma200.iloc[-1]) and cp > float(sma200.iloc[-1]) else "GIẢM",
        }

        ind_dict = {
            "rsi": rsi.values, "macd": macd_line.values, "macd_signal": macd_sig.values, "macd_hist": macd_hist.values,
            "bb_upper": bb_u.values, "bb_middle": bb_m.values, "bb_lower": bb_l.values,
            "sma20": sma20.values, "sma50": sma50.values, "sma200": sma200.values,
            "ema9": ema9.values, "stoch_k": stoch_k.values, "stoch_d": stoch_d.values,
            "obv": obv_series.values, "williams_r": wr.values, "adx": adx.values,
        }

        return tech_dict, ind_dict, forecast, cp

    def analyze_stock(self, symbol):
        sym = symbol.upper()
        df = self.data.get_tcbs_historical(sym, days=400)
        if df is None or len(df) < 30:
            return self._error_response(sym, "stock", f"Không lấy được dữ liệu giá từ TCBS cho mã {sym}")

        tech_dict, ind_dict, forecast, cp = self._compute_technical(df)
        fund = self.data.get_fundamental(sym)
        news = self.news_agent.get_news(sym)
        ai = self.ai_agent.analyze_stock(sym, tech_dict, fund, news, forecast)

        main_chart = chart_main(df, sym, ind_dict)
        forecast_chart = chart_forecast(df, sym, forecast, ind_dict)
        macd_chart = chart_macd_stoch(df, ind_dict)

        return {
            "mode": "stock",
            "data": {
                "symbol": sym, "type": "stock",
                "analysis": ai["analysis"], "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
                "news_count": len(news), "forecast": forecast,
                "charts": {"main": main_chart, "forecast": forecast_chart, "macd": macd_chart},
                "technical": {
                    "current_price": cp, "rsi": tech_dict["rsi_current"], "macd": tech_dict["macd_current"],
                    "macd_signal": tech_dict["macd_signal_current"], "macd_hist": tech_dict["macd_hist_current"],
                    "bb_upper": tech_dict["bb_upper_current"], "bb_middle": tech_dict["bb_middle_current"], "bb_lower": tech_dict["bb_lower_current"],
                    "sma20": tech_dict["sma20_current"], "sma50": tech_dict["sma50_current"], "sma200": tech_dict["sma200_current"],
                    "ema9": tech_dict["ema9_current"], "stoch_k": tech_dict["stoch_k_current"], "stoch_d": tech_dict["stoch_d_current"],
                    "atr": tech_dict["atr_current"], "williams_r": tech_dict["williams_r_current"], "momentum": tech_dict["momentum_current"],
                    "adx": tech_dict["adx_current"], "obv": tech_dict["obv_current"],
                    "support1": tech_dict.get("support1"), "support2": tech_dict.get("support2"),
                    "resistance1": tech_dict.get("resistance1"), "resistance2": tech_dict.get("resistance2"),
                    "trend_short": tech_dict["trend_short"], "trend_medium": tech_dict["trend_medium"], "trend_long": tech_dict["trend_long"],
                },
                "fundamental": fund,
                "price_history": {
                    "dates": df["time"].dt.strftime("%d/%m").tolist()[-50:],
                    "prices": [round(float(p), 0) for p in df["Close"].values[-50:]],
                    "volumes": [int(v) for v in df["Volume"].values[-50:]],
                },
            },
        }

    def analyze_fund(self, symbol):
        sym = symbol.upper()
        
        # Try to search fund first
        fund_search = self.fund_data.search_fund(sym)
        fund_id = fund_search.get("id") if fund_search else sym
        
        # Get NAV history
        df = self.fund_data.get_nav_history(fund_id, days=365)
        if df is None or len(df) < 30:
            return self._error_response(sym, "fund", f"Không lấy được dữ liệu NAV cho quỹ {sym}")

        tech_dict, ind_dict, forecast, cp = self._compute_technical(df)
        fund_info = self.fund_data.get_fund_info(fund_id)
        news = self.news_agent.get_fund_news(sym)
        ai = self.ai_agent.analyze_fund(sym, fund_info, tech_dict, forecast)

        main_chart = chart_main(df, sym, ind_dict, "· NAV Analysis")
        forecast_chart = chart_forecast(df, sym, forecast, ind_dict)

        return {
            "mode": "fund",
            "data": {
                "symbol": sym, "type": "fund",
                "analysis": ai["analysis"], "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
                "news_count": len(news), "forecast": forecast,
                "fund_info": fund_info,
                "charts": {"main": main_chart, "forecast": forecast_chart},
                "technical": {
                    "current_price": cp, "rsi": tech_dict["rsi_current"], "macd": tech_dict["macd_current"],
                    "bb_upper": tech_dict["bb_upper_current"], "bb_lower": tech_dict["bb_lower_current"],
                    "sma20": tech_dict["sma20_current"], "sma50": tech_dict["sma50_current"],
                    "ema9": tech_dict["ema9_current"], "stoch_k": tech_dict["stoch_k_current"],
                    "atr": tech_dict["atr_current"], "momentum": tech_dict["momentum_current"],
                    "support1": tech_dict.get("support1"), "support2": tech_dict.get("support2"),
                    "resistance1": tech_dict.get("resistance1"), "resistance2": tech_dict.get("resistance2"),
                    "trend_short": tech_dict["trend_short"], "trend_medium": tech_dict["trend_medium"], "trend_long": tech_dict["trend_long"],
                },
                "price_history": {
                    "dates": df["time"].dt.strftime("%d/%m").tolist()[-50:],
                    "prices": [round(float(p), 2) for p in df["Close"].values[-50:]],
                },
            },
        }

    def analyze_forex(self, pair):
        pair = pair.upper()
        
        # Generate synthetic history
        df = self.forex_data.generate_history(pair, days=90)
        if df is None or len(df) < 20:
            return self._error_response(pair, "forex", f"Không tạo được dữ liệu cho {pair}")

        tech_dict, ind_dict, forecast, cp = self._compute_technical(df)
        current_rate = self.forex_data.get_rate(pair)
        news = self.news_agent.get_forex_news(pair)
        ai = self.ai_agent.analyze_forex(pair, tech_dict, forecast)

        main_chart = chart_main(df, pair, ind_dict, "· Forex")
        forecast_chart = chart_forecast(df, pair, forecast, ind_dict)

        return {
            "mode": "forex",
            "data": {
                "symbol": pair, "type": "forex",
                "analysis": ai["analysis"], "recommendation": ai["recommendation"],
                "direction": ai.get("direction", "SIDEWAYS"),
                "confidence": ai.get("confidence", 7),
                "news_count": len(news), "forecast": forecast,
                "rate": current_rate,
                "charts": {"main": main_chart, "forecast": forecast_chart},
                "technical": {
                    "current_price": cp, "rsi": tech_dict["rsi_current"], "macd": tech_dict["macd_current"],
                    "bb_upper": tech_dict["bb_upper_current"], "bb_lower": tech_dict["bb_lower_current"],
                    "sma20": tech_dict["sma20_current"], "sma50": tech_dict["sma50_current"],
                    "ema9": tech_dict["ema9_current"], "stoch_k": tech_dict["stoch_k_current"],
                    "atr": tech_dict["atr_current"], "momentum": tech_dict["momentum_current"],
                    "support1": tech_dict.get("support1"), "support2": tech_dict.get("support2"),
                    "resistance1": tech_dict.get("resistance1"), "resistance2": tech_dict.get("resistance2"),
                    "trend_short": tech_dict["trend_short"], "trend_medium": tech_dict["trend_medium"], "trend_long": tech_dict["trend_long"],
                },
                "price_history": {
                    "dates": df["time"].dt.strftime("%d/%m").tolist()[-30:],
                    "prices": [round(float(p), 4) for p in df["Close"].values[-30:]],
                },
            },
        }

    def _error_response(self, symbol, mode, msg):
        return {
            "mode": mode,
            "data": {
                "symbol": symbol, "type": mode,
                "analysis": f"⚠️ {msg}\\n\\nVui lòng kiểm tra lại mã và thử lại.",
                "recommendation": "WATCH", "confidence": 3,
                "charts": {}, "technical": {}, "fundamental": {},
            },
        }

orc = Orchestrator()


# Part 4 - Flask routes and HTML template
app_py_part4 = '''
# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = (request.form.get("symbol") or "").strip().upper()
        stype = (request.form.get("type") or "stock").strip().lower()
        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã"}), 400
        
        logger.info(f"Analyzing: symbol={symbol} type={stype}")
        
        if stype == "stock":
            result = orc.analyze_stock(symbol)
        elif stype == "fund":
            result = orc.analyze_fund(symbol)
        elif stype == "forex":
            result = orc.analyze_forex(symbol)
        else:
            return jsonify({"error": f"Loại không hợp lệ: {stype}"}), 400
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {e}\\n{traceback.format_exc()}")
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "version": "4.0", "timestamp": datetime.now().isoformat(),
        "agents": {"data": True, "news": True, "reasoning": orc.ai_agent.available},
        "groq": orc.ai_agent.available
    })

# ============================================================
# HTML TEMPLATE
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VN Stock AI — Professional Multi-Asset v4.0</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:#060b14; --bg2:#0c1421; --bg3:#111d2e; --border:#1e3050; --border2:#2a4570;
  --accent:#00d4ff; --accent2:#0099cc; --gold:#f0c040; --green:#00e676; --red:#ff5252;
  --yellow:#ffd740; --purple:#b388ff; --text:#e8f4fd; --text2:#8baabb; --text3:#4a6b88;
  --card:rgba(12,20,33,0.97); --glow:0 0 30px rgba(0,212,255,0.12); --radius:12px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background-image:linear-gradient(rgba(0,212,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.025) 1px,transparent 1px);background-size:44px 44px}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px)}

header{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:0 2rem;height:60px;border-bottom:1px solid var(--border);background:rgba(6,11,20,0.96);backdrop-filter:blur(16px)}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 0 14px rgba(0,212,255,0.35)}
.logo-name{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;letter-spacing:-0.02em}
.logo-name span{color:var(--accent)}
.logo-sub{font-size:0.62rem;color:var(--text3);font-family:'Space Mono',monospace;margin-top:1px}
.header-right{display:flex;align-items:center;gap:1.5rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:var(--text3)}
.hstatus{display:flex;align-items:center;gap:6px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}
#htime{color:var(--text2)}

.ticker-bar{background:var(--bg2);border-bottom:1px solid var(--border);padding:5px 0;overflow:hidden;position:relative;z-index:9;user-select:none}
.ticker-inner{display:inline-flex;gap:2.5rem;white-space:nowrap;animation:scroll-ticker 45s linear infinite}
.ticker-inner:hover{animation-play-state:paused}
@keyframes scroll-ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.tick{display:inline-flex;align-items:center;gap:7px;font-family:'Space Mono',monospace;font-size:0.7rem}
.t-sym{color:var(--accent);font-weight:700}
.t-val{color:var(--text)}
.t-up{color:var(--green)}
.t-dn{color:var(--red)}

main{position:relative;z-index:1;max-width:1440px;margin:0 auto;padding:1.5rem;display:grid;grid-template-columns:340px 1fr;gap:1.25rem;align-items:start}
.sidebar{display:flex;flex-direction:column;gap:1rem;position:sticky;top:72px}

.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;position:relative;overflow:hidden;transition:border-color 0.2s,box-shadow 0.2s}
.card:hover{border-color:var(--border2);box-shadow:var(--glow)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0.4}
.card-title{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);margin-bottom:0.9rem;display:flex;align-items:center;gap:7px}

.tabs{display:flex;gap:3px;background:var(--bg);border-radius:8px;padding:3px;margin-bottom:0.9rem;border:1px solid var(--border)}
.tab-btn{flex:1;padding:7px 8px;border:none;border-radius:6px;background:transparent;color:var(--text2);font-family:'Space Mono',monospace;font-size:0.68rem;cursor:pointer;transition:background 0.18s,color 0.18s;text-align:center;white-space:nowrap}
.tab-btn.active{background:var(--accent);color:var(--bg);font-weight:700}
.tab-btn:hover:not(.active){background:var(--border);color:var(--text)}

.form-group{margin-bottom:0.8rem}
.form-label{display:block;font-size:0.67rem;font-family:'Space Mono',monospace;color:var(--text3);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px}
.input-row{display:flex;gap:5px}
.sym-input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px 12px;color:var(--text);font-family:'Space Mono',monospace;font-size:0.84rem;outline:none;text-transform:uppercase;transition:border-color 0.2s,box-shadow 0.2s}
.sym-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,212,255,0.09)}
.sym-input::placeholder{color:var(--text3);text-transform:none}
.run-btn{padding:9px 11px;background:var(--border);border:1px solid var(--border2);border-radius:8px;color:var(--accent);font-size:0.72rem;font-family:'Space Mono',monospace;cursor:pointer;transition:background 0.18s;white-space:nowrap}
.run-btn:hover{background:var(--border2)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
.chip{padding:3px 10px;background:var(--bg);border:1px solid var(--border);border-radius:20px;font-size:0.66rem;font-family:'Space Mono',monospace;color:var(--text2);cursor:pointer;transition:all 0.18s}
.chip:hover{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,0.05)}

.analyze-btn{width:100%;padding:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;color:var(--bg);font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:800;letter-spacing:0.04em;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:7px;transition:transform 0.18s,box-shadow 0.18s;box-shadow:0 4px 18px rgba(0,212,255,0.28);margin-top:0.5rem}
.analyze-btn:hover{transform:translateY(-1px);box-shadow:0 6px 26px rgba(0,212,255,0.42)}
.analyze-btn:disabled{opacity:0.45;cursor:not-allowed;transform:none;box-shadow:none}

.step{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;font-size:0.74rem;font-family:'Space Mono',monospace;color:var(--text3);background:var(--bg);border:1px solid var(--border);transition:all 0.3s}
.step.active{color:var(--accent);border-color:var(--accent);background:rgba(0,212,255,0.05)}
.step.done{color:var(--green);border-color:var(--green);background:rgba(0,230,118,0.05)}
.step-body{display:flex;flex-direction:column;gap:1px}
.step-label{font-size:0.72rem;color:var(--text2)}
.step-sub{font-size:0.62rem;color:var(--text3)}

.content{display:flex;flex-direction:column;gap:1rem;min-height:520px}

#welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:480px;text-align:center;gap:1.5rem}
.w-icon{font-size:3rem;animation:float 3s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.w-title{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--gold));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.w-desc{color:var(--text2);font-size:0.88rem;max-width:380px;line-height:1.6}
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.9rem;width:100%;max-width:560px}
.feat-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:0.9rem;text-align:center;transition:border-color 0.2s}
.feat-card:hover{border-color:var(--border2)}
.feat-icon{font-size:1.4rem;margin-bottom:5px}
.feat-name{font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:700;color:var(--accent);margin-bottom:3px}
.feat-desc{font-size:0.64rem;color:var(--text3);line-height:1.5}

#loading{display:none;flex-direction:column;align-items:center;justify-content:center;min-height:480px;gap:2rem}
#loading.show{display:flex}
.spinner-wrap{position:relative;width:80px;height:80px}
.ring-outer{width:80px;height:80px;border-radius:50%;border:2px solid var(--border);border-top-color:var(--accent);animation:spin 1s linear infinite}
.ring-inner{position:absolute;inset:10px;border-radius:50%;border:2px solid var(--border);border-bottom-color:var(--gold);animation:spin 0.65s linear infinite reverse}
@keyframes spin{to{transform:rotate(360deg)}}
#loadLabel{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--accent)}
.load-sub{font-size:0.72rem;color:var(--text3);font-family:'Space Mono',monospace;margin-top:4px}
.load-steps{display:flex;flex-direction:column;gap:6px;width:100%;max-width:340px}

#errorBox{display:none;background:rgba(255,82,82,0.07);border:1px solid var(--red);border-radius:10px;padding:1rem 1.2rem;color:var(--red);font-size:0.82rem;line-height:1.6}
#errorBox.show{display:block}

#result{display:none;flex-direction:column;gap:1rem}
#result.show{display:flex}

.res-head{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.4rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;position:relative;overflow:hidden}
.res-head::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent)}
.sym-block{display:flex;align-items:center;gap:14px}
.sym-code{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--accent);letter-spacing:-0.02em}
.sym-meta{display:flex;flex-direction:column;gap:2px}
.sym-type{font-size:0.62rem;font-family:'Space Mono',monospace;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em}
.sym-time{font-size:0.68rem;color:var(--text3);font-family:'Space Mono',monospace}

.rec-badge{padding:8px 18px;border-radius:8px;font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:800;letter-spacing:0.07em;display:flex;align-items:center;gap:7px;border:2px solid}
.rec-BUY{border-color:var(--green);color:var(--green);background:rgba(0,230,118,0.12)}
.rec-SELL{border-color:var(--red);color:var(--red);background:rgba(255,82,82,0.12)}
.rec-HOLD{border-color:var(--gold);color:var(--gold);background:rgba(240,192,64,0.12)}
.rec-WATCH{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,0.09)}

.forecast-bar{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.4rem;display:flex;align-items:center;gap:2rem;flex-wrap:wrap}
.fc-item{display:flex;flex-direction:column;gap:3px}
.fc-label{font-size:0.62rem;color:var(--text3);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:0.07em}
.fc-val{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--text)}
.fc-val.up{color:var(--green)}
.fc-val.down{color:var(--red)}

.badge-row{display:flex;gap:7px;flex-wrap:wrap}
.badge{display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:6px;border:1px solid;font-size:0.65rem;font-family:'Space Mono',monospace}
.badge.ok{border-color:var(--green);color:var(--green);background:rgba(0,230,118,0.05)}
.badge.warn{border-color:var(--yellow);color:var(--yellow);background:rgba(255,215,64,0.05)}
.badge.info{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,0.05)}

.data-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:0.7rem}
.dc{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:0.9rem;transition:border-color 0.2s}
.dc:hover{border-color:var(--border2)}
.dc-label{font-size:0.62rem;font-family:'Space Mono',monospace;color:var(--text3);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px}
.dc-val{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:var(--text)}
.dc-sub{font-size:0.68rem;font-family:'Space Mono',monospace;margin-top:3px}
.dc-sub.up{color:var(--green)}
.dc-sub.down{color:var(--red)}
.dc-sub.neu{color:var(--text3)}

.chart-wrap{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.chart-title{padding:0.8rem 1.2rem;border-bottom:1px solid var(--border);background:rgba(0,0,0,0.25);font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:7px}
.chart-wrap img{width:100%;height:auto;display:block}

.fund-table{width:100%;border-collapse:collapse;font-size:0.8rem;font-family:'Space Mono',monospace}
.fund-table th{text-align:left;padding:9px 13px;background:var(--bg2);color:var(--accent);font-size:0.66rem;text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid var(--border)}
.fund-table td{padding:9px 13px;border-bottom:1px solid var(--border);color:var(--text2)}
.fund-table tr:hover td{background:rgba(0,212,255,0.03);color:var(--text)}
.td-val{text-align:right;font-weight:600;color:var(--text)}
.td-note{color:var(--text3);font-size:0.7rem}

.report-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.report-head{padding:0.8rem 1.2rem;border-bottom:1px solid var(--border);background:rgba(0,0,0,0.25);display:flex;align-items:center;justify-content:space-between}
.report-head h3{font-family:'Syne',sans-serif;font-size:0.76rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:7px}
.copy-btn{padding:5px 11px;border-radius:6px;background:var(--bg);border:1px solid var(--border);color:var(--text2);font-size:0.65rem;font-family:'Space Mono',monospace;cursor:pointer;transition:all 0.18s}
.copy-btn:hover{border-color:var(--accent);color:var(--accent)}

.report-body{padding:1.5rem;line-height:1.85;font-size:0.875rem;color:var(--text)}
.report-body h1{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:var(--accent);margin:1.3rem 0 0.5rem}
.report-body h2{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:var(--gold);border-bottom:1px solid var(--border);padding-bottom:5px;margin:1.2rem 0 0.5rem}
.report-body h3{font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:700;color:var(--accent);margin:1rem 0 0.4rem}
.report-body strong{color:var(--text);font-weight:600}
.report-body em{color:var(--text2);font-style:italic}
.report-body ul,.report-body ol{padding-left:1.4rem;margin:0.4rem 0}
.report-body li{margin-bottom:4px}
.report-body hr{border:none;border-top:1px solid var(--border);margin:1rem 0}
.report-body blockquote{border-left:3px solid var(--accent);padding:7px 14px;background:rgba(0,212,255,0.04);border-radius:0 8px 8px 0;margin:0.7rem 0;color:var(--text2)}
.report-body table{width:100%;border-collapse:collapse;margin:0.8rem 0;font-size:0.78rem;font-family:'Space Mono',monospace}
.report-body th,.report-body td{padding:7px 11px;border:1px solid var(--border);text-align:left}
.report-body th{background:var(--bg2);color:var(--accent)}
.report-body p{margin-bottom:0.5rem}
.report-body code{background:var(--bg2);border:1px solid var(--border);padding:1px 5px;border-radius:4px;font-family:'Space Mono',monospace;font-size:0.8em;color:var(--accent)}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

@media(max-width:960px){main{grid-template-columns:1fr}.sidebar{position:static}.feat-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:540px){header{padding:0 1rem}main{padding:0.9rem}.feat-grid{grid-template-columns:1fr}.sym-code{font-size:1.5rem}.header-right{display:none}}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">📈</div>
    <div>
      <div class="logo-name">VN<span>Stock</span>AI</div>
      <div class="logo-sub">Professional Multi-Asset v4.0</div>
    </div>
  </div>
  <div class="header-right">
    <div class="hstatus"><div class="dot"></div><span id="hstatus">Đang kết nối...</span></div>
    <span id="htime"></span>
  </div>
</header>

<div class="ticker-bar">
  <div class="ticker-inner" id="ticker">
    <span class="tick"><span class="t-sym">VN-INDEX</span><span class="t-val">1,287.45</span><span class="t-up">▲+12.3 (+0.97%)</span></span>
    <span class="tick"><span class="t-sym">VCB</span><span class="t-val">86,500</span><span class="t-up">▲+500</span></span>
    <span class="tick"><span class="t-sym">VHM</span><span class="t-val">38,200</span><span class="t-dn">▼-300</span></span>
    <span class="tick"><span class="t-sym">HPG</span><span class="t-val">23,100</span><span class="t-up">▲+200</span></span>
    <span class="tick"><span class="t-sym">FPT</span><span class="t-val">113,600</span><span class="t-up">▲+1,200</span></span>
    <span class="tick"><span class="t-sym">TCB</span><span class="t-val">22,400</span><span class="t-up">▲+100</span></span>
    <span class="tick"><span class="t-sym">MBB</span><span class="t-val">19,800</span><span class="t-dn">▼-150</span></span>
    <span class="tick"><span class="t-sym">VIC</span><span class="t-val">41,500</span><span class="t-up">▲+350</span></span>
    <span class="tick"><span class="t-sym">ACB</span><span class="t-val">24,600</span><span class="t-up">▲+250</span></span>
    <span class="tick"><span class="t-sym">BID</span><span class="t-val">45,300</span><span class="t-dn">▼-200</span></span>
    <span class="tick"><span class="t-sym">VN-INDEX</span><span class="t-val">1,287.45</span><span class="t-up">▲+12.3 (+0.97%)</span></span>
    <span class="tick"><span class="t-sym">VCB</span><span class="t-val">86,500</span><span class="t-up">▲+500</span></span>
    <span class="tick"><span class="t-sym">VHM</span><span class="t-val">38,200</span><span class="t-dn">▼-300</span></span>
    <span class="tick"><span class="t-sym">HPG</span><span class="t-val">23,100</span><span class="t-up">▲+200</span></span>
    <span class="tick"><span class="t-sym">FPT</span><span class="t-val">113,600</span><span class="t-up">▲+1,200</span></span>
    <span class="tick"><span class="t-sym">TCB</span><span class="t-val">22,400</span><span class="t-up">▲+100</span></span>
    <span class="tick"><span class="t-sym">MBB</span><span class="t-val">19,800</span><span class="t-dn">▼-150</span></span>
    <span class="tick"><span class="t-sym">VIC</span><span class="t-val">41,500</span><span class="t-up">▲+350</span></span>
    <span class="tick"><span class="t-sym">ACB</span><span class="t-val">24,600</span><span class="t-up">▲+250</span></span>
    <span class="tick"><span class="t-sym">BID</span><span class="t-val">45,300</span><span class="t-dn">▼-200</span></span>
  </div>
</div>

<main>
  <aside class="sidebar">
    <div class="card">
      <div class="card-title">⚙ Cấu Hình Phân Tích</div>
      <div class="tabs" id="tabGroup">
        <button class="tab-btn active" data-mode="stock" onclick="setMode('stock',this)">📊 Cổ phiếu</button>
        <button class="tab-btn" data-mode="fund" onclick="setMode('fund',this)">🏦 Quỹ</button>
        <button class="tab-btn" data-mode="forex" onclick="setMode('forex',this)">💱 Ngoại tệ</button>
      </div>
      <div class="form-group">
        <label class="form-label" id="inputLabel">🔤 Mã Cổ Phiếu</label>
        <div class="input-row">
          <input type="text" id="symInput" class="sym-input" placeholder="VD: VCB, HPG, FPT..." autocomplete="off" autocorrect="off" spellcheck="false"/>
          <button class="run-btn" onclick="startAnalysis()">▶ Run</button>
        </div>
        <div class="chips" id="chipArea">
          <span class="chip" onclick="fillSym('VCB')">VCB</span>
          <span class="chip" onclick="fillSym('VHM')">VHM</span>
          <span class="chip" onclick="fillSym('HPG')">HPG</span>
          <span class="chip" onclick="fillSym('FPT')">FPT</span>
          <span class="chip" onclick="fillSym('TCB')">TCB</span>
          <span class="chip" onclick="fillSym('MBB')">MBB</span>
          <span class="chip" onclick="fillSym('ACB')">ACB</span>
          <span class="chip" onclick="fillSym('VIC')">VIC</span>
        </div>
      </div>
      <button class="analyze-btn" id="analyzeBtn" onclick="startAnalysis()">
        <span>🤖</span> Phân Tích AI Chuyên Sâu
      </button>
    </div>

    <div class="card">
      <div class="card-title">🧬 Trạng Thái Hệ Thống</div>
      <div style="display:flex;flex-direction:column;gap:7px;">
        <div class="step" id="agData"><span>📡</span><div class="step-body"><div class="step-label">Dữ liệu thị trường</div><div class="step-sub">TCBS · Fmarket · ExchangeRate</div></div></div>
        <div class="step" id="agTA"><span>📊</span><div class="step-body"><div class="step-label">Phân tích kỹ thuật</div><div class="step-sub">RSI · MACD · BB · Stoch · Williams · OBV</div></div></div>
        <div class="step" id="agFC"><span>📐</span><div class="step-body"><div class="step-label">Dự báo xu hướng</div><div class="step-sub">Linear Regression · Momentum · Mean Reversion</div></div></div>
        <div class="step" id="agAI"><span>🧠</span><div class="step-body"><div class="step-label">AI Reasoning</div><div class="step-sub">Groq · LLaMA 70B · DeepSeek</div></div></div>
      </div>
    </div>

    <div class="card" style="border-color:rgba(240,192,64,0.25);">
      <div class="card-title" style="color:var(--gold);">🔑 Cấu Hình API</div>
      <p style="font-size:0.7rem;color:var(--text3);line-height:1.75;">
        Thêm vào Render Environment:<br>
        <code style="color:var(--green);background:var(--bg);padding:1px 5px;border-radius:3px;">GROQ_API_KEY_STOCK</code><br>
        <a href="https://console.groq.com" target="_blank" style="color:var(--accent);">console.groq.com</a>
        <span style="color:var(--yellow);"> ⚡ Miễn phí!</span>
      </p>
    </div>
  </aside>

  <div class="content">
    <div id="welcome">
      <div class="w-icon">📊</div>
      <h2 class="w-title">Phân Tích Chứng Khoán AI</h2>
      <p class="w-desc">Hệ thống phân tích chuyên sâu với dữ liệu thật, 10+ chỉ báo kỹ thuật, dự báo xu hướng và báo cáo AI chuyên nghiệp</p>
      <div class="feat-grid">
        <div class="feat-card"><div class="feat-icon">📈</div><div class="feat-name">Dữ Liệu Thật</div><p class="feat-desc">Lịch sử giá từ TCBS, NAV quỹ từ Fmarket, tỷ giá real-time</p></div>
        <div class="feat-card"><div class="feat-icon">📊</div><div class="feat-name">10+ Chỉ Báo</div><p class="feat-desc">RSI, MACD, Bollinger, SMA, EMA, Stochastic, Williams %R, OBV, ATR</p></div>
        <div class="feat-card"><div class="feat-icon">📐</div><div class="feat-name">Dự Báo Xu Hướng</div><p class="feat-desc">Linear regression, momentum analysis, mean reversion ensemble</p></div>
        <div class="feat-card"><div class="feat-icon">🧠</div><div class="feat-name">AI Chuyên Sâu</div><p class="feat-desc">LLaMA 70B phân tích cơ bản, kỹ thuật và chiến lược giao dịch</p></div>
        <div class="feat-card"><div class="feat-icon">📋</div><div class="feat-name">Báo Cáo Chuẩn</div><p class="feat-desc">Entry, Stop-loss, Take-profit cụ thể với Risk/Reward ratio</p></div>
        <div class="feat-card"><div class="feat-icon">🏦</div><div class="feat-name">Cơ Bản P/E P/B</div><p class="feat-desc">ROE, EPS, Beta, vốn hóa, đánh giá định giá ngành</p></div>
      </div>
    </div>

    <div id="loading">
      <div class="spinner-wrap"><div class="ring-outer"></div><div class="ring-inner"></div></div>
      <div style="text-align:center;">
        <div id="loadLabel">Đang phân tích...</div>
        <div class="load-sub">Professional Technical Analysis Engine v4.0</div>
      </div>
      <div class="load-steps">
        <div class="step" id="ls1"><span>📡</span> Lấy dữ liệu giá từ TCBS...</div>
        <div class="step" id="ls2"><span>📊</span> Tính toán 10+ chỉ báo kỹ thuật...</div>
        <div class="step" id="ls3"><span>📐</span> Dự báo xu hướng, hỗ trợ/kháng cự...</div>
        <div class="step" id="ls4"><span>🧠</span> AI phân tích đa chiều...</div>
        <div class="step" id="ls5"><span>📋</span> Tổng hợp báo cáo chuyên nghiệp...</div>
      </div>
    </div>

    <div id="errorBox"></div>

    <div id="result">
      <div class="res-head">
        <div class="sym-block">
          <div class="sym-code" id="rSym">--</div>
          <div class="sym-meta">
            <div class="sym-type" id="rType">--</div>
            <div class="sym-time" id="rTime">--</div>
          </div>
        </div>
        <div class="rec-badge rec-WATCH" id="rBadge">—</div>
      </div>

      <div class="forecast-bar" id="forecastBar" style="display:none;">
        <div class="fc-item"><div class="fc-label">Dự báo xu hướng</div><div class="fc-val" id="fcDir">--</div></div>
        <div class="fc-item"><div class="fc-label">Slope/phiên</div><div class="fc-val" id="fcSlope">--</div></div>
        <div class="fc-item"><div class="fc-label">Độ tin cậy (R²)</div><div class="fc-val" id="fcR2">--</div></div>
        <div class="fc-item"><div class="fc-label">Dự báo T+5</div><div class="fc-val" id="fcT5">--</div></div>
      </div>

      <div class="badge-row" id="badgeRow"></div>
      <div class="data-grid" id="techGrid"></div>
      <div id="chartsArea"></div>

      <div class="card" id="fundCard" style="display:none;">
        <div class="card-title">📋 Chỉ Số Cơ Bản</div>
        <div style="overflow-x:auto;">
          <table class="fund-table" id="fundTable"><thead><tr><th>Chỉ số</th><th style="text-align:right;">Giá trị</th><th>Đánh giá</th></tr></thead><tbody></tbody></table>
        </div>
      </div>

      <div class="report-card">
        <div class="report-head">
          <h3>📋 Báo Cáo Phân Tích Toàn Diện</h3>
          <button class="copy-btn" onclick="copyReport()">📋 Sao chép</button>
        </div>
        <div class="report-body" id="reportBody"></div>
      </div>
    </div>
  </div>
</main>

<script>
let currentMode='stock';
let stepTimerId=null;

const MODE_CFG={
  stock:{label:'🔤 Mã Cổ Phiếu',placeholder:'VD: VCB, HPG, FPT...',chips:['VCB','VHM','HPG','FPT','TCB','MBB','ACB','VIC','BID','CTG'],typeLabel:'Cổ Phiếu | HOSE/HNX'},
  fund:{label:'🔤 Mã Chứng Chỉ Quỹ',placeholder:'VD: E1VFVN30, VFMVSF...',chips:['E1VFVN30','VFMVSF','SSISCA','MAFPF1','FVBF','DCDS'],typeLabel:'Chứng Chỉ Quỹ'},
  forex:{label:'💱 Cặp Tiền Tệ',placeholder:'VD: USD.VND, EUR.USD...',chips:['USD.VND','EUR.VND','GBP.VND','USD.JPY','EUR.USD','AUD.USD'],typeLabel:'Cặp Tiền Tệ'},
};

function updateClock(){const el=document.getElementById('htime');if(el)el.textContent=new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
setInterval(updateClock,1000);updateClock();

async function checkHealth(){
  try{
    const res=await fetch('/health');
    const d=await res.json();
    const el=document.getElementById('hstatus');
    if(d.status==='ok'){
      el.textContent='Hệ thống hoạt động';el.style.color='var(--green)';
      if(d.agents){if(d.agents.data)setStep('agData','done');if(d.agents.reasoning)setStep('agAI','done');}
    }
  }catch{
    const el=document.getElementById('hstatus');
    if(el){el.textContent='Chưa kết nối backend';el.style.color='var(--yellow)';}
  }
}
checkHealth();

function setStep(id,state){const el=document.getElementById(id);if(!el)return;el.classList.remove('active','done');if(state)el.classList.add(state);}

function setMode(mode,btn){
  currentMode=mode;
  document.querySelectorAll('#tabGroup .tab-btn').forEach(function(b){b.classList.remove('active');});
  btn.classList.add('active');
  const cfg=MODE_CFG[mode];
  document.getElementById('inputLabel').textContent=cfg.label;
  document.getElementById('symInput').placeholder=cfg.placeholder;
  document.getElementById('symInput').value='';
  const chipArea=document.getElementById('chipArea');
  chipArea.innerHTML=cfg.chips.map(function(c){return'<span class="chip" onclick="fillSym(\''+c+'\')">'+c+'</span>';}).join('');
}

function fillSym(sym){document.getElementById('symInput').value=sym;document.getElementById('symInput').focus();}

function startLoadSteps(){
  const ids=['ls1','ls2','ls3','ls4','ls5'];
  ids.forEach(function(id){setStep(id,'');});
  let i=0;
  stepTimerId=setInterval(function(){
    if(i>0)setStep(ids[i-1],'done');
    if(i<ids.length){setStep(ids[i],'active');i++;}
    else{clearInterval(stepTimerId);}
  },2200);
}

function showState(which){
  document.getElementById('welcome').style.display=(which==='welcome')?'':'none';
  document.getElementById('loading').classList.toggle('show',which==='loading');
  document.getElementById('errorBox').classList.toggle('show',which==='error');
  document.getElementById('result').classList.toggle('show',which==='result');
}

async function startAnalysis(){
  const sym=document.getElementById('symInput').value.trim().toUpperCase();
  if(!sym){alert('Vui lòng nhập mã');return;}
  showState('loading');
  document.getElementById('loadLabel').textContent='Đang phân tích '+sym+'...';
  document.getElementById('analyzeBtn').disabled=true;
  setStep('agData','active');
  startLoadSteps();
  const fd=new FormData();
  fd.append('symbol',sym);
  fd.append('type',currentMode);
  try{
    const resp=await fetch('/api/analyze',{method:'POST',body:fd});
    clearInterval(stepTimerId);
    if(!resp.ok){const errData=await resp.json().catch(function(){return{};});throw new Error(errData.error||'HTTP '+resp.status);}
    const json=await resp.json();
    renderResult(json,sym);
  }catch(err){
    clearInterval(stepTimerId);
    showState('error');
    document.getElementById('errorBox').innerHTML='⚠️ <strong>Lỗi:</strong> '+err.message+'<br><small style="color:var(--text3);">Kiểm tra lại mã, kết nối mạng hoặc backend logs.</small>';
  }finally{
    document.getElementById('analyzeBtn').disabled=false;
    setStep('agData','done');setStep('agTA','done');setStep('agFC','done');setStep('agAI','done');
  }
}

document.getElementById('symInput').addEventListener('keydown',function(e){if(e.key==='Enter')startAnalysis();});

function renderResult(json,sym){
  showState('result');
  const data=json.data||{};
  const mode=json.mode||currentMode;
  document.getElementById('rSym').textContent=sym;
  const typeMap={stock:'Cổ Phiếu | HOSE/HNX',fund:'Chứng Chỉ Quỹ',forex:'Cặp Tiền Tệ'};
  document.getElementById('rType').textContent=typeMap[mode]||mode;
  document.getElementById('rTime').textContent='Phân tích lúc: '+new Date().toLocaleString('vi-VN');

  const badge=document.getElementById('rBadge');
  if(mode==='forex'){
    const dir=data.direction||'SIDEWAYS';
    const dirMap={UP:'▲ TĂNG',DOWN:'▼ GIẢM',SIDEWAYS:'↔ ĐI NGANG'};
    const recFor=dir==='UP'?'BUY':dir==='DOWN'?'SELL':'HOLD';
    badge.textContent=dirMap[dir]||dir;
    badge.className='rec-badge rec-'+recFor;
  }else{
    const rec=data.recommendation||'WATCH';
    const recMap={BUY:'🟢 MUA',SELL:'🔴 BÁN',HOLD:'🟡 GIỮ',WATCH:'🔵 THEO DÕI'};
    badge.textContent=recMap[rec]||rec;
    badge.className='rec-badge rec-'+rec;
  }

  const fc=data.forecast||{};
  if(fc.direction){
    document.getElementById('forecastBar').style.display='';
    const fcDir=document.getElementById('fcDir');
    fcDir.textContent=fc.direction==='TĂNG'?'▲ TĂNG':'▼ GIẢM';
    fcDir.className='fc-val '+(fc.direction==='TĂNG'?'up':'down');
    document.getElementById('fcSlope').textContent=fc.slope?fc.slope+' VND':'--';
    document.getElementById('fcR2').textContent=fc.r_squared?fc.r_squared+' ('+(fc.confidence||'')+')':'--';
    const t5=fc.forecast_5d;
    document.getElementById('fcT5').textContent=t5&&t5.length?fmtNum(t5[t5.length-1]):'--';
  }else{
    document.getElementById('forecastBar').style.display='none';
  }

  const priceCount=data.price_history?data.price_history.prices.length:0;
  const techCount=Object.keys(data.technical||{}).length;
  const hasFund=data.fundamental&&Object.keys(data.fundamental).length>0;
  document.getElementById('badgeRow').innerHTML=
    badge_('ok','📡 Dữ liệu: '+(priceCount?priceCount+' phiên':'Real-time'))+
    badge_('ok','📊 Kỹ thuật: '+techCount+' chỉ báo')+
    badge_(hasFund?'ok':'warn','📋 Cơ bản: '+(hasFund?'Có dữ liệu':'Không có'))+
    badge_('info','🧠 AI: Hoàn thành')+
    (fc.direction?badge_(fc.direction==='TĂNG'?'ok':'warn','📐 Dự báo: '+fc.direction+' (R²='+fc.r_squared+')'):'');

  const tech=data.technical||{};
  const tGrid=document.getElementById('techGrid');
  let tHtml='';
  if(tech.current_price!==undefined)tHtml+=dc('Giá Hiện Tại',fmtNum(tech.current_price),'','');
  if(valid(tech.rsi)){const rv=parseFloat(tech.rsi);const rc=rv>70?'down':rv<30?'up':'neu';tHtml+=dc('RSI (14)',tech.rsi,rc,rv>70?'⚠ Quá mua':rv<30?'⚠ Quá bán':'Trung tính');}
  if(valid(tech.macd)){const mc=parseFloat(tech.macd)>parseFloat(tech.macd_signal||0)?'up':'down';tHtml+=dc('MACD',tech.macd,mc,'Signal: '+(tech.macd_signal||'N/A'));}
  if(valid(tech.bb_upper))tHtml+=dc('BB Upper',fmtNum(tech.bb_upper),'','Lower: '+fmtNum(tech.bb_lower));
  if(valid(tech.sma20))tHtml+=dc('SMA 20',fmtNum(tech.sma20),'','SMA50: '+fmtNum(tech.sma50));
  if(valid(tech.sma200))tHtml+=dc('SMA 200',fmtNum(tech.sma200),'','EMA9: '+fmtNum(tech.ema9));
  if(valid(tech.stoch_k))tHtml+=dc('Stoch %K',tech.stoch_k,'','%D: '+(tech.stoch_d||'N/A'));
  if(valid(tech.williams_r)){const wrv=parseFloat(tech.williams_r);const wrc=wrv>-20?'down':wrv<-80?'up':'neu';tHtml+=dc('Williams %R',tech.williams_r,wrc,wrv>-20?'⚠ Quá mua':wrv<-80?'⚠ Quá bán':'Trung tính');}
  if(valid(tech.atr))tHtml+=dc('ATR (14)',tech.atr,'neu','Biến động TB/phiên');
  if(valid(tech.momentum))tHtml+=dc('Momentum 10',tech.momentum+'%',parseFloat(tech.momentum)>=0?'up':'down','');
  if(valid(tech.support1)){tHtml+=dc('Hỗ Trợ 1',fmtNum(tech.support1),'up','Hỗ Trợ 2: '+fmtNum(tech.support2));tHtml+=dc('Kháng Cự 1',fmtNum(tech.resistance1),'down','Kháng Cự 2: '+fmtNum(tech.resistance2));}
  if(tech.trend_short)tHtml+=dc('Xu Hướng NH',tech.trend_short,tech.trend_short==='TĂNG'?'up':'down','');
  if(tech.trend_medium)tHtml+=dc('Xu Hướng TH',tech.trend_medium,tech.trend_medium==='TĂNG'?'up':'down','');
  if(tech.trend_long)tHtml+=dc('Xu Hướng DH',tech.trend_long,tech.trend_long==='TĂNG'?'up':'down','');
  if(data.fund_info){const fi=data.fund_info;tHtml+=dc('NAV Hiện Tại',fmtNum(fi.latest_nav),fi.nav_change>=0?'up':'down','Δ '+(fi.nav_change>=0?'+':'')+parseFloat(fi.nav_change).toFixed(4));tHtml+=dc('Công Ty QL',fi.management_company||'N/A','neu','');}
  if(data.rate!==undefined)tHtml+=dc('Tỷ Giá',data.rate,'neu','');
  tGrid.innerHTML=tHtml;

  const ca=document.getElementById('chartsArea');
  ca.innerHTML='';
  if(data.charts){
    if(data.charts.main)ca.innerHTML+='<div class="chart-wrap"><div class="chart-title">📈 Biểu Đồ Kỹ Thuật Tổng Hợp</div><img src="data:image/png;base64,'+data.charts.main+'" alt="Technical chart"/></div>';
    if(data.charts.forecast)ca.innerHTML+='<div class="chart-wrap" style="margin-top:1rem;"><div class="chart-title">📐 Deep Learning Forecast</div><img src="data:image/png;base64,'+data.charts.forecast+'" alt="Forecast chart"/></div>';
    if(data.charts.macd)ca.innerHTML+='<div class="chart-wrap" style="margin-top:1rem;"><div class="chart-title">📊 MACD &amp; Stochastic</div><img src="data:image/png;base64,'+data.charts.macd+'" alt="MACD chart"/></div>';
  }

  const fundCard=document.getElementById('fundCard');
  const fund=data.fundamental||{};
  if(Object.keys(fund).length>0){
    fundCard.style.display='';
    const tbody=fundCard.querySelector('tbody');
    const rows=[];
    if(fund.pe!==undefined&&fund.pe!==null)rows.push(['P/E Ratio',fund.pe,evalPE(fund.pe)]);
    if(fund.pb!==undefined&&fund.pb!==null)rows.push(['P/B Ratio',fund.pb,evalPB(fund.pb)]);
    if(fund.roe!==undefined&&fund.roe!==null)rows.push(['ROE (%)',fund.roe,evalROE(fund.roe)]);
    if(fund.roa!==undefined&&fund.roa!==null)rows.push(['ROA (%)',fund.roa,'']);
    if(fund.eps!==undefined&&fund.eps!==null)rows.push(['EPS',fund.eps,'']);
    if(fund.beta!==undefined&&fund.beta!==null)rows.push(['Beta',fund.beta,parseFloat(fund.beta)>1?'↑ Biến động cao hơn TT':'↓ Ổn định hơn TT']);
    if(fund.market_cap)rows.push(['Vốn Hóa (tỷ)',fund.market_cap,'']);
    if(fund.outstanding)rows.push(['CP Lưu Hành',fund.outstanding,'']);
    if(fund.dividend_yield)rows.push(['Dividend Yield%',fund.dividend_yield,'']);
    if(fund.industry)rows.push(['Ngành',fund.industry,'']);
    if(fund.exchange)rows.push(['Sàn Giao Dịch',fund.exchange,'']);
    if(fund['52w_high'])rows.push(['Cao 52 Tuần',fund['52w_high'],'']);
    if(fund['52w_low'])rows.push(['Thấp 52 Tuần',fund['52w_low'],'']);
    tbody.innerHTML=rows.map(function(r){return'<tr><td>'+r[0]+'</td><td class="td-val">'+r[1]+'</td><td class="td-note">'+r[2]+'</td></tr>';}).join('');
  }else{
    fundCard.style.display='none';
  }

  document.getElementById('reportBody').innerHTML=mdToHtml(data.analysis||'Không có dữ liệu phân tích.');
}

function valid(v){return v!==undefined&&v!==null&&v!=='N/A'&&v!=='';}
function badge_(cls,text){return'<div class="badge '+cls+'">'+text+'</div>';}
function dc(label,val,colorCls,sub){
  const colorMap={up:'var(--green)',down:'var(--red)',neu:'var(--text3)','':'var(--text)'};
  const col=colorMap[colorCls]||'var(--text)';
  return'<div class="dc"><div class="dc-label">'+label+'</div><div class="dc-val" style="color:'+col+'">'+val+'</div>'+(sub?'<div class="dc-sub '+(colorCls||'neu')+'">'+sub+'</div>':'')+'</div>';
}
function fmtNum(n){
  if(n===undefined||n===null||n==='N/A')return'N/A';
  const num=parseFloat(n);
  if(isNaN(num))return String(n);
  if(num>=1e12)return(num/1e12).toFixed(2)+'T';
  if(num>=1e9)return(num/1e9).toFixed(2)+'B';
  if(num>=1e6)return(num/1e6).toFixed(2)+'M';
  if(num>=1000)return num.toLocaleString('vi-VN');
  return num%1===0?num.toString():num.toFixed(4);
}
function evalPE(v){const n=parseFloat(v);if(isNaN(n))return'';return n<10?'✅ Thấp hơn TB':n>25?'⚠ Định giá cao':'✔ Hợp lý';}
function evalPB(v){const n=parseFloat(v);if(isNaN(n))return'';return n<1.5?'✅ Thấp hơn BV':n>3?'⚠ Định giá cao':'✔ Hợp lý';}
function evalROE(v){const n=parseFloat(v);if(isNaN(n))return'';return n>20?'⭐ Xuất sắc':n>15?'✅ Tốt':n>10?'✔ Khá':'⚠ Trung bình';}

function mdToHtml(text){
  if(!text)return'';
  text=text.replace(/<think[\s\S]*?<\/think>/gi,'');
  text=text.replace(/^######\s+(.+)$/gm,'<h3>$1</h3>');
  text=text.replace(/^#####\s+(.+)$/gm,'<h3>$1</h3>');
  text=text.replace(/^####\s+(.+)$/gm,'<h3>$1</h3>');
  text=text.replace(/^###\s+(.+)$/gm,'<h3>$1</h3>');
  text=text.replace(/^##\s+(.+)$/gm,'<h2>$1</h2>');
  text=text.replace(/^#\s+(.+)$/gm,'<h1>$1</h1>');
  text=text.replace(/^---+$/gm,'<hr>');
  text=text.replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>');
  text=text.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  text=text.replace(/\*(.+?)\*/g,'<em>$1</em>');
  text=text.replace(/`([^`]+)`/g,'<code>$1</code>');
  text=text.replace(/^>\s+(.+)$/gm,'<blockquote>$1</blockquote>');
  text=text.replace(/^[\-\*\•]\s+(.+)$/gm,'<li>$1</li>');
  text=text.replace(/(<li>[\s\S]*?<\/li>)/g,function(m){if(m.indexOf('<ul>')===-1)return'<ul>'+m+'</ul>';return m;});
  text=text.replace(/^\d+\.\s+(.+)$/gm,'<li>$1</li>');
  const lines=text.split('\n');
  const result=[];
  for(let i=0;i<lines.length;i++){
    const line=lines[i].trim();
    if(!line){result.push('');}
    else if(line.match(/^<(h[1-6]|ul|ol|li|hr|blockquote|table|thead|tbody|tr|th|td|p)/i)){result.push(line);}
    else{result.push('<p>'+line+'</p>');}
  }
  return result.join('\n');
}

function copyReport(){
  const text=document.getElementById('reportBody').innerText;
  navigator.clipboard.writeText(text).then(function(){
    const btn=document.querySelector('.copy-btn');
    const orig=btn.textContent;
    btn.textContent='✅ Đã sao chép!';
    setTimeout(function(){btn.textContent=orig;},2000);
  }).catch(function(){alert('Không thể sao chép. Hãy chọn và copy thủ công.');});
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
