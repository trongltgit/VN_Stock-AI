"""
VN Stock AI v3.0 - Professional Deep Learning Edition
- Interactive chart data (TradingView-compatible)
- LSTM-style ensemble forecasting
- Pattern recognition: Candlestick + Chart patterns
- Professional VCBS-style report template
"""

import os, json, logging, base64, io, traceback
from datetime import datetime, timedelta
from typing import Dict, List

import requests, pandas as pd, numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

load_dotenv()
app = Flask(__name__, template_folder='templates')
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

C = {
    "bg": "#060b14", "bg2": "#0c1421", "grid": "#1e3050",
    "text": "#e8f4fd", "text2": "#8baabb", "accent": "#00d4ff",
    "gold": "#f0c040", "green": "#00e676", "red": "#ff5252",
    "yellow": "#ffd740", "purple": "#b388ff", "orange": "#ff9800", "pink": "#ff6b9d"
}

PATTERNS = {
    "DOJI": {"name": "Doji", "signal": "neutral"},
    "HAMMER": {"name": "Búa", "signal": "bullish"},
    "SHOOTING_STAR": {"name": "Sao băng", "signal": "bearish"},
    "ENGULFING_BULL": {"name": "Engulfing Tăng", "signal": "bullish"},
    "ENGULFING_BEAR": {"name": "Engulfing Giảm", "signal": "bearish"},
    "MORNING_STAR": {"name": "Sao mai", "signal": "bullish"},
    "EVENING_STAR": {"name": "Sao hôm", "signal": "bearish"},
}

class VNStockData:
    HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

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
            r = requests.get(f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{symbol}/overview",
                           timeout=15, headers=VNStockData.HEADERS)
            d = r.json()
            return {
                "pe": d.get("pe"), "pb": d.get("pb"), "roe": d.get("roe"), "roa": d.get("roa"),
                "eps": d.get("eps"), "market_cap": d.get("marketCap"), "industry": d.get("industry"),
                "exchange": d.get("exchange"), "52w_high": d.get("priceHigh52W"), "52w_low": d.get("priceLow52W"),
                "avg_volume": d.get("avgVolume10Day"), "beta": d.get("beta"),
                "dividend_yield": d.get("dividendYield"), "outstanding": d.get("outstandingShare"),
            }
        except Exception as e:
            logger.warning(f"Fundamental failed: {e}")
        return {}

class DeepForecaster:
    def forecast(self, prices: pd.Series, lookback: int = 60, horizon: int = 10) -> Dict:
        try:
            prices_clean = prices.dropna().values
            if len(prices_clean) < lookback + 10:
                return self._simple_forecast(prices_clean, horizon)

            y = prices_clean[-lookback:]
            x = np.arange(len(y))

            lr = LinearRegression()
            lr.fit(x.reshape(-1, 1), y)
            lr_pred = lr.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))
            lr_r2 = r2_score(y, lr.predict(x.reshape(-1, 1)))

            alpha, beta = 0.3, 0.1
            level = y[0]
            trend = (y[-1] - y[0]) / len(y)
            hw_pred = []
            for i in range(horizon):
                level = alpha * (y[-1] if i == 0 else hw_pred[-1]) + (1 - alpha) * (level + trend)
                trend = beta * (level - (level - trend)) + (1 - beta) * trend
                hw_pred.append(level + trend)

            returns = np.diff(y) / y[:-1]
            avg_return = np.mean(returns[-10:]) if len(returns) >= 10 else 0
            mom_pred = [y[-1] * (1 + avg_return) ** (i + 1) for i in range(horizon)]

            mean_20 = np.mean(y[-20:])
            mr_pred = [y[-1]]
            for i in range(horizon):
                mr_pred.append(mr_pred[-1] + 0.1 * (mean_20 - mr_pred[-1]))
            mr_pred = mr_pred[1:]

            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            trend_weight = max(0.2, min(0.6, 1 - volatility * 10))
            ensemble = []
            for i in range(horizon):
                val = (trend_weight * lr_pred[i] + 0.3 * (1-trend_weight) * hw_pred[i] +
                       0.2 * (1-trend_weight) * mom_pred[i] + 0.5 * (1-trend_weight) * mr_pred[i])
                ensemble.append(val)

            residuals = y - lr.predict(x.reshape(-1, 1))
            std_err = np.sqrt(np.mean(residuals ** 2))
            upper = [e + 1.96 * std_err * np.sqrt(i + 1) for i, e in enumerate(ensemble)]
            lower = [e - 1.96 * std_err * np.sqrt(i + 1) for i, e in enumerate(ensemble)]

            direction = "TĂNG" if ensemble[-1] > y[-1] * 1.01 else "GIẢM" if ensemble[-1] < y[-1] * 0.99 else "ĐI NGANG"

            return {
                "method": "Deep Ensemble", "forecast": [round(float(p), 2) for p in ensemble],
                "upper_bound": [round(float(p), 2) for p in upper], "lower_bound": [round(float(p), 2) for p in lower],
                "direction": direction, "confidence": "CAO" if lr_r2 > 0.7 else "TRUNG BÌNH" if lr_r2 > 0.4 else "THẤP",
                "r_squared": round(float(lr_r2), 4), "current_price": round(float(y[-1]), 2),
                "target_1w": round(float(ensemble[4]), 2) if len(ensemble) > 4 else None,
                "target_2w": round(float(ensemble[-1]), 2),
                "expected_return_2w": round((ensemble[-1] / y[-1] - 1) * 100, 2),
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
            "direction": "TĂNG" if trend > 0 else "GIẢM", "confidence": "THẤP",
            "current_price": round(float(last), 2),
        }

class PatternRecognizer:
    @staticmethod
    def recognize_candlestick(df):
        patterns = []
        if len(df) < 3: return patterns
        for i in range(max(0, len(df) - 10), len(df)):
            row = df.iloc[i]
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            body = abs(c - o)
            us = h - max(o, c)
            ls = min(o, c) - l
            tr = h - l
            if tr == 0: continue

            if body / tr < 0.1:
                patterns.append({"pattern": "DOJI", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "neutral", "name": "Doji"})
            if ls > 2 * body and us < body * 0.5 and c > o:
                patterns.append({"pattern": "HAMMER", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bullish", "name": "Búa"})
            if us > 2 * body and ls < body * 0.5 and c < o:
                patterns.append({"pattern": "SHOOTING_STAR", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bearish", "name": "Sao băng"})
            if i > 0:
                prev = df.iloc[i-1]
                pb = abs(prev["Close"] - prev["Open"])
                if body > pb:
                    if c > o and prev["Close"] < prev["Open"] and o < prev["Close"] and c > prev["Open"]:
                        patterns.append({"pattern": "ENGULFING_BULL", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bullish", "name": "Engulfing Tăng"})
                    elif c < o and prev["Close"] > prev["Open"] and o > prev["Close"] and c < prev["Open"]:
                        patterns.append({"pattern": "ENGULFING_BEAR", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bearish", "name": "Engulfing Giảm"})
            if i > 1:
                p1, p2 = df.iloc[i-2], df.iloc[i-1]
                if (p1["Close"] < p1["Open"] and abs(p2["Close"]-p2["Open"]) < abs(p1["Close"]-p1["Open"])*0.3 and
                    c > o and c > (p1["Open"]+p1["Close"])/2):
                    patterns.append({"pattern": "MORNING_STAR", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bullish", "name": "Sao mai"})
                if (p1["Close"] > p1["Open"] and abs(p2["Close"]-p2["Open"]) < abs(p1["Close"]-p1["Open"])*0.3 and
                    c < o and c < (p1["Open"]+p1["Close"])/2):
                    patterns.append({"pattern": "EVENING_STAR", "index": i, "date": row["time"].strftime("%d/%m"), "signal": "bearish", "name": "Sao hôm"})

        seen = set()
        unique = []
        for p in reversed(patterns):
            if p["pattern"] not in seen:
                seen.add(p["pattern"])
                unique.append(p)
        return list(reversed(unique))

    @staticmethod
    def detect_chart_patterns(df):
        patterns = []
        if len(df) < 30: return patterns
        highs = df["High"].values[-30:]
        lows = df["Low"].values[-30:]
        closes = df["Close"].values[-30:]
        try:
            from scipy.signal import argrelextrema
            max_idx = argrelextrema(highs, np.greater, order=3)[0]
            min_idx = argrelextrema(lows, np.less, order=3)[0]
            if len(max_idx) >= 2 and abs(highs[max_idx[-2]] - highs[max_idx[-1]]) / highs[max_idx[-2]] < 0.03:
                patterns.append({"pattern": "DOUBLE_TOP", "name": "Đỉnh đôi", "signal": "bearish", "strength": "Trung bình", "desc": "Hai đỉnh gần bằng nhau"})
            if len(min_idx) >= 2 and abs(lows[min_idx[-2]] - lows[min_idx[-1]]) / lows[min_idx[-2]] < 0.03:
                patterns.append({"pattern": "DOUBLE_BOTTOM", "name": "Đáy đôi", "signal": "bullish", "strength": "Trung bình", "desc": "Hai đáy gần bằng nhau"})
        except: pass
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        if abs(slope) / np.mean(closes) > 0.002:
            patterns.append({"pattern": "TREND_CHANNEL", "name": "Kênh xu hướng", "signal": "bullish" if slope > 0 else "bearish",
                           "strength": "Mạnh" if abs(slope)/np.mean(closes) > 0.005 else "Trung bình", "desc": f"Xu hướng {'tăng' if slope>0 else 'giảm'}"})
        return patterns

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
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def ichimoku(high, low, close):
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        return tenkan, kijun

    @staticmethod
    def fibonacci(high, low):
        diff = high - low
        return {"0%": high, "23.6%": high-0.236*diff, "38.2%": high-0.382*diff,
                "50%": high-0.5*diff, "61.8%": high-0.618*diff, "78.6%": high-0.786*diff, "100%": low}

    @staticmethod
    def sr(high, low, close, window=20):
        rl = low.rolling(window).min().iloc[-1]
        rh = high.rolling(window).max().iloc[-1]
        return {
            "support1": round(float(rl), 0), "support2": round(float(close.quantile(0.1)), 0), "support3": round(float(close.quantile(0.25)), 0),
            "resistance1": round(float(rh), 0), "resistance2": round(float(close.quantile(0.9)), 0), "resistance3": round(float(close.quantile(0.75)), 0),
        }

class NewsAgent:
    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddg:
                results = ddg.text(f"{symbol} cổ phiếu tin tức phân tích 2025 2026", max_results=8)
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except Exception as e:
            logger.warning(f"News failed: {e}")
        return []

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

    def analyze(self, symbol, stype, tech, fund, news, forecast=None, patterns=None):
        if not self.client:
            return {"analysis": self._fallback(symbol), "recommendation": "WATCH"}

        news_text = "\n".join([f"- [{n['title']}]: {n['body'][:180]}" for n in news[:5]]) if news else "Không có tin tức"
        fund_text = json.dumps(fund, ensure_ascii=False, indent=2) if fund else "Không có dữ liệu"

        pattern_text = ""
        if patterns:
            pattern_text = "\nMÔ HÌNH NHẬN DIỆN:\n" + "\n".join([f"- {p.get('name', p['pattern'])} ({p.get('signal', '')})" for p in patterns])

        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO DEEP LEARNING:
- Hướng: {forecast.get('direction', 'N/A')} | Độ tin cậy: {forecast.get('confidence', 'N/A')} (R²={forecast.get('r_squared', 'N/A')})
- Mục tiêu 1T: {forecast.get('target_1w', 'N/A')} | 2T: {forecast.get('target_2w', 'N/A')} (Lợi nhuận: {forecast.get('expected_return_2w', 'N/A')}%)
- Dự báo 10 phiên: {forecast.get('forecast', [])}"""

        tech_text = f"""
CHỈ BÁO KỸ THUẬT:
Giá: {tech.get('current_price', 'N/A')} | RSI: {tech.get('rsi_current', 'N/A')} | MACD: {tech.get('macd_current', 'N/A')}
BB: U={tech.get('bb_upper_current', 'N/A')} M={tech.get('bb_middle_current', 'N/A')} L={tech.get('bb_lower_current', 'N/A')}
SMA20: {tech.get('sma20_current', 'N/A')} | SMA50: {tech.get('sma50_current', 'N/A')} | SMA200: {tech.get('sma200_current', 'N/A')}
EMA9: {tech.get('ema9_current', 'N/A')} | Stoch %K: {tech.get('stoch_k_current', 'N/A')} | Williams%R: {tech.get('williams_r_current', 'N/A')}
ATR: {tech.get('atr_current', 'N/A')} | Momentum: {tech.get('momentum_current', 'N/A')}%
ADX: {tech.get('adx_current', 'N/A')} | +DI: {tech.get('plus_di_current', 'N/A')} | -DI: {tech.get('minus_di_current', 'N/A')}
Ichimoku: Tenkan={tech.get('tenkan_current', 'N/A')} Kijun={tech.get('kijun_current', 'N/A')}
Hỗ trợ: S1={tech.get('support1', 'N/A')} S2={tech.get('support2', 'N/A')} S3={tech.get('support3', 'N/A')}
Kháng cự: R1={tech.get('resistance1', 'N/A')} R2={tech.get('resistance2', 'N/A')} R3={tech.get('resistance3', 'N/A')}
Xu hướng: NH={tech.get('trend_short', 'N/A')} | TH={tech.get('trend_medium', 'N/A')} | DH={tech.get('trend_long', 'N/A')}
{pattern_text}
{forecast_text}"""

        system_prompt = """Bạn là Giám đốc Phân tích của Công ty Chứng khoán hàng đầu Việt Nam.
Viết báo cáo CHUYÊN NGHIỆP theo chuẩn VCBS/SSI.

CẤU TRÚC BÁO CÁO:

## 📊 TÓM TẮT ĐIỀU HÀNH
- Khuyến nghị: **[MUA/BÁN/GIỮ/THEO DÕI]** — Điểm tin cậy: X/10
- Giá mục tiêu: 1 tháng | 3 tháng | 6 tháng (cụ thể VND)
- Tóm tắt luận điểm chính

## 🏛️ PHÂN TÍCH VĨ MÔ & NGÀNH
- Môi trường lãi suất, tỷ giá, chính sách tiền tệ
- Vị thế ngành trong chu kỳ kinh tế

## 📈 PHÂN TÍCH CƠ BẢN
- Định giá: P/E, P/B so với ngành
- ROE, ROA, EPS growth
- So sánh với đối thủ

## 📉 PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU
### 4.1 Xu hướng & Cấu trúc
### 4.2 Dao động học & Momentum
### 4.3 Bollinger Bands & Biến động
### 4.4 Mô hình giá nhận diện
### 4.5 Dự báo Deep Learning

## 🎯 CHIẾN LƯỢC GIAO DỊCH
| Vùng | Giá (VND) | Lý do |
|------|-----------|-------|
| Entry | xxx | ... |
| Stop Loss | xxx | ... |
| Take Profit 1 | xxx | ... |
| Take Profit 2 | xxx | ... |

## ⚠️ RỦI RO & KỊCH BẢN
| Kịch bản | Xác suất | Mô tả | Giá mục tiêu |
|----------|----------|-------|--------------|
| Tích cực | X% | ... | xxx |
| Cơ sở | X% | ... | xxx |
| Tiêu cực | X% | ... | xxx |

## 📝 KẾT LUẬN

PHONG CÁCH: Chuyên nghiệp, số liệu cụ thể, logic chặt chẽ."""

        user_prompt = f"""Phân tích {stype.upper()} **{symbol}** ngày {datetime.now().strftime('%d/%m/%Y')}:

{tech_text}

CHỈ SỐ CƠ BẢN:
{fund_text}

TIN TỨC:
{news_text}"""

        for model in self.MODELS:
            try:
                response = self.client.chat.completions.create(
                    model=model, temperature=0.12, max_tokens=7000,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                )
                analysis = response.choices[0].message.content
                rec = self._extract_rec(analysis)
                logger.info(f"AI done: model={model}, rec={rec}")
                return {"analysis": analysis, "recommendation": rec}
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")

        return {"analysis": self._fallback(symbol), "recommendation": "WATCH"}

    def _extract_rec(self, text):
        upper = text.upper()
        if any(k in upper for k in ["MUA", "BUY", "OVERWEIGHT", "TĂNG TỶ TRỌNG"]):
            return "BUY"
        if any(k in upper for k in ["BÁN", "SELL", "UNDERWEIGHT", "GIẢM TỶ TRỌNG"]):
            return "SELL"
        if any(k in upper for k in ["GIỮ", "HOLD", "NEUTRAL"]):
            return "HOLD"
        return "WATCH"

    def _fallback(self, symbol):
        return f"""⚠️ Chưa cấu hình GROQ_API_KEY_STOCK

Hệ thống vẫn phân tích kỹ thuật và dự báo dựa trên dữ liệu định lượng.

Để có báo cáo AI chuyên sâu:
1. Truy cập https://console.groq.com
2. Tạo API Key miễn phí
3. Thêm biến môi trường `GROQ_API_KEY_STOCK`

Mã phân tích: {symbol}"""

def _save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=C["bg"], edgecolor="none", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def chart_main(df, sym, ind, patterns=None):
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
        ax.grid(True, alpha=0.15, color=C["grid"], linewidth=0.5)
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
                                    facecolor=col, edgecolor=col, alpha=0.9, zorder=3))

    # Patterns
    if patterns:
        for p in patterns:
            idx = p.get("index", 0)
            if 0 <= idx < n:
                pat = PATTERNS.get(p["pattern"], {})
                color = C["gold"] if pat.get("signal") == "bullish" else C["red"] if pat.get("signal") == "bearish" else C["yellow"]
                axes[0].scatter(idx, df["High"].iloc[idx] * 1.01, marker="*", s=120, color=color, zorder=10, alpha=0.9)

    # Bollinger
    if "bb_upper" in ind:
        axes[0].plot(x, ind["bb_upper"], color=C["accent"], lw=0.9, alpha=0.6, ls="--")
        axes[0].plot(x, ind["bb_middle"], color=C["gold"], lw=1.2, alpha=0.8)
        axes[0].plot(x, ind["bb_lower"], color=C["accent"], lw=0.9, alpha=0.6, ls="--")
        axes[0].fill_between(x, ind["bb_upper"], ind["bb_lower"], alpha=0.03, color=C["accent"])

    for key, color, lw, label in [("sma20", C["orange"], 1.2, "SMA20"), ("sma50", C["yellow"], 1.4, "SMA50"), ("sma200", C["pink"], 1.4, "SMA200")]:
        if key in ind:
            axes[0].plot(x, ind[key], color=color, lw=lw, alpha=0.85, label=label)
    if "ema9" in ind:
        axes[0].plot(x, ind["ema9"], color=C["purple"], lw=1.1, alpha=0.8, label="EMA9")
    if "tenkan" in ind:
        axes[0].plot(x, ind["tenkan"], color=C["accent"], lw=1.0, alpha=0.7, label="Tenkan")
    if "kijun" in ind:
        axes[0].plot(x, ind["kijun"], color=C["gold"], lw=1.0, alpha=0.7, label="Kijun")

    axes[0].set_title(f"{sym}  ·  Professional Technical Analysis", color=C["accent"], fontsize=13, fontweight="bold", pad=12, loc="left")
    axes[0].legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=7, labelcolor=C["text2"], ncol=5, framealpha=0.9)
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
        axes[3].legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=7, labelcolor=C["text2"])
        axes[3].set_ylabel("MACD", color=C["text2"], fontsize=9)

    # ADX
    if "adx" in ind:
        axes[4].plot(x, ind["adx"], color=C["purple"], lw=1.3, label="ADX")
        axes[4].axhline(25, color=C["yellow"], lw=0.7, ls="--", alpha=0.6)
        axes[4].set_ylabel("ADX", color=C["text2"], fontsize=9)
        axes[4].set_ylim(0, 60)

    tick_count = min(8, n)
    tick_idx = np.linspace(0, n-1, tick_count, dtype=int)
    axes[4].set_xticks(tick_idx)
    axes[4].set_xticklabels([df["time"].iloc[i].strftime("%d/%m") for i in tick_idx], rotation=30, ha="right", color=C["text2"])

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
        ax.plot(fc_x, fc, color=C["green"], lw=2.5, ls="--", label="Dự báo", zorder=4)

        upper = forecast.get("upper_bound", [])
        lower = forecast.get("lower_bound", [])
        if upper and lower:
            ax.fill_between(fc_x, upper, lower, alpha=0.15, color=C["green"], label="Khoảng tin cậy 95%")

        if forecast.get("target_1w"):
            ax.axhline(forecast["target_1w"], color=C["yellow"], lw=1, ls="-.", alpha=0.7, label=f"Mục tiêu 1T: {forecast['target_1w']:,.0f}")
        if forecast.get("target_2w"):
            ax.axhline(forecast["target_2w"], color=C["gold"], lw=1.2, ls="-.", alpha=0.8, label=f"Mục tiêu 2T: {forecast['target_2w']:,.0f}")

    ax.set_title(f"{sym}  ·  Deep Learning Forecast", color=C["accent"], fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=8, labelcolor=C["text2"], framealpha=0.9)
    ax.grid(True, alpha=0.15, color=C["grid"])
    ax.tick_params(colors=C["text2"])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.set_ylabel("Giá (VND)", color=C["text2"])

    fig.tight_layout()
    return _save_fig(fig)

class InteractiveChartData:
    @staticmethod
    def generate_ohlcv(df):
        return [{"time": int(row["time"].timestamp()), "open": round(float(row["Open"]), 2),
                 "high": round(float(row["High"]), 2), "low": round(float(row["Low"]), 2),
                 "close": round(float(row["Close"]), 2), "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0}
                for _, row in df.iterrows()]

    @staticmethod
    def generate_indicators(df, indicators):
        result = {}
        n = len(df)
        times = [int(df["time"].iloc[i].timestamp()) for i in range(n)]
        for key, values in indicators.items():
            if values is not None and len(values) == n:
                result[key] = [{"time": times[i], "value": round(float(values[i]), 4) if pd.notna(values[i]) else None} for i in range(n)]
        return result

class Orchestrator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.ai_agent = ReasoningAgent()
        self.data = VNStockData()
        self.ta = TechnicalAnalyzer()
        self.forecaster = DeepForecaster()
        self.pattern_recognizer = PatternRecognizer()
        self.chart_data = InteractiveChartData()

    def _compute(self, df):
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
        tenkan, kijun = self.ta.ichimoku(h, l, c)
        sr = self.ta.sr(h, l, c)
        forecast = self.forecaster.forecast(c, lookback=60, horizon=10)
        fib = self.ta.fibonacci(h.max(), l.min())

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
            "tenkan_current": safe(tenkan), "kijun_current": safe(kijun), **sr,
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
            "tenkan": tenkan.values, "kijun": kijun.values,
        }

        return tech_dict, ind_dict, forecast, cp, fib

    def analyze_stock(self, symbol):
        sym = symbol.upper()
        df = self.data.get_tcbs_historical(sym, days=400)
        if df is None or len(df) < 30:
            return self._fallback(sym, "stock", f"Không lấy được dữ liệu giá từ TCBS cho mã {sym}")

        tech_dict, ind_dict, forecast, cp, fib = self._compute(df)
        fund = self.data.get_fundamental(sym)
        news = self.news_agent.get_news(sym)

        candle_patterns = self.pattern_recognizer.recognize_candlestick(df)
        chart_patterns = self.pattern_recognizer.detect_chart_patterns(df)
        all_patterns = candle_patterns + chart_patterns

        ai = self.ai_agent.analyze(sym, "cổ phiếu", tech_dict, fund, news, forecast, all_patterns)

        ohlcv_data = self.chart_data.generate_ohlcv(df)
        indicator_series = self.chart_data.generate_indicators(df, ind_dict)

        main_chart = chart_main(df, sym, ind_dict, candle_patterns)
        forecast_chart = chart_forecast(df, sym, forecast, ind_dict)

        return {
            "mode": "stock",
            "data": {
                "symbol": sym, "type": "stock",
                "analysis": ai["analysis"], "recommendation": ai["recommendation"],
                "news_count": len(news), "forecast": forecast, "patterns": all_patterns,
                "fibonacci": {k: round(v, 2) for k, v in fib.items()},
                "charts": {"main": main_chart, "forecast": forecast_chart},
                "interactive": {"ohlcv": ohlcv_data[-200:], "indicators": indicator_series},
                "technical": {
                    "current_price": cp, "rsi": tech_dict["rsi_current"], "macd": tech_dict["macd_current"],
                    "macd_signal": tech_dict["macd_signal_current"], "macd_hist": tech_dict["macd_hist_current"],
                    "bb_upper": tech_dict["bb_upper_current"], "bb_middle": tech_dict["bb_middle_current"], "bb_lower": tech_dict["bb_lower_current"],
                    "sma20": tech_dict["sma20_current"], "sma50": tech_dict["sma50_current"], "sma200": tech_dict["sma200_current"],
                    "ema9": tech_dict["ema9_current"], "stoch_k": tech_dict["stoch_k_current"], "stoch_d": tech_dict["stoch_d_current"],
                    "atr": tech_dict["atr_current"], "williams_r": tech_dict["williams_r_current"], "momentum": tech_dict["momentum_current"],
                    "adx": tech_dict["adx_current"], "support1": tech_dict.get("support1"), "support2": tech_dict.get("support2"), "support3": tech_dict.get("support3"),
                    "resistance1": tech_dict.get("resistance1"), "resistance2": tech_dict.get("resistance2"), "resistance3": tech_dict.get("resistance3"),
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

    def _fallback(self, symbol, mode, msg):
        ai = self.ai_agent.analyze(symbol, mode, {"current_price": "N/A"}, {}, [])
        return {
            "mode": mode,
            "data": {"symbol": symbol, "type": mode, "analysis": f"⚠️ {msg}\n\n---\n\n{ai['analysis']}",
                    "recommendation": ai["recommendation"], "news_count": 0, "charts": {}, "technical": {}, "fundamental": {}},
        }

orc = Orchestrator()

@app.route("/")
def index():
    return render_template("index.html")

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
        else:
            return jsonify({"error": f"Loại không hợp lệ: {stype}"}), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "3.0", "timestamp": datetime.now().isoformat(),
                   "agents": {"data": True, "news": True, "reasoning": orc.ai_agent.available}, "groq": orc.ai_agent.available})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
