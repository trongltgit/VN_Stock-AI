"""
VN Stock AI — Professional Edition v2.5 (LSTM + vnstock Fundamental)
Phiên bản nâng cấp chuyên nghiệp - Phân tích sâu như công ty chứng khoán hàng đầu
"""

import os
import json
import logging
import base64
import io
import traceback
from datetime import datetime, timedelta
from typing import Dict, List

import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# vnstock
try:
    from vnstock import Company
except ImportError:
    pass  # sẽ log warning sau

load_dotenv()
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== COLOR CONFIG ======================
C = {
    "bg": "#060b14", "bg2": "#0c1421", "grid": "#1e3050", "text": "#e8f4fd",
    "text2": "#8baabb", "accent": "#00d4ff", "gold": "#f0c040", "green": "#00e676",
    "red": "#ff5252", "yellow": "#ffd740", "purple": "#b388ff", "orange": "#ff9800",
}

def _save_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=C["bg"], edgecolor="none", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


# ====================== DATA LAYER ======================
class VNStockData:
    HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    @staticmethod
    def get_tcbs_historical(symbol: str, days: int = 400):
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
            params = {
                "ticker": symbol.upper(),
                "type": "stock",
                "resolution": "D",
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
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
            logger.warning(f"TCBS historical failed for {symbol}: {e}")
        return None

    @staticmethod
    def get_fundamental_vnstock(symbol: str) -> Dict:
        try:
            company = Company(symbol.upper(), source='VCI')
            overview = company.overview()
            financial = company.financial_ratio()
            return {
                "symbol": symbol.upper(),
                "company_name": overview.get("companyName", ""),
                "industry": overview.get("industry", ""),
                "exchange": overview.get("exchange", ""),
                "pe": financial.get("P/E", None),
                "pb": financial.get("P/B", None),
                "roe": financial.get("ROE", None),
                "roa": financial.get("ROA", None),
                "eps": financial.get("EPS", None),
                "market_cap": overview.get("marketCap", None),
                "beta": financial.get("Beta", None),
                "dividend_yield": financial.get("Dividend Yield", None),
            }
        except Exception as e:
            logger.warning(f"vnstock fundamental failed for {symbol}: {e}")
            return {"symbol": symbol.upper(), "error": str(e)}

    @staticmethod
    def get_fmarket_fund_nav(fund_code: str):
        try:
            r = requests.post("https://api.fmarket.vn/res/products/filter",
                              json={"types": ["FUND"], "page": 1, "pageSize": 20, "searchField": fund_code},
                              timeout=10)
            funds = r.json().get("data", {}).get("rows", [])
            if not funds:
                return None
            fund = funds[0]
            nav_r = requests.get(f"https://api.fmarket.vn/res/products/{fund['id']}/nav-histories", timeout=10)
            nav_data = nav_r.json().get("data", [])
            if nav_data:
                df = pd.DataFrame(nav_data)
                df["time"] = pd.to_datetime(df["navDate"])
                df["Close"] = pd.to_numeric(df["nav"], errors="coerce")
                df["Open"] = df["High"] = df["Low"] = df["Close"]
                df["Volume"] = 0
                df = df.sort_values("time").reset_index(drop=True).dropna(subset=["Close"])
                return {
                    "info": fund,
                    "df": df,
                    "latest_nav": float(df["Close"].iloc[-1]),
                    "nav_change": float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2]) if len(df) > 1 else 0,
                }
        except Exception as e:
            logger.warning(f"Fmarket failed for {fund_code}: {e}")
        return None


# ====================== TECHNICAL ANALYZER ======================
class TechnicalAnalyzer:
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def bollinger(prices: pd.Series, period=20, std=2):
        sma = prices.rolling(period).mean()
        stdev = prices.rolling(period).std()
        return sma + stdev * std, sma, sma - stdev * std

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def stochastic(high, low, close, k=14, d=3):
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        k_line = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
        return k_line, k_line.rolling(d).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def williams_r(high, low, close, period=14):
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    @staticmethod
    def support_resistance(high, low, close, window=20):
        recent_low = low.rolling(window).min().iloc[-1]
        recent_high = high.rolling(window).max().iloc[-1]
        return {
            "support1": round(float(recent_low), 0),
            "support2": round(float(close.quantile(0.1)), 0),
            "resistance1": round(float(recent_high), 0),
            "resistance2": round(float(close.quantile(0.9)), 0),
        }

    @staticmethod
    def lstm_forecast(df: pd.DataFrame, lookback=60, horizon=10, epochs=25):
        try:
            if len(df) < lookback + 50:
                return {"error": "Không đủ dữ liệu để train LSTM"}

            data = df["Close"].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            X, y = [], []
            for i in range(lookback, len(scaled_data) - horizon):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i:i+horizon, 0])

            X = np.array(X).reshape((len(X), lookback, 1))
            y = np.array(y)

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
            model.add(Dropout(0.25))
            model.add(LSTM(64))
            model.add(Dropout(0.25))
            model.add(Dense(horizon))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

            last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            pred_scaled = model.predict(last_sequence, verbose=0)
            pred_prices = scaler.inverse_transform(pred_scaled)[0]

            current_price = float(df["Close"].iloc[-1])
            direction = "TĂNG" if pred_prices[-1] > current_price else "GIẢM"

            return {
                "model": "LSTM",
                "current_price": round(current_price, 2),
                "forecast_10d": [round(float(p), 2) for p in pred_prices],
                "direction": direction,
                "confidence": "CAO" if abs(pred_prices[-1] - current_price) / current_price < 0.035 else "TRUNG BÌNH",
            }
        except Exception as e:
            logger.error(f"LSTM error: {e}")
            return {"error": str(e), "direction": "N/A"}


# ====================== CHART GENERATION ======================
def chart_main(df: pd.DataFrame, sym: str, ind: Dict) -> str:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 11), facecolor=C["bg"])
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[4, 1.2, 1.2, 0.8], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    n = len(df)
    x = np.arange(n)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(C["bg"])
        ax.grid(True, alpha=0.18, color=C["grid"], linewidth=0.6)
        ax.tick_params(colors=C["text2"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C["grid"])

    # Candlestick
    width = 0.6
    for i, row in df.iterrows():
        col = C["green"] if row["Close"] >= row["Open"] else C["red"]
        ax1.plot([i, i], [row["Low"], row["High"]], color=col, lw=0.9, alpha=0.85)
        body_h = abs(row["Close"] - row["Open"])
        body_y = min(row["Open"], row["Close"])
        ax1.add_patch(Rectangle((i - width/2, body_y), width, max(body_h, row["Close"]*0.001),
                                facecolor=col, edgecolor=col, alpha=0.92, zorder=3))

    if "bb_upper" in ind:
        ax1.plot(x, ind["bb_upper"], color=C["accent"], lw=1.0, alpha=0.7, linestyle="--")
        ax1.plot(x, ind["bb_middle"], color=C["gold"], lw=1.3, alpha=0.8)
        ax1.plot(x, ind["bb_lower"], color=C["accent"], lw=1.0, alpha=0.7, linestyle="--")

    if "sma20" in ind:
        ax1.plot(x, ind.get("sma20"), color=C["orange"], lw=1.3, alpha=0.85, label="SMA20")

    ax1.set_title(f"{sym} · Biểu Đồ Kỹ Thuật", color=C["accent"], fontsize=13, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", facecolor=C["bg2"], edgecolor=C["grid"], fontsize=8)
    ax1.set_ylabel("Giá (VND)", color=C["text2"])

    plt.setp(ax1.get_xticklabels(), visible=False)

    # Volume + RSI + Williams %R (giữ ngắn gọn)
    vol_colors = [C["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else C["red"] for i in range(n)]
    ax2.bar(x, df["Volume"], color=vol_colors, alpha=0.65, width=0.85)
    ax2.set_ylabel("Volume", color=C["text2"])

    if "rsi" in ind:
        ax3.plot(x, ind["rsi"], color=C["accent"], lw=1.6)
        ax3.axhline(70, color=C["red"], lw=0.8, ls="--")
        ax3.axhline(30, color=C["green"], lw=0.8, ls="--")
        ax3.set_ylim(0, 100)

    fig.tight_layout()
    return _save_fig(fig)


def chart_macd(df: pd.DataFrame, sym: str, ind: Dict) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), facecolor=C["bg"],
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    n = len(df)
    x = np.arange(n)
    if "macd" in ind:
        ax1.plot(x, ind["macd"], color=C["accent"], label="MACD")
        ax1.plot(x, ind.get("macd_signal"), color=C["gold"], label="Signal")
    ax1.legend()
    fig.tight_layout()
    return _save_fig(fig)


# ====================== AI AGENTS ======================
class NewsAgent:
    def get_news(self, symbol: str) -> List[Dict]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddg:
                results = ddg.text(f"{symbol} cổ phiếu tin tức 2025 2026", max_results=8)
                return [{"title": r["title"], "body": r["body"], "href": r["href"]} for r in results]
        except:
            return []


class ReasoningAgent:
    MODELS = ["llama-3.3-70b-versatile", "llama3-70b-8192"]

    def __init__(self):
        key = os.getenv("GROQ_API_KEY_STOCK")
        self.available = bool(key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
            except Exception as e:
                logger.error(f"Groq init error: {e}")
                self.available = False

    def analyze(self, symbol: str, stype: str, tech: Dict, fund: Dict, news: List[Dict], forecast: Dict = None) -> Dict:
        if not self.client:
            return {"analysis": "⚠️ Chưa cấu hình GROQ_API_KEY_STOCK", "recommendation": "WATCH"}

        # Xử lý tin tức an toàn (sửa lỗi f-string)
        news_text = "\n".join([f"- {n.get('title', '')}" for n in news[:6]]) if news else "Không có tin tức mới"

        forecast_text = ""
        if forecast and isinstance(forecast, dict) and "forecast_10d" in forecast:
            forecast_text = f"""
DỰ BÁO LSTM:
- Hướng: {forecast.get('direction', 'N/A')}
- Dự báo 10 phiên: {forecast.get('forecast_10d', [])}
- Độ tin cậy: {forecast.get('confidence', 'N/A')}"""

        system_prompt = "Bạn là chuyên gia phân tích chứng khoán Việt Nam cấp cao. Viết báo cáo chuyên nghiệp bằng tiếng Việt."

        user_prompt = f"""Phân tích {stype.upper()} **{symbol}** ngày {datetime.now().strftime('%d/%m/%Y')}:

{forecast_text}

CHỈ BÁO KỸ THUẬT:
Giá: {tech.get('current_price', 'N/A')} VND
RSI: {tech.get('rsi_current', 'N/A')}
MACD: {tech.get('macd_current', 'N/A')}

CƠ BẢN:
{json.dumps(fund, ensure_ascii=False, indent=2)}

TIN TỨC:
{news_text}"""

        for model in self.MODELS:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0.15,
                    max_tokens=4500,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                analysis = response.choices[0].message.content

                upper = analysis.upper()
                rec = "BUY" if any(k in upper for k in ["MUA", "TĂNG TỶ TRỌNG"]) else \
                      "SELL" if any(k in upper for k in ["BÁN", "GIẢM TỶ TRỌNG"]) else "HOLD"

                return {"analysis": analysis, "recommendation": rec}
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")

        return {"analysis": "Lỗi khi gọi AI.", "recommendation": "WATCH"}


# ====================== ORCHESTRATOR & ROUTES ======================
class Orchestrator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.ai_agent = ReasoningAgent()
        self.data = VNStockData()
        self.ta = TechnicalAnalyzer()

    def _compute_indicators(self, df: pd.DataFrame):
        if df is None or len(df) < 50:
            return {}, {}, {}, 0.0

        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        rsi = self.ta.rsi(c)
        macd_line, macd_sig, macd_hist = self.ta.macd(c)
        bb_u, bb_m, bb_l = self.ta.bollinger(c)
        sma20 = self.ta.sma(c, 20)

        lstm_result = self.ta.lstm_forecast(df)

        tech_dict = {
            "current_price": round(float(c.iloc[-1]), 2),
            "rsi_current": round(float(rsi.iloc[-1]), 2) if len(rsi) > 0 else None,
            "lstm_forecast": lstm_result,
        }

        ind_dict = {"rsi": rsi.values, "macd": macd_line.values, "macd_signal": macd_sig.values, "bb_upper": bb_u.values}

        return tech_dict, ind_dict, lstm_result, float(c.iloc[-1])

    def analyze_stock(self, symbol: str):
        df = self.data.get_tcbs_historical(symbol, days=400)
        if df is None or len(df) < 100:
            return {"error": f"Không lấy được dữ liệu cho {symbol}"}

        tech_dict, ind_dict, forecast, cp = self._compute_indicators(df)
        fund = self.data.get_fundamental_vnstock(symbol)
        news = self.news_agent.get_news(symbol)
        ai = self.ai_agent.analyze(symbol, "cổ phiếu", tech_dict, fund, news, forecast)

        main_chart = chart_main(df, symbol, ind_dict)
        macd_chart = chart_macd(df, symbol, ind_dict)

        return {
            "mode": "stock",
            "data": {
                "symbol": symbol,
                "analysis": ai["analysis"],
                "recommendation": ai["recommendation"],
                "forecast": forecast,
                "charts": {"main": main_chart, "macd": macd_chart},
                "technical": tech_dict,
                "fundamental": fund,
            }
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

        if stype == "stock":
            result = orc.analyze_stock(symbol)
        else:
            result = {"error": "Chưa hỗ trợ loại này"}

        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
