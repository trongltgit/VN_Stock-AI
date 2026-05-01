"""
VN Stock AI v5.0 — Professional Multi-Asset Analysis
Deep Learning Forecast · Interactive Charts · VCBS-Style AI Reports
"""
import os, json, logging, traceback, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

load_dotenv()
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# DATA PROVIDERS
# ══════════════════════════════════════════════════════════════════════

class VNStockData:
    H = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://tcinvest.tcbs.com.vn/",
    }

    @staticmethod
    def get_historical(symbol: str, days: int = 400):
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
            r = requests.get(url, params={
                "ticker": symbol, "type": "stock", "resolution": "D",
                "from": int(start.timestamp()), "to": int(end.timestamp()),
            }, timeout=20, headers=VNStockData.H)
            data = r.json()
            if data.get("data"):
                df = pd.DataFrame(data["data"])
                df["time"] = pd.to_datetime(df["tradingDate"])
                df = df.rename(columns={
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"
                })
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.sort_values("time").reset_index(drop=True).dropna(subset=["Close"])
        except Exception as e:
            logger.warning(f"TCBS error {symbol}: {e}")
        return None

    @staticmethod
    def get_fundamental(symbol: str) -> dict:
        try:
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{symbol}/overview"
            r = requests.get(url, timeout=15, headers=VNStockData.H)
            d = r.json()
            return {
                "pe": d.get("pe"), "pb": d.get("pb"),
                "roe": d.get("roe"), "roa": d.get("roa"), "eps": d.get("eps"),
                "market_cap": d.get("marketCap"), "industry": d.get("industry"),
                "exchange": d.get("exchange"), "52w_high": d.get("priceHigh52W"),
                "52w_low": d.get("priceLow52W"), "avg_volume": d.get("avgVolume10Day"),
                "beta": d.get("beta"), "dividend_yield": d.get("dividendYield"),
                "outstanding": d.get("outstandingShare"), "company_name": d.get("shortName"),
            }
        except Exception as e:
            logger.warning(f"Fundamental error: {e}")
        return {}


class FundData:
    H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    @staticmethod
    def search(query: str) -> dict:
        try:
            r = requests.get("https://api.fmarket.vn/api/search",
                             params={"q": query, "type": "fund"},
                             timeout=15, headers=FundData.H)
            d = r.json()
            rows = d.get("data", {}).get("rows") or d.get("data") or []
            return rows[0] if isinstance(rows, list) and rows else {}
        except Exception as e:
            logger.warning(f"Fund search: {e}")
        return {}

    @staticmethod
    def get_nav(fund_id, days=365):
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            r = requests.get(f"https://api.fmarket.vn/api/fund/{fund_id}/nav-history",
                             params={"from": start.strftime("%Y-%m-%d"),
                                     "to": end.strftime("%Y-%m-%d")},
                             timeout=20, headers=FundData.H)
            items = r.json().get("data", [])
            if not items:
                return None
            df = pd.DataFrame(items)
            date_col = "navDate" if "navDate" in df.columns else "date"
            nav_col = "nav" if "nav" in df.columns else "navPerShare"
            df["time"] = pd.to_datetime(df[date_col])
            df["Close"] = pd.to_numeric(df.get(nav_col, 0), errors="coerce")
            df["Open"] = df["Close"].shift(1).fillna(df["Close"])
            df["High"] = df["Close"] * 1.002
            df["Low"] = df["Close"] * 0.998
            df["Volume"] = 0
            return df.sort_values("time").reset_index(drop=True).dropna(subset=["Close"])
        except Exception as e:
            logger.warning(f"Fund NAV error: {e}")
        return None

    @staticmethod
    def get_info(fund_id) -> dict:
        try:
            r = requests.get(f"https://api.fmarket.vn/api/fund/{fund_id}",
                             timeout=15, headers=FundData.H)
            d = r.json().get("data", {})
            return {
                "fund_name": d.get("name"),
                "management_company": (d.get("managementCompany") or {}).get("name", "N/A"),
                "fund_type": d.get("fundType"),
                "risk_level": d.get("riskLevel"),
                "inception_date": d.get("inceptionDate"),
                "management_fee": d.get("managementFee"),
                "latest_nav": d.get("latestNav"),
                "nav_change": d.get("latestNavChange", 0),
                "aum": d.get("aum"),
            }
        except Exception as e:
            logger.warning(f"Fund info: {e}")
        return {}


class ForexData:
    RATES = {
        "USD.VND": 25250, "EUR.VND": 27300, "GBP.VND": 31800,
        "JPY.VND": 168.5, "AUD.VND": 16600, "CAD.VND": 18500,
        "CHF.VND": 28800, "CNY.VND": 3480, "SGD.VND": 18800,
        "EUR.USD": 1.082, "GBP.USD": 1.26, "USD.JPY": 149.8,
        "AUD.USD": 0.658, "USD.CNY": 7.24,
    }

    @staticmethod
    def gen_history(pair: str, days=120):
        base = ForexData.RATES.get(pair.upper(), 25000)
        vol = 0.008 if base > 100 else 0.015
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        rets = np.random.normal(0.0001, vol, len(dates))
        prices = [base]
        for r in rets[1:]:
            prices.append(prices[-1] * (1 + r))
        prices = prices[:len(dates)]
        df = pd.DataFrame({
            "time": dates,
            "Open": pd.Series(prices).shift(1).fillna(base).values,
            "Close": prices,
        })
        rng = abs(np.random.normal(0, vol * 0.4, len(dates)))
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + rng)
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - rng)
        df["Volume"] = np.random.randint(500000, 5000000, len(dates))
        return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════

class TA:
    @staticmethod
    def rsi(c, p=14):
        d = c.diff()
        g = d.where(d > 0, 0.0).ewm(com=p - 1, min_periods=p).mean()
        l = (-d.where(d < 0, 0.0)).ewm(com=p - 1, min_periods=p).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    @staticmethod
    def macd(c, f=12, s=26, sig=9):
        ml = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        sl = ml.ewm(span=sig, adjust=False).mean()
        return ml, sl, ml - sl

    @staticmethod
    def bbands(c, p=20, k=2):
        m = c.rolling(p).mean()
        s = c.rolling(p).std()
        return m + k * s, m, m - k * s

    @staticmethod
    def sma(c, p): return c.rolling(p).mean()

    @staticmethod
    def ema(c, p): return c.ewm(span=p, adjust=False).mean()

    @staticmethod
    def stoch(h, l, c, k=14, d=3):
        ks = 100 * (c - l.rolling(k).min()) / (
            h.rolling(k).max() - l.rolling(k).min() + 1e-9)
        return ks, ks.rolling(d).mean()

    @staticmethod
    def atr(h, l, c, p=14):
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()

    @staticmethod
    def williams(h, l, c, p=14):
        return -100 * (h.rolling(p).max() - c) / (
            h.rolling(p).max() - l.rolling(p).min() + 1e-9)

    @staticmethod
    def obv(c, v):
        return (np.sign(c.diff().fillna(0)) * v).cumsum()

    @staticmethod
    def adx(h, l, c, p=14):
        pdm = h.diff().clip(lower=0)
        ndm = (-l.diff()).clip(lower=0)
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        atr_ = tr.ewm(span=p, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / atr_.replace(0, np.nan)
        ndi = 100 * ndm.ewm(span=p, adjust=False).mean() / atr_.replace(0, np.nan)
        dx = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-9)
        return dx.ewm(span=p, adjust=False).mean(), pdi, ndi

    @staticmethod
    def support_resistance(h, l, c, w=30):
        return {
            "support1": round(float(l.rolling(w).min().iloc[-1]), 0),
            "support2": round(float(c.quantile(0.1)), 0),
            "resistance1": round(float(h.rolling(w).max().iloc[-1]), 0),
            "resistance2": round(float(c.quantile(0.9)), 0),
        }

    @staticmethod
    def ichimoku(h, l, c):
        """Ichimoku Cloud"""
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        chikou = c.shift(-26)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    @staticmethod
    def vwap(h, l, c, v):
        typical = (h + l + c) / 3
        cumtp = (typical * v).cumsum()
        cumv = v.cumsum()
        return cumtp / cumv.replace(0, np.nan)

    @staticmethod
    def cci(h, l, c, p=20):
        tp = (h + l + c) / 3
        return (tp - tp.rolling(p).mean()) / (0.015 * tp.rolling(p).std())

    @staticmethod
    def mfi(h, l, c, v, p=14):
        tp = (h + l + c) / 3
        pmf = (tp * v).where(tp > tp.shift(1), 0).rolling(p).sum()
        nmf = (tp * v).where(tp < tp.shift(1), 0).rolling(p).sum()
        return 100 - (100 / (1 + pmf / nmf.replace(0, np.nan)))


# ══════════════════════════════════════════════════════════════════════
# DEEP ENSEMBLE FORECASTER (Neural Network + Classical)
# ══════════════════════════════════════════════════════════════════════

class Forecaster:
    LOOKBACK = 25
    HORIZON = 10

    def _nn(self, pn: np.ndarray):
        """Feed-forward NN with temporal window (LSTM-style feature engineering)"""
        try:
            lb = self.LOOKBACK
            if len(pn) < lb + 20:
                return None
            X, y = [], []
            for i in range(lb, len(pn)):
                w = pn[i - lb:i]
                # Feature engineering: window + gradients + volatility
                g1 = float(np.diff(w[-5:]).mean()) if len(w) >= 5 else 0.0
                g2 = float(np.diff(np.diff(w[-5:])).mean()) if len(w) >= 6 else 0.0
                vol = float(np.std(np.diff(w[-10:]))) if len(w) >= 11 else 0.01
                X.append(np.append(w, [g1, g2, vol]))
                y.append(pn[i])
            X, y = np.array(X), np.array(y)
            mdl = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="tanh", solver="adam",
                max_iter=800, random_state=42,
                early_stopping=True, n_iter_no_change=20,
                learning_rate_init=0.001, alpha=0.0001,
            )
            mdl.fit(X, y)
            seq = list(pn[-lb:])
            preds = []
            for _ in range(self.HORIZON):
                w = np.array(seq[-lb:])
                g1 = float(np.diff(w[-5:]).mean())
                g2 = float(np.diff(np.diff(w[-5:])).mean())
                vol = float(np.std(np.diff(w[-10:])))
                p = mdl.predict([np.append(w, [g1, g2, vol])])[0]
                preds.append(p)
                seq.append(p)
            return np.array(preds)
        except Exception as e:
            logger.warning(f"NN forecast: {e}")
            return None

    def predict(self, prices: pd.Series) -> dict:
        raw = prices.dropna().values
        if len(raw) < 40:
            return {"direction": "N/A", "confidence": "THẤP",
                    "forecast": [], "nn_available": False}
        last = float(raw[-1])
        h = self.HORIZON
        x = np.arange(len(raw))

        # Normalize for NN
        scaler = MinMaxScaler()
        pn = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
        nn_n = self._nn(pn)
        nn_pred = (scaler.inverse_transform(nn_n.reshape(-1, 1)).flatten()
                   if nn_n is not None else None)

        # Polynomial
        poly_m = make_pipeline(PolynomialFeatures(2), Ridge(0.1))
        poly_m.fit(x.reshape(-1, 1), raw)
        poly_p = poly_m.predict(np.arange(len(raw), len(raw) + h).reshape(-1, 1))
        poly_r2 = r2_score(raw, poly_m.predict(x.reshape(-1, 1)))

        # Linear
        lr = LinearRegression()
        lr.fit(x[-80:].reshape(-1, 1), raw[-80:])
        lr_p = lr.predict(np.arange(len(raw), len(raw) + h).reshape(-1, 1))
        lr_r2 = r2_score(raw[-80:], lr.predict(x[-80:].reshape(-1, 1)))

        # Holt-Winters
        al, bl = 0.25, 0.08
        lev, trnd = raw[-1], (raw[-1] - raw[-20]) / 20
        hw_p = []
        for i in range(h):
            lev = al * (hw_p[-1] if hw_p else raw[-1]) + (1 - al) * (lev + trnd)
            trnd = bl * (lev - (lev - trnd)) + (1 - bl) * trnd
            hw_p.append(lev + trnd)
        hw_p = np.array(hw_p)

        # AR(5) ARIMA-style
        try:
            lag = 5
            Xa = np.array([raw[i - lag:i] for i in range(lag, len(raw))])
            ya = raw[lag:]
            arm = LinearRegression().fit(Xa, ya)
            seq = list(raw[-lag:])
            ar_p = []
            for _ in range(h):
                v = arm.predict([seq[-lag:]])[0]
                ar_p.append(v)
                seq.append(v)
            ar_p = np.array(ar_p)
        except Exception:
            ar_p = np.full(h, last)

        # Momentum
        rets = np.diff(raw) / (raw[:-1] + 1e-9)
        avg_ret = float(np.mean(rets[-10:]))
        mom_p = np.array([last * (1 + avg_ret) ** (i + 1) for i in range(h)])

        # Mean Reversion
        mean20 = float(np.mean(raw[-20:]))
        mr = [last]
        for _ in range(h):
            mr.append(mr[-1] + 0.2 * (mean20 - mr[-1]))
        mr_p = np.array(mr[1:])

        # Ensemble weights
        vol = float(np.std(rets[-20:])) if len(rets) >= 20 else 0.02
        ts = abs(float(lr.coef_[0])) / (float(np.mean(raw)) + 1e-9)

        if nn_pred is not None:
            w = ([0.32, 0.17, 0.14, 0.14, 0.07, 0.08, 0.08] if ts > 0.004 else
                 [0.26, 0.14, 0.15, 0.18, 0.12, 0.05, 0.10])
            ens = (w[0]*nn_pred + w[1]*poly_p + w[2]*lr_p +
                   w[3]*hw_p + w[4]*ar_p + w[5]*mom_p + w[6]*mr_p)
        else:
            w = ([0.25, 0.22, 0.20, 0.13, 0.10, 0.10] if ts > 0.004 else
                 [0.20, 0.18, 0.24, 0.17, 0.07, 0.14])
            ens = w[0]*poly_p + w[1]*lr_p + w[2]*hw_p + w[3]*ar_p + w[4]*mom_p + w[5]*mr_p

        resid = raw[-80:] - lr.predict(x[-80:].reshape(-1, 1))
        se = float(np.sqrt(np.mean(resid ** 2)))
        upper = ens + 1.96 * se * np.sqrt(np.arange(1, h + 1))
        lower = ens - 1.96 * se * np.sqrt(np.arange(1, h + 1))
        lower = np.maximum(lower, 0)

        avg_r2 = (poly_r2 + lr_r2) / 2
        direction = ("TĂNG" if ens[-1] > last * 1.005 else
                     "GIẢM" if ens[-1] < last * 0.995 else "ĐI NGANG")
        confidence = ("CAO" if avg_r2 > 0.75 else
                      "TRUNG BÌNH" if avg_r2 > 0.45 else "THẤP")
        stop_pct = max(vol * 3, 0.05)

        return {
            "method": "Neural Network + Deep Ensemble v5.0",
            "direction": direction,
            "confidence": confidence,
            "r_squared": round(avg_r2, 4),
            "slope": round(float(lr.coef_[0]), 2),
            "volatility": round(vol, 4),
            "current_price": round(last, 2),
            "target_1w": round(float(ens[4]), 2) if h > 4 else None,
            "target_2w": round(float(ens[-1]), 2),
            "expected_return_2w": round((float(ens[-1]) / last - 1) * 100, 2),
            "stop_loss": round(last * (1 - stop_pct), 0),
            "take_profit_1": round(float(ens[-1]) * 1.015, 0),
            "take_profit_2": round(float(upper[-1]), 0),
            "forecast": [round(float(v), 2) for v in ens],
            "forecast_5d": [round(float(v), 2) for v in ens[:5]],
            "upper_bound": [round(float(v), 2) for v in upper],
            "lower_bound": [round(float(v), 2) for v in lower],
            "nn_available": nn_pred is not None,
        }


# ══════════════════════════════════════════════════════════════════════
# AI REASONING AGENT (Groq)
# ══════════════════════════════════════════════════════════════════════

class AIAgent:
    MODELS = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self):
        key = os.getenv("GROQ_API_KEY_STOCK")
        self.ok = bool(key)
        self.client = None
        if self.ok:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                logger.info("Groq AI ready")
            except Exception as e:
                logger.error(f"Groq init: {e}")
                self.ok = False

    # ── STOCK REPORT ──────────────────────────────────────────
    def analyze_stock(self, sym, tech, fund, forecast):
        if not self.client:
            return {"analysis": self._no_key(sym), "recommendation": "WATCH", "confidence": 5}

        sys_p = """Bạn là Trưởng phòng Phân tích của công ty chứng khoán hàng đầu Việt Nam (VCBS/SSI/VPS).
Viết báo cáo phân tích CỔ PHIẾU theo chuẩn chuyên nghiệp, đầy đủ 8 phần, dùng Markdown:

# [KÝ HIỆU] — BÁO CÁO PHÂN TÍCH CỔ PHIẾU
**Khuyến nghị:** [MUA / GIỮ / BÁN / THEO DÕI] | **Giá mục tiêu:** [X,XXX VND] | **Độ tin cậy:** [X/10]

## 1. 📊 TÓM TẮT ĐIỀU HÀNH
- Khuyến nghị rõ ràng, giá mục tiêu, upside/downside %
- 3-4 luận điểm chính

## 2. 🏛️ BỐI CẢNH VĨ MÔ & NGÀNH
- Lãi suất, tỷ giá, chính sách tiền tệ
- Vị thế ngành, đối thủ cạnh tranh

## 3. 📋 PHÂN TÍCH CƠ BẢN
- Bảng P/E, P/B, ROE, EPS so ngành
- Đánh giá định giá (rẻ/hợp lý/đắt)

## 4. 📈 PHÂN TÍCH KỸ THUẬT
- Xu hướng, cấu trúc giá, volume
- Bollinger, RSI, MACD, SMA confluence
- Vùng hỗ trợ / kháng cự then chốt

## 5. 📐 DỰ BÁO NEURAL NETWORK
- Phân tích kết quả mô hình AI
- Kịch bản giá T+5, T+10

## 6. 🎯 CHIẾN LƯỢC GIAO DỊCH
| Tham số | Giá trị |
|---------|---------|
| Điểm mua | X,XXX – Y,YYY |
| Stop Loss | X,XXX (–Z%) |
| Take Profit 1 | X,XXX (+Z%) |
| Take Profit 2 | X,XXX (+Z%) |
| Tỷ lệ R/R | 1:X.X |
| Khung thời gian | X tuần |

## 7. ⚠️ RỦI RO & KỊCH BẢN
- Kịch bản tích cực / cơ sở / tiêu cực
- Yếu tố rủi ro chính

## 8. 📝 KẾT LUẬN
PHONG CÁCH: Chuyên nghiệp, số liệu cụ thể, bảng biểu rõ ràng, ngắn gọn súc tích."""

        f_text = ""
        if forecast:
            f_text = f"""
DỰ BÁO NEURAL NETWORK:
- Phương pháp: {forecast.get('method','N/A')}
- Hướng: {forecast.get('direction','N/A')} | R²={forecast.get('r_squared','N/A')} | Độ tin cậy: {forecast.get('confidence','N/A')}
- NN: {'✅ Có' if forecast.get('nn_available') else '⚠️ Không'}
- Mục tiêu T+5: {forecast.get('target_1w','N/A')} | T+10: {forecast.get('target_2w','N/A')}
- Lợi nhuận kỳ vọng 2T: {forecast.get('expected_return_2w','N/A')}%
- Stop Loss gợi ý: {forecast.get('stop_loss','N/A')}
- Take Profit 1: {forecast.get('take_profit_1','N/A')} | TP2: {forecast.get('take_profit_2','N/A')}"""

        user_p = f"""Phân tích CỔ PHIẾU **{sym}** — {datetime.now().strftime('%d/%m/%Y')}

CHỈ BÁO KỸ THUẬT:
Giá: {tech.get('current_price','N/A')} | RSI14: {tech.get('rsi','N/A')} | MACD: {tech.get('macd','N/A')}/{tech.get('macd_signal','N/A')}
BB: U={tech.get('bb_upper','N/A')} / M={tech.get('bb_middle','N/A')} / L={tech.get('bb_lower','N/A')}
SMA: 20={tech.get('sma20','N/A')} / 50={tech.get('sma50','N/A')} / 200={tech.get('sma200','N/A')}
EMA9={tech.get('ema9','N/A')} | Stoch %K={tech.get('stoch_k','N/A')} | Williams%R={tech.get('williams_r','N/A')}
ATR14={tech.get('atr','N/A')} | Momentum={tech.get('momentum','N/A')}% | ADX={tech.get('adx','N/A')}
Hỗ trợ: S1={tech.get('support1','N/A')} / S2={tech.get('support2','N/A')}
Kháng cự: R1={tech.get('resistance1','N/A')} / R2={tech.get('resistance2','N/A')}
Xu hướng: NH={tech.get('trend_short','N/A')} / TH={tech.get('trend_medium','N/A')} / DH={tech.get('trend_long','N/A')}

CHỈ SỐ CƠ BẢN:
P/E={fund.get('pe','N/A')} | P/B={fund.get('pb','N/A')} | ROE={fund.get('roe','N/A')}%
EPS={fund.get('eps','N/A')} | Beta={fund.get('beta','N/A')} | Ngành: {fund.get('industry','N/A')}
Vốn hóa={fund.get('market_cap','N/A')} tỷ | DY={fund.get('dividend_yield','N/A')}%
52W: H={fund.get('52w_high','N/A')} / L={fund.get('52w_low','N/A')}
{f_text}"""

        return self._call(sys_p, user_p, sym)

    # ── FUND REPORT ───────────────────────────────────────────
    def analyze_fund(self, sym, info, tech, forecast):
        if not self.client:
            return {"analysis": self._no_key(sym), "recommendation": "HOLD", "confidence": 5}

        sys_p = """Bạn là chuyên gia phân tích quỹ đầu tư cấp cao.
Viết báo cáo phân tích CHỨNG CHỈ QUỸ chuyên nghiệp:

# [MÃ QUỸ] — BÁO CÁO PHÂN TÍCH CHỨNG CHỈ QUỸ
**Khuyến nghị:** [MUA/GIỮ/BÁN] | **NAV mục tiêu:** [X.XX VND] | **Kỳ vọng:** [X%/năm]

## 1. TỔNG QUAN QUỸ
## 2. HIỆU SUẤT & BENCHMARK  
## 3. PHÂN TÍCH NAV KỸ THUẬT
## 4. RỦI RO (Risk Level, Sharpe, Drawdown)
## 5. DỰ BÁO NAV
## 6. KHUYẾN NGHỊ & PHÂN BỔ
## 7. KẾT LUẬN"""

        user_p = f"""Quỹ **{sym}** — {datetime.now().strftime('%d/%m/%Y')}
Thông tin quỹ: {json.dumps(info, ensure_ascii=False)}
NAV hiện tại: {tech.get('current_price','N/A')}
RSI: {tech.get('rsi','N/A')} | Xu hướng: {tech.get('trend_short','N/A')}/{tech.get('trend_medium','N/A')}
Dự báo hướng: {forecast.get('direction','N/A')} | R²={forecast.get('r_squared','N/A')}"""

        return self._call(sys_p, user_p, sym)

    # ── FOREX REPORT ──────────────────────────────────────────
    def analyze_forex(self, pair, tech, forecast):
        if not self.client:
            return {"analysis": self._no_key(pair), "direction": "SIDEWAYS", "confidence": 5}

        sys_p = """Bạn là chuyên gia phân tích ngoại hối tại ngân hàng thương mại lớn.
Viết báo cáo phân tích TỶ GIÁ chuyên nghiệp:

# [CẶP TIỀN] — BÁO CÁO PHÂN TÍCH TỶ GIÁ
**Xu hướng:** [TĂNG/GIẢM/ĐI NGANG] | **Mục tiêu:** [X.XX] | **SL:** [X.XX]

## 1. TÓM TẮT XU HƯỚNG
## 2. YẾU TỐ VĨ MÔ (Fed, SBV, lạm phát, thương mại)
## 3. PHÂN TÍCH KỸ THUẬT
## 4. MÔ HÌNH AI DỰ BÁO
## 5. CHIẾN LƯỢC GIAO DỊCH
## 6. RỦI RO & CAN THIỆP NHNN
## 7. KẾT LUẬN"""

        user_p = f"""{pair} — {datetime.now().strftime('%d/%m/%Y')}
Tỷ giá: {tech.get('current_price','N/A')}
RSI: {tech.get('rsi','N/A')} | MACD: {tech.get('macd','N/A')}
BB: {tech.get('bb_upper','N/A')}/{tech.get('bb_lower','N/A')}
S1={tech.get('support1','N/A')} R1={tech.get('resistance1','N/A')}
Xu hướng: {tech.get('trend_short','N/A')}/{tech.get('trend_medium','N/A')}
Dự báo: {forecast.get('direction','N/A')} | Mục tiêu: {forecast.get('target_2w','N/A')}"""

        r = self._call(sys_p, user_p, pair)
        r["direction"] = self._dir(r.get("analysis", ""))
        return r

    def _call(self, sys_p, user_p, sym):
        for model in self.MODELS:
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=0.10, max_tokens=9000,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                )
                text = resp.choices[0].message.content
                rec = self._rec(text)
                conf = self._conf(text)
                logger.info(f"AI: model={model}, rec={rec}, sym={sym}")
                return {"analysis": text, "recommendation": rec, "confidence": conf}
            except Exception as e:
                logger.warning(f"AI model {model}: {e}")
        return {"analysis": self._no_key(sym), "recommendation": "WATCH", "confidence": 5}

    def _rec(self, t):
        u = t.upper()
        if any(k in u for k in ["MUA", "BUY", "OVERWEIGHT", "TĂNG TỶ TRỌNG"]): return "BUY"
        if any(k in u for k in ["BÁN", "SELL", "UNDERWEIGHT"]): return "SELL"
        if any(k in u for k in ["GIỮ", "HOLD", "NEUTRAL"]): return "HOLD"
        return "WATCH"

    def _dir(self, t):
        u = t.upper()
        if any(k in u for k in ["XU HƯỚNG TĂNG", "▲ TĂNG", "TĂNG GIÁ"]): return "UP"
        if any(k in u for k in ["XU HƯỚNG GIẢM", "▼ GIẢM", "GIẢM GIÁ"]): return "DOWN"
        return "SIDEWAYS"

    def _conf(self, t):
        import re
        m = re.search(r"(\d+)/10", t)
        return int(m.group(1)) if m else 7

    def _no_key(self, sym):
        return (
            f"## ⚠️ Chưa cấu hình GROQ_API_KEY_STOCK\n\n"
            f"Hệ thống đã hoàn tất **phân tích kỹ thuật** và **dự báo Neural Network** cho `{sym}`.\n\n"
            f"Để có báo cáo AI chuyên sâu:\n"
            f"1. Truy cập **https://console.groq.com** → Tạo API Key miễn phí\n"
            f"2. Thêm biến môi trường: `GROQ_API_KEY_STOCK=<key>`\n"
            f"3. Khởi động lại server\n\n"
            f"Model sẽ dùng: **LLaMA-3.3 70B** (Groq, miễn phí, nhanh nhất)"
        )


# ══════════════════════════════════════════════════════════════════════
# NEWS AGENT
# ══════════════════════════════════════════════════════════════════════

class NewsAgent:
    @staticmethod
    def get(query: str):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                r = d.text(query, max_results=5)
                return [{"title": i["title"], "body": i["body"][:200]} for i in r]
        except Exception as e:
            logger.debug(f"News: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# DATA FORMATTER (OHLCV → Lightweight Charts format)
# ══════════════════════════════════════════════════════════════════════

def df_to_ohlcv(df: pd.DataFrame):
    """Convert DataFrame to Lightweight Charts OHLCV format"""
    out = []
    for _, row in df.iterrows():
        t = row["time"]
        ts = int(t.timestamp()) if hasattr(t, "timestamp") else int(pd.Timestamp(t).timestamp())
        out.append({
            "time": ts,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row.get("Volume", 0) or 0),
        })
    return out


def series_to_points(times, values):
    """Convert pandas Series → [{time, value}] for Lightweight Charts"""
    out = []
    for t, v in zip(times, values):
        if pd.isna(v):
            continue
        ts = int(pd.Timestamp(t).timestamp())
        out.append({"time": ts, "value": round(float(v), 4)})
    return out


def hist_bars(times, values):
    """For MACD histogram → [{time, value, color}]"""
    out = []
    for t, v in zip(times, values):
        if pd.isna(v):
            continue
        ts = int(pd.Timestamp(t).timestamp())
        out.append({
            "time": ts,
            "value": round(float(v), 4),
            "color": "#00e676" if v >= 0 else "#ff5252",
        })
    return out


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self):
        self.vn = VNStockData()
        self.fd = FundData()
        self.fx = ForexData()
        self.ta = TA()
        self.fc = Forecaster()
        self.ai = AIAgent()
        self.news = NewsAgent()

    def _indicators(self, df: pd.DataFrame):
        """Compute all TA indicators, return tech_summary + chart_data dicts"""
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        times = df["time"]

        rsi = self.ta.rsi(c)
        macd_l, macd_s, macd_h = self.ta.macd(c)
        bb_u, bb_m, bb_l = self.ta.bbands(c)
        sma20 = self.ta.sma(c, 20)
        sma50 = self.ta.sma(c, 50)
        sma200 = self.ta.sma(c, 200)
        ema9 = self.ta.ema(c, 9)
        ema21 = self.ta.ema(c, 21)
        sk, sd = self.ta.stoch(h, l, c)
        atr_ = self.ta.atr(h, l, c)
        wr = self.ta.williams(h, l, c)
        obv_ = self.ta.obv(c, v)
        adx_, pdi, ndi = self.ta.adx(h, l, c)
        cci_ = self.ta.cci(h, l, c)
        mfi_ = self.ta.mfi(h, l, c, v)
        vwap_ = self.ta.vwap(h, l, c, v)
        mom = c / c.shift(10) * 100 - 100
        sr = self.ta.support_resistance(h, l, c)
        ten, kij, sa, sb, chi = self.ta.ichimoku(h, l, c)

        def sv(s): 
            val = s.dropna().iloc[-1] if len(s.dropna()) else np.nan
            return round(float(val), 4) if not np.isnan(val) else "N/A"

        cp = float(c.iloc[-1])
        tech = {
            "current_price": cp,
            "rsi": sv(rsi), "macd": sv(macd_l), "macd_signal": sv(macd_s),
            "macd_hist": sv(macd_h), "bb_upper": sv(bb_u), "bb_middle": sv(bb_m),
            "bb_lower": sv(bb_l), "sma20": sv(sma20), "sma50": sv(sma50),
            "sma200": sv(sma200), "ema9": sv(ema9), "ema21": sv(ema21),
            "stoch_k": sv(sk), "stoch_d": sv(sd), "atr": sv(atr_),
            "williams_r": sv(wr), "obv": sv(obv_), "adx": sv(adx_),
            "plus_di": sv(pdi), "minus_di": sv(ndi), "cci": sv(cci_),
            "mfi": sv(mfi_), "vwap": sv(vwap_), "momentum": sv(mom),
            "tenkan": sv(ten), "kijun": sv(kij),
            **sr,
            "trend_short": "TĂNG" if cp > sv(sma20) else "GIẢM",
            "trend_medium": "TĂNG" if cp > sv(sma50) else "GIẢM",
            "trend_long": "TĂNG" if cp > sv(sma200) else "GIẢM",
        }

        charts = {
            "sma20":   series_to_points(times, sma20),
            "sma50":   series_to_points(times, sma50),
            "sma200":  series_to_points(times, sma200),
            "ema9":    series_to_points(times, ema9),
            "ema21":   series_to_points(times, ema21),
            "bb_upper":  series_to_points(times, bb_u),
            "bb_middle": series_to_points(times, bb_m),
            "bb_lower":  series_to_points(times, bb_l),
            "rsi":       series_to_points(times, rsi),
            "macd_line":   series_to_points(times, macd_l),
            "macd_signal": series_to_points(times, macd_s),
            "macd_hist":   hist_bars(times, macd_h),
            "stoch_k":   series_to_points(times, sk),
            "stoch_d":   series_to_points(times, sd),
            "williams_r": series_to_points(times, wr),
            "adx":       series_to_points(times, adx_),
            "obv":       series_to_points(times, obv_),
            "cci":       series_to_points(times, cci_),
            "mfi":       series_to_points(times, mfi_),
            "vwap":      series_to_points(times, vwap_),
            "tenkan":    series_to_points(times, ten),
            "kijun":     series_to_points(times, kij),
            "ichimoku_a": series_to_points(times, sa),
            "ichimoku_b": series_to_points(times, sb),
        }
        return tech, charts

    def _forecast_points(self, df: pd.DataFrame, fc: dict):
        """Generate forecast time points for Lightweight Charts"""
        last_time = df["time"].iloc[-1]
        future_pts, upper_pts, lower_pts = [], [], []
        for i, (f, u, lo) in enumerate(zip(
            fc.get("forecast", []),
            fc.get("upper_bound", []),
            fc.get("lower_bound", []),
        )):
            dt = pd.Timestamp(last_time) + timedelta(days=i + 1)
            # Skip weekends
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
            ts = int(dt.timestamp())
            future_pts.append({"time": ts, "value": round(f, 2)})
            upper_pts.append({"time": ts, "value": round(u, 2)})
            lower_pts.append({"time": ts, "value": round(lo, 2)})
        fc["forecast_points"] = future_pts
        fc["upper_points"] = upper_pts
        fc["lower_points"] = lower_pts
        return fc

    # ── STOCK ────────────────────────────────────────────────
    def analyze_stock(self, symbol: str):
        sym = symbol.upper()
        df = self.vn.get_historical(sym)
        if df is None or len(df) < 30:
            return self._err(sym, "stock",
                f"Không lấy được dữ liệu giá từ TCBS cho mã {sym}. "
                "Kiểm tra mã cổ phiếu và kết nối mạng.")

        tech, charts = self._indicators(df)
        fund = self.vn.get_fundamental(sym)
        fc = self.fc.predict(df["Close"])
        fc = self._forecast_points(df, fc)
        ai = self.ai.analyze_stock(sym, tech, fund, fc)

        return {
            "mode": "stock",
            "data": {
                "symbol": sym, "type": "stock",
                "ohlcv": df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": fund,
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
            },
        }

    # ── FUND ─────────────────────────────────────────────────
    def analyze_fund(self, symbol: str):
        sym = symbol.upper()
        info_s = self.fd.search(sym)
        fund_id = info_s.get("id") or info_s.get("shortName") or sym
        df = self.fd.get_nav(fund_id)
        if df is None or len(df) < 20:
            return self._err(sym, "fund",
                f"Không lấy được dữ liệu NAV cho quỹ {sym}.")

        tech, charts = self._indicators(df)
        fund_info = self.fd.get_info(fund_id)
        fc = self.fc.predict(df["Close"])
        fc = self._forecast_points(df, fc)
        ai = self.ai.analyze_fund(sym, fund_info, tech, fc)

        return {
            "mode": "fund",
            "data": {
                "symbol": sym, "type": "fund",
                "ohlcv": df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": {},
                "fund_info": fund_info,
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
            },
        }

    # ── FOREX ────────────────────────────────────────────────
    def analyze_forex(self, pair: str):
        pair = pair.upper()
        df = self.fx.gen_history(pair)
        if df is None or len(df) < 20:
            return self._err(pair, "forex", f"Lỗi tạo dữ liệu {pair}.")

        tech, charts = self._indicators(df)
        rate = ForexData.RATES.get(pair, float(df["Close"].iloc[-1]))
        fc = self.fc.predict(df["Close"])
        fc = self._forecast_points(df, fc)
        ai = self.ai.analyze_forex(pair, tech, fc)

        return {
            "mode": "forex",
            "data": {
                "symbol": pair, "type": "forex",
                "ohlcv": df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": {},
                "rate": rate,
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai.get("recommendation", "WATCH"),
                "direction": ai.get("direction", "SIDEWAYS"),
                "confidence": ai.get("confidence", 7),
            },
        }

    def _err(self, sym, mode, msg):
        return {
            "mode": mode,
            "data": {
                "symbol": sym, "type": mode,
                "ohlcv": [], "indicators": {}, "technical": {},
                "fundamental": {}, "forecast": {},
                "analysis": f"## ⚠️ Lỗi dữ liệu\n\n{msg}",
                "recommendation": "WATCH", "confidence": 3,
            },
        }


orc = Orchestrator()


# ══════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        sym = (request.form.get("symbol") or "").strip().upper()
        stype = (request.form.get("type") or "stock").lower()
        if not sym:
            return jsonify({"error": "Vui lòng nhập mã"}), 400
        logger.info(f"[API] {stype}:{sym}")
        if stype == "stock":
            result = orc.analyze_stock(sym)
        elif stype == "fund":
            result = orc.analyze_fund(sym)
        elif stype == "forex":
            result = orc.analyze_forex(sym)
        else:
            return jsonify({"error": f"Loại không hỗ trợ: {stype}"}), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"[API] Error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "version": "5.0",
        "timestamp": datetime.now().isoformat(),
        "ai": orc.ai.ok,
        "forecaster": "Neural Network + Ensemble",
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
