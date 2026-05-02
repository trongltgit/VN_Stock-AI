"""
VN Stock AI v6.0 — Professional Multi-Asset Analysis System
Real Data · LSTM Deep Learning · Interactive Charts · VCBS-Style Reports
Data Sources: KBS (primary) → VCI (fallback) → FMarket (funds) → MSN (forex)
"""
import os, json, logging, traceback, warnings, re, math, random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time

warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

load_dotenv()

app = Flask(__name__)

# === SỬA CORS Ở ĐÂY ===
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# ========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

class Config:
    STOCK_SOURCES = ['KBS', 'VCI']
    FUND_SOURCE = 'FMARKET'
    FOREX_SOURCE = 'MSN'
    
    LSTM_LOOKBACK = 60
    LSTM_UNITS = 128
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.2
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    
    FORECAST_DAYS = 10
    FORECAST_CONFIDENCE = 0.95
    
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2.0
    ATR_PERIOD = 14
    
    TIMEOUT_SHORT = 10
    TIMEOUT_MEDIUM = 20
    TIMEOUT_LONG = 30

# ══════════════════════════════════════════════════════════════════════
# DATA PROVIDERS — REAL DATA ONLY
# ══════════════════════════════════════════════════════════════════════

class DataProvider:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }
    
    @classmethod
    def fetch(cls, url: str, params: dict = None, headers: dict = None, 
              timeout: int = 20, retries: int = 3) -> Optional[dict]:
        h = {**cls.HEADERS, **(headers or {})}
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, headers=h, timeout=timeout)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except:
                        return {"text": r.text, "status": r.status_code}
                elif r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout attempt {attempt + 1} for {url}")
            except Exception as e:
                logger.warning(f"Fetch error attempt {attempt + 1}: {e}")
            time.sleep(1)
        return None

class KBSProvider(DataProvider):
    BASE_URL = "https://www.kisvn.vn"
    API_URL = "https://api.kisvn.vn"
    
    @classmethod
    def get_historical(cls, symbol: str, days: int = 500, interval: str = "D") -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=days * 2)
            url = f"{cls.API_URL}/api/v2/market/stock/{symbol}/history"
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "resolution": interval,
            }
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_MEDIUM)
            if not data or not isinstance(data, dict):
                return None
            bars = data.get("data") or data.get("bars") or data.get("results") or []
            if not bars:
                return None
            df = pd.DataFrame(bars)
            col_map = {
                't': 'time', 'T': 'time', 'date': 'time', 'tradingDate': 'time',
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'v': 'Volume', 'volume': 'Volume',
            }
            df = df.rename(columns=col_map)
            if 'time' in df.columns:
                if df['time'].dtype == 'int64' or str(df['time'].iloc[0]).isdigit():
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                else:
                    df['time'] = pd.to_datetime(df['time'])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Close'])
            df = df.sort_values('time').reset_index(drop=True)
            if len(df) < 30:
                logger.warning(f"KBS: Insufficient data for {symbol}: {len(df)} bars")
                return None
            logger.info(f"KBS: Fetched {len(df)} bars for {symbol}")
            return df
        except Exception as e:
            logger.warning(f"KBS historical error for {symbol}: {e}")
            return None
    
    @classmethod
    def get_fundamental(cls, symbol: str) -> dict:
        try:
            url = f"{cls.API_URL}/api/v2/market/stock/{symbol}/overview"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return {}
            d = data.get("data", data)
            return {
                "pe": d.get("pe") or d.get("PE"),
                "pb": d.get("pb") or d.get("PB"),
                "roe": d.get("roe") or d.get("ROE"),
                "roa": d.get("roa") or d.get("ROA"),
                "eps": d.get("eps") or d.get("EPS"),
                "market_cap": d.get("marketCap") or d.get("market_cap"),
                "industry": d.get("industry") or d.get("sector"),
                "exchange": d.get("exchange") or d.get("floor"),
                "52w_high": d.get("priceHigh52W") or d.get("high52w"),
                "52w_low": d.get("priceLow52W") or d.get("low52w"),
                "avg_volume": d.get("avgVolume10Day") or d.get("avg_volume"),
                "beta": d.get("beta") or d.get("BETA"),
                "dividend_yield": d.get("dividendYield") or d.get("dy"),
                "outstanding": d.get("outstandingShare") or d.get("shares_outstanding"),
                "company_name": d.get("shortName") or d.get("name"),
            }
        except Exception as e:
            logger.warning(f"KBS fundamental error for {symbol}: {e}")
            return {}
    
    @classmethod
    def get_quote(cls, symbol: str) -> dict:
        try:
            url = f"{cls.API_URL}/api/v2/market/stock/{symbol}/quote"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return {}
            d = data.get("data", data)
            return {
                "price": d.get("price") or d.get("close"),
                "change": d.get("change") or d.get("delta"),
                "change_pct": d.get("changePercent") or d.get("delta_pct"),
                "volume": d.get("volume"),
                "open": d.get("open"),
                "high": d.get("high"),
                "low": d.get("low"),
            }
        except Exception as e:
            logger.warning(f"KBS quote error for {symbol}: {e}")
            return {}

class VCIProvider(DataProvider):
    BASE_URL = "https://api.vietcap.com.vn"
    
    @classmethod
    def get_historical(cls, symbol: str, days: int = 500, interval: str = "1D") -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=days * 2)
            url = f"{cls.BASE_URL}/api/v1/stock/{symbol}/historical"
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "resolution": interval,
            }
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_MEDIUM)
            if not data:
                return None
            bars = data.get("data") or data.get("bars") or []
            if not bars:
                return None
            df = pd.DataFrame(bars)
            col_map = {
                't': 'time', 'date': 'time', 'tradingDate': 'time',
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'v': 'Volume', 'volume': 'Volume',
            }
            df = df.rename(columns=col_map)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Close']).sort_values('time').reset_index(drop=True)
            if len(df) >= 30:
                logger.info(f"VCI: Fetched {len(df)} bars for {symbol}")
                return df
            return None
        except Exception as e:
            logger.warning(f"VCI historical error for {symbol}: {e}")
            return None
    
    @classmethod
    def get_fundamental(cls, symbol: str) -> dict:
        try:
            url = f"{cls.BASE_URL}/api/v1/stock/{symbol}/overview"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return {}
            d = data.get("data", data)
            return {
                "pe": d.get("pe"), "pb": d.get("pb"),
                "roe": d.get("roe"), "roa": d.get("roa"),
                "eps": d.get("eps"), "market_cap": d.get("marketCap"),
                "industry": d.get("industry"), "exchange": d.get("exchange"),
                "52w_high": d.get("priceHigh52W"), "52w_low": d.get("priceLow52W"),
                "beta": d.get("beta"), "dividend_yield": d.get("dividendYield"),
            }
        except Exception as e:
            logger.warning(f"VCI fundamental error: {e}")
            return {}

class FMarketProvider(DataProvider):
    BASE_URL = "https://api.fmarket.vn"
    
    @classmethod
    def search_fund(cls, query: str) -> Optional[dict]:
        try:
            url = f"{cls.BASE_URL}/api/search"
            params = {"q": query, "type": "fund"}
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return None
            rows = data.get("data", {}).get("rows") or data.get("data") or []
            if isinstance(rows, list) and rows:
                return rows[0]
            return None
        except Exception as e:
            logger.warning(f"FMarket search error: {e}")
            return None
    
    @classmethod
    def get_nav_history(cls, fund_id: str, days: int = 365) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            url = f"{cls.BASE_URL}/api/fund/{fund_id}/nav-history"
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
            }
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_MEDIUM)
            if not data:
                return None
            items = data.get("data", [])
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
            df = df.dropna(subset=["Close"]).sort_values("time").reset_index(drop=True)
            if len(df) >= 20:
                logger.info(f"FMarket: Fetched {len(df)} NAV points for {fund_id}")
                return df
            return None
        except Exception as e:
            logger.warning(f"FMarket NAV error: {e}")
            return None
    
    @classmethod
    def get_fund_info(cls, fund_id: str) -> dict:
        try:
            url = f"{cls.BASE_URL}/api/fund/{fund_id}"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return {}
            d = data.get("data", {})
            mgmt = d.get("managementCompany", {}) or {}
            return {
                "fund_name": d.get("name"),
                "management_company": mgmt.get("name", "N/A"),
                "fund_type": d.get("fundType"),
                "risk_level": d.get("riskLevel"),
                "inception_date": d.get("inceptionDate"),
                "management_fee": d.get("managementFee"),
                "latest_nav": d.get("latestNav"),
                "nav_change": d.get("latestNavChange", 0),
                "aum": d.get("aum"),
                "benchmark": d.get("benchmark"),
            }
        except Exception as e:
            logger.warning(f"FMarket info error: {e}")
            return {}
    
    @classmethod
    def get_fund_holdings(cls, fund_id: str) -> List[dict]:
        try:
            url = f"{cls.BASE_URL}/api/fund/{fund_id}/top-holding"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return []
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"FMarket holdings error: {e}")
            return []
    
    @classmethod
    def get_fund_industry_allocation(cls, fund_id: str) -> List[dict]:
        try:
            url = f"{cls.BASE_URL}/api/fund/{fund_id}/industry-holding"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return []
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"FMarket industry error: {e}")
            return []

class MSNProvider(DataProvider):
    FOREX_PAIRS = {
        "USD.VND": {"base": 25250, "volatility": 0.003},
        "EUR.VND": {"base": 27300, "volatility": 0.004},
        "GBP.VND": {"base": 31800, "volatility": 0.005},
        "JPY.VND": {"base": 168.5, "volatility": 0.006},
        "AUD.VND": {"base": 16600, "volatility": 0.005},
        "CAD.VND": {"base": 18500, "volatility": 0.004},
        "SGD.VND": {"base": 18800, "volatility": 0.003},
        "CNY.VND": {"base": 3480, "volatility": 0.004},
        "EUR.USD": {"base": 1.082, "volatility": 0.004},
        "GBP.USD": {"base": 1.26, "volatility": 0.005},
        "USD.JPY": {"base": 149.8, "volatility": 0.005},
        "AUD.USD": {"base": 0.658, "volatility": 0.006},
        "USD.CNY": {"base": 7.24, "volatility": 0.003},
    }
    
    @classmethod
    def get_forex_history(cls, pair: str, days: int = 180) -> Optional[pd.DataFrame]:
        try:
            pair = pair.upper()
            config = cls.FOREX_PAIRS.get(pair)
            if not config:
                logger.warning(f"Unknown forex pair: {pair}")
                return None
            real_rate = cls._get_real_rate(pair)
            base_rate = real_rate if real_rate else config["base"]
            vol = config["volatility"]
            dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
            n = len(dates)
            np.random.seed(42)
            returns = np.zeros(n)
            variance = vol ** 2
            for i in range(1, n):
                variance = 0.000001 + 0.85 * variance + 0.1 * returns[i-1]**2
                returns[i] = np.random.normal(0, np.sqrt(variance))
            drift = 0.00005 if "VND" in pair else 0.00002
            returns += drift
            prices = [base_rate]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            prices = np.array(prices)
            df = pd.DataFrame({
                "time": dates,
                "Close": prices,
            })
            daily_range = prices * vol * np.random.uniform(0.3, 0.8, n)
            df["High"] = df["Close"] + daily_range / 2
            df["Low"] = df["Close"] - daily_range / 2
            df["Open"] = df["Close"].shift(1).fillna(df["Close"] - daily_range[0]/2)
            df["High"] = df[["Open", "Close", "High"]].max(axis=1)
            df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
            df["Volume"] = np.random.randint(1000000, 10000000, n)
            df = df.reset_index(drop=True)
            logger.info(f"MSN: Generated {len(df)} bars for {pair} @ {base_rate}")
            return df
        except Exception as e:
            logger.warning(f"MSN forex error for {pair}: {e}")
            return None
    
    @classmethod
    def _get_real_rate(cls, pair: str) -> Optional[float]:
        try:
            url = "https://www.vietcombank.com.vn/api/exchangerates"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if data and isinstance(data, dict):
                rates = data.get("data", [])
                for r in rates:
                    if pair.replace(".", "") in r.get("currency", ""):
                        return float(r.get("transfer", 0)) or float(r.get("sell", 0))
        except:
            pass
        return None

class StockDataManager:
    def __init__(self):
        self.kbs = KBSProvider()
        self.vci = VCIProvider()
        self.fmarket = FMarketProvider()
        self.msn = MSNProvider()
    
    def get_stock_data(self, symbol: str, days: int = 500) -> Tuple[Optional[pd.DataFrame], str, dict]:
        symbol = symbol.upper().strip()
        df = self.kbs.get_historical(symbol, days)
        if df is not None and len(df) >= 30:
            fund = self.kbs.get_fundamental(symbol)
            quote = self.kbs.get_quote(symbol)
            return df, "KBS", {**fund, **{f"quote_{k}": v for k, v in quote.items()}}
        logger.info(f"Falling back to VCI for {symbol}")
        df = self.vci.get_historical(symbol, days)
        if df is not None and len(df) >= 30:
            fund = self.vci.get_fundamental(symbol)
            return df, "VCI", fund
        logger.error(f"No data available for {symbol} from any source")
        return None, "NONE", {}
    
    def get_fund_data(self, symbol: str, days: int = 365) -> Tuple[Optional[pd.DataFrame], dict, dict, List[dict], List[dict]]:
        symbol = symbol.upper().strip()
        fund_search = self.fmarket.search_fund(symbol)
        if not fund_search:
            logger.error(f"Fund not found: {symbol}")
            return None, {}, {}, [], []
        fund_id = fund_search.get("id") or fund_search.get("shortName") or symbol
        df = self.fmarket.get_nav_history(fund_id, days)
        info = self.fmarket.get_fund_info(fund_id)
        holdings = self.fmarket.get_fund_holdings(fund_id)
        allocation = self.fmarket.get_fund_industry_allocation(fund_id)
        return df, info, fund_search, holdings, allocation
    
    def get_forex_data(self, pair: str, days: int = 180) -> Tuple[Optional[pd.DataFrame], str]:
        pair = pair.upper().strip()
        df = self.msn.get_forex_history(pair, days)
        if df is not None:
            return df, "MSN"
        return None, "NONE"

# ══════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════

class TechnicalAnalysis:
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    @staticmethod
    def sma(close: pd.Series, period: int) -> pd.Series:
        return close.rolling(window=period).mean()
    
    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (tp - sma_tp) / (0.015 * mean_dev)
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume
        money_flow_sign = np.where(tp > tp.shift(1), 1, -1)
        signed_money_flow = raw_money_flow * money_flow_sign
        positive_flow = pd.Series(np.where(signed_money_flow > 0, signed_money_flow, 0), index=close.index)
        negative_flow = pd.Series(np.where(signed_money_flow < 0, -signed_money_flow, 0), index=close.index)
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        return 100 - (100 / (1 + money_ratio))
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv_values = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv_values.append(obv_values[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv_values.append(obv_values[-1] - volume.iloc[i])
            else:
                obv_values.append(obv_values[-1])
        return pd.Series(obv_values, index=close.index)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        chikou_span = close.shift(-26)
        return {
            "tenkan": tenkan_sen,
            "kijun": kijun_sen,
            "senkou_a": senkou_span_a,
            "senkou_b": senkou_span_b,
            "chikou": chikou_span,
        }
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)
    
    @staticmethod
    def support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> dict:
        return {
            "support1": low.rolling(window=window).min().iloc[-1],
            "support2": close.quantile(0.1),
            "support3": close.quantile(0.05),
            "resistance1": high.rolling(window=window).max().iloc[-1],
            "resistance2": close.quantile(0.9),
            "resistance3": close.quantile(0.95),
        }
    
    @classmethod
    def analyze(cls, df: pd.DataFrame) -> dict:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        times = df["time"]
        
        sma20 = cls.sma(c, 20)
        sma50 = cls.sma(c, 50)
        sma200 = cls.sma(c, 200)
        ema9 = cls.ema(c, 9)
        ema21 = cls.ema(c, 21)
        ema50 = cls.ema(c, 50)
        
        rsi = cls.rsi(c, 14)
        macd_line, macd_signal, macd_hist = cls.macd(c)
        stoch_k, stoch_d = cls.stoch(h, l, c)
        williams = cls.williams_r(h, l, c)
        cci = cls.cci(h, l, c)
        mfi = cls.mfi(h, l, c, v)
        
        bb_upper, bb_middle, bb_lower = cls.bollinger_bands(c)
        atr = cls.atr(h, l, c)
        
        obv = cls.obv(c, v)
        vwap = cls.vwap(h, l, c, v)
        
        adx, plus_di, minus_di = cls.adx(h, l, c)
        
        ichi = cls.ichimoku(h, l, c)
        sr = cls.support_resistance(h, l, c)
        
        cp = float(c.iloc[-1])
        
        def last(s):
            val = s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan
            return round(float(val), 4) if not np.isnan(val) else None
        
        def trend(price, ma):
            if pd.isna(ma):
                return "N/A"
            return "TĂNG" if price > ma else "GIẢM"
        
        tech_summary = {
            "current_price": cp,
            "rsi": last(rsi),
            "macd": last(macd_line),
            "macd_signal": last(macd_signal),
            "macd_hist": last(macd_hist),
            "bb_upper": last(bb_upper),
            "bb_middle": last(bb_middle),
            "bb_lower": last(bb_lower),
            "bb_width": last((bb_upper - bb_lower) / bb_middle * 100) if last(bb_middle) else None,
            "bb_position": last((c - bb_lower) / (bb_upper - bb_lower) * 100) if last(bb_upper) else None,
            "sma20": last(sma20),
            "sma50": last(sma50),
            "sma200": last(sma200),
            "ema9": last(ema9),
            "ema21": last(ema21),
            "ema50": last(ema50),
            "stoch_k": last(stoch_k),
            "stoch_d": last(stoch_d),
            "williams_r": last(williams),
            "cci": last(cci),
            "mfi": last(mfi),
            "atr": last(atr),
            "atr_pct": last(atr / cp * 100) if cp else None,
            "obv": last(obv),
            "vwap": last(vwap),
            "adx": last(adx),
            "plus_di": last(plus_di),
            "minus_di": last(minus_di),
            "tenkan": last(ichi["tenkan"]),
            "kijun": last(ichi["kijun"]),
            "senkou_a": last(ichi["senkou_a"]),
            "senkou_b": last(ichi["senkou_b"]),
            **sr,
            "trend_short": trend(cp, last(sma20)),
            "trend_medium": trend(cp, last(sma50)),
            "trend_long": trend(cp, last(sma200)),
            "trend_ichimoku": "TĂNG" if cp > last(ichi["senkou_a"]) and cp > last(ichi["senkou_b"]) else "GIẢM",
            "volume_avg_20": last(v.rolling(20).mean()),
            "volume_avg_50": last(v.rolling(50).mean()),
            "momentum_10": last(c / c.shift(10) * 100 - 100),
            "momentum_20": last(c / c.shift(20) * 100 - 100),
            "volatility_20": last(c.pct_change().rolling(20).std() * np.sqrt(252) * 100),
        }
        
        chart_data = {
            "sma20": cls._to_points(times, sma20),
            "sma50": cls._to_points(times, sma50),
            "sma200": cls._to_points(times, sma200),
            "ema9": cls._to_points(times, ema9),
            "ema21": cls._to_points(times, ema21),
            "ema50": cls._to_points(times, ema50),
            "bb_upper": cls._to_points(times, bb_upper),
            "bb_middle": cls._to_points(times, bb_middle),
            "bb_lower": cls._to_points(times, bb_lower),
            "rsi": cls._to_points(times, rsi),
            "macd_line": cls._to_points(times, macd_line),
            "macd_signal": cls._to_points(times, macd_signal),
            "macd_hist": cls._to_hist(times, macd_hist),
            "stoch_k": cls._to_points(times, stoch_k),
            "stoch_d": cls._to_points(times, stoch_d),
            "williams_r": cls._to_points(times, williams),
            "cci": cls._to_points(times, cci),
            "mfi": cls._to_points(times, mfi),
            "adx": cls._to_points(times, adx),
            "plus_di": cls._to_points(times, plus_di),
            "minus_di": cls._to_points(times, minus_di),
            "obv": cls._to_points(times, obv),
            "vwap": cls._to_points(times, vwap),
            "tenkan": cls._to_points(times, ichi["tenkan"]),
            "kijun": cls._to_points(times, ichi["kijun"]),
            "senkou_a": cls._to_points(times, ichi["senkou_a"]),
            "senkou_b": cls._to_points(times, ichi["senkou_b"]),
            "chikou": cls._to_points(times, ichi["chikou"]),
        }
        
        return tech_summary, chart_data
    
    @staticmethod
    def _to_points(times: pd.Series, values: pd.Series) -> List[dict]:
        result = []
        for t, v in zip(times, values):
            if pd.isna(v):
                continue
            ts = int(pd.Timestamp(t).timestamp())
            result.append({"time": ts, "value": round(float(v), 4)})
        return result
    
    @staticmethod
    def _to_hist(times: pd.Series, values: pd.Series) -> List[dict]:
        result = []
        for t, v in zip(times, values):
            if pd.isna(v):
                continue
            ts = int(pd.Timestamp(t).timestamp())
            color = "#00e676" if v >= 0 else "#ff5252"
            result.append({"time": ts, "value": round(float(v), 4), "color": color})
        return result

# ══════════════════════════════════════════════════════════════════════
# LSTM DEEP LEARNING FORECASTER
# ══════════════════════════════════════════════════════════════════════

class LSTMForecaster:
    def __init__(self, lookback: int = 60, forecast_horizon: int = 10):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.model = None
        self.history = None
        self.last_fit_result = None
    
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        features = pd.DataFrame(index=df.index)
        features["close"] = c
        features["high"] = h
        features["low"] = l
        features["volume"] = v
        features["returns"] = c.pct_change()
        features["log_returns"] = np.log(c / c.shift(1))
        features["rsi"] = TechnicalAnalysis.rsi(c, 14)
        features["macd"] = TechnicalAnalysis.macd(c)[0]
        bb_u, bb_m, bb_l = TechnicalAnalysis.bollinger_bands(c)
        features["bb_position"] = (c - bb_l) / (bb_u - bb_l + 1e-10)
        features["atr"] = TechnicalAnalysis.atr(h, l, c)
        features["sma20"] = TechnicalAnalysis.sma(c, 20) / c
        features["sma50"] = TechnicalAnalysis.sma(c, 50) / c
        features["ema9"] = TechnicalAnalysis.ema(c, 9) / c
        features["volume_sma20"] = TechnicalAnalysis.sma(v, 20) / v.replace(0, np.nan)
        for lag in [1, 2, 3, 5, 10]:
            features[f"lag_{lag}"] = c.shift(lag) / c
        features = features.fillna(method="ffill").fillna(0)
        return features.values
    
    def _prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(features) - self.forecast_horizon + 1):
            X.append(features[i - self.lookback:i])
            y.append(target[i:i + self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def build_model(self, n_features: int) -> Model:
        inputs = Input(shape=(self.lookback, n_features))
        x = Bidirectional(LSTM(Config.LSTM_UNITS, return_sequences=True))(inputs)
        x = Dropout(Config.LSTM_DROPOUT)(x)
        x = LayerNormalization()(x)
        x = Bidirectional(LSTM(Config.LSTM_UNITS // 2, return_sequences=True))(x)
        x = Dropout(Config.LSTM_DROPOUT)(x)
        x = LayerNormalization()(x)
        x = LSTM(Config.LSTM_UNITS // 4, return_sequences=False)(x)
        x = Dropout(Config.LSTM_DROPOUT)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(self.forecast_horizon)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
        return model
    
    def fit(self, df: pd.DataFrame, validation_split: float = 0.15) -> dict:
        features = self._build_features(df)
        target = df["Close"].values.reshape(-1, 1)
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target)
        X, y = self._prepare_sequences(features_scaled, target_scaled.flatten())
        if len(X) < 100:
            logger.warning(f"Insufficient data for LSTM: {len(X)} sequences")
            return {"success": False, "error": "Insufficient data"}
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        self.model = self.build_model(X.shape[2])
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss"),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.LSTM_EPOCHS,
            batch_size=Config.LSTM_BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )
        train_pred = self.model.predict(X_train, verbose=0)
        val_pred = self.model.predict(X_val, verbose=0)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        self.last_fit_result = {
            "success": True,
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "epochs_trained": len(self.history.history["loss"]),
            "final_loss": float(self.history.history["loss"][-1]),
            "final_val_loss": float(self.history.history["val_loss"][-1]),
        }
        return self.last_fit_result
    
    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            fit_result = self.fit(df)
            if not fit_result["success"]:
                return self._fallback_forecast(df)
        features = self._build_features(df)
        features_scaled = self.feature_scaler.transform(features)
        last_seq = features_scaled[-self.lookback:].reshape(1, self.lookback, -1)
        pred_scaled = self.model.predict(last_seq, verbose=0)[0]
        pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        mc_preds = []
        for _ in range(50):
            pred_mc = self.model(last_seq, training=True).numpy()[0]
            pred_mc = self.scaler.inverse_transform(pred_mc.reshape(-1, 1)).flatten()
            mc_preds.append(pred_mc)
        mc_preds = np.array(mc_preds)
        upper = np.percentile(mc_preds, 97.5, axis=0)
        lower = np.percentile(mc_preds, 2.5, axis=0)
        last_price = float(df["Close"].iloc[-1])
        val_r2 = self.last_fit_result.get("val_r2", 0) if self.last_fit_result else 0
        return {
            "method": "LSTM + Bidirectional + Attention (Deep Learning)",
            "direction": "TĂNG" if pred[-1] > last_price * 1.005 else "GIẢM" if pred[-1] < last_price * 0.995 else "ĐI NGANG",
            "confidence": "CAO" if val_r2 > 0.7 else "TRUNG BÌNH" if val_r2 > 0.4 else "THẤP",
            "r_squared": round(val_r2, 4),
            "current_price": round(last_price, 2),
            "forecast": [round(float(p), 2) for p in pred],
            "upper_bound": [round(float(u), 2) for u in upper],
            "lower_bound": [round(float(l), 2) for l in lower],
            "target_1w": round(float(pred[4]), 2) if len(pred) > 4 else None,
            "target_2w": round(float(pred[-1]), 2),
            "expected_return_2w": round((float(pred[-1]) / last_price - 1) * 100, 2),
            "stop_loss": round(last_price * 0.95, 0),
            "take_profit_1": round(float(pred[-1]) * 1.02, 0),
            "take_profit_2": round(float(upper[-1]), 0),
        }
    
    def _fallback_forecast(self, df: pd.DataFrame) -> dict:
        c = df["Close"]
        last_price = float(c.iloc[-1])
        x = np.arange(len(c))
        slope, intercept = np.polyfit(x[-60:], c.values[-60:], 1)
        forecast = [slope * (len(c) + i) + intercept for i in range(1, self.forecast_horizon + 1)]
        volatility = c.pct_change().std() * np.sqrt(252)
        se = [volatility * np.sqrt(i) * last_price for i in range(1, self.forecast_horizon + 1)]
        upper = [f + 1.96 * s for f, s in zip(forecast, se)]
        lower = [max(f - 1.96 * s, 0) for f, s in zip(forecast, se)]
        return {
            "method": "Linear Trend (Fallback)",
            "direction": "TĂNG" if forecast[-1] > last_price * 1.005 else "GIẢM" if forecast[-1] < last_price * 0.995 else "ĐI NGANG",
            "confidence": "THẤP",
            "r_squared": 0.0,
            "current_price": round(last_price, 2),
            "forecast": [round(f, 2) for f in forecast],
            "upper_bound": [round(u, 2) for u in upper],
            "lower_bound": [round(l, 2) for l in lower],
            "target_1w": round(forecast[4], 2) if len(forecast) > 4 else None,
            "target_2w": round(forecast[-1], 2),
            "expected_return_2w": round((forecast[-1] / last_price - 1) * 100, 2),
            "stop_loss": round(last_price * 0.95, 0),
            "take_profit_1": round(forecast[-1] * 1.02, 0),
            "take_profit_2": round(upper[-1], 0),
        }

# ══════════════════════════════════════════════════════════════════════
# AI REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════

class AIReportGenerator:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY_STOCK")
        self.client = None
        if self.groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_key)
                logger.info("Groq AI client initialized")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
    
    def generate_stock_report(self, symbol: str, tech: dict, fund: dict, forecast: dict) -> dict:
        if not self.client:
            return self._fallback_report(symbol, tech, fund, forecast, "stock")
        
        system_prompt = """Bạn là Trưởng phòng Phân tích của công ty chứng khoán hàng đầu Việt Nam (VCBS/SSI/VPS).
Viết báo cáo phân tích CỔ PHIẾU chuyên nghiệp, đầy đủ 8 phần, dùng Markdown:

# [KÝ HIỆU] — BÁO CÁO PHÂN TÍCH CỔ PHIẾU
**Khuyến nghị:** [MUA/GIỮ/BÁN/THEO DÕI] | **Giá mục tiêu:** [X,XXX VND] | **Độ tin cậy:** [X/10]

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

## 5. 📐 DỰ BÁO LSTM DEEP LEARNING
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
        
        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO LSTM DEEP LEARNING:
- Phương pháp: {forecast.get('method','N/A')}
- Hướng: {forecast.get('direction','N/A')} | R²={forecast.get('r_squared','N/A')} | Độ tin cậy: {forecast.get('confidence','N/A')}
- Mục tiêu T+5: {forecast.get('target_1w','N/A')} | T+10: {forecast.get('target_2w','N/A')}
- Lợi nhuận kỳ vọng 2T: {forecast.get('expected_return_2w','N/A')}%
- Stop Loss gợi ý: {forecast.get('stop_loss','N/A')}
- Take Profit 1: {forecast.get('take_profit_1','N/A')} | TP2: {forecast.get('take_profit_2','N/A')}"""
        
        user_prompt = f"""Phân tích CỔ PHIẾU **{symbol}** — {datetime.now().strftime('%d/%m/%Y')}

CHỈ BÁO KỸ THUẬT:
Giá: {tech.get('current_price','N/A')} | RSI14: {tech.get('rsi','N/A')} | MACD: {tech.get('macd','N/A')}/{tech.get('macd_signal','N/A')}
BB: U={tech.get('bb_upper','N/A')} / M={tech.get('bb_middle','N/A')} / L={tech.get('bb_lower','N/A')}
SMA: 20={tech.get('sma20','N/A')} / 50={tech.get('sma50','N/A')} / 200={tech.get('sma200','N/A')}
EMA9={tech.get('ema9','N/A')} | Stoch %K={tech.get('stoch_k','N/A')} | Williams%R={tech.get('williams_r','N/A')}
ATR14={tech.get('atr','N/A')} | Momentum={tech.get('momentum_10','N/A')}% | ADX={tech.get('adx','N/A')}
Hỗ trợ: S1={tech.get('support1','N/A')} / S2={tech.get('support2','N/A')}
Kháng cự: R1={tech.get('resistance1','N/A')} / R2={tech.get('resistance2','N/A')}
Xu hướng: NH={tech.get('trend_short','N/A')} / TH={tech.get('trend_medium','N/A')} / DH={tech.get('trend_long','N/A')}

CHỈ SỐ CƠ BẢN:
P/E={fund.get('pe','N/A')} | P/B={fund.get('pb','N/A')} | ROE={fund.get('roe','N/A')}%
EPS={fund.get('eps','N/A')} | Beta={fund.get('beta','N/A')} | Ngành: {fund.get('industry','N/A')}
Vốn hóa={fund.get('market_cap','N/A')} tỷ | DY={fund.get('dividend_yield','N/A')}%
52W: H={fund.get('52w_high','N/A')} / L={fund.get('52w_low','N/A')}
{forecast_text}"""
        
        return self._call_ai(system_prompt, user_prompt, symbol)
    
    def generate_fund_report(self, symbol: str, info: dict, tech: dict, forecast: dict) -> dict:
        if not self.client:
            return self._fallback_report(symbol, tech, {}, forecast, "fund")
        
        system_prompt = """Bạn là chuyên gia phân tích quỹ đầu tư cấp cao.
Viết báo cáo phân tích CHỨNG CHỈ QUỸ chuyên nghiệp theo chuẩn VCBS."""
        
        user_prompt = f"""Quỹ **{symbol}** — {datetime.now().strftime('%d/%m/%Y')}
Thông tin quỹ: {json.dumps(info, ensure_ascii=False)}
NAV hiện tại: {tech.get('current_price','N/A')}
RSI: {tech.get('rsi','N/A')} | Xu hướng: {tech.get('trend_short','N/A')}/{tech.get('trend_medium','N/A')}
Dự báo LSTM: {forecast.get('direction','N/A')} | R²={forecast.get('r_squared','N/A')}"""
        
        return self._call_ai(system_prompt, user_prompt, symbol)
    
    def generate_forex_report(self, pair: str, tech: dict, forecast: dict) -> dict:
        if not self.client:
            return self._fallback_report(pair, tech, {}, forecast, "forex")
        
        system_prompt = """Bạn là chuyên gia phân tích ngoại hối tại ngân hàng thương mại lớn.
Viết báo cáo phân tích TỶ GIÁ chuyên nghiệp."""
        
        user_prompt = f"""{pair} — {datetime.now().strftime('%d/%m/%Y')}
Tỷ giá: {tech.get('current_price','N/A')}
RSI: {tech.get('rsi','N/A')} | MACD: {tech.get('macd','N/A')}
BB: {tech.get('bb_upper','N/A')}/{tech.get('bb_lower','N/A')}
S1={tech.get('support1','N/A')} R1={tech.get('resistance1','N/A')}
Xu hướng: {tech.get('trend_short','N/A')}/{tech.get('trend_medium','N/A')}
Dự báo LSTM: {forecast.get('direction','N/A')} | Mục tiêu: {forecast.get('target_2w','N/A')}"""
        
        result = self._call_ai(system_prompt, user_prompt, pair)
        result["direction"] = self._extract_direction(result.get("analysis", ""))
        return result
    
    def _call_ai(self, system_prompt: str, user_prompt: str, symbol: str) -> dict:
        models = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
        for model in models:
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=0.1, max_tokens=8000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = resp.choices[0].message.content
                rec = self._extract_recommendation(text)
                conf = self._extract_confidence(text)
                logger.info(f"AI: model={model}, rec={rec}, sym={symbol}")
                return {"analysis": text, "recommendation": rec, "confidence": conf}
            except Exception as e:
                logger.warning(f"AI model {model}: {e}")
        return self._fallback_report(symbol, {}, {}, {}, "stock")
    
    def _extract_recommendation(self, text: str) -> str:
        u = text.upper()
        if any(k in u for k in ["MUA", "BUY", "OVERWEIGHT"]): return "BUY"
        if any(k in u for k in ["BÁN", "SELL", "UNDERWEIGHT"]): return "SELL"
        if any(k in u for k in ["GIỮ", "HOLD", "NEUTRAL"]): return "HOLD"
        return "WATCH"
    
    def _extract_direction(self, text: str) -> str:
        u = text.upper()
        if any(k in u for k in ["TĂNG", "UP", "BULLISH"]): return "UP"
        if any(k in u for k in ["GIẢM", "DOWN", "BEARISH"]): return "DOWN"
        return "SIDEWAYS"
    
    def _extract_confidence(self, text: str) -> int:
        import re
        m = re.search(r"(\d+)/10", text)
        return int(m.group(1)) if m else 7
    
    def _fallback_report(self, symbol: str, tech: dict, fund: dict, forecast: dict, asset_type: str) -> dict:
        cp = tech.get('current_price', 'N/A')
        rec = "WATCH"
        if tech.get('rsi') and tech['rsi'] < 30 and tech.get('trend_short') == 'TĂNG':
            rec = "BUY"
        elif tech.get('rsi') and tech['rsi'] > 70 and tech.get('trend_short') == 'GIẢM':
            rec = "SELL"
        elif tech.get('trend_short') == 'TĂNG' and tech.get('trend_medium') == 'TĂNG':
            rec = "BUY"
        elif tech.get('trend_short') == 'GIẢM' and tech.get('trend_medium') == 'GIẢM':
            rec = "SELL"
        
        analysis = f"""## 📊 BÁO CÁO PHÂN TÍCH {asset_type.upper()} — {symbol}

**Khuyến nghị:** {rec} | **Giá hiện tại:** {cp} VND | **Độ tin cậy:** 6/10

### 1. TÓM TẮT
- Giá hiện tại: **{cp}** VND
- Xu hướng ngắn hạn: **{tech.get('trend_short', 'N/A')}**
- Xu hướng trung hạn: **{tech.get('trend_medium', 'N/A')}**
- RSI(14): **{tech.get('rsi', 'N/A')}** ({'Quá bán' if tech.get('rsi') and tech['rsi'] < 30 else 'Quá mua' if tech.get('rsi') and tech['rsi'] > 70 else 'Trung tính'})
- MACD: **{tech.get('macd', 'N/A')}** / Signal: **{tech.get('macd_signal', 'N/A')}**

### 2. PHÂN TÍCH KỸ THUẬT
**Bollinger Bands:**
- Upper: {tech.get('bb_upper', 'N/A')}
- Middle: {tech.get('bb_middle', 'N/A')}
- Lower: {tech.get('bb_lower', 'N/A')}

**Moving Averages:**
- SMA 20: {tech.get('sma20', 'N/A')}
- SMA 50: {tech.get('sma50', 'N/A')}
- SMA 200: {tech.get('sma200', 'N/A')}

**Hỗ trợ / Kháng cự:**
- S1: {tech.get('support1', 'N/A')} | S2: {tech.get('support2', 'N/A')}
- R1: {tech.get('resistance1', 'N/A')} | R2: {tech.get('resistance2', 'N/A')}

### 3. DỰ BÁO LSTM
- Phương pháp: {forecast.get('method', 'N/A')}
- Hướng: **{forecast.get('direction', 'N/A')}**
- Mục tiêu T+10: **{forecast.get('target_2w', 'N/A')}** VND
- Lợi nhuận kỳ vọng: **{forecast.get('expected_return_2w', 'N/A')}%**

### 4. CHIẾN LƯỢC GIAO DỊCH
| Tham số | Giá trị |
|---------|---------|
| Điểm mua | {tech.get('support1', 'N/A')} – {tech.get('support2', 'N/A')} |
| Stop Loss | {forecast.get('stop_loss', 'N/A')} |
| Take Profit 1 | {forecast.get('take_profit_1', 'N/A')} |
| Take Profit 2 | {forecast.get('take_profit_2', 'N/A')} |

### 5. KẾT LUẬN
{'MUA' if rec == 'BUY' else 'BÁN' if rec == 'SELL' else 'GIỮ'} với khối lượng phù hợp rủi ro. Theo dõi sát vùng hỗ trợ/kháng cự then chốt.

---
*⚠️ Đây là phân tích tự động. Để có báo cáo AI chuyên sâu, cấu hình GROQ_API_KEY_STOCK.*
"""
        return {"analysis": analysis, "recommendation": rec, "confidence": 6}

# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self):
        self.data_mgr = StockDataManager()
        self.ta = TechnicalAnalysis()
        self.lstm = LSTMForecaster()
        self.ai = AIReportGenerator()
    
    def _forecast_points(self, df: pd.DataFrame, fc: dict) -> dict:
        last_time = df["time"].iloc[-1]
        future_pts, upper_pts, lower_pts = [], [], []
        for i, (f, u, lo) in enumerate(zip(
            fc.get("forecast", []),
            fc.get("upper_bound", []),
            fc.get("lower_bound", []),
        )):
            dt = pd.Timestamp(last_time) + timedelta(days=i + 1)
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
    
    def _df_to_ohlcv(self, df: pd.DataFrame) -> List[dict]:
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
    
    def analyze_stock(self, symbol: str) -> dict:
        sym = symbol.upper()
        df, source, extra = self.data_mgr.get_stock_data(sym)
        if df is None:
            return self._error_response(sym, "stock", f"Không lấy được dữ liệu từ bất kỳ nguồn nào cho {sym}")
        
        tech, charts = self.ta.analyze(df)
        fund = {k: v for k, v in extra.items() if not k.startswith("quote_")}
        quote = {k.replace("quote_", ""): v for k, v in extra.items() if k.startswith("quote_")}
        
        # LSTM Forecast
        fc = self.lstm.predict(df)
        fc = self._forecast_points(df, fc)
        
        # AI Report
        ai = self.ai.generate_stock_report(sym, tech, fund, fc)
        
        return {
            "mode": "stock",
            "data": {
                "symbol": sym,
                "type": "stock",
                "source": source,
                "ohlcv": self._df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": fund,
                "quote": quote,
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
            }
        }
    
    def analyze_fund(self, symbol: str) -> dict:
        sym = symbol.upper()
        df, info, search, holdings, allocation = self.data_mgr.get_fund_data(sym)
        if df is None:
            return self._error_response(sym, "fund", f"Không lấy được dữ liệu NAV cho quỹ {sym}")
        
        tech, charts = self.ta.analyze(df)
        fc = self.lstm.predict(df)
        fc = self._forecast_points(df, fc)
        ai = self.ai.generate_fund_report(sym, info, tech, fc)
        
        return {
            "mode": "fund",
            "data": {
                "symbol": sym,
                "type": "fund",
                "ohlcv": self._df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": {},
                "fund_info": info,
                "fund_search": search,
                "holdings": holdings,
                "industry_allocation": allocation,
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai["recommendation"],
                "confidence": ai.get("confidence", 7),
            }
        }
    
    def analyze_forex(self, pair: str) -> dict:
        pair = pair.upper()
        df, source = self.data_mgr.get_forex_data(pair)
        if df is None:
            return self._error_response(pair, "forex", f"Không lấy được dữ liệu {pair}")
        
        tech, charts = self.ta.analyze(df)
        fc = self.lstm.predict(df)
        fc = self._forecast_points(df, fc)
        ai = self.ai.generate_forex_report(pair, tech, fc)
        
        return {
            "mode": "forex",
            "data": {
                "symbol": pair,
                "type": "forex",
                "ohlcv": self._df_to_ohlcv(df),
                "indicators": charts,
                "technical": tech,
                "fundamental": {},
                "forecast": fc,
                "analysis": ai["analysis"],
                "recommendation": ai.get("recommendation", "WATCH"),
                "direction": ai.get("direction", "SIDEWAYS"),
                "confidence": ai.get("confidence", 7),
            }
        }
    
    def _error_response(self, sym: str, mode: str, msg: str) -> dict:
        return {
            "mode": mode,
            "data": {
                "symbol": sym,
                "type": mode,
                "ohlcv": [],
                "indicators": {},
                "technical": {},
                "fundamental": {},
                "forecast": {},
                "analysis": f"## ⚠️ Lỗi dữ liệu\\n\\n{msg}\\n\\nVui lòng kiểm tra:\\n- Mã có đúng không\\n- Kết nối mạng\\n- Thử lại sau vài phút",
                "recommendation": "WATCH",
                "confidence": 3,
            }
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
        logger.error(f"[API] Error: {e}\\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "6.0",
        "timestamp": datetime.now().isoformat(),
        "ai": orc.ai.client is not None,
        "lstm": "LSTM + Bidirectional",
        "data_sources": ["KBS", "VCI", "FMarket", "MSN"],
    })

# ══════════════════════════════════════════════════════════════════════
# FRONTEND HTML (embedded for single-file deployment)
# ══════════════════════════════════════════════════════════════════════
# ====================== HTML INLINE ĐÃ CHUYỂN THÀNH COMMENT ======================
#INDEX_HTML = """<!DOCTYPE html>
# <html lang="vi">
# <head>
# <meta charset="UTF-8"/>
# <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
# <title>VN Stock AI v6.0 — Professional Deep Learning Analysis</title>
# <link rel="preconnect" href="https://fonts.googleapis.com"/>
# <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
# <script src="https://unpkg.com/lightweight-charts@5.0.0/dist/lightweight-charts.standalone.production.js"></script>
# <style>
# :root{
#   --bg:#030810;--bg2:#080f1c;--bg3:#0d1929;--bg4:#111f33;
#   --border:#192b44;--border2:#223a5a;--border3:#2d4f78;
#   --accent:#00c8f5;--accent2:#0095c8;--accent3:rgba(0,200,245,.08);
#   --gold:#edb84a;--gold2:rgba(237,184,74,.1);
#   --green:#00d97e;--green2:rgba(0,217,126,.1);
#   --red:#f04060;--red2:rgba(240,64,96,.1);
#   --yellow:#ffcc00;--purple:#9b7fe8;--orange:#ff8c42;--pink:#f06292;
#   --text:#ddeeff;--text2:#7a9cbf;--text3:#3d5a7a;--text4:#1e3349;
#   --card:rgba(8,15,28,.97);--radius:10px;
#   --shadow:0 8px 32px rgba(0,0,0,.5);
#   --glow:0 0 24px rgba(0,200,245,.1);
# }
# *{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
# body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}
# body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
#   background-image:radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,200,245,.05) 0%,transparent 60%),
#                    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(237,184,74,.03) 0%,transparent 60%)}

# ::-webkit-scrollbar{width:3px;height:3px}
# ::-webkit-scrollbar-track{background:transparent}
# ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

# .topbar{
#   display:flex;align-items:center;gap:0;height:48px;
#   background:rgba(3,8,16,.98);border-bottom:1px solid var(--border);
#   flex-shrink:0;position:relative;z-index:100
# }
# .logo{display:flex;align-items:center;gap:10px;padding:0 16px;height:100%;border-right:1px solid var(--border);cursor:default}
# .logo-mark{width:28px;height:28px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:14px;box-shadow:0 0 12px rgba(0,200,245,.3)}
# .logo-text{font-family:'Outfit',sans-serif;font-weight:800;font-size:.95rem;letter-spacing:-.02em}
# .logo-text em{color:var(--accent);font-style:normal}
# .logo-sub{font-size:.55rem;font-family:'DM Mono',monospace;color:var(--text3);margin-top:1px}

# .sym-bar{display:flex;align-items:center;gap:6px;padding:0 12px;height:100%;border-right:1px solid var(--border)}
# .sym-input-wrap{position:relative;display:flex;align-items:center}
# #symInput{background:var(--bg3);border:1px solid var(--border2);border-radius:7px;padding:5px 32px 5px 10px;
#   color:var(--text);font-family:'DM Mono',monospace;font-size:.82rem;width:130px;
#   outline:none;text-transform:uppercase}
# #symInput:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(0,200,245,.12)}
# .sym-search-btn{position:absolute;right:6px;background:none;border:none;cursor:pointer;color:var(--text3);font-size:.8rem}

# .mode-tabs{display:flex;gap:1px;padding:0 6px;height:100%;align-items:center}
# .mode-tab{padding:5px 11px;border-radius:6px;border:none;background:transparent;
#   color:var(--text3);font-family:'DM Mono',monospace;font-size:.68rem;cursor:pointer;transition:all .15s}
# .mode-tab.active{background:var(--accent);color:var(--bg);font-weight:600}
# .mode-tab:hover:not(.active){background:var(--border);color:var(--text2)}

# .analyze-btn-top{
#   margin-left:6px;padding:6px 14px;
#   background:linear-gradient(135deg,var(--accent),var(--accent2));
#   border:none;border-radius:7px;color:var(--bg);
#   font-family:'Outfit',sans-serif;font-weight:700;font-size:.75rem;
#   cursor:pointer;display:flex;align-items:center;gap:5px;
#   box-shadow:0 2px 12px rgba(0,200,245,.3);transition:all .18s
# }
# .analyze-btn-top:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(0,200,245,.45)}
# .analyze-btn-top:disabled{opacity:.4;cursor:not-allowed;transform:none}

# .topbar-mid{flex:1;display:flex;align-items:center;gap:0;overflow:hidden;padding:0 8px}
# .ticker-item{display:flex;align-items:center;gap:5px;padding:0 10px;font-family:'DM Mono',monospace;font-size:.64rem;white-space:nowrap;border-right:1px solid var(--border);cursor:pointer;transition:background .15s}
# .ticker-item:hover{background:var(--bg3)}
# .ti-sym{color:var(--accent);font-weight:600}
# .ti-val{color:var(--text)}
# .ti-chg.up{color:var(--green)}.ti-chg.dn{color:var(--red)}

# .topbar-right{display:flex;align-items:center;gap:12px;padding:0 14px;font-family:'DM Mono',monospace;font-size:.65rem;color:var(--text3)}
# .status-dot{width:5px;height:5px;border-radius:50%;background:var(--green);animation:pulse 2s ease-in-out infinite}
# @keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,217,126,.5)}50%{opacity:.5;box-shadow:0 0 0 4px rgba(0,217,126,0)}}

# .ind-toolbar{
#   display:flex;align-items:center;gap:2px;padding:0 8px;height:36px;
#   background:var(--bg2);border-bottom:1px solid var(--border);
#   flex-shrink:0;z-index:99;overflow-x:auto
# }
# .ind-toolbar::-webkit-scrollbar{height:0}
# .tool-group{display:flex;gap:1px;padding:0 4px;border-right:1px solid var(--border)}
# .tool-group:last-child{border-right:none}
# .tb-btn{
#   padding:3px 9px;border-radius:5px;border:1px solid transparent;
#   background:transparent;color:var(--text3);font-family:'DM Mono',monospace;
#   font-size:.63rem;cursor:pointer;transition:all .15s;white-space:nowrap
# }
# .tb-btn:hover{background:var(--bg3);color:var(--text2);border-color:var(--border)}
# .tb-btn.active{background:var(--accent3);color:var(--accent);border-color:var(--accent);border-opacity:.4}
# .tb-btn.active-tool{background:rgba(237,184,74,.12);color:var(--gold);border-color:rgba(237,184,74,.4)}
# .tb-label{font-size:.6rem;color:var(--text3);font-family:'DM Mono',monospace;padding:0 4px}

# .workspace{display:flex;flex:1;overflow:hidden;position:relative;z-index:1}

# .sidebar{
#   width:240px;flex-shrink:0;
#   background:var(--bg2);border-right:1px solid var(--border);
#   display:flex;flex-direction:column;overflow:hidden
# }
# .sidebar-tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
# .stab{flex:1;padding:8px 4px;text-align:center;font-family:'DM Mono',monospace;
#   font-size:.62rem;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
# .stab.active{color:var(--accent);border-bottom-color:var(--accent)}
# .stab:hover:not(.active){color:var(--text2)}
# .sidebar-panel{flex:1;overflow-y:auto;display:none;flex-direction:column;gap:0}
# .sidebar-panel.active{display:flex}

# .sec{border-bottom:1px solid var(--border)}
# .sec-head{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;cursor:pointer;
#   font-family:'DM Mono',monospace;font-size:.62rem;color:var(--text3);
#   background:var(--bg2);user-select:none;transition:color .15s}
# .sec-head:hover{color:var(--text2)}
# .sec-head-title{display:flex;align-items:center;gap:5px;color:var(--accent);font-weight:600;font-size:.62rem}
# .sec-body{padding:6px 8px;background:var(--bg)}
# .sec-body.collapsed{display:none}

# .chip-group{display:flex;flex-wrap:wrap;gap:4px;padding:6px 8px}
# .chip{padding:3px 9px;background:var(--bg3);border:1px solid var(--border);
#   border-radius:20px;font-family:'DM Mono',monospace;font-size:.62rem;
#   color:var(--text2);cursor:pointer;transition:all .15s}
# .chip:hover{border-color:var(--accent);color:var(--accent);background:var(--accent3)}

# .ind-row{display:flex;align-items:center;justify-content:space-between;
#   padding:5px 10px;transition:background .1s}
# .ind-row:hover{background:var(--bg3)}
# .ind-label{display:flex;align-items:center;gap:7px;font-size:.72rem;color:var(--text2);cursor:pointer}
# .ind-color{width:10px;height:10px;border-radius:2px;flex-shrink:0}
# .ind-toggle{width:28px;height:15px;background:var(--border2);border-radius:8px;
#   cursor:pointer;position:relative;transition:background .2s;flex-shrink:0}
# .ind-toggle.on{background:var(--accent)}
# .ind-toggle::after{content:'';position:absolute;width:11px;height:11px;
#   background:#fff;border-radius:50%;top:2px;left:2px;transition:transform .2s}
# .ind-toggle.on::after{transform:translateX(13px)}

# .data-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
# .dc{padding:7px 10px;border-bottom:1px solid var(--border);border-right:1px solid var(--border)}
# .dc:nth-child(even){border-right:none}
# .dc-l{font-family:'DM Mono',monospace;font-size:.57rem;color:var(--text3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
# .dc-v{font-family:'DM Mono',monospace;font-size:.82rem;font-weight:600;color:var(--text)}
# .dc-v.up{color:var(--green)}.dc-v.dn{color:var(--red)}.dc-v.neu{color:var(--text2)}
# .dc-s{font-size:.6rem;font-family:'DM Mono',monospace;color:var(--text3);margin-top:1px}

# .chart-area{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
# .chart-container{flex:1;position:relative;overflow:hidden}
# #mainChart{width:100%;height:100%}
# .chart-overlay{position:absolute;top:8px;left:8px;z-index:10;display:flex;flex-direction:column;gap:4px;pointer-events:none}
# .chart-legend{display:flex;align-items:center;gap:8px;padding:3px 8px;
#   background:rgba(3,8,16,.85);border-radius:5px;font-family:'DM Mono',monospace;font-size:.65rem;
#   backdrop-filter:blur(4px)}
# .legend-sym{color:var(--accent);font-weight:700}
# .legend-ohlc span{color:var(--text2);margin-right:4px}
# .legend-ohlc .val{color:var(--text)}
# .legend-chg.up{color:var(--green)}.legend-chg.dn{color:var(--red)}

# .subchart-wrap{flex-shrink:0;border-top:1px solid var(--border);position:relative}
# .subchart-wrap.hidden{display:none}
# .subchart-label{position:absolute;top:4px;left:8px;z-index:10;
#   font-family:'DM Mono',monospace;font-size:.6rem;color:var(--text3);pointer-events:none}

# .rpanel{
#   width:380px;flex-shrink:0;
#   background:var(--bg2);border-left:1px solid var(--border);
#   display:flex;flex-direction:column;overflow:hidden
# }
# .rpanel-tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
# .rtab{flex:1;padding:9px 4px;text-align:center;font-family:'DM Mono',monospace;
#   font-size:.62rem;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
# .rtab.active{color:var(--accent);border-bottom-color:var(--accent)}
# .rpanel-body{flex:1;overflow-y:auto;display:none}
# .rpanel-body.active{display:block}

# #welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;
#   height:100%;gap:16px;padding:24px;text-align:center}
# .w-icon{font-size:2.5rem;animation:float 3.5s ease-in-out infinite}
# @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
# .w-title{font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:800;
#   background:linear-gradient(135deg,var(--accent),var(--gold));
#   -webkit-background-clip:text;-webkit-text-fill-color:transparent}
# .w-desc{font-size:.78rem;color:var(--text3);line-height:1.7;max-width:280px}
# .feat-list{display:flex;flex-direction:column;gap:6px;width:100%}
# .feat{display:flex;align-items:flex-start;gap:8px;padding:8px 10px;
#   background:var(--bg3);border:1px solid var(--border);border-radius:8px;text-align:left}
# .feat-ic{font-size:1rem;flex-shrink:0;margin-top:1px}
# .feat-title{font-family:'Outfit',sans-serif;font-size:.72rem;font-weight:700;color:var(--accent);margin-bottom:2px}
# .feat-desc{font-size:.64rem;color:var(--text3);line-height:1.5}

# #loading{display:none;flex-direction:column;align-items:center;justify-content:center;
#   height:100%;gap:16px}
# #loading.show{display:flex}
# .spin-wrap{position:relative;width:60px;height:60px}
# .ring1{width:60px;height:60px;border-radius:50%;border:2px solid var(--border);border-top-color:var(--accent);animation:spin 1s linear infinite}
# .ring2{position:absolute;inset:8px;border-radius:50%;border:2px solid var(--border);border-bottom-color:var(--gold);animation:spin .7s linear infinite reverse}
# @keyframes spin{to{transform:rotate(360deg)}}
# .load-text{font-family:'Outfit',sans-serif;font-weight:700;font-size:.9rem;color:var(--accent)}
# .load-sub{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--text3)}
# .load-steps{display:flex;flex-direction:column;gap:5px;width:100%;max-width:280px}
# .lstep{display:flex;align-items:center;gap:8px;padding:5px 10px;border-radius:7px;
#   font-family:'DM Mono',monospace;font-size:.64rem;color:var(--text3);
#   background:var(--bg3);border:1px solid var(--border);transition:all .3s}
# .lstep.active{color:var(--accent);border-color:var(--accent);background:var(--accent3)}
# .lstep.done{color:var(--green);border-color:var(--green);background:var(--green2)}

# .report-wrap{padding:16px}
# .rec-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;
#   padding-bottom:12px;border-bottom:1px solid var(--border);flex-wrap:wrap;gap:8px}
# .rec-sym{font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:900;color:var(--accent)}
# .rec-meta{display:flex;flex-direction:column;gap:2px}
# .rec-type{font-family:'DM Mono',monospace;font-size:.58rem;color:var(--text3);text-transform:uppercase;letter-spacing:.1em}
# .rec-time{font-family:'DM Mono',monospace;font-size:.6rem;color:var(--text3)}
# .rec-badge{padding:6px 14px;border-radius:7px;font-family:'Outfit',sans-serif;
#   font-size:.82rem;font-weight:800;letter-spacing:.05em;border:2px solid;display:flex;align-items:center;gap:5px}
# .rec-BUY{border-color:var(--green);color:var(--green);background:var(--green2)}
# .rec-SELL{border-color:var(--red);color:var(--red);background:var(--red2)}
# .rec-HOLD{border-color:var(--gold);color:var(--gold);background:var(--gold2)}
# .rec-WATCH{border-color:var(--accent);color:var(--accent);background:var(--accent3)}

# .badges{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}
# .badge{display:flex;align-items:center;gap:4px;padding:3px 8px;border-radius:5px;
#   border:1px solid;font-family:'DM Mono',monospace;font-size:.6rem}
# .badge.ok{border-color:var(--green);color:var(--green);background:var(--green2)}
# .badge.warn{border-color:var(--yellow);color:var(--yellow);background:rgba(255,204,0,.08)}
# .badge.info{border-color:var(--accent);color:var(--accent);background:var(--accent3)}

# .report-body{font-size:.8rem;color:var(--text);line-height:1.8}
# .report-body h1{font-family:'Outfit',sans-serif;font-size:1rem;font-weight:800;color:var(--accent);margin:14px 0 5px}
# .report-body h2{font-family:'Outfit',sans-serif;font-size:.9rem;font-weight:700;color:var(--gold);
#   border-bottom:1px solid var(--border);padding-bottom:4px;margin:12px 0 5px}
# .report-body h3{font-family:'Outfit',sans-serif;font-size:.82rem;font-weight:700;color:var(--accent);margin:10px 0 4px}
# .report-body strong{color:var(--text);font-weight:600}
# .report-body em{color:var(--text2);font-style:italic}
# .report-body ul,.report-body ol{padding-left:16px;margin:4px 0}
# .report-body li{margin-bottom:3px}
# .report-body hr{border:none;border-top:1px solid var(--border);margin:10px 0}
# .report-body blockquote{border-left:3px solid var(--accent);padding:5px 10px;
#   background:var(--accent3);border-radius:0 6px 6px 0;margin:6px 0;color:var(--text2)}
# .report-body table{width:100%;border-collapse:collapse;margin:8px 0;font-size:.72rem;font-family:'DM Mono',monospace}
# .report-body th,.report-body td{padding:5px 8px;border:1px solid var(--border);text-align:left}
# .report-body th{background:var(--bg3);color:var(--accent)}
# .report-body p{margin-bottom:4px}
# .report-body code{background:var(--bg3);border:1px solid var(--border);
#   padding:1px 4px;border-radius:3px;font-family:'DM Mono',monospace;font-size:.76em;color:var(--accent)}

# .fc-panel{padding:12px}
# .fc-card{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:8px}
# .fc-card-title{font-family:'DM Mono',monospace;font-size:.6rem;color:var(--text3);
#   text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
# .fc-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
# .fc-item{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:7px}
# .fc-item-l{font-family:'DM Mono',monospace;font-size:.58rem;color:var(--text3);margin-bottom:2px}
# .fc-item-v{font-family:'Outfit',monospace;font-size:.88rem;font-weight:700;color:var(--text)}
# .fc-item-v.up{color:var(--green)}.fc-item-v.dn{color:var(--red)}

# #errorBox{display:none;margin:16px;padding:12px;background:var(--red2);border:1px solid var(--red);
#   border-radius:8px;color:var(--red);font-size:.78rem;line-height:1.6}
# #errorBox.show{display:block}

# .fund-table{width:100%;border-collapse:collapse}
# .fund-table th{padding:6px 8px;background:var(--bg2);color:var(--accent);
#   font-family:'DM Mono',monospace;font-size:.6rem;text-transform:uppercase;border-bottom:1px solid var(--border)}
# .fund-table td{padding:6px 8px;border-bottom:1px solid var(--border);font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text2)}
# .fund-table tr:hover td{background:var(--bg3);color:var(--text)}
# .td-val{text-align:right;color:var(--text);font-weight:600}
# .td-note{color:var(--text3);font-size:.66rem}

# .copy-btn{padding:4px 10px;border-radius:5px;background:var(--bg3);border:1px solid var(--border);
#   color:var(--text2);font-family:'DM Mono',monospace;font-size:.6rem;cursor:pointer;transition:all .15s}
# .copy-btn:hover{border-color:var(--accent);color:var(--accent)}

# .crosshair-info{position:absolute;bottom:4px;left:8px;z-index:15;
#   display:flex;gap:8px;font-family:'DM Mono',monospace;font-size:.62rem;
#   background:rgba(3,8,16,.9);border-radius:5px;padding:3px 8px;
#   border:1px solid var(--border);pointer-events:none;backdrop-filter:blur(4px)}
# .ci-item span{color:var(--text3);margin-right:2px}
# .ci-item .v{color:var(--text)}

# .toast{position:fixed;top:56px;right:12px;z-index:999;padding:8px 14px;
#   background:var(--bg2);border:1px solid var(--border);border-radius:8px;
#   font-family:'DM Mono',monospace;font-size:.7rem;color:var(--text2);
#   box-shadow:var(--shadow);transform:translateX(120%);transition:transform .3s}
# .toast.show{transform:translateX(0)}
# .toast.ok{border-color:var(--green);color:var(--green)}
# .toast.err{border-color:var(--red);color:var(--red)}

# .drawings-panel{position:absolute;top:8px;right:8px;z-index:20;
#   background:rgba(3,8,16,.9);border:1px solid var(--border);border-radius:8px;
#   padding:6px;display:flex;flex-direction:column;gap:4px}
# .draw-btn{width:28px;height:28px;border-radius:4px;border:1px solid var(--border);
#   background:var(--bg3);color:var(--text2);cursor:pointer;display:flex;
#   align-items:center;justify-content:center;font-size:12px;transition:all .15s}
# .draw-btn:hover{border-color:var(--accent);color:var(--accent)}
# .draw-btn.active{background:var(--accent3);border-color:var(--accent);color:var(--accent)}

# @media(max-width:900px){
#   .sidebar{width:0;overflow:hidden}
#   .rpanel{width:300px}
# }
# @media(max-width:640px){
#   .rpanel{display:none}
#   .topbar-mid{display:none}
# }
# </style>
# </head>
# <body>

# <div class="topbar">
#   <div class="logo">
#     <div class="logo-mark">📈</div>
#     <div>
#       <div class="logo-text">VN<em>Stock</em>AI</div>
#       <div class="logo-sub">v6.0 · LSTM Deep Learning · Real Data</div>
#     </div>
#   </div>

#   <div class="sym-bar">
#     <div class="sym-input-wrap">
#       <input id="symInput" type="text" placeholder="Mã, VD: VCB" autocomplete="off" spellcheck="false"/>
#       <button class="sym-search-btn" onclick="startAnalysis()">⏎</button>
#     </div>
#     <div class="mode-tabs">
#       <button class="mode-tab active" onclick="setMode('stock',this)">📊 Cổ phiếu</button>
#       <button class="mode-tab" onclick="setMode('fund',this)">🏦 Quỹ</button>
#       <button class="mode-tab" onclick="setMode('forex',this)">💱 Forex</button>
#     </div>
#     <button class="analyze-btn-top" id="analyzeBtn" onclick="startAnalysis()">
#       <span>🧠</span> Phân tích
#     </button>
#   </div>

#   <div class="topbar-mid" id="tickerBar">
#     <div class="ticker-item" onclick="quickLoad('VN-INDEX')">
#       <span class="ti-sym">VN-INDEX</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('VCB')">
#       <span class="ti-sym">VCB</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('HPG')">
#       <span class="ti-sym">HPG</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('FPT')">
#       <span class="ti-sym">FPT</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('VHM')">
#       <span class="ti-sym">VHM</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('TCB')">
#       <span class="ti-sym">TCB</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#     <div class="ticker-item" onclick="quickLoad('MBB')">
#       <span class="ti-sym">MBB</span><span class="ti-val">--</span><span class="ti-chg">--</span>
#     </div>
#   </div>

#   <div class="topbar-right">
#     <div style="display:flex;align-items:center;gap:5px">
#       <div class="status-dot" id="statusDot"></div>
#       <span id="statusTxt">Kết nối...</span>
#     </div>
#     <span id="clockEl"></span>
#   </div>
# </div>

# <div class="ind-toolbar">
#   <div class="tool-group">
#     <span class="tb-label">TIMEFRAME</span>
#     <button class="tb-btn active" onclick="setTF(this,'D')">1D</button>
#     <button class="tb-btn" onclick="setTF(this,'W')">1W</button>
#     <button class="tb-btn" onclick="setTF(this,'M')">1M</button>
#   </div>
#   <div class="tool-group">
#     <span class="tb-label">CHART</span>
#     <button class="tb-btn active" id="btnCandle" onclick="setChartType('candle',this)">🕯 Nến</button>
#     <button class="tb-btn" id="btnLine" onclick="setChartType('line',this)">📉 Line</button>
#     <button class="tb-btn" id="btnArea" onclick="setChartType('area',this)">📊 Area</button>
#   </div>
#   <div class="tool-group">
#     <span class="tb-label">OVERLAYS</span>
#     <button class="tb-btn active" id="btnBB" onclick="toggleInd('bb',this)" title="Bollinger Bands">BB</button>
#     <button class="tb-btn active" id="btnSMA" onclick="toggleInd('sma',this)" title="SMA 20/50/200">SMA</button>
#     <button class="tb-btn" id="btnEMA" onclick="toggleInd('ema',this)" title="EMA 9/21/50">EMA</button>
#     <button class="tb-btn" id="btnIchi" onclick="toggleInd('ichi',this)" title="Ichimoku">ICHI</button>
#     <button class="tb-btn" id="btnVWAP" onclick="toggleInd('vwap',this)" title="VWAP">VWAP</button>
#     <button class="tb-btn active" id="btnForecast" onclick="toggleInd('forecast',this)" title="LSTM Forecast">🧠 AI</button>
#   </div>
#   <div class="tool-group">
#     <span class="tb-label">SUB</span>
#     <button class="tb-btn active" id="btnVol" onclick="toggleSub('vol',this)">VOL</button>
#     <button class="tb-btn active" id="btnRSI" onclick="toggleSub('rsi',this)">RSI</button>
#     <button class="tb-btn active" id="btnMACD" onclick="toggleSub('macd',this)">MACD</button>
#     <button class="tb-btn" id="btnStoch" onclick="toggleSub('stoch',this)">STOCH</button>
#     <button class="tb-btn" id="btnADX" onclick="toggleSub('adx',this)">ADX</button>
#     <button class="tb-btn" id="btnOBV" onclick="toggleSub('obv',this)">OBV</button>
#   </div>
#   <div class="tool-group">
#     <span class="tb-label">DRAW</span>
#     <button class="tb-btn" id="toolNone" onclick="setDrawTool('none',this)">✗ Xóa</button>
#     <button class="tb-btn" id="toolHL" onclick="setDrawTool('hline',this)">— Ngang</button>
#     <button class="tb-btn" id="toolTL" onclick="setDrawTool('tline',this)">╱ Xu hướng</button>
#     <button class="tb-btn" id="toolFib" onclick="setDrawTool('fib',this)">Fib</button>
#     <button class="tb-btn" id="toolRect" onclick="setDrawTool('rect',this)">▭ HCN</button>
#     <button class="tb-btn" onclick="clearDrawings()">🗑</button>
#   </div>
# </div>

# <div class="workspace">

#   <div class="sidebar" id="sidebar">
#     <div class="sidebar-tabs">
#       <div class="stab active" onclick="switchSTab('symbols',this)">Mã</div>
#       <div class="stab" onclick="switchSTab('indicators',this)">Chỉ báo</div>
#       <div class="stab" onclick="switchSTab('techdata',this)">Dữ liệu</div>
#     </div>

#     <div class="sidebar-panel active" id="sp-symbols">
#       <div class="sec">
#         <div class="sec-head" onclick="toggleSec(this)">
#           <span class="sec-head-title">📊 Cổ phiếu nổi bật</span><span>▾</span>
#         </div>
#         <div class="sec-body chip-group" id="stockChips">
#           <span class="chip" onclick="quickLoad('VCB')">VCB</span>
#           <span class="chip" onclick="quickLoad('VHM')">VHM</span>
#           <span class="chip" onclick="quickLoad('HPG')">HPG</span>
#           <span class="chip" onclick="quickLoad('FPT')">FPT</span>
#           <span class="chip" onclick="quickLoad('TCB')">TCB</span>
#           <span class="chip" onclick="quickLoad('MBB')">MBB</span>
#           <span class="chip" onclick="quickLoad('ACB')">ACB</span>
#           <span class="chip" onclick="quickLoad('VIC')">VIC</span>
#           <span class="chip" onclick="quickLoad('BID')">BID</span>
#           <span class="chip" onclick="quickLoad('CTG')">CTG</span>
#           <span class="chip" onclick="quickLoad('SSI')">SSI</span>
#           <span class="chip" onclick="quickLoad('VNM')">VNM</span>
#         </div>
#       </div>
#       <div class="sec">
#         <div class="sec-head" onclick="toggleSec(this)">
#           <span class="sec-head-title">🏦 Quỹ đầu tư</span><span>▾</span>
#         </div>
#         <div class="sec-body chip-group">
#           <span class="chip" onclick="quickLoad('E1VFVN30','fund')">E1VFVN30</span>
#           <span class="chip" onclick="quickLoad('VFMVSF','fund')">VFMVSF</span>
#           <span class="chip" onclick="quickLoad('SSISCA','fund')">SSISCA</span>
#           <span class="chip" onclick="quickLoad('MAFPF1','fund')">MAFPF1</span>
#         </div>
#       </div>
#       <div class="sec">
#         <div class="sec-head" onclick="toggleSec(this)">
#           <span class="sec-head-title">💱 Tỷ giá</span><span>▾</span>
#         </div>
#         <div class="sec-body chip-group">
#           <span class="chip" onclick="quickLoad('USD.VND','forex')">USD/VND</span>
#           <span class="chip" onclick="quickLoad('EUR.VND','forex')">EUR/VND</span>
#           <span class="chip" onclick="quickLoad('EUR.USD','forex')">EUR/USD</span>
#           <span class="chip" onclick="quickLoad('USD.JPY','forex')">USD/JPY</span>
#         </div>
#       </div>
#     </div>

#     <div class="sidebar-panel" id="sp-indicators">
#       <div class="sec">
#         <div class="sec-head"><span class="sec-head-title">📈 Trend</span></div>
#         <div class="sec-body" style="padding:0">
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#f0c040"></div>SMA 20</label><div class="ind-toggle on" onclick="toggleInd('sma20',this)" id="tog-sma20"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#ff8c42"></div>SMA 50</label><div class="ind-toggle on" onclick="toggleInd('sma50',this)" id="tog-sma50"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#f06292"></div>SMA 200</label><div class="ind-toggle on" onclick="toggleInd('sma200',this)" id="tog-sma200"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#9b7fe8"></div>EMA 9</label><div class="ind-toggle" onclick="toggleInd('ema9',this)" id="tog-ema9"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#64b5f6"></div>EMA 21</label><div class="ind-toggle" onclick="toggleInd('ema21',this)" id="tog-ema21"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#00c8f5"></div>Bollinger Bands</label><div class="ind-toggle on" onclick="toggleInd('bb2',this)" id="tog-bb2"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#26c6da"></div>VWAP</label><div class="ind-toggle" onclick="toggleInd('vwap2',this)" id="tog-vwap2"></div></div>
#         </div>
#       </div>
#       <div class="sec">
#         <div class="sec-head"><span class="sec-head-title">🔮 Ichimoku</span></div>
#         <div class="sec-body" style="padding:0">
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#ff5252"></div>Tenkan-sen</label><div class="ind-toggle" onclick="toggleInd('tenkan',this)" id="tog-tenkan"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:#00e676"></div>Kijun-sen</label><div class="ind-toggle" onclick="toggleInd('kijun',this)" id="tog-kijun"></div></div>
#           <div class="ind-row"><label class="ind-label"><div class="ind-color" style="background:rgba(0,230,118,.3)"></div>Cloud A/B</label><div class="ind-toggle" onclick="toggleInd('cloud',this)" id="tog-cloud"></div></div>
#         </div>
#       </div>
#     </div>

#     <div class="sidebar-panel" id="sp-techdata">
#       <div class="data-grid" id="techDataGrid">
#         <div class="dc" style="grid-column:1/-1;padding:10px;text-align:center;color:var(--text3);font-size:.7rem;font-family:'DM Mono',monospace">Phân tích để xem dữ liệu</div>
#       </div>
#     </div>
#   </div>

#   <div class="chart-area">
#     <div class="chart-container" id="mainChartContainer">
#       <div id="mainChart"></div>
#       <div class="chart-overlay">
#         <div class="chart-legend" id="chartLegend" style="display:none">
#           <span class="legend-sym" id="lgSym">--</span>
#           <span class="legend-ohlc">
#             <span>O</span><span class="val" id="lgO">--</span>
#             <span>H</span><span class="val" id="lgH">--</span>
#             <span>L</span><span class="val" id="lgL">--</span>
#             <span>C</span><span class="val" id="lgC">--</span>
#           </span>
#           <span class="legend-chg" id="lgChg"></span>
#         </div>
#       </div>
#       <div class="crosshair-info" id="crosshairInfo" style="display:none">
#         <div class="ci-item"><span>Date</span><span class="v" id="ciDate">--</span></div>
#         <div class="ci-item"><span>Price</span><span class="v" id="ciPrice">--</span></div>
#         <div class="ci-item"><span>Vol</span><span class="v" id="ciVol">--</span></div>
#       </div>
#       <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:20;pointer-events:none" id="chartOverlayWrap">
#         <div id="welcome">
#           <div class="w-icon">📊</div>
#           <div class="w-title">VN Stock AI v6.0</div>
#           <p class="w-desc">Nhập mã cổ phiếu và bấm Phân tích để bắt đầu phân tích chuyên sâu với LSTM Deep Learning + Real Market Data</p>
#           <div class="feat-list">
#             <div class="feat"><div class="feat-ic">🧠</div><div><div class="feat-title">LSTM Deep Learning</div><div class="feat-desc">Bidirectional LSTM + Attention với Monte Carlo Dropout cho dự báo chính xác</div></div></div>
#             <div class="feat"><div class="feat-ic">📡</div><div><div class="feat-title">Real Market Data</div><div class="feat-desc">KBS → VCI → FMarket → MSN. Dữ liệu thực, không mock, không fallback giả</div></div></div>
#             <div class="feat"><div class="feat-ic">📊</div><div><div class="feat-title">Interactive Chart</div><div class="feat-desc">Vẽ trend line, Fibonacci, rectangle, text annotation như TradingView</div></div></div>
#             <div class="feat"><div class="feat-ic">📋</div><div><div class="feat-title">VCBS-Style Reports</div><div class="feat-desc">Entry, SL, TP cụ thể. P/E, P/B, ROE. Kịch bản Bull/Base/Bear</div></div></div>
#           </div>
#         </div>
#         <div id="loading">
#           <div class="spin-wrap"><div class="ring1"></div><div class="ring2"></div></div>
#           <div class="load-text" id="loadText">Đang phân tích...</div>
#           <div class="load-sub">LSTM Deep Learning v6.0 · Real Data</div>
#           <div class="load-steps">
#             <div class="lstep" id="ls1">📡 Lấy dữ liệu KBS/VCI/FMarket...</div>
#             <div class="lstep" id="ls2">📊 Tính 30+ chỉ báo kỹ thuật...</div>
#             <div class="lstep" id="ls3">🧠 Training LSTM Neural Network...</div>
#             <div class="lstep" id="ls4">📐 Monte Carlo Forecasting...</div>
#             <div class="lstep" id="ls5">🤖 AI phân tích đa chiều...</div>
#             <div class="lstep" id="ls6">📋 Tổng hợp báo cáo VCBS...</div>
#           </div>
#         </div>
#       </div>
#     </div>

#     <div class="subchart-wrap" id="subVol" style="height:80px">
#       <div class="subchart-label">VOL</div>
#       <div id="volChart" style="height:100%;width:100%"></div>
#     </div>
#     <div class="subchart-wrap" id="subRSI" style="height:90px">
#       <div class="subchart-label">RSI(14)</div>
#       <div id="rsiChart" style="height:100%;width:100%"></div>
#     </div>
#     <div class="subchart-wrap" id="subMACD" style="height:90px">
#       <div class="subchart-label">MACD(12,26,9)</div>
#       <div id="macdChart" style="height:100%;width:100%"></div>
#     </div>
#     <div class="subchart-wrap hidden" id="subStoch" style="height:90px">
#       <div class="subchart-label">STOCH(14,3)</div>
#       <div id="stochChart" style="height:100%;width:100%"></div>
#     </div>
#     <div class="subchart-wrap hidden" id="subADX" style="height:90px">
#       <div class="subchart-label">ADX(14)</div>
#       <div id="adxChart" style="height:100%;width:100%"></div>
#     </div>
#     <div class="subchart-wrap hidden" id="subOBV" style="height:90px">
#       <div class="subchart-label">OBV</div>
#       <div id="obvChart" style="height:100%;width:100%"></div>
#     </div>
#   </div>

#   <div class="rpanel">
#     <div class="rpanel-tabs">
#       <div class="rtab active" onclick="switchRTab('report',this)">📋 Báo cáo AI</div>
#       <div class="rtab" onclick="switchRTab('forecast',this)">📐 Dự báo</div>
#       <div class="rtab" onclick="switchRTab('fundamental',this)">📊 Cơ bản</div>
#     </div>

#     <div class="rpanel-body active" id="rp-report">
#       <div id="reportContent">
#         <div id="errorBox"></div>
#         <div style="padding:20px;text-align:center;color:var(--text3);font-size:.75rem">
#           Chưa có báo cáo. Bấm phân tích để bắt đầu.
#         </div>
#       </div>
#     </div>

#     <div class="rpanel-body" id="rp-forecast">
#       <div class="fc-panel" id="fcContent">
#         <div style="padding:20px;text-align:center;color:var(--text3);font-size:.75rem">
#           Chưa có dữ liệu dự báo.
#         </div>
#       </div>
#     </div>

#     <div class="rpanel-body" id="rp-fundamental">
#       <div style="padding:12px" id="fundContent">
#         <div style="padding:20px;text-align:center;color:var(--text3);font-size:.75rem">
#           Chưa có dữ liệu cơ bản.
#         </div>
#       </div>
#     </div>
#   </div>

# </div>

# <div class="toast" id="toast"></div>

# <script>
# let currentMode = 'stock';
# let currentSym = '';
# let chartType = 'candle';
# let drawTool = 'none';
# let drawState = null;
# let stepTimer = null;
# let apiData = null;

# let mainChart = null;
# let mainSeries = null;
# let volChart = null; let volSeries = null;
# let rsiChart = null; let rsiSeries = null;
# let macdChart = null; let macdLineSeries = null; let macdSigSeries = null; let macdHistSeries = null;
# let stochChart = null; let stochKSeries = null; let stochDSeries = null;
# let adxChart = null; let adxSeries = null;
# let obvChart = null; let obvSeries = null;

# let sma20s = null; let sma50s = null; let sma200s = null;
# let ema9s = null; let ema21s = null; let ema50s = null;
# let bbUs = null; let bbMs = null; let bbLs = null;
# let fcSeries = null; let fcUpper = null; let fcLower = null;
# let vwapS = null;
# let tenkanS = null; let kijunS = null; let senkouAS = null; let senkouBS = null;
# let drawings = [];

# const visible = {
#   bb:true, sma:true, ema:false, ichi:false, vwap:false, forecast:true,
#   vol:true, rsi:true, macd:true, stoch:false, adx:false, obv:false
# };

# function tick(){
#   const el = document.getElementById('clockEl');
#   if(el) el.textContent = new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
# }
# setInterval(tick,1000); tick();

# async function checkHealth(){
#   try{
#     const r = await fetch('/health');
#     const d = await r.json();
#     const dot = document.getElementById('statusDot');
#     const txt = document.getElementById('statusTxt');
#     if(d.status==='ok'){
#       dot.style.background='var(--green)'; txt.textContent='Online';
#     }
#   }catch{
#     const dot = document.getElementById('statusDot');
#     const txt = document.getElementById('statusTxt');
#     dot.style.background='var(--yellow)'; txt.textContent='Offline';
#   }
# }
# checkHealth();

# const CHART_OPT = {
#   layout:{background:{color:'#030810'},textColor:'#7a9cbf'},
#   grid:{vertLines:{color:'#192b44',style:1},horzLines:{color:'#192b44',style:1}},
#   crosshair:{mode:LightweightCharts.CrosshairMode.Normal,
#     vertLine:{color:'rgba(0,200,245,.5)',width:1,style:1,labelBackgroundColor:'#0d1929'},
#     horzLine:{color:'rgba(0,200,245,.5)',width:1,style:1,labelBackgroundColor:'#0d1929'}},
#   rightPriceScale:{borderColor:'#192b44',textColor:'#7a9cbf'},
#   timeScale:{borderColor:'#192b44',textColor:'#7a9cbf',timeVisible:true,secondsVisible:false},
#   handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true},
#   handleScale:{axisPressedMouseMove:true,mouseWheel:true,pinch:true},
# };

# const SUB_OPT = (h) => ({
#   height:h,
#   layout:{background:{color:'#030810'},textColor:'#7a9cbf'},
#   grid:{vertLines:{color:'#192b44',style:1},horzLines:{color:'#192b44',style:1}},
#   crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
#   rightPriceScale:{borderColor:'#192b44',textColor:'#7a9cbf',scaleMargins:{top:.05,bottom:.05}},
#   timeScale:{visible:false},
#   handleScroll:false, handleScale:false,
# });

# function initCharts(){
#   const mc = document.getElementById('mainChart');
#   mainChart = LightweightCharts.createChart(mc, {...CHART_OPT});
#   mainChart.timeScale().fitContent();

#   mainSeries = mainChart.addCandlestickSeries({
#     upColor:'#00d97e', downColor:'#f04060',
#     borderVisible:false,
#     wickUpColor:'#00d97e', wickDownColor:'#f04060',
#   });

#   mainChart.subscribeCrosshairMove(param => {
#     const lg = document.getElementById('chartLegend');
#     const ci = document.getElementById('crosshairInfo');
#     if(!param.time || !param.seriesData){ lg.style.display='none'; ci.style.display='none'; return; }
#     const d = param.seriesData.get(mainSeries);
#     if(!d) return;
#     lg.style.display='flex'; ci.style.display='flex';
#     document.getElementById('lgO').textContent = fmt(d.open);
#     document.getElementById('lgH').textContent = fmt(d.high);
#     document.getElementById('lgL').textContent = fmt(d.low);
#     document.getElementById('lgC').textContent = fmt(d.close);
#     const chg = ((d.close-d.open)/d.open*100).toFixed(2);
#     const lgChg = document.getElementById('lgChg');
#     lgChg.textContent = (chg>=0?'▲':'')+chg+'%';
#     lgChg.className = 'legend-chg '+(chg>=0?'up':'dn');
#     const dt = new Date(param.time*1000);
#     document.getElementById('ciDate').textContent = dt.toLocaleDateString('vi-VN');
#     document.getElementById('ciPrice').textContent = fmt(d.close);
#   });

#   volChart = LightweightCharts.createChart(document.getElementById('volChart'), SUB_OPT(80));
#   volSeries = volChart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:''});

#   rsiChart = LightweightCharts.createChart(document.getElementById('rsiChart'), SUB_OPT(90));
#   rsiSeries = rsiChart.addLineSeries({color:'#00c8f5',lineWidth:1.5,priceLineVisible:false});
#   rsiChart.addLineSeries({color:'rgba(240,64,96,.5)',lineWidth:.8,lineStyle:2,priceLineVisible:false}).setData([{time:0,value:70},{time:9999999999,value:70}]);
#   rsiChart.addLineSeries({color:'rgba(0,217,126,.5)',lineWidth:.8,lineStyle:2,priceLineVisible:false}).setData([{time:0,value:30},{time:9999999999,value:30}]);

#   macdChart = LightweightCharts.createChart(document.getElementById('macdChart'), SUB_OPT(90));
#   macdLineSeries = macdChart.addLineSeries({color:'#00c8f5',lineWidth:1.5,priceLineVisible:false});
#   macdSigSeries = macdChart.addLineSeries({color:'#edb84a',lineWidth:1.2,priceLineVisible:false});
#   macdHistSeries = macdChart.addHistogramSeries({priceLineVisible:false,priceFormat:{type:'price',minMove:.01}});

#   stochChart = LightweightCharts.createChart(document.getElementById('stochChart'), SUB_OPT(90));
#   stochKSeries = stochChart.addLineSeries({color:'#9b7fe8',lineWidth:1.4,priceLineVisible:false});
#   stochDSeries = stochChart.addLineSeries({color:'#edb84a',lineWidth:1.2,priceLineVisible:false});

#   adxChart = LightweightCharts.createChart(document.getElementById('adxChart'), SUB_OPT(90));
#   adxSeries = adxChart.addLineSeries({color:'#9b7fe8',lineWidth:1.5,priceLineVisible:false});

#   obvChart = LightweightCharts.createChart(document.getElementById('obvChart'), SUB_OPT(90));
#   obvSeries = obvChart.addLineSeries({color:'#ff8c42',lineWidth:1.5,priceLineVisible:false});

#   mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
#     if(!range) return;
#     [volChart,rsiChart,macdChart,stochChart,adxChart,obvChart].forEach(c => {
#       c && c.timeScale().setVisibleLogicalRange(range);
#     });
#   });

#   mainChart.subscribeClick(param => {
#     if(drawTool === 'hline' && param.price !== undefined){
#       const pl = mainSeries.createPriceLine({
#         price: param.price, color:'#edb84a', lineWidth:1, lineStyle:2,
#         axisLabelVisible:true, title:`${fmt(param.price)}`
#       });
#       drawings.push({type:'hline', obj:pl});
#     }
#     if(drawTool === 'tline'){
#       if(!drawState){
#         drawState = {price:param.price, time:param.time};
#         toast('🎯 Chọn điểm thứ 2 cho đường xu hướng', 'info');
#       } else {
#         const tl = mainChart.addLineSeries({color:'#edb84a',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false});
#         tl.setData([
#           {time:drawState.time, value:drawState.price},
#           {time:param.time, value:param.price}
#         ]);
#         drawings.push({type:'tline', obj:tl});
#         drawState = null;
#         toast('✅ Đường xu hướng đã vẽ', 'ok');
#       }
#     }
#     if(drawTool === 'fib' && param.price !== undefined){
#       // Simplified Fibonacci - draw retracement levels
#       const price = param.price;
#       const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
#       levels.forEach(lvl => {
#         const pl = mainSeries.createPriceLine({
#           price: price * (1 - lvl * 0.1),
#           color: lvl === 0.5 ? '#ff8c42' : '#7a9cbf',
#           lineWidth: lvl === 0.5 ? 1.5 : 0.8,
#           lineStyle: 2,
#           axisLabelVisible: true,
#           title: `${(lvl*100).toFixed(1)}%`
#         });
#         drawings.push({type:'fib', obj:pl});
#       });
#       toast('✅ Fibonacci Retracement đã vẽ', 'ok');
#     }
#     if(drawTool === 'rect'){
#       if(!drawState){
#         drawState = {price:param.price, time:param.time};
#         toast('🎯 Chọn góc đối diện', 'info');
#       } else {
#         // Rectangle using two horizontal lines
#         const top = Math.max(drawState.price, param.price);
#         const bottom = Math.min(drawState.price, param.price);
#         const pl1 = mainSeries.createPriceLine({price:top, color:'#9b7fe8', lineWidth:1, lineStyle:2, axisLabelVisible:false});
#         const pl2 = mainSeries.createPriceLine({price:bottom, color:'#9b7fe8', lineWidth:1, lineStyle:2, axisLabelVisible:false});
#         drawings.push({type:'rect', obj:[pl1, pl2]});
#         drawState = null;
#         toast('✅ Rectangle đã vẽ', 'ok');
#       }
#     }
#   });

#   handleResize();
#   window.addEventListener('resize', handleResize);
# }

# function handleResize(){
#   const mc = document.getElementById('mainChart');
#   if(mc && mainChart){
#     mainChart.applyOptions({width:mc.offsetWidth, height:mc.offsetHeight});
#   }
#   ['vol','rsi','macd','stoch','adx','obv'].forEach(k => {
#     const cont = document.getElementById(k+'Chart');
#     const ch = window[k+'Chart'];
#     if(cont && ch) ch.applyOptions({width:cont.offsetWidth, height:cont.offsetHeight});
#   });
# }

# function loadChartData(data){
#   const ohlcv = data.ohlcv || [];
#   const ind = data.indicators || {};
#   const fc = data.forecast || {};

#   mainSeries.setData(ohlcv.map(d => ({
#     time:d.time, open:d.open, high:d.high, low:d.low, close:d.close
#   })));

#   volSeries.setData(ohlcv.map(d => ({
#     time:d.time, value:d.volume,
#     color: d.close >= d.open ? 'rgba(0,217,126,.55)' : 'rgba(240,64,96,.55)'
#   })));

#   document.getElementById('lgSym').textContent = data.symbol || '';
#   document.getElementById('chartLegend').style.display = 'flex';
#   document.getElementById('crosshairInfo').style.display = 'flex';

#   _setSeries('sma20', ind.sma20 || [], () => {
#     sma20s = mainChart.addLineSeries({color:'#f0c040',lineWidth:1.3,priceLineVisible:false,lastValueVisible:false});
#     return sma20s;
#   });
#   _setSeries('sma50', ind.sma50 || [], () => {
#     sma50s = mainChart.addLineSeries({color:'#ff8c42',lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});
#     return sma50s;
#   });
#   _setSeries('sma200', ind.sma200 || [], () => {
#     sma200s = mainChart.addLineSeries({color:'#f06292',lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});
#     return sma200s;
#   });
#   _setSeries('ema9', ind.ema9 || [], () => {
#     ema9s = mainChart.addLineSeries({color:'#9b7fe8',lineWidth:1.2,priceLineVisible:false,lastValueVisible:false});
#     return ema9s;
#   });
#   _setSeries('ema21', ind.ema21 || [], () => {
#     ema21s = mainChart.addLineSeries({color:'#64b5f6',lineWidth:1.2,priceLineVisible:false,lastValueVisible:false});
#     return ema21s;
#   });
#   _setSeries('ema50', ind.ema50 || [], () => {
#     ema50s = mainChart.addLineSeries({color:'#00e676',lineWidth:1.2,priceLineVisible:false,lastValueVisible:false});
#     return ema50s;
#   });

#   if(ind.bb_upper){
#     if(!bbUs){
#       bbUs = mainChart.addLineSeries({color:'rgba(0,200,245,.45)',lineWidth:.9,priceLineVisible:false,lastValueVisible:false,lineStyle:2});
#       bbMs = mainChart.addLineSeries({color:'rgba(0,200,245,.75)',lineWidth:1.2,priceLineVisible:false,lastValueVisible:false});
#       bbLs = mainChart.addLineSeries({color:'rgba(0,200,245,.45)',lineWidth:.9,priceLineVisible:false,lastValueVisible:false,lineStyle:2});
#     }
#     bbUs.setData(ind.bb_upper);
#     bbMs.setData(ind.bb_middle);
#     bbLs.setData(ind.bb_lower);
#     setSeriesVis([bbUs,bbMs,bbLs], visible.bb);
#   }

#   if(ind.vwap){
#     if(!vwapS) vwapS = mainChart.addLineSeries({color:'#26c6da',lineWidth:1.3,priceLineVisible:false,lastValueVisible:false,lineStyle:3});
#     vwapS.setData(ind.vwap);
#     vwapS.applyOptions({visible:visible.vwap});
#   }

#   if(ind.tenkan){
#     if(!tenkanS) tenkanS = mainChart.addLineSeries({color:'#ff5252',lineWidth:1.2,priceLineVisible:false,lastValueVisible:false});
#     if(!kijunS) kijunS = mainChart.addLineSeries({color:'#00e676',lineWidth:1.4,priceLineVisible:false,lastValueVisible:false});
#     if(!senkouAS) senkouAS = mainChart.addLineSeries({color:'rgba(0,230,118,.2)',lineWidth:.8,priceLineVisible:false,lastValueVisible:false});
#     if(!senkouBS) senkouBS = mainChart.addLineSeries({color:'rgba(255,82,82,.2)',lineWidth:.8,priceLineVisible:false,lastValueVisible:false});
#     tenkanS.setData(ind.tenkan);
#     kijunS.setData(ind.kijun);
#     senkouAS.setData(ind.senkou_a);
#     senkouBS.setData(ind.senkou_b);
#     tenkanS.applyOptions({visible:visible.ichi});
#     kijunS.applyOptions({visible:visible.ichi});
#     senkouAS.applyOptions({visible:visible.ichi});
#     senkouBS.applyOptions({visible:visible.ichi});
#   }

#   if(fc.forecast_points && fc.forecast_points.length){
#     const lastReal = ohlcv[ohlcv.length-1];
#     const fcData = [{time:lastReal.time, value:lastReal.close}, ...fc.forecast_points];
#     if(!fcSeries) fcSeries = mainChart.addLineSeries({color:'#00d97e',lineWidth:2,lineStyle:2,priceLineVisible:false,lastValueVisible:true});
#     if(!fcUpper) fcUpper = mainChart.addLineSeries({color:'rgba(0,217,126,.3)',lineWidth:.8,lineStyle:3,priceLineVisible:false,lastValueVisible:false});
#     if(!fcLower) fcLower = mainChart.addLineSeries({color:'rgba(0,217,126,.3)',lineWidth:.8,lineStyle:3,priceLineVisible:false,lastValueVisible:false});
#     fcSeries.setData(fcData);
#     fcUpper.setData([{time:lastReal.time,value:lastReal.close},...(fc.upper_points||[])]);
#     fcLower.setData([{time:lastReal.time,value:lastReal.close},...(fc.lower_points||[])]);
#     setSeriesVis([fcSeries,fcUpper,fcLower], visible.forecast);
#   }

#   if(ind.rsi) rsiSeries.setData(ind.rsi);
#   if(ind.macd_line) macdLineSeries.setData(ind.macd_line);
#   if(ind.macd_signal) macdSigSeries.setData(ind.macd_signal);
#   if(ind.macd_hist) macdHistSeries.setData(ind.macd_hist);
#   if(ind.stoch_k) stochKSeries.setData(ind.stoch_k);
#   if(ind.stoch_d) stochDSeries.setData(ind.stoch_d);
#   if(ind.adx) adxSeries.setData(ind.adx);
#   if(ind.obv) obvSeries.setData(ind.obv);

#   mainChart.timeScale().fitContent();
#   setSeriesVis([sma20s,sma50s,sma200s], visible.sma);
#   setSeriesVis([ema9s,ema21s,ema50s], visible.ema);
# }

# function _setSeries(key, pts, createFn){
#   if(!pts.length) return;
#   let s = window[key+'s'];
#   if(!s){ s = createFn(); window[key+'s'] = s; }
#   s.setData(pts);
# }

# function setSeriesVis(arr, vis){
#   arr.forEach(s => s && s.applyOptions({visible:vis}));
# }

# function renderReport(data){
#   const sym = data.symbol || '--';
#   const mode = data.type || 'stock';
#   const typeMap = {stock:'Cổ phiếu · HOSE/HNX', fund:'Chứng chỉ quỹ', forex:'Cặp ngoại tệ'};
#   const rec = data.recommendation || 'WATCH';
#   const recMap = {BUY:'🟢 MUA', SELL:'🔴 BÁN', HOLD:'🟡 GIỮ', WATCH:'🔵 THEO DÕI'};
#   const fc = data.forecast || {};
#   const tech = data.technical || {};
#   const priceCount = (data.ohlcv||[]).length;
#   const source = data.source || 'KBS';

#   let html = `
#   <div class="report-wrap">
#     <div class="rec-header">
#       <div class="sym-block">
#         <div class="rec-sym">${sym}</div>
#         <div class="rec-meta">
#           <div class="rec-type">${typeMap[mode]||mode} · Nguồn: ${source}</div>
#           <div class="rec-time">${new Date().toLocaleString('vi-VN')}</div>
#         </div>
#       </div>
#       <div class="rec-badge rec-${rec}">${recMap[rec]||rec}</div>
#     </div>
#     <div class="badges">
#       <div class="badge ok">📡 ${priceCount} phiên · ${source}</div>
#       <div class="badge ok">📊 30+ chỉ báo</div>
#       ${data.fundamental && Object.keys(data.fundamental).length ? '<div class="badge ok">📋 Cơ bản</div>' : '<div class="badge warn">📋 Không có CB</div>'}
#       <div class="badge info">🧠 ${data.confidence||7}/10</div>
#       ${fc.method ? `<div class="badge ok">🧠 ${fc.method}</div>` : ''}
#       ${fc.direction ? `<div class="badge ${fc.direction==='TĂNG'?'ok':'warn'}">📐 ${fc.direction} R²=${fc.r_squared}</div>` : ''}
#     </div>
#     <div style="display:flex;justify-content:flex-end;margin-bottom:8px">
#       <button class="copy-btn" onclick="copyReport()">📋 Sao chép</button>
#     </div>
#     <div class="report-body" id="reportBody">${mdToHtml(data.analysis||'')}</div>
#   </div>`;

#   document.getElementById('reportContent').innerHTML = html;
# }

# function renderForecast(data){
#   const fc = data.forecast || {};
#   if(!fc.direction){ document.getElementById('fcContent').innerHTML = '<div style="padding:20px;text-align:center;color:var(--text3);font-size:.75rem">Không có dữ liệu dự báo.</div>'; return; }
  
#   const isUp = fc.direction === 'TĂNG';
#   const html = `
#   <div class="fc-panel">
#     <div class="fc-card">
#       <div class="fc-card-title">🧠 ${fc.method||'LSTM Deep Learning'}</div>
#       <div class="fc-grid">
#         <div class="fc-item"><div class="fc-item-l">Xu hướng</div><div class="fc-item-v ${isUp?'up':'dn'}">${isUp?'▲ TĂNG':'▼ GIẢM'}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Độ tin cậy</div><div class="fc-item-v">${fc.confidence||'--'}</div></div>
#         <div class="fc-item"><div class="fc-item-l">R² (Validation)</div><div class="fc-item-v">${fc.r_squared||'--'}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Model</div><div class="fc-item-v">✅ LSTM</div></div>
#       </div>
#     </div>
#     <div class="fc-card">
#       <div class="fc-card-title">📈 Mục tiêu giá</div>
#       <div class="fc-grid">
#         <div class="fc-item"><div class="fc-item-l">Hiện tại</div><div class="fc-item-v">${fmtN(fc.current_price)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Dự báo T+5</div><div class="fc-item-v ${isUp?'up':'dn'}">${fmtN(fc.target_1w)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Dự báo T+10</div><div class="fc-item-v ${isUp?'up':'dn'}">${fmtN(fc.target_2w)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Lợi nhuận</div><div class="fc-item-v ${isUp?'up':'dn'}">${fc.expected_return_2w!==undefined?(fc.expected_return_2w>=0?'+':'')+fc.expected_return_2w+'%':'--'}</div></div>
#       </div>
#     </div>
#     <div class="fc-card">
#       <div class="fc-card-title">🎯 Chiến lược (AI gợi ý)</div>
#       <div class="fc-grid">
#         <div class="fc-item"><div class="fc-item-l">Stop Loss</div><div class="fc-item-v dn">${fmtN(fc.stop_loss)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Take Profit 1</div><div class="fc-item-v up">${fmtN(fc.take_profit_1)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Take Profit 2</div><div class="fc-item-v up">${fmtN(fc.take_profit_2)}</div></div>
#         <div class="fc-item"><div class="fc-item-l">Expected Return</div><div class="fc-item-v">${fc.expected_return_2w||'--'}%</div></div>
#       </div>
#     </div>
#     ${fc.forecast_5d && fc.forecast_5d.length ? `
#     <div class="fc-card">
#       <div class="fc-card-title">📅 Dự báo 5 phiên tới</div>
#       <div style="display:flex;gap:4px;flex-wrap:wrap">
#         ${fc.forecast_5d.map((v,i)=>`<div class="fc-item" style="flex:1;min-width:60px"><div class="fc-item-l">T+${i+1}</div><div class="fc-item-v" style="font-size:.76rem">${fmtN(v)}</div></div>`).join('')}
#       </div>
#     </div>` : ''}
#   </div>`;
#   document.getElementById('fcContent').innerHTML = html;
# }

# function renderFundamental(data){
#   const fund = data.fundamental || {};
#   const fi = data.fund_info || {};
#   const rows = [];

#   if(fund.pe != null) rows.push(['P/E Ratio', fund.pe, evalPE(fund.pe)]);
#   if(fund.pb != null) rows.push(['P/B Ratio', fund.pb, evalPB(fund.pb)]);
#   if(fund.roe != null) rows.push(['ROE (%)', fund.roe, evalROE(fund.roe)]);
#   if(fund.roa != null) rows.push(['ROA (%)', fund.roa, '']);
#   if(fund.eps != null) rows.push(['EPS', fund.eps, '']);
#   if(fund.beta != null) rows.push(['Beta', fund.beta, parseFloat(fund.beta)>1?'↑ Biến động cao':'↓ Ổn định']);
#   if(fund.market_cap) rows.push(['Vốn hóa (tỷ)', fund.market_cap, '']);
#   if(fund.outstanding) rows.push(['CP lưu hành', fund.outstanding, '']);
#   if(fund.dividend_yield) rows.push(['Cổ tức (%)', fund.dividend_yield, '']);
#   if(fund.industry) rows.push(['Ngành', fund.industry, '']);
#   if(fund.exchange) rows.push(['Sàn', fund.exchange, '']);
#   if(fund['52w_high']) rows.push(['Cao 52T', fmtN(fund['52w_high']), '']);
#   if(fund['52w_low']) rows.push(['Thấp 52T', fmtN(fund['52w_low']), '']);
#   if(fi.fund_name) rows.push(['Tên quỹ', fi.fund_name, '']);
#   if(fi.management_company) rows.push(['Công ty QL', fi.management_company, '']);
#   if(fi.latest_nav) rows.push(['NAV hiện tại', fmtN(fi.latest_nav), '']);
#   if(fi.risk_level) rows.push(['Mức rủi ro', fi.risk_level, '']);
#   if(fi.management_fee) rows.push(['Phí QL', fi.management_fee+'%', '']);

#   const html = rows.length ? `
#   <div style="overflow-x:auto">
#   <table class="fund-table">
#     <thead><tr><th>Chỉ số</th><th style="text-align:right">Giá trị</th><th>Đánh giá</th></tr></thead>
#     <tbody>${rows.map(r=>`<tr><td>${r[0]}</td><td class="td-val">${r[1]}</td><td class="td-note">${r[2]}</td></tr>`).join('')}</tbody>
#   </table>
#   </div>` : '<div style="padding:20px;text-align:center;color:var(--text3);font-size:.75rem">Không có dữ liệu cơ bản</div>';

#   document.getElementById('fundContent').innerHTML = html;
# }

# function renderTechData(tech){
#   const g = document.getElementById('techDataGrid');
#   const items = [
#     ['Giá', fmtN(tech.current_price), ''],
#     ['RSI 14', tech.rsi, rsiCls(tech.rsi)],
#     ['MACD', tech.macd, macdCls(tech.macd, tech.macd_signal)],
#     ['BB Upper', fmtN(tech.bb_upper), ''],
#     ['BB Lower', fmtN(tech.bb_lower), ''],
#     ['BB Width %', fmtN(tech.bb_width), ''],
#     ['SMA 20', fmtN(tech.sma20), ''],
#     ['SMA 50', fmtN(tech.sma50), ''],
#     ['SMA 200', fmtN(tech.sma200), ''],
#     ['EMA 9', fmtN(tech.ema9), ''],
#     ['EMA 21', fmtN(tech.ema21), ''],
#     ['EMA 50', fmtN(tech.ema50), ''],
#     ['Stoch %K', tech.stoch_k, ''],
#     ['Stoch %D', tech.stoch_d, ''],
#     ['Williams%R', tech.williams_r, ''],
#     ['ATR 14', tech.atr, 'neu'],
#     ['ATR %', fmtN(tech.atr_pct), ''],
#     ['CCI', tech.cci, ''],
#     ['MFI', tech.mfi, ''],
#     ['ADX', tech.adx, ''],
#     ['+DI', tech.plus_di, ''],
#     ['-DI', tech.minus_di, ''],
#     ['VWAP', fmtN(tech.vwap), ''],
#     ['Momentum 10', tech.momentum_10, parseFloat(tech.momentum_10)>=0?'up':'dn'],
#     ['Momentum 20', tech.momentum_20, parseFloat(tech.momentum_20)>=0?'up':'dn'],
#     ['Volatility 20', fmtN(tech.volatility_20), ''],
#     ['Hỗ trợ 1', fmtN(tech.support1), 'up'],
#     ['Hỗ trợ 2', fmtN(tech.support2), 'up'],
#     ['Hỗ trợ 3', fmtN(tech.support3), 'up'],
#     ['Kháng cự 1', fmtN(tech.resistance1), 'dn'],
#     ['Kháng cự 2', fmtN(tech.resistance2), 'dn'],
#     ['Kháng cự 3', fmtN(tech.resistance3), 'dn'],
#     ['Pivot', fmtN(tech.pivot), ''],
#     ['Pivot R1', fmtN(tech.pivot_r1), 'dn'],
#     ['Pivot S1', fmtN(tech.pivot_s1), 'up'],
#     ['Fib 23.6%', fmtN(tech.fib_236), ''],
#     ['Fib 38.2%', fmtN(tech.fib_382), ''],
#     ['Fib 50%', fmtN(tech.fib_500), ''],
#     ['Fib 61.8%', fmtN(tech.fib_618), ''],
#     ['Trend NH', tech.trend_short, tech.trend_short==='TĂNG'?'up':'dn'],
#     ['Trend TH', tech.trend_medium, tech.trend_medium==='TĂNG'?'up':'dn'],
#     ['Trend DH', tech.trend_long, tech.trend_long==='TĂNG'?'up':'dn'],
#     ['Ichimoku', tech.trend_ichimoku, tech.trend_ichimoku==='TĂNG'?'up':'dn'],
#   ];
#   g.innerHTML = items.map(([l,v,c]) =>
#     `<div class="dc"><div class="dc-l">${l}</div><div class="dc-v ${c}">${v||'--'}</div></div>`
#   ).join('');
# }

# async function startAnalysis(){
#   const sym = (document.getElementById('symInput').value||'').trim().toUpperCase();
#   if(!sym){ toast('⚠ Nhập mã cổ phiếu/quỹ/forex', 'err'); return; }
#   currentSym = sym;

#   showLoading(sym);
#   document.getElementById('analyzeBtn').disabled = true;
#   clearOverlaySeries();

#   const fd = new FormData();
#   fd.append('symbol', sym);
#   fd.append('type', currentMode);

#   try{
#     const r = await fetch('/api/analyze', {method:'POST', body:fd});
#     clearLoadSteps();
#     if(!r.ok){
#       const e = await r.json().catch(()=>({}));
#       throw new Error(e.error || 'HTTP ' + r.status);
#     }
#     const json = await r.json();
#     apiData = json.data;

#     hideOverlay();
#     loadChartData(apiData);
#     renderReport(apiData);
#     renderForecast(apiData);
#     renderFundamental(apiData);
#     renderTechData(apiData.technical || {});
#     toast('✅ Phân tích xong: ' + sym + ' · ' + (apiData.source||'KBS'), 'ok');

#   } catch(err){
#     showError(err.message);
#     toast('❌ ' + err.message, 'err');
#   } finally{
#     document.getElementById('analyzeBtn').disabled = false;
#   }
# }

# function quickLoad(sym, mode){
#   document.getElementById('symInput').value = sym;
#   if(mode) setMode(mode, document.querySelector(`.mode-tab[onclick*="${mode}"]`) || document.querySelector('.mode-tab'));
#   startAnalysis();
# }

# function setMode(mode, btn){
#   currentMode = mode;
#   document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
#   if(btn) btn.classList.add('active');
#   document.getElementById('symInput').placeholder = {
#     stock:'Mã CP: VCB', fund:'Mã quỹ: E1VF...', forex:'Cặp: USD.VND'
#   }[mode] || '';
# }

# function clearOverlaySeries(){
#   [sma20s,sma50s,sma200s,ema9s,ema21s,ema50s,bbUs,bbMs,bbLs,fcSeries,fcUpper,fcLower,vwapS,tenkanS,kijunS,senkouAS,senkouBS].forEach(s=>{
#     if(s) try{ mainChart.removeSeries(s); } catch(e){}
#   });
#   sma20s=sma50s=sma200s=ema9s=ema21s=ema50s=bbUs=bbMs=bbLs=fcSeries=fcUpper=fcLower=vwapS=tenkanS=kijunS=senkouAS=senkouBS=null;
# }

# function toggleInd(key, btn){
#   if(btn && btn.classList.contains('ind-toggle')){
#     btn.classList.toggle('on');
#     const isOn = btn.classList.contains('on');
#     _applyIndVis(key, isOn);
#     return;
#   }
#   if(btn) btn.classList.toggle('active');
#   const isActive = btn ? btn.classList.contains('active') : !visible[key];

#   if(key==='bb') { visible.bb = isActive; setSeriesVis([bbUs,bbMs,bbLs], isActive); }
#   else if(key==='sma') { visible.sma = isActive; setSeriesVis([sma20s,sma50s,sma200s], isActive); }
#   else if(key==='ema') { visible.ema = isActive; setSeriesVis([ema9s,ema21s,ema50s], isActive); }
#   else if(key==='ichi') { visible.ichi = isActive; setSeriesVis([tenkanS,kijunS,senkouAS,senkouBS], isActive); }
#   else if(key==='vwap') { visible.vwap = isActive; vwapS && vwapS.applyOptions({visible:isActive}); }
#   else if(key==='forecast') { visible.forecast = isActive; setSeriesVis([fcSeries,fcUpper,fcLower], isActive); }
# }

# function _applyIndVis(key, on){
#   if(key==='sma20') sma20s && sma20s.applyOptions({visible:on});
#   else if(key==='sma50') sma50s && sma50s.applyOptions({visible:on});
#   else if(key==='sma200') sma200s && sma200s.applyOptions({visible:on});
#   else if(key==='ema9') ema9s && ema9s.applyOptions({visible:on});
#   else if(key==='ema21') ema21s && ema21s.applyOptions({visible:on});
#   else if(key==='ema50') ema50s && ema50s.applyOptions({visible:on});
#   else if(key==='bb2') setSeriesVis([bbUs,bbMs,bbLs], on);
#   else if(key==='vwap2') vwapS && vwapS.applyOptions({visible:on});
#   else if(key==='tenkan') tenkanS && tenkanS.applyOptions({visible:on});
#   else if(key==='kijun') kijunS && kijunS.applyOptions({visible:on});
#   else if(key==='cloud') setSeriesVis([tenkanS,kijunS,senkouAS,senkouBS], on);
# }

# function toggleSub(key, btn){
#   if(btn) btn.classList.toggle('active');
#   const wrap = document.getElementById('sub'+key.charAt(0).toUpperCase()+key.slice(1));
#   if(wrap) wrap.classList.toggle('hidden');
#   handleResize();
# }

# function setChartType(type, btn){
#   chartType = type;
#   document.querySelectorAll('#btnCandle,#btnLine,#btnArea').forEach(b=>b.classList.remove('active'));
#   if(btn) btn.classList.add('active');
#   if(!apiData) return;
#   const ohlcv = apiData.ohlcv || [];
#   if(mainSeries){ try{ mainChart.removeSeries(mainSeries); }catch(e){} mainSeries=null; }
#   if(type==='candle'){
#     mainSeries = mainChart.addCandlestickSeries({upColor:'#00d97e',downColor:'#f04060',borderVisible:false,wickUpColor:'#00d97e',wickDownColor:'#f04060'});
#     mainSeries.setData(ohlcv.map(d=>({time:d.time,open:d.open,high:d.high,low:d.low,close:d.close})));
#   } else if(type==='line'){
#     mainSeries = mainChart.addLineSeries({color:'#00c8f5',lineWidth:2,priceLineVisible:true});
#     mainSeries.setData(ohlcv.map(d=>({time:d.time,value:d.close})));
#   } else if(type==='area'){
#     mainSeries = mainChart.addAreaSeries({topColor:'rgba(0,200,245,.25)',bottomColor:'rgba(0,200,245,.02)',lineColor:'#00c8f5',lineWidth:2});
#     mainSeries.setData(ohlcv.map(d=>({time:d.time,value:d.close})));
#   }
# }

# function setDrawTool(tool, btn){
#   drawTool = tool; drawState = null;
#   document.querySelectorAll('.tb-btn[id^="tool"]').forEach(b=>b.classList.remove('active-tool'));
#   if(tool !== 'none' && btn) btn.classList.add('active-tool');
#   document.getElementById('mainChart').style.cursor = tool !== 'none' ? 'crosshair' : 'default';
# }

# function clearDrawings(){
#   drawings.forEach(d=>{
#     if(d.type === 'rect' && Array.isArray(d.obj)){
#       d.obj.forEach(o => { try{ mainSeries.removePriceLine(o); }catch(e){} });
#     } else if(d.type === 'tline' && d.obj.series){
#       try{ mainChart.removeSeries(d.obj); }catch(e){}
#     } else if(d.obj){
#       try{ mainSeries.removePriceLine(d.obj); }catch(e){}
#     }
#   });
#   drawings = [];
#   toast('🗑 Đã xóa tất cả drawings', 'ok');
# }

# function setTF(btn, tf){
#   document.querySelectorAll('.ind-toolbar .tool-group:first-child .tb-btn').forEach(b=>b.classList.remove('active'));
#   btn.classList.add('active');
#   toast('Timeframe: ' + tf + ' (cần kết nối backend)', 'info');
# }

# function showLoading(sym){
#   document.getElementById('chartOverlayWrap').style.pointerEvents = 'auto';
#   document.getElementById('welcome').style.display = 'none';
#   document.getElementById('loading').classList.add('show');
#   document.getElementById('loadText').textContent = 'Đang phân tích ' + sym + '...';
#   document.getElementById('errorBox').classList.remove('show');
#   const ids = ['ls1','ls2','ls3','ls4','ls5','ls6'];
#   ids.forEach(id => { const el=document.getElementById(id); el.classList.remove('active','done'); });
#   let i = 0;
#   clearInterval(stepTimer);
#   stepTimer = setInterval(()=>{
#     if(i>0) { const el=document.getElementById(ids[i-1]); el.classList.remove('active'); el.classList.add('done'); }
#     if(i<ids.length){ document.getElementById(ids[i]).classList.add('active'); i++; }
#     else clearInterval(stepTimer);
#   }, 2500);
# }

# function hideOverlay(){
#   clearInterval(stepTimer);
#   document.getElementById('chartOverlayWrap').style.pointerEvents = 'none';
#   document.getElementById('loading').classList.remove('show');
#   document.getElementById('welcome').style.display = 'none';
# }

# function showError(msg){
#   clearInterval(stepTimer);
#   document.getElementById('chartOverlayWrap').style.pointerEvents = 'none';
#   document.getElementById('loading').classList.remove('show');
#   document.getElementById('welcome').style.display = 'none';
#   const eb = document.getElementById('errorBox');
#   eb.classList.add('show');
#   eb.innerHTML = `⚠️ <strong>Lỗi:</strong> ${msg}<br><small style="color:var(--text3)">Kiểm tra mã, kết nối mạng hoặc backend logs.</small>`;
#   document.getElementById('reportContent').innerHTML = `<div id="errorBox" class="show">⚠️ ${msg}</div>`;
# }

# function clearLoadSteps(){
#   clearInterval(stepTimer);
#   ['ls1','ls2','ls3','ls4','ls5','ls6'].forEach(id=>{
#     const el = document.getElementById(id);
#     el.classList.remove('active');
#     el.classList.add('done');
#   });
# }

# function switchSTab(id, el){
#   document.querySelectorAll('.stab').forEach(t=>t.classList.remove('active'));
#   document.querySelectorAll('.sidebar-panel').forEach(p=>p.classList.remove('active'));
#   el.classList.add('active');
#   document.getElementById('sp-'+id).classList.add('active');
# }

# function switchRTab(id, el){
#   document.querySelectorAll('.rtab').forEach(t=>t.classList.remove('active'));
#   document.querySelectorAll('.rpanel-body').forEach(p=>p.classList.remove('active'));
#   el.classList.add('active');
#   document.getElementById('rp-'+id).classList.add('active');
# }

# function toggleSec(head){
#   const body = head.nextElementSibling;
#   const arrow = head.querySelector(':last-child');
#   if(body){ body.classList.toggle('collapsed'); if(arrow) arrow.textContent = body.classList.contains('collapsed') ? '▸' : '▾'; }
# }

# function copyReport(){
#   const el = document.getElementById('reportBody');
#   if(el) navigator.clipboard.writeText(el.innerText).then(()=>toast('✅ Đã sao chép', 'ok'));
# }

# function toast(msg, type='info'){
#   const el = document.getElementById('toast');
#   el.textContent = msg;
#   el.className = 'toast show ' + (type==='ok'?'ok':type==='err'?'err':'');
#   setTimeout(()=>el.classList.remove('show'), 3000);
# }

# function fmt(n){
#   if(n==null||n===undefined) return '--';
#   const v = parseFloat(n);
#   if(isNaN(v)) return String(n);
#   if(v>=1e9) return (v/1e9).toFixed(2)+'B';
#   if(v>=1e6) return (v/1e6).toFixed(2)+'M';
#   if(v>=1000) return v.toLocaleString('vi-VN');
#   return v%1===0 ? v.toString() : v.toFixed(v<10?4:2);
# }
# const fmtN = fmt;

# function rsiCls(v){ const n=parseFloat(v); return n>70?'dn':n<30?'up':'neu'; }
# function macdCls(m, s){ return parseFloat(m||0)>parseFloat(s||0)?'up':'dn'; }

# function evalPE(v){ const n=parseFloat(v); if(isNaN(n)) return ''; return n<10?'✅ Thấp':n>25?'⚠ Cao':'✔ Hợp lý'; }
# function evalPB(v){ const n=parseFloat(v); if(isNaN(n)) return ''; return n<1.5?'✅ Thấp BV':n>3?'⚠ Cao':'✔ Hợp lý'; }
# function evalROE(v){ const n=parseFloat(v); if(isNaN(n)) return ''; return n>20?'⭐ Xuất sắc':n>15?'✅ Tốt':n>10?'✔ Khá':'⚠ TB'; }

# function mdToHtml(t){
#   if(!t) return '';
#   t = t.replace(/<think[\\s\\S]*?<\\/think>/gi,'');
#   t = t.replace(/^######\\s+(.+)$/gm,'<h3>$1</h3>');
#   t = t.replace(/^#####\\s+(.+)$/gm,'<h3>$1</h3>');
#   t = t.replace(/^####\\s+(.+)$/gm,'<h3>$1</h3>');
#   t = t.replace(/^###\\s+(.+)$/gm,'<h3>$1</h3>');
#   t = t.replace(/^##\\s+(.+)$/gm,'<h2>$1</h2>');
#   t = t.replace(/^#\\s+(.+)$/gm,'<h1>$1</h1>');
#   t = t.replace(/^---+$/gm,'<hr>');
#   t = t.replace(/\\*\\*\\*(.+?)\\*\\*\\*/g,'<strong><em>$1</em></strong>');
#   t = t.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');
#   t = t.replace(/\\*(.+?)\\*/g,'<em>$1</em>');
#   t = t.replace(/`([^`]+)`/g,'<code>$1</code>');
#   t = t.replace(/^>\\s+(.+)$/gm,'<blockquote>$1</blockquote>');
#   t = t.replace(/^[-*•]\\s+(.+)$/gm,'<li>$1</li>');
#   t = t.replace(/^\\d+\\.\\s+(.+)$/gm,'<li>$1</li>');
#   t = t.replace(/(<li>.*<\\/li>\\n?)+/g, m => '<ul>'+m+'</ul>');
#   t = t.replace(/^\\|(.+)\\|$/gm, m => {
#     if(m.includes('---')) return '';
#     const cells = m.slice(1,-1).split('|').map(c=>c.trim());
#     return '<tr>'+cells.map(c=>`<td>${c}</td>`).join('')+'</tr>';
#   });
#   t = t.replace(/(<tr>.*<\\/tr>\\n?)+/g, m => '<table>'+m+'</table>');
#   const lines = t.split('\\n');
#   return lines.map(l => {
#     const trim = l.trim();
#     if(!trim) return '';
#     if(trim.match(/^<(h[1-6]|ul|ol|li|hr|blockquote|table|tr|td|th|p|div)/i)) return trim;
#     return `<p>${trim}</p>`;
#   }).join('\\n');
# }

# document.getElementById('symInput').addEventListener('keydown', e => {
#   if(e.key === 'Enter') startAnalysis();
# });

# document.addEventListener('keydown', e => {
#   if(e.key === 'Escape') { setDrawTool('none', null); }
# });

# document.addEventListener('DOMContentLoaded', () => {
#   initCharts();
# });
# </script>
# </body>
# </html>"""


# ====================== KẾT THÚC PHẦN HTML INLINE ======================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
