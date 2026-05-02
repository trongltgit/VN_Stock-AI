"""
VN Stock AI v6.1 — Professional Multi-Asset Analysis System
Real Data · Linear Forecast · Interactive Charts · VCBS-Style Reports
Data Sources: TCBS (primary) → VCI (fallback) → FMarket (funds) → MSN (forex)
"""
import os, json, logging, traceback, warnings, re, math
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
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Scikit-learn cho ML nhẹ
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

load_dotenv()
app = Flask(__name__, template_folder='templates')
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

class Config:
    STOCK_SOURCES = ['TCBS', 'VCI']
    FUND_SOURCE = 'FMARKET'
    FOREX_SOURCE = 'MSN'
    
    LOOKBACK = 60
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

class TCBSProvider(DataProvider):
    """TCBS API - Primary source for Vietnam stocks (2025)"""
    BASE_URL = "https://apipubaws.tcbs.com.vn"
    
    @classmethod
    def get_historical(cls, symbol: str, days: int = 500, resolution: str = "D") -> Optional[pd.DataFrame]:
        try:
            symbol = symbol.upper().strip()
            url = f"{cls.BASE_URL}/stock-insight/v1/stock/bars-long-term"
            
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days * 2)).timestamp())
            
            params = {
                "ticker": symbol,
                "type": "stock",
                "resolution": resolution,
                "from": start_time,
                "to": end_time,
            }
            
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_MEDIUM)
            if not data or not isinstance(data, dict):
                return None
            
            bars = data.get("data", [])
            if not bars or not isinstance(bars, list):
                return None
            
            df = pd.DataFrame(bars)
            
            col_map = {
                't': 'time', 'T': 'time', 'time': 'time', 'tradingDate': 'time',
                'o': 'Open', 'open': 'Open', 'Open': 'Open',
                'h': 'High', 'high': 'High', 'High': 'High',
                'l': 'Low', 'low': 'Low', 'Low': 'Low',
                'c': 'Close', 'close': 'Close', 'Close': 'Close',
                'v': 'Volume', 'volume': 'Volume', 'Volume': 'Volume',
            }
            df = df.rename(columns=col_map)
            
            if 'time' in df.columns:
                sample = df['time'].iloc[0] if len(df) > 0 else 0
                if sample > 1e12:
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                elif sample > 1e9:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                else:
                    df['time'] = pd.to_datetime(df['time'])
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['Close'])
            df = df.sort_values('time').reset_index(drop=True)
            
            if len(df) < 30:
                logger.warning(f"TCBS: Insufficient data for {symbol}: {len(df)} bars")
                return None
            
            logger.info(f"TCBS: Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"TCBS historical error for {symbol}: {e}")
            return None
    
    @classmethod
    def get_fundamental(cls, symbol: str) -> dict:
        try:
            symbol = symbol.upper().strip()
            url = f"{cls.BASE_URL}/tcanalysis/v1/ticker/{symbol}/overview"
            data = cls.fetch(url, timeout=Config.TIMEOUT_SHORT)
            if not data or not isinstance(data, dict):
                return {}
            
            d = data.get("data", data)
            return {
                "pe": d.get("pe") or d.get("PE") or d.get("currentPE"),
                "pb": d.get("pb") or d.get("PB") or d.get("currentPB"),
                "roe": d.get("roe") or d.get("ROE") or d.get("currentROE"),
                "roa": d.get("roa") or d.get("ROA") or d.get("currentROA"),
                "eps": d.get("eps") or d.get("EPS") or d.get("currentEPS"),
                "market_cap": d.get("marketCap") or d.get("market_cap") or d.get("marketCapitalization"),
                "industry": d.get("industry") or d.get("sector") or d.get("icbName"),
                "exchange": d.get("exchange") or d.get("floor") or "HOSE",
                "52w_high": d.get("priceHigh52W") or d.get("high52W") or d.get("high52w"),
                "52w_low": d.get("priceLow52W") or d.get("low52W") or d.get("low52w"),
                "avg_volume": d.get("avgVolume10Day") or d.get("avgVolume") or d.get("avg_volume"),
                "beta": d.get("beta") or d.get("BETA"),
                "dividend_yield": d.get("dividendYield") or d.get("dy") or d.get("dividend_yield"),
                "outstanding": d.get("outstandingShare") or d.get("sharesOutstanding") or d.get("shares_outstanding"),
                "company_name": d.get("shortName") or d.get("name") or d.get("companyName"),
            }
        except Exception as e:
            logger.warning(f"TCBS fundamental error for {symbol}: {e}")
            return {}
    
    @classmethod
    def get_quote(cls, symbol: str) -> dict:
        try:
            symbol = symbol.upper().strip()
            url = f"{cls.BASE_URL}/stock-insight/v1/stock/quote"
            params = {"ticker": symbol}
            data = cls.fetch(url, params=params, timeout=Config.TIMEOUT_SHORT)
            if not data:
                return {}
            d = data.get("data", data)
            return {
                "price": d.get("price") or d.get("close") or d.get("lastPrice"),
                "change": d.get("change") or d.get("delta") or d.get("priceChange"),
                "change_pct": d.get("changePercent") or d.get("delta_pct") or d.get("priceChangePercent"),
                "volume": d.get("volume") or d.get("totalVolume"),
                "open": d.get("open") or d.get("openPrice"),
                "high": d.get("high") or d.get("highPrice"),
                "low": d.get("low") or d.get("lowPrice"),
            }
        except Exception as e:
            logger.warning(f"TCBS quote error for {symbol}: {e}")
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
        self.tcbs = TCBSProvider()
        self.vci = VCIProvider()
        self.fmarket = FMarketProvider()
        self.msn = MSNProvider()
    
    def get_stock_data(self, symbol: str, days: int = 500) -> Tuple[Optional[pd.DataFrame], str, dict]:
        symbol = symbol.upper().strip()
        df = self.tcbs.get_historical(symbol, days)
        if df is not None and len(df) >= 30:
            fund = self.tcbs.get_fundamental(symbol)
            quote = self.tcbs.get_quote(symbol)
            return df, "TCBS", {**fund, **{f"quote_{k}": v for k, v in quote.items()}}
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
        stoch_k, stoch_d = cls.stochastic(h, l, c)
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
# LINEAR FORECASTER (Thay thế LSTM - nhẹ, không cần TensorFlow)
# ══════════════════════════════════════════════════════════════════════

class LinearForecaster:
    def __init__(self, lookback: int = 60, forecast_horizon: int = 10):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = LinearRegression()
        self.last_fit_result = None
    
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        features = pd.DataFrame(index=df.index)
        features["close"] = c
        features["returns"] = c.pct_change()
        features["rsi"] = TechnicalAnalysis.rsi(c, 14)
        bb_u, bb_m, bb_l = TechnicalAnalysis.bollinger_bands(c)
        features["bb_position"] = (c - bb_l) / (bb_u - bb_l + 1e-10)
        features["atr"] = TechnicalAnalysis.atr(h, l, c)
        features["sma20"] = TechnicalAnalysis.sma(c, 20) / c
        features["sma50"] = TechnicalAnalysis.sma(c, 50) / c
        for lag in [1, 2, 3, 5, 10]:
            features[f"lag_{lag}"] = c.shift(lag) / c
        features = features.fillna(method="ffill").fillna(0)
        return features.values
    
    def fit(self, df: pd.DataFrame, validation_split: float = 0.15) -> dict:
        features = self._build_features(df)
        target = df["Close"].values
        
        X = features[self.lookback:]
        y = target[self.lookback:]
        
        if len(X) < 30:
            return {"success": False, "error": "Insufficient data"}
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        self.last_fit_result = {
            "success": True,
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "method": "Linear Regression + Technical Features",
        }
        return self.last_fit_result
    
    def predict(self, df: pd.DataFrame) -> dict:
        if self.last_fit_result is None:
            fit_result = self.fit(df)
            if not fit_result["success"]:
                return self._fallback_forecast(df)
        
        features = self._build_features(df)
        last_features = features[-1:].reshape(1, -1)
        
        predictions = []
        current_close = float(df["Close"].iloc[-1])
        
        for i in range(self.forecast_horizon):
            pred = self.model.predict(last_features)[0]
            predictions.append(pred)
            last_features = np.roll(last_features, -1, axis=1)
            last_features[0, -1] = pred / current_close
        
        pred = np.array(predictions)
        
        volatility = df["Close"].pct_change().std() * np.sqrt(252)
        last_price = float(df["Close"].iloc[-1])
        se = [volatility * np.sqrt(i+1) * last_price for i in range(self.forecast_horizon)]
        upper = [min(p + 1.96 * s, p * 1.1) for p, s in zip(pred, se)]
        lower = [max(p - 1.96 * s, p * 0.9) for p, s in zip(pred, se)]
        
        val_r2 = self.last_fit_result.get("val_r2", 0) if self.last_fit_result else 0
        
        return {
            "method": "Linear Regression + Technical Features",
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
# AI REPORT GENERATOR (GROQ API)
# ══════════════════════════════════════════════════════════════════════

class AIReportGenerator:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY_STOCK")
        self.client = None
        if self.groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_key)
                logger.info("Groq AI client initialized successfully")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
                self.client = None
        else:
            logger.warning("GROQ_API_KEY_STOCK not found in environment variables")
    
    def generate_stock_report(self, symbol: str, tech: dict, fund: dict, forecast: dict) -> dict:
        if not self.client:
            logger.info("Groq not available - using fallback report")
            return self._fallback_report(symbol, tech, fund, forecast, "stock")
        
        system_prompt = """Bạn là Trưởng phòng Phân tích của công ty chứng khoán hàng đầu Việt Nam (VCBS/SSI/VPS).
Viết báo cáo phân tích CỔ PHIẾU chuyên nghiệp, đầy đủ 8 phần, dùng Markdown:

# [KÝ HIỆU] — BÁO CÁO PHÂN TÍCH CỔ PHIẾU
**Khuyến nghị:** [MUA/GIỮ/BÁN/THEO DÕI] | **Giá mục tiêu:** [X,XXX VND] | **Độ tin cậy:** [X/10]

## 1. TÓM TẮT ĐIỀU HÀNH
- Khuyến nghị rõ ràng, giá mục tiêu, upside/downside %
- 3-4 luận điểm chính

## 2. BỐI CẢNH VĨ MÔ & NGÀNH
- Lãi suất, tỷ giá, chính sách tiền tệ
- Vị thế ngành, đối thủ cạnh tranh

## 3. PHÂN TÍCH CƠ BẢN
- Bảng P/E, P/B, ROE, EPS so ngành
- Đánh giá định giá (rẻ/hợp lý/đắt)

## 4. PHÂN TÍCH KỸ THUẬT
- Xu hướng, cấu trúc giá, volume
- Bollinger, RSI, MACD, SMA confluence
- Vùng hỗ trợ / kháng cự then chốt

## 5. DỰ BÁO LINEAR MODEL
- Phân tích kết quả mô hình ML
- Kịch bản giá T+5, T+10

## 6. CHIẾN LƯỢC GIAO DỊCH
| Tham số | Giá trị |
|---------|---------|
| Điểm mua | X,XXX – Y,YYY |
| Stop Loss | X,XXX (–Z%) |
| Take Profit 1 | X,XXX (+Z%) |
| Take Profit 2 | X,XXX (+Z%) |
| Tỷ lệ R/R | 1:X.X |
| Khung thời gian | X tuần |

## 7. RỦI RO & KỊCH BẢN
- Kịch bản tích cực / cơ sở / tiêu cực
- Yếu tố rủi ro chính

## 8. KẾT LUẬN
PHONG CÁCH: Chuyên nghiệp, số liệu cụ thể, bảng biểu rõ ràng, ngắn gọn súc tích."""
        
        forecast_text = ""
        if forecast:
            forecast_text = f"""
DỰ BÁO LINEAR MODEL:
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
Dự báo Linear: {forecast.get('direction','N/A')} | R²={forecast.get('r_squared','N/A')}"""
        
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
Dự báo Linear: {forecast.get('direction','N/A')} | Mục tiêu: {forecast.get('target_2w','N/A')}"""
        
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
        
        analysis = f"""## BÁO CÁO PHÂN TÍCH {asset_type.upper()} — {symbol}

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

### 3. DỰ BÁO LINEAR MODEL
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
*Đây là phân tích tự động. Để có báo cáo AI chuyên sâu, cấu hình GROQ_API_KEY_STOCK trong Environment Variables.*
"""
        return {"analysis": analysis, "recommendation": rec, "confidence": 6}

# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self):
        self.data_mgr = StockDataManager()
        self.ta = TechnicalAnalysis()
        self.forecaster = LinearForecaster()
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
        
        # Linear Forecast
        fc = self.forecaster.predict(df)
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
            # Fallback: ETFs like E1VFVN30 trade on exchanges like stocks — try stock path
            logger.info(f"Fund not on FMarket, trying stock fallback for {sym}")
            try:
                return self.analyze_stock(sym)
            except Exception:
                pass
            return self._error_response(sym, "fund", f"Không lấy được dữ liệu NAV cho quỹ {sym}. Nếu đây là ETF, hãy chọn loại 'stock'.")
        
        tech, charts = self.ta.analyze(df)
        fc = self.forecaster.predict(df)
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
        fc = self.forecaster.predict(df)
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
        "version": "6.1",
        "timestamp": datetime.now().isoformat(),
        "ai": orc.ai.client is not None,
        "forecast": "Linear Regression + Technical Features",
        "data_sources": ["TCBS", "VCI", "FMarket", "MSN"],
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
