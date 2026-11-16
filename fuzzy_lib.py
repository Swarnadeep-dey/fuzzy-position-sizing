"""CA-3: indicator functions (Step 2). Plain NumPy/pandas, each function maps to one equation in Section 8 of the Phase 1 document."""
import numpy as np
import pandas as pd

STOCKS = ["HDFCBANK", "HINDUNILVR", "TCS", "RELIANCE", "TATASTEEL"]

def load(name):
    d = pd.read_csv(f"{name}.csv", parse_dates=["Date"], index_col="Date")
    d = d[d["Volume"] > 0]            # drop the 2 placeholder rows (no trading, OHLC = previous close)
    return d[["Open", "High", "Low", "Close", "Volume"]]

def rsi_wilder(close, n=14):
    """RSI with Wilder smoothing. First AvgU/AvgD = simple mean of first n moves, then
    Avg_t = ((n-1)*Avg_{t-1} + move_t) / n.  RSI = 100 - 100/(1+RS); if AvgD = 0 then RSI = 100."""
    p = np.asarray(close, float)
    u = np.maximum(np.diff(p), 0.0)
    d = np.maximum(-np.diff(p), 0.0)
    out = np.full(len(p), np.nan)
    if len(u) < n:
        return pd.Series(out, index=getattr(close, "index", None))
    au, ad = u[:n].mean(), d[:n].mean()
    def rsi(a, b):
        return 100.0 if b == 0 else 100.0 - 100.0 / (1.0 + a / b)
    out[n] = rsi(au, ad)
    for t in range(n, len(u)):
        au = ((n - 1) * au + u[t]) / n
        ad = ((n - 1) * ad + d[t]) / n
        out[t + 1] = rsi(au, ad)
    return pd.Series(out, index=getattr(close, "index", None))

def ema(x, span):
    """EMA_t = a*x_t + (1-a)*EMA_{t-1}, a = 2/(span+1), started from the first value."""
    x = np.asarray(x, float)
    a = 2.0 / (span + 1)
    out = np.empty(len(x)); out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out

def macd_hist_pct(close, fast=12, slow=26, sig=9):
    """x2 = 100 * (MACD - Signal) / Price."""
    p = np.asarray(close, float)
    macd = ema(p, fast) - ema(p, slow)
    signal = ema(macd, sig)
    return pd.Series(100.0 * (macd - signal) / p, index=getattr(close, "index", None))

def volatility20(close, n=20):
    """x3 = sample std (ddof=1) of the last n daily returns, in %."""
    r = pd.Series(np.asarray(close, float)).pct_change()
    return pd.Series(100.0 * r.rolling(n).std(ddof=1).values, index=getattr(close, "index", None))

def rel_volume(volume, n=20):
    """x4 = V_t / mean(V_{t-n} .. V_{t-1})  (previous n days, today excluded)."""
    v = pd.Series(np.asarray(volume, float))
    return pd.Series((v / v.shift(1).rolling(n).mean()).values, index=getattr(volume, "index", None))

WARMUP = 60      # trading days dropped so the starting values fade (after 60 days about 1% of the EMA26 start and 3% of the RSI start remain)
CALIB = 252      # calibration window = first 252 usable trading days (about 12 months)

def build(name):
    d = load(name)
    f = pd.DataFrame(index=d.index)
    f["close"] = d["Close"]
    f["ret"] = d["Close"].pct_change()
    f["x1_rsi"] = rsi_wilder(d["Close"])
    f["x2_macd"] = macd_hist_pct(d["Close"])
    f["x3_vol"] = volatility20(d["Close"])
    f["x4_relvol"] = rel_volume(d["Volume"])
    f = f.iloc[WARMUP:].copy()
    f["period"] = ["calib"] * CALIB + ["test"] * (len(f) - CALIB)
    return f
