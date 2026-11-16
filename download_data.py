"""Download daily data for the five CA-3 stocks from Yahoo Finance.
Run once:  pip install yfinance pandas   then   python download_data.py
Creates one CSV per stock in the same folder. The five CSV files are already included in this package.
"""
import datetime as dt
import pandas as pd
import yfinance as yf

STOCKS = {
    "HDFCBANK": "HDFCBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "TATASTEEL": "TATASTEEL.NS",
}
START, END = "2023-09-01", "2026-10-21"   # a little extra at the start for indicator warm-up

print("Downloaded on:", dt.date.today())   # use this as the 'access date' in your report
for name, ticker in STOCKS.items():
    df = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True, actions=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
    df.to_csv(f"{name}.csv", index_label="Date")
    print(f"\n{name}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    ev = df[df["Stock Splits"] != 0]
    if len(ev):
        print("  split/bonus events (Yahoo ratio):")
        for d, r in ev["Stock Splits"].items():
            w = df.loc[d - pd.Timedelta(days=6): d + pd.Timedelta(days=6), "Volume"]
            print(f"    {d.date()}  ratio {r}  volume nearby: {[int(v) for v in w.values]}")
    else:
        print("  no split/bonus events")
