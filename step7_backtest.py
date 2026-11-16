import numpy as np, pandas as pd
pd.set_option("display.width",220)
ALL=pd.read_pickle("test_results.pkl")
def metrics(r):
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return pd.Series({"total_ret%":100*(eq.iloc[-1]-1),"max_dd%":100*dd,"vol%(ann)":100*r.std(ddof=1)*np.sqrt(252)})
strats={"Buy&hold":None,"Crisp":"y_crisp","Mamdani":"y_mamdani","Wavg":"y_wavg"}
out={}; rets={}
for n,t in ALL.items():
    r=t.ret.shift(-1).dropna()  # return of day t+1
    rr={}
    for s,col in strats.items():
        w=pd.Series(1.0,index=r.index) if col is None else t[col].loc[r.index]/100   # weight set at close t, earns r_{t+1}
        rr[s]=w*r
    rets[n]=pd.DataFrame(rr)
    for s in strats:
        m=metrics(rets[n][s]); m["avg_exposure%"]=100 if strats[s] is None else t[strats[s]].loc[r.index].mean(); 
        m["turnover/day"]=0 if strats[s] is None else t[strats[s]].diff().abs().mean()
        out[(n,s)]=m
R=pd.DataFrame(out).T; print(R.round(1).to_string())
print("\n-- Equal-weight portfolio of the 5 stocks (daily-rebalanced)")
P=sum(rets.values())/5; print(pd.DataFrame({s:metrics(P[s]) for s in strats}).T.round(1))
print("\n-- Average across the 5 stocks")
print(R.groupby(level=1).mean().loc[list(strats)].round(1))
# in how many stocks did fuzzy beat crisp on drawdown / vol
for m in ["max_dd%","vol%(ann)"]:
    print(m,"Mamdani better (less negative / lower) than Buy&hold in", sum(abs(R.loc[(n,'Mamdani'),m])<abs(R.loc[(n,'Buy&hold'),m]) for n in ALL),"of 5; than Crisp in",sum(abs(R.loc[(n,'Mamdani'),m])<abs(R.loc[(n,'Crisp'),m]) for n in ALL),"of 5")
R.to_pickle("backtest.pkl"); pd.to_pickle(rets,"bt_rets.pkl")
print("\nperiod:",min(t.index[0] for t in ALL.values()).date(),"to",max(t.index[-1] for t in ALL.values()).date())
