import numpy as np, pandas as pd
from fuzzy_model import *
pd.set_option("display.width",220)
F=load_all(); A=pooled_anchors(F); M=FuzzyPositionModel(A); C=CrispModel(A); W=WeightedAvgModel(A)
rows=[]; ALL={}
for n,f in F.items():
    t=f[f.period=="test"].copy(); x=[t[c].values for c in COLS]
    r=M.run(*x,detail=True)
    t["S"]=r["S"]; t["Q"]=r["Q"]; t["y_mamdani"]=r["y"]; t["y_wavg"]=W.run(*x); t["y_crisp"]=C.run(*x)
    ALL[n]=t
    y=t.y_mamdani
    rows.append([n,len(t),y.mean(),y.median(),y.std(),y.min(),y.max(),100*(y<35).mean(),100*((y>=35)&(y<=65)).mean(),100*(y>65).mean(),t.S.mean(),t.Q.mean(),t.y_crisp.mean(),t.y_wavg.mean()])
D=pd.DataFrame(rows,columns=["stock","days","mean","median","std","min","max","%small(<35)","%medium","%large(>65)","meanS","meanQ","crisp_mean","wavg_mean"])
print(D.round(1).to_string(index=False))
pd.to_pickle(ALL,"test_results.pkl")
print("\nSpread between stocks in mean y: max-min = %.1f points"%(D["mean"].max()-D["mean"].min()))
print("Correlation of daily y between stocks (Mamdani):"); print(pd.DataFrame({n:t.y_mamdani for n,t in ALL.items()}).corr().round(2))
print("\nMean test-period inputs per stock:"); print(pd.DataFrame({n:t[COLS].mean() for n,t in ALL.items()}).T.round(3))
