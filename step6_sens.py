import numpy as np, pandas as pd
from fuzzy_model import *
pd.set_option("display.width",220)
F=load_all(); A=pooled_anchors(F)
T=pd.concat([f[f.period=="test"] for f in F.values()]); X={c:T[c].values for c in COLS}
iqr={c:A[c][2]-A[c][0] for c in COLS}
sign={"x1_rsi":1,"x2_macd":1,"x3_vol":-1,"x4_relvol":1}   # +1 = higher is favourable
print("cols",COLS,"IQR",{k:round(v,3) for k,v in iqr.items()})
models={"Mamdani":FuzzyPositionModel(A),"Weighted-avg":WeightedAvgModel(A),"Crisp":CrispModel(A)}
base={k:m.run(*[X[c] for c in COLS]) for k,m in models.items()}
def run(m,shift):  # shift: dict col-> amount added to input
    return m.run(*[X[c]+shift.get(c,0) for c in COLS])
P=[.10,.20,.30]
rows=[]
for name,m in models.items():
    for c in COLS:
        for p in P:
            for d,lab in [(+1,"improve"),(-1,"worsen")]:
                y=run(m,{c:d*sign[c]*p*iqr[c]}); dy=y-base[name]
                rows.append([name,c,p,lab,dy.mean(),np.abs(dy).mean(),np.abs(dy).max(),100*(np.abs(dy)>1e-9).mean()])
R=pd.DataFrame(rows,columns=["model","input","pct","dir","mean_dy","mean_abs","max_abs","%days_changed"])
R.to_pickle("sens_inputs.pkl")
for name in models:
    print("\n==",name,"— mean |change in y| (points) when ONE input moves by p x IQR (average of improve & worsen)")
    t=R[R.model==name].groupby(["input","pct"]).mean_abs.mean().unstack().loc[COLS]; print(t.round(2).to_string())
print("\n== Mamdani: signed mean change (improve / worsen) at 30%"); print(R[(R.model=="Mamdani")&(R.pct==.3)].pivot(index="input",columns="dir",values="mean_dy").loc[COLS].round(2))
print("\n== Mamdani: %days where y moved at 10/20/30"); print(R[(R.model=="Mamdani")].groupby(["input","pct"])["%days_changed"].mean().unstack().loc[COLS].round(0))
print("\n== All four inputs improve together / worsen together (Mamdani)")
for p in P:
    up=run(models["Mamdani"],{c:sign[c]*p*iqr[c] for c in COLS})-base["Mamdani"]; dn=run(models["Mamdani"],{c:-sign[c]*p*iqr[c] for c in COLS})-base["Mamdani"]
    print(f"{int(p*100)}%: improve mean {up.mean():+.2f}, worsen mean {dn.mean():+.2f}")
# anchor (membership boundary) shifts: all three anchors of ONE input shifted by p x IQR, and all inputs together
print("\n== Membership boundaries moved by p x IQR (Mamdani): mean |change| of y, and change of stock mean")
rows=[]
for c in COLS+["ALL"]:
    for p in P:
        for d in (+1,-1):
            A2={k:tuple(v+(d*p*iqr[k] if (c==k or c=="ALL") else 0) for v in A[k]) for k in A}
            y=FuzzyPositionModel(A2).run(*[X[k] for k in COLS]); dy=y-base["Mamdani"]
            rows.append([c,p,d,np.abs(dy).mean(),dy.mean(),np.abs(dy).max()])
B=pd.DataFrame(rows,columns=["input","pct","dir","mean_abs","mean_dy","max_abs"])
print(B.groupby(["input","pct"]).mean_abs.mean().unstack().loc[COLS+["ALL"]].round(2))
B.to_pickle("sens_anchors.pkl")
# stock ranking stability under boundary shift ALL +-30%
names=list(F.keys()); print("\nStock ranking (mean y) under boundary shift of all inputs:")
rk={}
for lab,d in [("-30%",-.3),("base",0),("+30%",.3)]:
    A2={k:tuple(v+d*iqr[k] for v in A[k]) for k in A}; mm=FuzzyPositionModel(A2)
    rk[lab]={n:mm.run(*[F[n][F[n].period=="test"][c].values for c in COLS]).mean() for n in names}
print(pd.DataFrame(rk).round(1).sort_values("base",ascending=False))
