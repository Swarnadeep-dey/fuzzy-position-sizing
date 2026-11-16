"""Orientation-aware verification of the model (run after any change)."""
import numpy as np, pandas as pd, io, contextlib, warnings
warnings.filterwarnings("ignore")
from fuzzy_model import *
import fuzzy_model as fm
import fuzzy_lib as L
ok=lambda b:"PASS" if b else "FAIL"
F=load_all(); A=pooled_anchors(F); M=FuzzyPositionModel(A); C=CrispModel(A)
FAVLOW={"x1_rsi":fm.RSI_FAV_LOW,"x2_macd":False,"x3_vol":True,"x4_relvol":False}
print("RSI_FAV_LOW =",fm.RSI_FAV_LOW)
# V1 partition of unity on all data
mx=max(np.abs(fuzzify(np.concatenate([pd.concat([x[c] for x in F.values()]).values,[-1e6,1e6]]),*A[c]).sum(0)-1).max() for c in COLS)
print(f"V1 memberships sum to 1 (max err {mx:.1e}):",ok(mx<1e-12))
# V2 extremes: build inputs that are far on the favourable / unfavourable side of each set
def far(k,fav):
    q=A[k]; span=q[2]-q[0]; high=q[2]+span; low=q[0]-span
    return low if (FAVLOW[k]==fav) else high
fav=M.run(*[[far(k,True)] for k in COLS]); unf=M.run(*[[far(k,False)] for k in COLS]); neu=M.run(*[[A[k][1]] for k in COLS])
print(f"V2 all favourable {fav[0]:.2f} (expect 81.43), all unfavourable {unf[0]:.2f} (expect 18.57), all neutral {neu[0]:.2f} (expect 50):",ok(abs(fav[0]-81.43)<.01 and abs(unf[0]-18.57)<.01 and abs(neu[0]-50)<1e-6))
# V3 monotone with others at median, each input
allok=True
for k in COLS:
    q=A[k]; g=np.linspace(q[0]-(q[2]-q[0]),q[2]+(q[2]-q[0]),200); base={c:np.full(200,A[c][1]) for c in COLS}; base[k]=g
    y=M.run(*[base[c] for c in COLS]); d=np.diff(y); allok&=bool((d<=1e-9).all() if FAVLOW[k] else (d>=-1e-9).all())
print("V3 monotone in each input (others at median):",ok(allok))
# V4 centroid of stage 3 vs fine-grid integration on 3 days
t=F["TCS"].iloc[[10,300,600]]; r=M.run(*[t[c].values for c in COLS],detail=True); fine=np.linspace(0,100,400001); good=True
for i in range(3):
    w=r["firing"][2][i]; agg=np.zeros_like(fine)
    for a in range(3):
        for b in range(3): agg=np.maximum(agg,np.minimum(w[a,b],trapmf(fine,*OUT[band(a,b)])))
    good&=abs(trapz(agg*fine,fine)/trapz(agg,fine)-r["y"][i])<1e-3
print("V4 centroid equals fine-grid integral:",ok(good))
# V5 independent library
import skfuzzy as fuzz
from skfuzzy import control as ctrl
def sets_for(q,favlow,lo,hi):
    q1,q2,q3=q; lowset=(lo,lo,q1,q2); mid=(q1,q2,q2,q3); high=(q2,q3,hi,hi); return [high,mid,lowset] if favlow else [lowset,mid,high]
def stage_sim(uA,sA,uB,sB):
    a=ctrl.Antecedent(uA,"a"); b=ctrl.Antecedent(uB,"b"); out=ctrl.Consequent(GRID,"o")
    for i in range(3): a[f"s{i}"]=fuzz.trapmf(uA,list(sA[i])); b[f"s{i}"]=fuzz.trapmf(uB,list(sB[i]))
    for i,o in enumerate(OUT): out[f"l{i}"]=fuzz.trapmf(GRID,list(o))
    out.defuzzify_method="centroid"
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([ctrl.Rule(a[f"s{i}"]&b[f"s{j}"],out[f"l{band(i,j)}"]) for i in range(3) for j in range(3)]))
worst=0; n_days=0
for name in ["TATASTEEL","HINDUNILVR"]:
    f=F[name]; test=f[f.period=="test"].iloc[::20]
    U={c:(min(f[c].min(),A[c][0])-(A[c][2]-A[c][0]),max(f[c].max(),A[c][2])+(A[c][2]-A[c][0])) for c in COLS}
    uni={c:np.linspace(*U[c],4001) for c in COLS}
    S1=stage_sim(uni["x1_rsi"],sets_for(A["x1_rsi"],FAVLOW["x1_rsi"],*U["x1_rsi"]),uni["x2_macd"],sets_for(A["x2_macd"],False,*U["x2_macd"]))
    S2=stage_sim(uni["x3_vol"],sets_for(A["x3_vol"],True,*U["x3_vol"]),uni["x4_relvol"],sets_for(A["x4_relvol"],False,*U["x4_relvol"]))
    S3=stage_sim(GRID,[tuple(o) for o in OUT],GRID,[tuple(o) for o in OUT])
    mine=M.run(*[test[c].values for c in COLS]); sk=[]
    for r in test.itertuples():
        S1.input["a"]=r.x1_rsi; S1.input["b"]=r.x2_macd; S1.compute(); S2.input["a"]=r.x3_vol; S2.input["b"]=r.x4_relvol; S2.compute()
        S3.input["a"]=S1.output["o"]; S3.input["b"]=S2.output["o"]; S3.compute(); sk.append(S3.output["o"])
    worst=max(worst,np.abs(mine-np.array(sk)).max()); n_days+=len(test)
print(f"V5 vs scikit-fuzzy on {n_days} days: max |diff| = {worst:.4f}:",ok(worst<0.1))
# V6 crisp = manual count; V7 no NaN, range
r=F["RELIANCE"].sample(8,random_state=3); m={k:v[1] for k,v in A.items()}
hand=[25*(int((x.x1_rsi<m['x1_rsi']) if fm.RSI_FAV_LOW else (x.x1_rsi>m['x1_rsi']))+int(x.x2_macd>m['x2_macd'])+int(x.x3_vol<m['x3_vol'])+int(x.x4_relvol>m['x4_relvol'])) for x in r.itertuples()]
print("V6 crisp baseline equals manual count:",ok(np.allclose(C.run(r.x1_rsi,r.x2_macd,r.x3_vol,r.x4_relvol),hand)))
Y=np.concatenate([M.run(*[f[c].values for c in COLS]) for f in F.values()]); print(f"V7 all {len(Y)} stock-days: range {Y.min():.2f}-{Y.max():.2f}, NaN {int(np.isnan(Y).sum())}:",ok(not np.isnan(Y).any() and Y.min()>18.5 and Y.max()<81.5))
