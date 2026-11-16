import numpy as np, pandas as pd, matplotlib, warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuzzy_model import *
BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; INK2="#52514e"; GRID_C="#e4e3df"; BG="#fcfcfb"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":INK2,"text.color":INK,
                     "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False})
F=load_all(); A=pooled_anchors(F); M=FuzzyPositionModel(A); C=CrispModel(A)
ALL=pd.read_pickle("test_results.pkl")
NAMES={"HDFCBANK":"HDFC Bank","HINDUNILVR":"Hindustan Unilever","TCS":"TCS","RELIANCE":"Reliance","TATASTEEL":"Tata Steel"}
# ---- Figure 4: distribution of fuzzy position size per stock (test period) ----
fig,axs=plt.subplots(1,5,figsize=(13,3.6),sharey=True,sharex=True)
for ax,(n,t) in zip(axs,ALL.items()):
    ax.hist(t.y_mamdani,bins=np.arange(15,86,5),color=BLUE,edgecolor=BG,lw=1.2)
    mu=t.y_mamdani.mean(); ax.axvline(mu,color=INK,lw=1.2,ls=(0,(4,3))); ax.set_ylim(0,105); ax.text(mu+1.5,101,f"mean {mu:.1f}",ha="left",va="top",fontsize=8.5)
    ax.set_title(NAMES[n],fontsize=10,loc="left"); ax.set_xlabel("Position size (%)"); ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.set_xlim(15,85)
axs[0].set_ylabel("Number of trading days")
fig.suptitle("Figure 5. Fuzzy position size on the 458 test days, by stock (dashed line = mean)",fontsize=11,x=0.01,ha="left")
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("figs/fig5_distribution_by_stock.png",dpi=170); plt.close(fig)
# ---- Figure 5: crisp staircase vs fuzzy curve, MACD swept, other inputs fixed at TCS 20 Jun 2025 ----
row=F["TCS"].loc[pd.Timestamp("2025-06-20")]; g=np.linspace(-0.6,0.8,561)
def sweep(model_fn):
    return model_fn(np.full_like(g,row.x1_rsi),g,np.full_like(g,row.x3_vol),np.full_like(g,row.x4_relvol))
yf=M.run(np.full_like(g,row.x1_rsi),g,np.full_like(g,row.x3_vol),np.full_like(g,row.x4_relvol))
yc=C.run(np.full_like(g,row.x1_rsi),g,np.full_like(g,row.x3_vol),np.full_like(g,row.x4_relvol))
fig,ax=plt.subplots(figsize=(8.8,4.2))
ax.step(g,yc,where="post",color=ORANGE,lw=2,label="Fixed-threshold rule (crisp)"); ax.plot(g,yf,color=BLUE,lw=2,label="Fuzzy model")
med=A["x2_macd"][1]; ax.axvline(med,color=INK2,lw=.9,ls=(0,(3,3))); ax.text(med+0.015,6,"cut-off = calibration median",fontsize=9,color=INK2)
ax.set_xlabel("MACD histogram (% of price), other three inputs held fixed"); ax.set_ylabel("Position size y (%)"); ax.set_ylim(0,100); ax.set_xlim(g[0],g[-1])
ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=2)
ax.set_title("Figure 6. A tiny change across the cut-off: the crisp rule jumps, the fuzzy model glides",fontsize=10.5,loc="left")
fig.tight_layout(); fig.savefig("figs/fig6_crisp_vs_fuzzy.png",dpi=170); plt.close(fig)
print("held fixed:",{k:round(float(row[k]),3) for k in ["x1_rsi","x3_vol","x4_relvol"]})
i0=np.searchsorted(g,med-0.02); i1=np.searchsorted(g,med+0.02)
print("MACD %.3f -> crisp %.0f fuzzy %.2f ; MACD %.3f -> crisp %.0f fuzzy %.2f"%(g[i0],yc[i0],yf[i0],g[i1],yc[i1],yf[i1]))
