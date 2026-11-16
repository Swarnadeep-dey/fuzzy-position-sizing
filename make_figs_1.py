import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuzzy_model import *
BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; INK2="#52514e"; GRID_C="#e4e3df"; BG="#fcfcfb"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":INK2,"text.color":INK,
                     "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False})
F=load_all(); A=pooled_anchors(F)
# ---------- Figure 1: input membership functions ----------
titles={"x1_rsi":("RSI (0 to 100)","RSI value","High RSI = favourable"),"x2_macd":("MACD histogram (% of price)","MACD histogram (% of price)","High = favourable"),
        "x3_vol":("Volatility (20-day std of daily returns)","Volatility (% per day)","Low = favourable"),"x4_relvol":("Relative volume (today / 20-day average)","Relative volume (ratio)","High = favourable")}
fav_low={"x1_rsi":RSI_FAV_LOW,"x2_macd":False,"x3_vol":True,"x4_relvol":False}
pool=pd.concat([f[f.period=="calib"][COLS] for f in F.values()])
fig,axs=plt.subplots(2,2,figsize=(11,7.2))
for ax,k in zip(axs.ravel(),COLS):
    q1,q2,q3=A[k]; iq=q3-q1; lo=q1-1.5*iq; hi=q3+1.5*iq; lo=max(lo,0.0) if k in ('x3_vol','x4_relvol') else lo; x=np.linspace(lo,hi,600)
    mu=to_scores(fuzzify(x,q1,q2,q3),fav_low[k])       # rows: Unfavourable, Neutral, Favourable
    for row,(lab,c) in enumerate([("Unfavourable",ORANGE),("Neutral",AQUA),("Favourable",BLUE)]):
        ax.plot(x,mu[row],color=c,lw=2,label=lab)
    for v,t in [(q1,"Q1"),(q2,"median"),(q3,"Q3")]:
        ax.axvline(v,color=INK2,lw=0.8,ls=(0,(3,3)),alpha=.7); ax.text(v,1.07,f"{t}\n{v:.2f}",ha="center",va="bottom",fontsize=8,color=INK2)
    ax.set_ylim(0,1.02); ax.set_xlim(x[0],x[-1]); ax.set_xlabel(titles[k][1]); ax.set_ylabel("Degree of membership")
    ax.set_title(titles[k][0]+"  |  "+titles[k][2],fontsize=10,loc="left",pad=30); ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True)
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,frameon=False,fontsize=10,bbox_to_anchor=(0.5,0.0))
fig.suptitle("Figure 1. Fuzzy sets for the four inputs (corners = 25%, 50%, 75% points of the calibration year, five stocks together)",fontsize=11,x=0.01,ha="left",y=0.995)
fig.tight_layout(rect=(0,0.04,1,0.96)); fig.savefig("figs/fig1_input_membership.png",dpi=170); plt.close(fig)

# ---------- Figure 2: output sets ----------
fig,ax=plt.subplots(figsize=(8.5,3.8))
for mu,(lab,c) in zip(OUT_MU,[("Small",ORANGE),("Medium",AQUA),("Large",BLUE)]): ax.plot(GRID,mu,color=c,lw=2,label=lab)
ax.set_xlabel("Position size y (% of maximum capital)"); ax.set_ylabel("Degree of membership"); ax.set_ylim(0,1.05); ax.set_xlim(0,100)
ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.2))
ax.set_title("Figure 2. Output sets Small / Medium / Large (the same shapes are used for S and Q)",fontsize=10.5,loc="left",pad=10)
fig.tight_layout(); fig.savefig("figs/fig2_output_sets.png",dpi=170); plt.close(fig)
print("ok")
