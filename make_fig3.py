import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuzzy_model import *
BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; INK2="#52514e"; GRID_C="#e4e3df"; BG="#fcfcfb"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":INK2,"text.color":INK,
                     "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False})
F=load_all(); A=pooled_anchors(F); M=FuzzyPositionModel(A)
row=F["HDFCBANK"].loc[[pd.Timestamp("2024-12-11")]]
r=M.run(*[row[c].values for c in COLS],detail=True); y=float(r["y"][0]); w=r["firing"][2][0]
fig,ax=plt.subplots(figsize=(8.5,4.2))
agg=r["agg"][2][0]
ax.fill_between(GRID,agg,color=BLUE,alpha=.18,label="Combined shape (all clipped shapes stacked)")
for a in range(3):
    for b in range(3):
        if w[a,b]>1e-9:
            lab=["Small","Medium","Large"][band(a,b)]; c={"Small":ORANGE,"Medium":AQUA,"Large":BLUE}[lab]
            ax.plot(GRID,np.minimum(w[a,b],OUT_MU[band(a,b)]),color=c,lw=2,label=f"{lab} clipped at {w[a,b]:.3f}")
ax.axvline(y,color=INK,lw=1.2,ls=(0,(4,3))); ax.text(y+1,0.96,f"balance point y = {y:.2f}",fontsize=10,va="top")
crisp=CrispModel(A).run(*[row[c].values for c in COLS])[0]; ax.axvline(crisp,color=INK2,lw=1,ls=":"); ax.text(crisp+1,0.40,f"fixed rule = {crisp:.0f}",fontsize=9,color=INK2,va="top")
ax.set_xlim(0,100); ax.set_ylim(0,1.05); ax.set_xlabel("Position size y (% of maximum capital)"); ax.set_ylabel("Degree of membership")
ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True)
ax.set_title("Figure 4. Stage 3 for HDFC Bank on 11 Dec 2024: two rules fire, the balance point gives the position size",fontsize=10.5,loc="left")
ax.legend(frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=2)
fig.tight_layout(); fig.savefig("figs/fig4_worked_example.png",dpi=170); plt.close(fig)
