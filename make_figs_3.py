import numpy as np, pandas as pd, matplotlib, warnings
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; INK2="#52514e"; GRID_C="#e4e3df"; BG="#fcfcfb"; GREY="#8a8984"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":INK2,"text.color":INK,
                     "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False})
# Fig 6: sensitivity
R=pd.read_pickle("sens_inputs.pkl"); R=R[R.model=="Mamdani"].groupby(["input","pct"]).mean_abs.mean().unstack()
lab={"x1_rsi":"RSI","x2_macd":"MACD","x3_vol":"Volatility","x4_relvol":"Volume"}
cols={"x1_rsi":BLUE,"x2_macd":ORANGE,"x3_vol":AQUA,"x4_relvol":"#7a4fc8"}; mk={"x1_rsi":"o","x2_macd":"s","x3_vol":"^","x4_relvol":"D"}
fig,ax=plt.subplots(figsize=(8.2,4.4))
for i,c in enumerate(R.index):
    ax.plot(R.columns*100,R.loc[c].values,color=cols[c],lw=2,marker=mk[c],ms=7,mec=BG,mew=1.5,label=lab[c])
ax.set_xticks([10,20,30]); ax.set_xticklabels(["10%","20%","30%"]); ax.set_xlim(7,33); ax.set_ylim(0,5)
ax.set_xlabel("Size of change in ONE input (as % of its typical range, the 25th–75th percentile gap)"); ax.set_ylabel("Average change in position size (points of 100)")
ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.2))
ax.set_title("Figure 7. All four inputs move the position size by almost the same amount",fontsize=10.5,loc="left")
fig.tight_layout(); fig.savefig("figs/fig7_sensitivity.png",dpi=170); plt.close(fig)
# Fig 7: equal-weight portfolio growth of 100
rets=pd.read_pickle("bt_rets.pkl"); P=sum(rets.values())/5
eq=100*(1+P).cumprod(); eq=pd.concat([pd.DataFrame([[100]*4],columns=P.columns,index=[P.index[0]-pd.Timedelta(days=1)]),eq])
fig,ax=plt.subplots(figsize=(9,4.4))
for s,c,l in [("Buy&hold",GREY,"Buy and hold"),("Crisp",ORANGE,"Fixed-threshold rule (crisp)"),("Mamdani",BLUE,"Fuzzy model")]:
    ax.plot(eq.index,eq[s],color=c,lw=2,label=l)
ax.axhline(100,color=INK2,lw=.8,ls=(0,(3,3)))
ax.set_ylabel("Value of 100 invested (equal mix of 5 stocks)"); ax.set_xlabel("Date")
ax.grid(axis="y",color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.17))
ax.set_title("Figure 8. Backtest: the position-size rules fall less than buy and hold (Dec 2024 – Oct 2026)",fontsize=10.5,loc="left")
fig.tight_layout(); fig.savefig("figs/fig8_backtest.png",dpi=170); plt.close(fig)
print(eq.iloc[-1].round(1).to_dict())
# Fig 8: Mamdani vs weighted average
ALL=pd.read_pickle("test_results.pkl"); T=pd.concat(ALL.values())
fig,ax=plt.subplots(figsize=(5.6,5.2))
ax.scatter(T.y_wavg,T.y_mamdani,s=7,color=BLUE,alpha=.35,edgecolors="none"); ax.plot([15,85],[15,85],color=INK2,lw=1,ls=(0,(3,3)))
ax.set_xlim(15,85); ax.set_ylim(15,85); ax.set_xlabel("Weighted-average method: position size (%)"); ax.set_ylabel("Mamdani method: position size (%)")
ax.grid(color=GRID_C,lw=.6); ax.set_axisbelow(True); ax.text(17,80,"dashed line = identical answers",fontsize=9,color=INK2)
ax.set_title("Figure 3. The two fuzzy methods agree (2,290 days)",fontsize=10.5,loc="left")
fig.tight_layout(); fig.savefig("figs/fig3_method_agreement.png",dpi=170); plt.close(fig)
# case inputs
import fuzzy_model as fm
for n,d in [("HDFCBANK","2024-12-11"),("HDFCBANK","2025-03-25"),("TATASTEEL","2025-01-15"),("TATASTEEL","2025-03-12"),("TCS","2025-06-19"),("TCS","2025-06-20")]:
    r=ALL[n].loc[pd.Timestamp(d)]; print(n,d,{k:round(float(r[k]),3) for k in fm.COLS},"S %.1f Q %.1f y %.2f crisp %.0f wavg %.2f"%(r.S,r.Q,r.y_mamdani,r.y_crisp,r.y_wavg))
A=fm.pooled_anchors(fm.load_all()); print({k:[round(float(x),3) for x in v] for k,v in A.items()})
