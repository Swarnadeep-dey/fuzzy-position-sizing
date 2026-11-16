"""CA-3: the fuzzy model (Steps 3-4). Mamdani, three small stages, 27 rules in total.
Stage 1: RSI + MACD      -> S  (Weak / Moderate / Strong)   9 rules
Stage 2: Volatility + RelVolume -> Q (Poor / Fair / Good)    9 rules
Stage 3: S + Q           -> y  (Small / Medium / Large)      9 rules
"""
import numpy as np
import pandas as pd
from fuzzy_lib import STOCKS, build

INF = 1e9
RSI_FAV_LOW = False     # decided 20 Oct 2026: momentum reading, HIGH RSI is favourable
GRID = np.linspace(0, 100, 2001)          # output universe 0..100, step 0.05
trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapezoid needs NumPy 2; older NumPy uses np.trapz

def trapmf(x, a, b, c, d):
    """Trapezoid: 0 below a, rises to 1 at b, stays 1 until c, falls to 0 at d. (b=c gives a triangle.)"""
    x = np.asarray(x, float)
    if a <= -INF / 2:      # open left shoulder: exactly 1 up to c
        up = np.ones_like(x)
    else:
        up = np.where(b > a, (x - a) / ((b - a) if b > a else 1.0), (x >= b).astype(float))
    if d >= INF / 2:       # open right shoulder: exactly 1 from b onwards
        dn = np.ones_like(x)
    else:
        dn = np.where(d > c, (d - x) / ((d - c) if d > c else 1.0), (x <= c).astype(float))
    return np.clip(np.minimum(up, dn), 0.0, 1.0)

# ---- output sets (also used for S and Q): label 0 = bad end, 1 = middle, 2 = good end -------------
OUT = [(0, 0, 20, 50), (20, 50, 50, 80), (50, 80, 100, 100)]   # Small/Weak/Poor, Medium/Moderate/Fair, Large/Strong/Good
OUT_MU = [trapmf(GRID, *s) for s in OUT]

def band(a, b):
    """Equal-importance rule: score sum 0-1 -> level 0, 2 -> level 1, 3-4 -> level 2."""
    s = a + b
    return 0 if s <= 1 else (1 if s == 2 else 2)

def fuzzify(x, q1, q2, q3):
    """Three sets from the calibration quartiles: Low [.. q1 -> q2], Mid (q1,q2,q3), High [q2 -> q3 ..]."""
    x = np.asarray(x, float)
    return np.stack([trapmf(x, -INF, -INF, q1, q2), trapmf(x, q1, q2, q2, q3), trapmf(x, q2, q3, INF, INF)])

def to_scores(mu, favourable_is_low):
    """Re-order Low/Mid/High into score 0 (Unfavourable), 1 (Neutral), 2 (Favourable)."""
    return mu[::-1] if favourable_is_low else mu          # volatility: Low is favourable (RSI, MACD, volume: High is favourable)

def stage(mu_a, mu_b, return_detail=False):
    """One Mamdani stage. mu_a, mu_b: arrays (3, n) of memberships for scores 0,1,2.
    9 rules: IF A is score a AND B is score b THEN out = band(a+b).  AND = min, clip, aggregate = max, centroid."""
    n = mu_a.shape[1]
    agg = np.zeros((n, len(GRID)))
    firing = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(3):
            w = np.minimum(mu_a[a], mu_b[b])                    # rule strength (AND = min)
            firing[:, a, b] = w
            clipped = np.minimum(w[:, None], OUT_MU[band(a, b)][None, :])
            agg = np.maximum(agg, clipped)                      # aggregation = max
    y = trapz(agg * GRID, GRID, axis=1) / trapz(agg, GRID, axis=1)   # centroid = integral(y*mu) / integral(mu)
    return (y, firing, agg) if return_detail else y

class FuzzyPositionModel:
    def __init__(self, anchors):
        """anchors: dict input -> (q1, q2, q3), from the pooled calibration year."""
        self.anchors = anchors
    def run(self, rsi, macd, vol, relvol, detail=False):
        A = self.anchors
        m1 = to_scores(fuzzify(rsi,    *A["x1_rsi"]),   RSI_FAV_LOW)
        m2 = to_scores(fuzzify(macd,   *A["x2_macd"]),  False)
        m3 = to_scores(fuzzify(vol,    *A["x3_vol"]),   True)
        m4 = to_scores(fuzzify(relvol, *A["x4_relvol"]), False)
        S, f1, a1 = stage(m1, m2, True)
        Q, f2, a2 = stage(m3, m4, True)
        mS, mQ = fuzzify_out(S), fuzzify_out(Q)
        y, f3, a3 = stage(mS, mQ, True)
        if detail:
            return dict(m=[m1, m2, m3, m4], S=S, Q=Q, y=y, firing=[f1, f2, f3], agg=[a1, a2, a3], mS=mS, mQ=mQ)
        return y

def fuzzify_out(v):
    """Memberships (score 0,1,2) of a stage output on the 0-100 scale."""
    v = np.asarray(v, float)
    return np.stack([trapmf(v, *o) for o in OUT])

class CrispModel:
    """Baseline: y = 25 x (number of the four indicators on the favourable side of the calibration median)."""
    def __init__(self, anchors):
        self.med = {k: v[1] for k, v in anchors.items()}
    def run(self, rsi, macd, vol, relvol):
        m = self.med
        passes = ((np.asarray(rsi) < m["x1_rsi"]) if RSI_FAV_LOW else (np.asarray(rsi) > m["x1_rsi"])).astype(int) + (np.asarray(macd) > m["x2_macd"]) + \
                 (np.asarray(vol) < m["x3_vol"]).astype(int) + (np.asarray(relvol) > m["x4_relvol"])
        return 25.0 * np.asarray(passes, float)

COLS = ["x1_rsi", "x2_macd", "x3_vol", "x4_relvol"]

def load_all():
    return {n: build(n) for n in STOCKS}

def pooled_anchors(F):
    pool = pd.concat([f[f.period == "calib"][COLS] for f in F.values()])
    return {c: tuple(pool[c].quantile([.25, .50, .75]).values) for c in COLS}


class WeightedAvgModel:
    """Second method (Sugeno-style): same fuzzy sets and the same 27 rules, but each stage output is the
    weighted average of the rule outputs (Small/Weak/Poor = 18.57, Medium = 50, Large = 81.43) instead of the centroid."""
    CEN = [18.57, 50.0, 81.43]
    def __init__(self, anchors):
        self.anchors = anchors
    @staticmethod
    def _stage(ma, mb):
        num = 0.0; den = 0.0
        for a in range(3):
            for b in range(3):
                w = np.minimum(ma[a], mb[b]); num = num + w * WeightedAvgModel.CEN[band(a, b)]; den = den + w
        return num / den
    def run(self, rsi, macd, vol, relvol):
        A = self.anchors
        m1 = to_scores(fuzzify(rsi, *A["x1_rsi"]), RSI_FAV_LOW); m2 = to_scores(fuzzify(macd, *A["x2_macd"]), False)
        m3 = to_scores(fuzzify(vol, *A["x3_vol"]), True);        m4 = to_scores(fuzzify(relvol, *A["x4_relvol"]), False)
        S = self._stage(m1, m2); Q = self._stage(m3, m4)
        return self._stage(fuzzify_out(S), fuzzify_out(Q))
