"""
calculations.py
===============
Logica quantitativa per l'analisi dell'entropia dei mercati finanziari.

Studi implementati:
  Studio 1 — Shannon Entropy su Returns, Volatility, Skewness
  Studio 2 — Permutation Entropy normalizzata
  Studio 3 — Regime (Bassa/Media/Alta entropia) → Forward Returns medi
  Studio 4 — Scatter Entropia × Forward Returns (1M/3M/6M/12M)
  Studio 5 — Percentile storico dell'entropia (zone di allerta)
  Studio 6 — Cross-correlazione rolling + Heatmap stagionale

PREDITTIVITÀ — note metodologiche (Studi 3 e 4)
-----------------------------------------------
La domanda operativa è: "il regime/livello di entropia al tempo t determina, con
valore statistico, un rendimento positivo a distanza x?". Per rispondere in modo
ONESTO servono tre accorgimenti, tutti implementati qui:

  1) Soglie di regime POINT-IN-TIME (expanding): il regime al tempo t usa solo
     la storia fino a t → niente look-ahead, replicabile dal vivo.
  2) Significatività corretta per la SOVRAPPOSIZIONE dei forward returns: i forward
     a N giorni si sovrappongono (autocorrelazione ~1−1/N) → la N efficace è ~n/N.
     Si usano errori standard HAC / Newey-West (lag = orizzonte) sia per la
     pendenza dello scatter sia per la media di ciascun regime (via regressione su
     dummy). Si riporta anche la N efficace.
  3) Lettura della forma: media condizionata per decile di entropia (curva di
     predittività) + smoother kernel, oltre alla sola retta OLS.

Tutto è calcolato sia per la Shannon Entropy dei returns sia per la Permutation
Entropy (PE), così la misura di prevedibilità migliore non resta scollegata dai
forward returns.

Riferimenti:
  - Shannon, C.E. (1948). A Mathematical Theory of Communication.
  - Bandt, C. & Pompe, B. (2002). Permutation Entropy. PRL 88, 174102.
  - Newey, W.K. & West, K.D. (1987). A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  - Lo, A.W. (2004). The Adaptive Markets Hypothesis.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial, log2

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, pearsonr, spearmanr, norm

# ================================================================
# COSTANTI PARAMETRI
# ================================================================

SHANNON_WINDOW: int = 63    # finestra rolling Shannon ≈ 1 trimestre (63 trading days)
SHANNON_BINS: int = 10      # bin per discretizzazione serie temporale
PE_ORDER: int = 3           # embedding dimension Permutation Entropy
PE_WINDOW: int = 63         # finestra rolling PE (stessa di Shannon per coerenza)
REGIME_CORR_WINDOW: int = 126  # finestra correlazione rolling cross-entropy (≈6M)

# Storia minima prima di assegnare un regime point-in-time (expanding)
REGIME_MIN_PERIODS: int = 252

FORWARD_PERIODS: dict[str, int] = {
    "1M":  21,
    "3M":  63,
    "6M":  126,
    "12M": 252,
}

# Percentili soglia per allerta Studio 5
ALERT_HIGH: int = 80   # zona rossa (alta entropia → incertezza elevata)
ALERT_LOW: int = 20    # zona verde (bassa entropia → mercato più prevedibile)

# Soglie regime (tertili)
REGIME_P_LOW: float = 1 / 3
REGIME_P_HIGH: float = 2 / 3

# Misure di entropia analizzate per la predittività (colonna feat → etichetta)
ENTROPY_MEASURES: dict[str, str] = {
    "shannon_ret":  "Shannon Entropy (returns)",
    "perm_entropy": "Permutation Entropy",
}


# ================================================================
# DATACLASS RISULTATO
# ================================================================

@dataclass
class EntropyResult:
    """Contenitore con tutti i risultati dell'analisi di entropia."""

    feat: pd.DataFrame          # DataFrame completo delle feature
    ticker: str
    ticker_label: str
    n_raw: int                  # barre grezze dopo filtro data
    n_feat: int                 # righe con feature complete (dopo dropna)

    # Parametri usati
    shannon_window: int
    pe_order: int
    regime_min_periods: int

    # Soglie regime CORRENTI (ultimo valore expanding) — solo per display
    regime_p33: float
    regime_p67: float
    pe_p33: float
    pe_p67: float

    # Studio 3: forward returns medi per regime (Shannon, retro-compatibilità)
    regime_fwd: pd.DataFrame    # colonne: periodo, regime, fwd_mean

    # Studio 4: correlazioni scatter (Shannon) → {periodo: (r, p_hac)}
    scatter_corr: dict[str, tuple[float, float]]

    # Predittività completa per misura: {"shannon_ret": {...}, "perm_entropy": {...}}
    # Ogni voce: regime_stats (DataFrame), slope_stats (DataFrame),
    #            binned (dict periodo→DataFrame), smoother (dict periodo→(x,y))
    predictivity: dict[str, dict]

    # Studio 6: heatmap e cross-correlazione
    heatmap_data: pd.DataFrame
    cross_corr: pd.Series

    log_lines: list[str] = field(default_factory=list)


# ================================================================
# CORE — MISURE DI ENTROPIA
# ================================================================

def _shannon_entropy_window(arr: np.ndarray, bins: int = SHANNON_BINS) -> float:
    """
    Calcola la Shannon Entropy di un array tramite discretizzazione in `bins` bin.

    H = -Σ p_i · log2(p_i)

    dove p_i è la frequenza relativa dell'i-esimo bin.
    Ritorna 0.0 se tutti i valori cadono in un solo bin.
    """
    counts, _ = np.histogram(arr, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts[counts > 0] / total
    return float(scipy_entropy(probs, base=2))


def shannon_entropy_series(
    series: pd.Series,
    window: int = SHANNON_WINDOW,
    bins: int = SHANNON_BINS,
) -> pd.Series:
    """
    Applica Shannon Entropy su una finestra rolling di `window` osservazioni.
    """
    return series.rolling(window).apply(
        lambda x: _shannon_entropy_window(x, bins=bins),
        raw=True,
    )


def _permutation_entropy_window(arr: np.ndarray, order: int = PE_ORDER) -> float:
    """
    Calcola la Permutation Entropy normalizzata (Bandt & Pompe, 2002).
    PE ≈ 1 → alta complessità (casuale); PE << 1 → struttura (prevedibilità).
    """
    n = len(arr)
    if n < order:
        return np.nan

    all_patterns = list(itertools.permutations(range(order)))
    counts: dict = {p: 0 for p in all_patterns}

    for i in range(n - order + 1):
        pattern = tuple(np.argsort(arr[i: i + order], kind="stable"))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return np.nan

    probs = np.array([c / total for c in counts.values() if c > 0])
    h = -np.sum(probs * np.log2(probs))
    h_max = log2(factorial(order))

    return float(h / h_max) if h_max > 0 else np.nan


def permutation_entropy_series(
    series: pd.Series,
    order: int = PE_ORDER,
    window: int = PE_WINDOW,
) -> pd.Series:
    """Applica la Permutation Entropy normalizzata su una finestra rolling."""
    return series.rolling(window).apply(
        lambda x: _permutation_entropy_window(x, order=order),
        raw=True,
    )


# ================================================================
# REGIME POINT-IN-TIME (expanding tertiles)
# ================================================================

def assign_regime_expanding(
    series: pd.Series, min_periods: int = REGIME_MIN_PERIODS
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Assegna il regime (Bassa/Media/Alta) usando i tertili EXPANDING: per ogni t
    le soglie P33/P67 sono calcolate solo sulla storia fino a t (incluso t).

    Niente look-ahead: il regime è quello effettivamente conoscibile in tempo reale.
    Le righe con meno di `min_periods` osservazioni ricevono regime "n/d".

    Returns
    -------
    (regime, p33_exp, p67_exp)
    """
    p33 = series.expanding(min_periods=min_periods).quantile(REGIME_P_LOW)
    p67 = series.expanding(min_periods=min_periods).quantile(REGIME_P_HIGH)

    val = series.to_numpy(dtype=float)
    lo = p33.to_numpy(dtype=float)
    hi = p67.to_numpy(dtype=float)

    regime = np.full(len(series), "n/d", dtype=object)
    valid = ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(val)
    regime[valid & (val <= lo)] = "Bassa"
    regime[valid & (val > lo) & (val <= hi)] = "Media"
    regime[valid & (val > hi)] = "Alta"

    return pd.Series(regime, index=series.index), p33, p67


# ================================================================
# MOTORE STATISTICO — HAC / Newey-West
# ================================================================

def _hac_ols(X: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """
    OLS con matrice di covarianza HAC (Newey-West, kernel di Bartlett).

    Restituisce per ogni coefficiente: beta, se (HAC), t, p (normale), n_eff
    (N efficace = n · Var_OLS / Var_HAC, cappata a n).

    `lag` è il numero massimo di ritardi del kernel (≈ orizzonte dei forward
    returns, per assorbire la loro sovrapposizione).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    u = X * resid[:, None]            # contributi allo score (n × k)
    S = u.T @ u                       # lag 0
    L = int(max(0, min(lag, n - 1)))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)

    cov_hac = XtX_inv @ S @ XtX_inv
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov_ols = sigma2 * XtX_inv

    var_hac = np.clip(np.diag(cov_hac), 0.0, None)
    var_ols = np.clip(np.diag(cov_ols), 0.0, None)
    se = np.sqrt(var_hac)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
        p = 2.0 * norm.sf(np.abs(t))
        n_eff = np.where(var_hac > 0, n * var_ols / var_hac, float(n))
    n_eff = np.minimum(n_eff, float(n))

    return {"beta": beta, "se": se, "t": t, "p": p, "n_eff": n_eff, "n": n}


# ================================================================
# CURVA DI PREDITTIVITÀ — binning + smoother
# ================================================================

def _binned_conditional(
    x: np.ndarray, y: np.ndarray, horizon: int, n_bins: int = 10
) -> pd.DataFrame:
    """
    Media condizionata del forward return per decile del livello di entropia.
    L'errore standard usa una N efficace ≈ n_bin/horizon per tenere conto della
    sovrapposizione (approssimazione conservativa, etichettata come tale nell'UI).
    """
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin_center", "mean", "se", "hit_rate", "n"])
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb < 5:
            continue
        yy = y[m]
        n_eff = max(1.0, nb / horizon)
        se = float(np.std(yy, ddof=1) / np.sqrt(n_eff)) if nb > 1 else np.nan
        rows.append({
            "bin_center": float(np.median(x[m])),
            "mean":       float(yy.mean()),
            "se":         se,
            "hit_rate":   float((yy > 0).mean() * 100.0),
            "n":          nb,
        })
    return pd.DataFrame(rows)


def _kernel_smoother(
    x: np.ndarray, y: np.ndarray, grid_size: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smoother di Nadaraya-Watson (kernel gaussiano) per rivelare relazioni
    non-lineari/non-monotone tra entropia e forward return. Bandwidth di Silverman.
    """
    lo, hi = np.quantile(x, [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
    xs = np.linspace(lo, hi, grid_size)

    h = 1.06 * np.std(x) * len(x) ** (-1 / 5)
    if h <= 0:
        h = (hi - lo) / 20 if hi > lo else 1e-6

    yhat = np.empty(grid_size)
    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((x - xi) / h) ** 2)
        sw = w.sum()
        yhat[i] = float((w @ y) / sw) if sw > 0 else np.nan
    return xs, yhat


def _compute_predictivity(
    feat_fwd: pd.DataFrame, entropy_col: str, regime_col: str
) -> dict:
    """
    Calcola tutta la predittività per UNA misura di entropia:

      regime_stats : DataFrame [periodo, regime, n, mean_pct, hit_rate, t_hac, p_hac, sig]
                     (media del forward return per regime, con test HAC che media≠0)
      slope_stats  : DataFrame [periodo, pearson_r, spearman_rho, slope, p_hac, n_eff]
                     (pendenza/correlazione entropia→forward, p-value HAC, N efficace)
      binned       : {periodo: DataFrame} curva media-per-decile
      smoother     : {periodo: (xgrid, yhat)} smoother kernel
    """
    regime_rows: list[dict] = []
    slope_rows: list[dict] = []
    binned: dict[str, pd.DataFrame] = {}
    smoother: dict[str, tuple] = {}

    regimi = ["Bassa", "Media", "Alta"]

    for periodo, days in FORWARD_PERIODS.items():
        col = f"fwd_{periodo}"
        sub = feat_fwd[[entropy_col, regime_col, col]].dropna()
        sub = sub[sub[regime_col].isin(regimi)]
        if len(sub) < 30:
            continue

        x = sub[entropy_col].to_numpy(dtype=float)
        y = sub[col].to_numpy(dtype=float)

        # ── Scatter: pendenza/correlazione con p-value HAC ──────────
        Xs = np.column_stack([np.ones(len(x)), x])
        res = _hac_ols(Xs, y, lag=days)
        r = float(pearsonr(x, y)[0]) if np.std(x) > 0 else np.nan
        rho = float(spearmanr(x, y).statistic) if np.std(x) > 0 else np.nan
        slope_rows.append({
            "periodo":      periodo,
            "pearson_r":    round(r, 4),
            "spearman_rho": round(rho, 4),
            "slope":        float(res["beta"][1]),
            "p_hac":        float(res["p"][1]),
            "n_eff":        float(res["n_eff"][1]),
            "n":            int(res["n"]),
        })

        # ── Regime: media per regime con test HAC (media ≠ 0) ───────
        present = [g for g in regimi if (sub[regime_col] == g).any()]
        D = np.column_stack([(sub[regime_col] == g).to_numpy(dtype=float) for g in present])
        resd = _hac_ols(D, y, lag=days)
        for i, g in enumerate(present):
            mask = (sub[regime_col] == g).to_numpy()
            regime_rows.append({
                "periodo":  periodo,
                "regime":   g,
                "n":        int(mask.sum()),
                "mean_pct": float(resd["beta"][i]),
                "hit_rate": float((y[mask] > 0).mean() * 100.0),
                "t_hac":    float(resd["t"][i]),
                "p_hac":    float(resd["p"][i]),
                "sig":      bool(resd["p"][i] < 0.05),
            })

        # ── Curva di predittività ──────────────────────────────────
        binned[periodo] = _binned_conditional(x, y, horizon=days)
        smoother[periodo] = _kernel_smoother(x, y)

    return {
        "regime_stats": pd.DataFrame(regime_rows),
        "slope_stats":  pd.DataFrame(slope_rows),
        "binned":       binned,
        "smoother":     smoother,
    }


# ================================================================
# BUILD FEATURES — funzione principale
# ================================================================

def build_features(
    df: pd.DataFrame,
    ticker: str,
    ticker_label: str,
    shannon_window: int = SHANNON_WINDOW,
    pe_order: int = PE_ORDER,
    regime_min_periods: int = REGIME_MIN_PERIODS,
    log_lines: list[str] | None = None,
) -> EntropyResult:
    """
    Costruisce tutte le feature di entropia a partire dai prezzi OHLCV e calcola
    la predittività (Studi 3-4) in modo point-in-time e con statistiche HAC.
    """
    if log_lines is None:
        log_lines = []

    close = df["adjusted_close"].dropna()
    n_raw = len(close)
    log_lines.append(f"Barre disponibili: {n_raw} ({close.index[0].date()} → {close.index[-1].date()})")

    # ── 1. Log-returns ──────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))

    # ── 2. Features di input per Shannon ────────────────────────
    vol = log_ret.rolling(21).std()
    skew = log_ret.rolling(63).skew()
    log_lines.append("Returns / Volatilità / Skewness calcolati")

    # ── 3. Shannon Entropy rolling ───────────────────────────────
    sh_ret  = shannon_entropy_series(log_ret, window=shannon_window)
    sh_vol  = shannon_entropy_series(vol,     window=shannon_window)
    sh_skew = shannon_entropy_series(skew,    window=shannon_window)

    # ── 4. Permutation Entropy rolling ──────────────────────────
    pe = permutation_entropy_series(log_ret, order=pe_order, window=PE_WINDOW)

    # ── 5. Assembla feature DataFrame (solo feature entropiche) ──
    feat = pd.DataFrame({
        "close":        close,
        "log_ret":      log_ret,
        "vol_21":       vol,
        "skew_63":      skew,
        "shannon_ret":  sh_ret,
        "shannon_vol":  sh_vol,
        "shannon_skew": sh_skew,
        "perm_entropy": pe,
    })

    feat = feat.dropna(subset=["shannon_ret", "perm_entropy"])
    n_feat = len(feat)
    log_lines.append(f"Feature complete: {n_feat:,} righe ({n_raw - n_feat} NaN rimossi)")
    log_lines.append(f"Ultima candela: {feat.index[-1].date()}")

    # ── 6. Regime POINT-IN-TIME (expanding tertiles) ─────────────
    regime_sh, sh_p33_exp, sh_p67_exp = assign_regime_expanding(
        feat["shannon_ret"], min_periods=regime_min_periods
    )
    regime_pe, pe_p33_exp, pe_p67_exp = assign_regime_expanding(
        feat["perm_entropy"], min_periods=regime_min_periods
    )
    feat["regime"]    = regime_sh
    feat["regime_pe"] = regime_pe
    log_lines.append("Regime point-in-time (expanding tertiles) ✓ — niente look-ahead")

    # Soglie correnti (ultimo valore expanding) per display
    regime_p33 = float(sh_p33_exp.iloc[-1])
    regime_p67 = float(sh_p67_exp.iloc[-1])
    pe_p33     = float(pe_p33_exp.iloc[-1])
    pe_p67     = float(pe_p67_exp.iloc[-1])

    # ── 7. Forward returns (DataFrame separato + colonne in feat) ─
    fwd_series = {}
    for label, days in FORWARD_PERIODS.items():
        fwd_series[f"fwd_{label}"] = log_ret.rolling(days).sum().shift(-days) * 100

    feat_fwd = feat.copy()
    for col, series in fwd_series.items():
        feat_fwd[col] = series
        feat[col] = series

    # ── 8. Predittività per Shannon e per PE ─────────────────────
    predictivity = {
        "shannon_ret":  _compute_predictivity(feat_fwd, "shannon_ret", "regime"),
        "perm_entropy": _compute_predictivity(feat_fwd, "perm_entropy", "regime_pe"),
    }
    log_lines.append("Studi 3-4 (Regime/Scatter → Forward) con HAC ✓ — Shannon + PE")

    # regime_fwd (Shannon) per retro-compatibilità con la bar chart/tabella
    sh_reg = predictivity["shannon_ret"]["regime_stats"]
    if not sh_reg.empty:
        regime_fwd = sh_reg.rename(columns={"mean_pct": "fwd_mean"})[
            ["periodo", "regime", "fwd_mean"]
        ].copy()
    else:
        regime_fwd = pd.DataFrame(columns=["periodo", "regime", "fwd_mean"])

    # scatter_corr (Shannon) → {periodo: (r, p_hac)}
    sh_slope = predictivity["shannon_ret"]["slope_stats"]
    scatter_corr = {
        row["periodo"]: (float(row["pearson_r"]), float(row["p_hac"]))
        for _, row in sh_slope.iterrows()
    }

    # ── 9. Heatmap mensile (Studio 6) ───────────────────────────
    feat["anno"] = feat.index.year
    feat["mese"] = feat.index.month
    heatmap_data = (
        feat.groupby(["anno", "mese"])["shannon_ret"]
        .mean()
        .unstack(level="mese")
    )
    mesi_ita = {1:"Gen", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mag", 6:"Giu",
                7:"Lug", 8:"Ago", 9:"Set", 10:"Ott", 11:"Nov", 12:"Dic"}
    heatmap_data.columns = [mesi_ita[m] for m in heatmap_data.columns]
    log_lines.append("Studio 6 (Heatmap stagionale) ✓")

    # ── 10. Cross-correlazione rolling PE ↔ Shannon (Studio 6) ──
    cross_corr = (
        feat["perm_entropy"]
        .rolling(REGIME_CORR_WINDOW)
        .corr(feat["shannon_ret"])
    )

    log_lines.append("✅ Completato")

    return EntropyResult(
        feat=feat,
        ticker=ticker,
        ticker_label=ticker_label,
        n_raw=n_raw,
        n_feat=n_feat,
        shannon_window=shannon_window,
        pe_order=pe_order,
        regime_min_periods=regime_min_periods,
        regime_p33=regime_p33,
        regime_p67=regime_p67,
        pe_p33=pe_p33,
        pe_p67=pe_p67,
        regime_fwd=regime_fwd,
        scatter_corr=scatter_corr,
        predictivity=predictivity,
        heatmap_data=heatmap_data,
        cross_corr=cross_corr,
        log_lines=log_lines,
    )


# ================================================================
# UTILITY — percentile storico rolling (Studio 5)
# ================================================================

def compute_percentile_series(series: pd.Series) -> pd.Series:
    """
    Per ogni punto t, percentile della Shannon Entropy corrente rispetto a tutta
    la storia fino a t (percentile expanding, point-in-time).
    """
    expanding_rank = series.expanding().rank(pct=True) * 100
    return expanding_rank
