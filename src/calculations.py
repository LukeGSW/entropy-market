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

Riferimenti:
  - Shannon, C.E. (1948). A Mathematical Theory of Communication.
  - Bandt, C. & Pompe, B. (2002). Permutation Entropy. PRL 88, 174102.
  - Pincus, S.M. (1991). Approximate entropy as a measure of system complexity.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial, log2

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, pearsonr, linregress

# ================================================================
# COSTANTI PARAMETRI
# ================================================================

SHANNON_WINDOW: int = 63    # finestra rolling Shannon ≈ 1 trimestre (63 trading days)
SHANNON_BINS: int = 10      # bin per discretizzazione serie temporale
PE_ORDER: int = 3           # embedding dimension Permutation Entropy
PE_WINDOW: int = 63         # finestra rolling PE (stessa di Shannon per coerenza)
REGIME_CORR_WINDOW: int = 126  # finestra correlazione rolling cross-entropy (≈6M)

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

    # Soglie regime (valori numerici di shannon_ret ai percentili 33/67)
    regime_p33: float
    regime_p67: float

    # Studio 2: soglie PE per il grafico
    pe_p33: float
    pe_p67: float

    # Studio 3: forward returns medi per regime
    regime_fwd: pd.DataFrame    # colonne: periodo, regime, fwd_mean

    # Studio 4: correlazioni scatter
    scatter_corr: dict[str, tuple[float, float]]   # {periodo: (r, pvalue)}

    # Studio 6: heatmap e cross-correlazione
    heatmap_data: pd.DataFrame   # pivot: index=anno, columns=mese
    cross_corr: pd.Series        # correlazione rolling PE ↔ Shannon

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

    Parameters
    ----------
    series : Serie temporale dei valori (es. log-return, volatilità)
    window : Lunghezza della finestra rolling (default 63 = ~1 trimestre)
    bins   : Numero di bin per la discretizzazione (default 10)

    Returns
    -------
    pd.Series con la Shannon Entropy rolling (NaN per i primi window-1 valori)
    """
    return series.rolling(window).apply(
        lambda x: _shannon_entropy_window(x, bins=bins),
        raw=True,
    )


def _permutation_entropy_window(arr: np.ndarray, order: int = PE_ORDER) -> float:
    """
    Calcola la Permutation Entropy di un array (Bandt & Pompe, 2002).

    1. Si generano tutti i pattern di lunghezza `order` (ordinal patterns)
    2. Si contano le frequenze relative di ciascun pattern
    3. H_perm = -Σ p_i · log2(p_i) / log2(order!)   [normalizzata 0-1]

    Una PE ≈ 1 indica alta complessità (mercato casuale/efficiente).
    Una PE << 1 indica struttura temporale (possibile prevedibilità).
    """
    n = len(arr)
    if n < order:
        return np.nan

    # Genera tutti i possibili pattern ordinali
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
    h_max = log2(factorial(order))   # entropia massima teorica

    return float(h / h_max) if h_max > 0 else np.nan


def permutation_entropy_series(
    series: pd.Series,
    order: int = PE_ORDER,
    window: int = PE_WINDOW,
) -> pd.Series:
    """
    Applica la Permutation Entropy normalizzata su una finestra rolling.

    Parameters
    ----------
    series : Serie temporale (tipicamente log-return)
    order  : Ordine/embedding dimension m (default 3; range consigliato 3-6)
    window : Lunghezza della finestra rolling (default 63)

    Returns
    -------
    pd.Series con PE normalizzata in [0, 1]
    """
    return series.rolling(window).apply(
        lambda x: _permutation_entropy_window(x, order=order),
        raw=True,
    )


# ================================================================
# BUILD FEATURES — funzione principale
# ================================================================

def build_features(
    df: pd.DataFrame,
    ticker: str,
    ticker_label: str,
    shannon_window: int = SHANNON_WINDOW,
    pe_order: int = PE_ORDER,
    log_lines: list[str] | None = None,
) -> EntropyResult:
    """
    Costruisce tutte le feature di entropia a partire dai prezzi OHLCV.

    Pipeline:
      1. Log-return giornalieri
      2. Volatilità rolling (21g) e skewness rolling (63g)
      3. Shannon Entropy rolling su: returns, volatilità, skewness
      4. Permutation Entropy normalizzata rolling sui returns
      5. Forward returns cumulativi: 1M/3M/6M/12M
      6. Dropna e calcolo statistiche regime/scatter/heatmap

    Parameters
    ----------
    df            : DataFrame OHLCV con colonna 'adjusted_close'
    ticker        : Ticker EODHD (es. 'GSPC.INDX')
    ticker_label  : Etichetta leggibile (es. 'S&P 500 (SPX)')
    shannon_window: Finestra rolling Shannon (default 63)
    pe_order      : Embedding dimension PE (default 3)
    log_lines     : Lista dove appendere messaggi di log per la UI

    Returns
    -------
    EntropyResult con tutte le feature e le statistiche derivate
    """
    if log_lines is None:
        log_lines = []

    close = df["adjusted_close"].dropna()
    n_raw = len(close)
    log_lines.append(f"Barre disponibili: {n_raw} ({close.index[0].date()} → {close.index[-1].date()})")

    # ── 1. Log-returns ──────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))

    # ── 2. Features di input per Shannon ────────────────────────
    # Volatilità: rolling std 21g (≈ 1 mese) → misura dispersione locale
    vol = log_ret.rolling(21).std()

    # Skewness rolling 63g → cattura asimmetria della distribuzione locale
    skew = log_ret.rolling(63).skew()

    log_lines.append("Returns / Volatilità / Skewness calcolati")

    # ── 3. Shannon Entropy rolling ───────────────────────────────
    sh_ret  = shannon_entropy_series(log_ret, window=shannon_window)
    sh_vol  = shannon_entropy_series(vol,     window=shannon_window)
    sh_skew = shannon_entropy_series(skew,    window=shannon_window)

    # ── 4. Permutation Entropy rolling ──────────────────────────
    pe = permutation_entropy_series(log_ret, order=pe_order, window=PE_WINDOW)

    # ── 5. Assembla feature DataFrame (solo feature entropiche) ──
    # IMPORTANTE: i forward return NON entrano in questo DataFrame.
    # shift(-N) renderebbe NaN le ultime N righe → dropna taglierebbe
    # l'ultimo anno di dati (12M = 252 giorni ≈ 1 anno).
    # Le feature entropiche arrivano fino all'ultima candela disponibile.
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

    # Dropna solo sulle colonne entropiche (rimuove warm-up iniziale)
    feat = feat.dropna(subset=["shannon_ret", "perm_entropy"])
    n_feat = len(feat)
    log_lines.append(f"Feature complete: {n_feat:,} righe ({n_raw - n_feat} NaN rimossi)")
    log_lines.append(f"Ultima candela: {feat.index[-1].date()}")

    # ── 6. Soglie regime (tertili di shannon_ret) ────────────────
    regime_p33 = float(feat["shannon_ret"].quantile(REGIME_P_LOW))
    regime_p67 = float(feat["shannon_ret"].quantile(REGIME_P_HIGH))
    pe_p33     = float(feat["perm_entropy"].quantile(REGIME_P_LOW))
    pe_p67     = float(feat["perm_entropy"].quantile(REGIME_P_HIGH))

    # ── 7. Etichette regime ──────────────────────────────────────
    def _assign_regime(val: float) -> str:
        if val <= regime_p33:
            return "Bassa"
        elif val <= regime_p67:
            return "Media"
        return "Alta"

    feat["regime"] = feat["shannon_ret"].apply(_assign_regime)

    # ── 8. Forward returns su DataFrame separato (Studio 3 e 4) ──
    # shift(-N) crea NaN nelle ultime N righe: accettabile qui perché
    # feat_fwd serve solo per statistiche storiche, non per i grafici
    # di serie temporale. Le righe recenti con NaN vengono escluse
    # automaticamente nelle analisi successive tramite dropna() locale.
    fwd_series = {}
    for label, days in FORWARD_PERIODS.items():
        fwd_series[f"fwd_{label}"] = log_ret.rolling(days).sum().shift(-days) * 100

    feat_fwd = feat.copy()
    for col, series in fwd_series.items():
        feat_fwd[col] = series

    # ── 9. Forward returns medi per regime (Studio 3) ────────────
    rows = []
    for periodo, days in FORWARD_PERIODS.items():
        col = f"fwd_{periodo}"
        valid = feat_fwd[["regime", col]].dropna()
        for regime in ["Bassa", "Media", "Alta"]:
            mask = valid["regime"] == regime
            mean_ret = float(valid.loc[mask, col].mean()) if mask.any() else 0.0
            rows.append({"periodo": periodo, "regime": regime, "fwd_mean": mean_ret})
    regime_fwd = pd.DataFrame(rows)

    log_lines.append("Studio 3 (Regime → Forward Returns) ✓")

    # ── 10. Correlazioni scatter (Studio 4) ─────────────────────
    scatter_corr: dict[str, tuple[float, float]] = {}
    for periodo in FORWARD_PERIODS:
        col = f"fwd_{periodo}"
        valid = feat_fwd[["shannon_ret", col]].dropna()
        if len(valid) > 10:
            r, p = pearsonr(valid["shannon_ret"], valid[col])
            scatter_corr[periodo] = (round(float(r), 4), round(float(p), 4))
        else:
            scatter_corr[periodo] = (np.nan, np.nan)

    log_lines.append("Studio 4 (Scatter Entropia × Returns) ✓")

    # ── 11. Percentile storico (Studio 5) ───────────────────────
    log_lines.append("Studio 5 (Percentile Entropia) ✓")

    # ── 12. Heatmap mensile (Studio 6) ──────────────────────────
    feat["anno"]  = feat.index.year
    feat["mese"]  = feat.index.month
    heatmap_data = (
        feat.groupby(["anno", "mese"])["shannon_ret"]
        .mean()
        .unstack(level="mese")
    )
    mesi_ita = {1:"Gen", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mag", 6:"Giu",
                7:"Lug", 8:"Ago", 9:"Set", 10:"Ott", 11:"Nov", 12:"Dic"}
    heatmap_data.columns = [mesi_ita[m] for m in heatmap_data.columns]

    log_lines.append("Studio 6 (Heatmap stagionale) ✓")

    # ── 13. Cross-correlazione rolling PE ↔ Shannon (Studio 6) ──
    cross_corr = (
        feat["perm_entropy"]
        .rolling(REGIME_CORR_WINDOW)
        .corr(feat["shannon_ret"])
    )

    # ── 14. Aggiungi forward return a feat per scatter (Studio 4) ─
    # Solo le colonne fwd vanno aggiunte a feat per uso in charts.py
    for col, series in fwd_series.items():
        feat[col] = series

    log_lines.append("✅ Completato")

    return EntropyResult(
        feat=feat,
        ticker=ticker,
        ticker_label=ticker_label,
        n_raw=n_raw,
        n_feat=n_feat,
        shannon_window=shannon_window,
        pe_order=pe_order,
        regime_p33=regime_p33,
        regime_p67=regime_p67,
        pe_p33=pe_p33,
        pe_p67=pe_p67,
        regime_fwd=regime_fwd,
        scatter_corr=scatter_corr,
        heatmap_data=heatmap_data,
        cross_corr=cross_corr,
        log_lines=log_lines,
    )


# ================================================================
# UTILITY — percentile storico rolling (Studio 5)
# ================================================================

def compute_percentile_series(series: pd.Series) -> pd.Series:
    """
    Per ogni punto t, calcola il percentile della Shannon Entropy corrente
    rispetto a tutta la storia fino a t (percentile expanding).

    Returns
    -------
    pd.Series con valori in [0, 100]
    """
    # rank expanding / expanding().count() → percentile espandente
    expanding_rank = series.expanding().rank(pct=True) * 100
    return expanding_rank
