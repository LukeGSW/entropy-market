"""
export.py — entropy-market
============================
Costruisce il JSON di esportazione dell'analisi entropica, strutturato
per la ricerca di insight e alpha su dati storici.

Sezioni del JSON:
  metadata          → parametri dell'analisi, asset, periodo, configurazione
  time_series       → serie completa giornaliera con tutte le feature entropiche
                      + forward return (NaN per le ultime N barre)
  regime_alpha      → statistiche per regime (Bassa/Media/Alta entropia):
                      n, mean, std, hit_rate, Sharpe per ogni orizzonte
  correlations      → r di Pearson e p-value entropia × forward return
  alpha_signals     → eventi pre-filtrati: low_entropy_entry, extreme_percentile,
                      pe_compression, regime_transitions
  seasonal_entropy  → heatmap mensile (media H per anno × mese)
  statistical_summary → distribuzione H, autocorrelazione, confronto PE vs Shannon

Uso in app.py:
    from src.export import build_entropy_export
    payload = build_entropy_export(result)
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇ Esporta JSON", json_bytes, "entropy_export.json", "application/json")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date

from src.calculations import EntropyResult, compute_percentile_series, FORWARD_PERIODS


# ================================================================
# UTILITY
# ================================================================

def _safe_float(x) -> float | None:
    """Converte in float Python nativo; None se NaN/inf."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 8)
    except (TypeError, ValueError):
        return None


def _sharpe(series: pd.Series) -> float | None:
    """Sharpe annualizzato: (mean / std) × √252."""
    s = series.dropna()
    if len(s) < 5 or s.std() == 0:
        return None
    return _safe_float((s.mean() / s.std()) * np.sqrt(252))


def _hit_rate(series: pd.Series) -> float | None:
    """Percentuale osservazioni con rendimento > 0."""
    s = series.dropna()
    if len(s) == 0:
        return None
    return round(float((s > 0).sum() / len(s) * 100), 2)


def _sortino(series: pd.Series) -> float | None:
    """Sortino ratio: mean / downside_std × √252."""
    s = series.dropna()
    if len(s) < 5:
        return None
    downside = s[s < 0]
    if len(downside) == 0 or downside.std() == 0:
        return None
    return _safe_float((s.mean() / downside.std()) * np.sqrt(252))


# ================================================================
# SEZIONE 1 — TIME SERIES
# ================================================================

def _build_time_series(feat: pd.DataFrame, pct_series: pd.Series) -> list[dict]:
    """
    Restituisce la lista di record giornalieri con tutte le feature.
    Le colonne fwd_* sono incluse ma possono essere None nelle ultime N righe
    (forward return non disponibile perché siamo vicini all'oggi).
    """
    records = []
    fwd_cols = [f"fwd_{p}" for p in FORWARD_PERIODS]

    for dt, row in feat.iterrows():
        rec = {
            "date":           dt.strftime("%Y-%m-%d"),
            "close":          _safe_float(row.get("close")),
            "log_ret":        _safe_float(row.get("log_ret")),
            "shannon_ret":    _safe_float(row.get("shannon_ret")),
            "shannon_vol":    _safe_float(row.get("shannon_vol")),
            "shannon_skew":   _safe_float(row.get("shannon_skew")),
            "perm_entropy":   _safe_float(row.get("perm_entropy")),
            "regime":         str(row.get("regime", "")),
            "percentile":     _safe_float(pct_series.get(dt)),
        }
        for col in fwd_cols:
            rec[col] = _safe_float(row.get(col))
        records.append(rec)

    return records


# ================================================================
# SEZIONE 2 — REGIME ALPHA
# ================================================================

def _build_regime_alpha(feat: pd.DataFrame) -> dict:
    """
    Per ogni regime (Bassa / Media / Alta) e ogni orizzonte calcola:
    n, mean, std, hit_rate, sharpe, sortino, p10, median, p90.
    """
    result: dict[str, dict] = {}
    fwd_periods = list(FORWARD_PERIODS.keys())

    for regime in ["Bassa", "Media", "Alta"]:
        sub = feat[feat["regime"] == regime]
        regime_stats: dict[str, dict] = {
            "_count": int(len(sub)),
            "_pct_of_total": round(float(len(sub) / len(feat) * 100), 2),
            "_shannon_mean": _safe_float(sub["shannon_ret"].mean()),
            "_pe_mean":      _safe_float(sub["perm_entropy"].mean()),
        }
        for periodo in fwd_periods:
            col = f"fwd_{periodo}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            regime_stats[periodo] = {
                "n":        int(len(s)),
                "mean_pct": _safe_float(s.mean()),
                "std_pct":  _safe_float(s.std()),
                "hit_rate": _hit_rate(s),
                "sharpe":   _sharpe(s),
                "sortino":  _sortino(s),
                "p10":      _safe_float(s.quantile(0.10)),
                "median":   _safe_float(s.median()),
                "p90":      _safe_float(s.quantile(0.90)),
                "max_drawdown": _safe_float(s.min()),
            }
        result[regime] = regime_stats

    # Differenziale Bassa vs Alta (alpha spread)
    spread: dict[str, float | None] = {}
    for periodo in fwd_periods:
        bassa = result.get("Bassa", {}).get(periodo, {}).get("mean_pct")
        alta  = result.get("Alta",  {}).get(periodo, {}).get("mean_pct")
        if bassa is not None and alta is not None:
            spread[periodo] = _safe_float(bassa - alta)
        else:
            spread[periodo] = None
    result["_spread_bassa_vs_alta"] = spread

    return result


# ================================================================
# SEZIONE 3 — ALPHA SIGNALS (eventi pre-filtrati)
# ================================================================

def _build_alpha_signals(
    feat: pd.DataFrame,
    pct_series: pd.Series,
    regime_p33: float,
    regime_p67: float,
    pe_p33: float,
    pe_p67: float,
) -> list[dict]:
    """
    Pre-filtra i giorni significativi come segnali operativi:
      LOW_ENTROPY_ENTRY   : entrée in regime Bassa (crossover P33 dal basso)
      HIGH_ENTROPY_ENTRY  : entrée in regime Alta (crossover P67 verso l'alto)
      REGIME_EXIT_LOW     : uscita da regime Bassa → possibile inversione
      EXTREME_PCT_LOW     : percentile < 10 (entropia storicamente bassa)
      EXTREME_PCT_HIGH    : percentile > 90 (entropia storicamente alta)
      PE_COMPRESSION      : PE < pe_p33 (struttura temporale forte)
    """
    signals = []
    prev_regime = None
    fwd_cols = list(FORWARD_PERIODS.keys())

    for dt, row in feat.iterrows():
        regime    = str(row.get("regime", ""))
        h_ret     = float(row.get("shannon_ret", np.nan))
        pe_val    = float(row.get("perm_entropy", np.nan))
        pct_val   = float(pct_series.get(dt, np.nan))

        signal_type = None

        # Transizioni di regime (rilevate dal confronto con il giorno precedente)
        if prev_regime is not None and prev_regime != regime:
            if regime == "Bassa":
                signal_type = "LOW_ENTROPY_ENTRY"
            elif regime == "Alta":
                signal_type = "HIGH_ENTROPY_ENTRY"
            elif prev_regime == "Bassa" and regime in ("Media", "Alta"):
                signal_type = "REGIME_EXIT_LOW"

        # Percentile estremo (override se più rilevante)
        if not np.isnan(pct_val):
            if pct_val < 10:
                signal_type = "EXTREME_PCT_LOW"
            elif pct_val > 90:
                signal_type = "EXTREME_PCT_HIGH"

        # Compressione PE (struttura temporale forte)
        if not np.isnan(pe_val) and pe_val < pe_p33:
            if signal_type is None:
                signal_type = "PE_COMPRESSION"

        if signal_type:
            rec = {
                "date":         dt.strftime("%Y-%m-%d"),
                "signal":       signal_type,
                "regime":       regime,
                "shannon_ret":  _safe_float(h_ret),
                "perm_entropy": _safe_float(pe_val),
                "percentile":   _safe_float(pct_val),
                "regime_p33":   _safe_float(regime_p33),
                "regime_p67":   _safe_float(regime_p67),
            }
            for periodo in fwd_cols:
                rec[f"fwd_{periodo}"] = _safe_float(row.get(f"fwd_{periodo}"))
            signals.append(rec)

        prev_regime = regime

    return signals


# ================================================================
# SEZIONE 4 — SEASONAL ENTROPY (heatmap in formato lista)
# ================================================================

def _build_seasonal_entropy(heatmap_data: pd.DataFrame) -> list[dict]:
    """
    Converte la pivot table (anno × mese) in lista di dict per il JSON.
    Ogni record: {anno, Gen, Feb, ..., Dic, mean_annual, max_month, min_month}
    """
    records = []
    for year, row in heatmap_data.iterrows():
        valid = row.dropna()
        rec = {"anno": int(year)}
        for mese, val in row.items():
            rec[str(mese)] = _safe_float(val)
        rec["mean_annual"] = _safe_float(valid.mean()) if len(valid) > 0 else None
        rec["max_month"]   = str(valid.idxmax()) if len(valid) > 0 else None
        rec["min_month"]   = str(valid.idxmin()) if len(valid) > 0 else None
        records.append(rec)
    return records


# ================================================================
# SEZIONE 5 — STATISTICAL SUMMARY
# ================================================================

def _build_statistical_summary(feat: pd.DataFrame) -> dict:
    """
    Distribuzione di shannon_ret e perm_entropy, autocorrelazione,
    e confronto tra le due misure di entropia.
    """
    h  = feat["shannon_ret"].dropna()
    pe = feat["perm_entropy"].dropna()

    def _dist(s: pd.Series) -> dict:
        return {
            "mean":      _safe_float(s.mean()),
            "std":       _safe_float(s.std()),
            "skewness":  _safe_float(s.skew()),
            "kurtosis":  _safe_float(s.kurt()),
            "p5":        _safe_float(s.quantile(0.05)),
            "p25":       _safe_float(s.quantile(0.25)),
            "median":    _safe_float(s.median()),
            "p75":       _safe_float(s.quantile(0.75)),
            "p95":       _safe_float(s.quantile(0.95)),
            "min":       _safe_float(s.min()),
            "max":       _safe_float(s.max()),
        }

    autocorr_h = {
        f"lag_{lag}": _safe_float(h.autocorr(lag=lag))
        for lag in [1, 5, 21, 63, 126]
    }
    autocorr_pe = {
        f"lag_{lag}": _safe_float(pe.autocorr(lag=lag))
        for lag in [1, 5, 21, 63, 126]
    }

    # Correlazione globale PE vs Shannon
    common = feat[["shannon_ret", "perm_entropy"]].dropna()
    pe_sh_corr = None
    if len(common) > 10:
        pe_sh_corr = _safe_float(common["shannon_ret"].corr(common["perm_entropy"]))

    return {
        "shannon_ret_distribution":  _dist(h),
        "perm_entropy_distribution": _dist(pe),
        "autocorrelation_shannon":   autocorr_h,
        "autocorrelation_pe":        autocorr_pe,
        "pe_vs_shannon_correlation": pe_sh_corr,
        "regime_distribution": {
            regime: int((feat["regime"] == regime).sum())
            for regime in ["Bassa", "Media", "Alta"]
        },
    }


# ================================================================
# SEZIONE 6 — PREDITTIVITÀ (Studi 3-4, statistiche HAC, Shannon + PE)
# ================================================================

def _build_predictivity(result: EntropyResult) -> dict:
    """
    Serializza la predittività point-in-time + HAC per ogni misura di entropia:
      regime_forward : media/hit-rate/p_HAC per regime e orizzonte
      scatter_slope  : pendenza/correlazione, p_HAC e N efficace per orizzonte
    """
    out: dict[str, dict] = {}
    for key, pred in result.predictivity.items():
        reg = pred.get("regime_stats", pd.DataFrame())
        slope = pred.get("slope_stats", pd.DataFrame())

        regime_list = [] if reg.empty else [
            {
                "periodo":          str(r["periodo"]),
                "regime":           str(r["regime"]),
                "n":                int(r["n"]),
                "mean_pct":         _safe_float(r["mean_pct"]),
                "hit_rate":         _safe_float(r["hit_rate"]),
                "t_hac":            _safe_float(r["t_hac"]),
                "p_hac":            _safe_float(r["p_hac"]),
                "significant_5pct": bool(r["sig"]),
            }
            for _, r in reg.iterrows()
        ]
        slope_list = [] if slope.empty else [
            {
                "periodo":      str(r["periodo"]),
                "pearson_r":    _safe_float(r["pearson_r"]),
                "spearman_rho": _safe_float(r["spearman_rho"]),
                "slope":        _safe_float(r["slope"]),
                "p_hac":        _safe_float(r["p_hac"]),
                "n_eff":        _safe_float(r["n_eff"]),
                "n":            int(r["n"]),
            }
            for _, r in slope.iterrows()
        ]
        out[key] = {"regime_forward": regime_list, "scatter_slope": slope_list}
    return out


# ================================================================
# ENTRY POINT
# ================================================================

def build_entropy_export(result: EntropyResult) -> dict:
    """
    Costruisce il dizionario completo da serializzare come JSON.

    Parameters
    ----------
    result : EntropyResult dall'analisi principale (build_features)

    Returns
    -------
    dict serializzabile con json.dumps()
    """
    feat = result.feat
    pct_series = compute_percentile_series(feat["shannon_ret"])

    return {
        "metadata": {
            "export_date":   date.today().isoformat(),
            "study":         "entropy_market",
            "ticker":        result.ticker,
            "ticker_label":  result.ticker_label,
            "start_date":    feat.index[0].strftime("%Y-%m-%d"),
            "end_date":      feat.index[-1].strftime("%Y-%m-%d"),
            "n_observations": result.n_feat,
            "parameters": {
                "shannon_window":  result.shannon_window,
                "shannon_bins":    result.shannon_bins,
                "shannon_bias_correction": "miller_madow",
                "pe_order":        result.pe_order,
                "pe_window":       result.shannon_window,
                "forward_periods": list(FORWARD_PERIODS.keys()),
                "regime_method":   "expanding_tertiles_point_in_time",
                "regime_min_periods": result.regime_min_periods,
                "significance":    "HAC_Newey_West_lag_eq_horizon",
                "regime_p33":      _safe_float(result.regime_p33),
                "regime_p67":      _safe_float(result.regime_p67),
                "pe_p33":          _safe_float(result.pe_p33),
                "pe_p67":          _safe_float(result.pe_p67),
            },
            "current_state": {
                "shannon_ret":    _safe_float(feat["shannon_ret"].iloc[-1]),
                "perm_entropy":   _safe_float(feat["perm_entropy"].iloc[-1]),
                "regime":         str(feat["regime"].iloc[-1]),
                "percentile":     _safe_float(pct_series.iloc[-1]),
            },
        },
        "time_series":       _build_time_series(feat, pct_series),
        "regime_alpha":      _build_regime_alpha(feat),
        "correlations":      {
            periodo: {
                "r":            _safe_float(r),
                "p_value_hac":  _safe_float(p),
                "significant_5pct": (p is not None and p < 0.05),
            }
            for periodo, (r, p) in result.scatter_corr.items()
        },
        "predictivity":      _build_predictivity(result),
        "alpha_signals":     _build_alpha_signals(
            feat, pct_series,
            result.regime_p33, result.regime_p67,
            result.pe_p33, result.pe_p67,
        ),
        "seasonal_entropy":  _build_seasonal_entropy(result.heatmap_data),
        "statistical_summary": _build_statistical_summary(feat),
    }
