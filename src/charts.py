"""
charts.py
=========
Funzioni di visualizzazione per i 6 studi di Analisi dell'Entropia.

Ogni funzione riceve dati già calcolati e restituisce un go.Figure Plotly
pronto per essere renderizzato con st.plotly_chart().

Palette dark theme coerente con Kriterion Quant:
  CYAN     → Shannon Returns (principale)
  ORANGE   → Shannon Volatility
  GREEN    → Shannon Skewness / regime bassa entropia
  PURPLE   → Permutation Entropy
  RED      → allerta alta / regime alta entropia
  YELLOW   → regime media entropia
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

from src.calculations import (
    EntropyResult,
    ALERT_HIGH,
    ALERT_LOW,
    REGIME_CORR_WINDOW,
    FORWARD_PERIODS,
    ENTROPY_MEASURES,
    compute_percentile_series,
)

# Mappa misura → (colonna entropia, colonna regime)
_MEASURE_COLS = {
    "shannon_ret":  ("shannon_ret", "regime"),
    "perm_entropy": ("perm_entropy", "regime_pe"),
}

# ================================================================
# PALETTE COLORI E LAYOUT BASE
# ================================================================

C = {
    "bg":       "#0D0D0D",
    "plot":     "#111111",
    "grid":     "rgba(255,255,255,0.08)",
    "text":     "#E0E0E0",
    "cyan":     "#00BCD4",
    "orange":   "#FF9800",
    "green":    "#4CAF50",
    "red":      "#F44336",
    "yellow":   "#FFC107",
    "purple":   "#AB47BC",
    "white":    "#FFFFFF",
    "grey":     "#9E9E9E",
}

_FONT = dict(family="Inter, Arial, sans-serif", color=C["text"])


def _base_layout(**kwargs) -> dict:
    """Layout Plotly dark theme comune a tutti i grafici."""
    return dict(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["plot"],
        font=_FONT,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        **kwargs,
    )


def _axis(title: str = "", log: bool = False, **kw) -> dict:
    """Asse Plotly con stile dark theme."""
    d = dict(
        title=dict(text=title, font=dict(size=11, color=C["grey"])),
        showgrid=True,
        gridcolor=C["grid"],
        zeroline=False,
        tickfont=dict(color=C["text"], size=10),
        color=C["text"],
        **kw,
    )
    if log:
        d["type"] = "log"
    return d


# ================================================================
# STUDIO 1 — Shannon Entropy: panoramica
# ================================================================

def build_shannon_overview(result: EntropyResult) -> go.Figure:
    """
    Studio 1 — 4 subplot impilati (shared x-axis):
      - Row 1: Prezzo (scala log)
      - Row 2: Shannon Entropy dei log-returns
      - Row 3: Shannon Entropy della volatilità rolling
      - Row 4: Shannon Entropy dello skewness rolling
    """
    feat = result.feat
    label = result.ticker_label

    subplot_titles = [
        f"{label} — Prezzo (scala log)",
        "Entropia Shannon — Returns",
        "Entropia Shannon — Volatilità",
        "Entropia Shannon — Skewness",
    ]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        row_heights=[0.35, 0.22, 0.22, 0.21],
    )

    # ── Row 1: Prezzo ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["close"],
        name="Prezzo",
        line=dict(color=C["cyan"], width=1.2),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Prezzo: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # ── Row 2: Shannon Returns ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["shannon_ret"],
        name="Ent Returns",
        line=dict(color=C["cyan"], width=1.0),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>H(ret): %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # ── Row 3: Shannon Volatility ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["shannon_vol"],
        name="Ent Volatility",
        line=dict(color=C["orange"], width=1.0),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>H(vol): %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    # ── Row 4: Shannon Skewness ─────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["shannon_skew"],
        name="Ent Skewness",
        line=dict(color=C["green"], width=1.0),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>H(skew): %{y:.3f}<extra></extra>",
    ), row=4, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Studio 1 — Entropia di Shannon: panoramica",
                font=dict(size=15, color=C["text"]),
            ),
            height=820,
        )
    )

    # Asse y1 in scala log
    fig.update_yaxes(_axis("Prezzo", log=True), row=1, col=1)
    fig.update_yaxes(_axis("H (bit)"), row=2, col=1)
    fig.update_yaxes(_axis("H (bit)"), row=3, col=1)
    fig.update_yaxes(_axis("H (bit)"), row=4, col=1)
    fig.update_xaxes(_axis(), row=4, col=1)

    # Titoli subplot: stile coerente
    for ann in fig.layout.annotations:
        ann.font.color = C["grey"]
        ann.font.size = 11

    return fig


# ================================================================
# STUDIO 2 — Permutation Entropy
# ================================================================

def build_perm_entropy_chart(result: EntropyResult) -> go.Figure:
    """
    Studio 2 — 2 subplot:
      - Row 1: Permutation Entropy (viola) con linee tratteggiate a P33 e P67
      - Row 2: Prezzo (scala log)
    """
    feat = result.feat
    label = result.ticker_label

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Permutation Entropy (normalizzata 0–1)",
            "Prezzo (log)",
        ],
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
    )

    # ── Permutation Entropy ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["perm_entropy"],
        name="Perm. Entropy",
        line=dict(color=C["purple"], width=1.1),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>PE: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Linee orizzontali P33 e P67
    for val, color, tag in [
        (result.pe_p33, C["green"],  f"P33={result.pe_p33:.4f}"),
        (result.pe_p67, C["orange"], f"P67={result.pe_p67:.4f}"),
    ]:
        fig.add_hline(
            y=val, line_dash="dash", line_color=color, line_width=1.2,
            annotation_text=tag,
            annotation_font_color=color,
            annotation_position="top right",
            row=1, col=1,
        )

    # ── Prezzo ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["close"],
        name="Prezzo",
        line=dict(color=C["cyan"], width=1.0),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Prezzo: %{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Studio 2 — Permutation Entropy: complessità locale dei rendimenti",
                font=dict(size=15, color=C["text"]),
            ),
            height=600,
        )
    )

    fig.update_yaxes(_axis("PE (0–1)"), row=1, col=1)
    fig.update_yaxes(_axis("Prezzo", log=True), row=2, col=1)
    fig.update_xaxes(_axis(), row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = C["grey"]
        ann.font.size = 11

    return fig


# ================================================================
# STUDIO 3 — Regime → Forward Returns
# ================================================================

def build_regime_bar_chart(result: EntropyResult, entropy_key: str = "shannon_ret") -> go.Figure:
    """
    Studio 3 — Bar chart raggruppato: forward return medio per regime e orizzonte,
    con test di significatività HAC (media ≠ 0) e hit-rate.

    Le barre con `★` hanno media statisticamente ≠ 0 al 5% (p-value Newey-West,
    lag = orizzonte, che corregge la sovrapposizione dei forward returns).
    Il testo sopra ogni barra mostra il return medio; l'hover aggiunge hit-rate,
    N e p_HAC.
    """
    reg = result.predictivity.get(entropy_key, {}).get("regime_stats", pd.DataFrame())
    label = ENTROPY_MEASURES.get(entropy_key, entropy_key)
    colori = {"Bassa": C["green"], "Media": C["yellow"], "Alta": C["red"]}
    periodi = list(FORWARD_PERIODS.keys())

    fig = go.Figure()

    for regime in ["Bassa", "Media", "Alta"]:
        means, texts, hits, ps, ns = [], [], [], [], []
        for p in periodi:
            row = reg[(reg["regime"] == regime) & (reg["periodo"] == p)] if not reg.empty else reg
            if not reg.empty and len(row):
                m = float(row["mean_pct"].iloc[0])
                sig = bool(row["sig"].iloc[0])
                means.append(m)
                texts.append(f"{m:.2f}%{' ★' if sig else ''}")
                hits.append(float(row["hit_rate"].iloc[0]))
                ps.append(float(row["p_hac"].iloc[0]))
                ns.append(int(row["n"].iloc[0]))
            else:
                means.append(0.0); texts.append(""); hits.append(np.nan); ps.append(np.nan); ns.append(0)

        customdata = np.column_stack([hits, ps, ns])
        fig.add_trace(go.Bar(
            name=f"Entropia {regime}",
            x=periodi,
            y=means,
            marker_color=colori[regime],
            text=texts,
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                f"Regime: {regime}<br>Orizzonte: %{{x}}<br>"
                "Return medio: %{y:.2f}%<br>"
                "Hit-rate: %{customdata[0]:.1f}%<br>"
                "p_HAC: %{customdata[1]:.3f}  ·  N: %{customdata[2]:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Studio 3 — {label}: Regime → Forward Returns (★ = significativo 5% HAC)",
                font=dict(size=15, color=C["text"]),
            ),
            barmode="group",
            height=480,
            xaxis=_axis("Orizzonte"),
            yaxis=_axis("Forward Return medio (%)"),
        )
    )
    fig.add_hline(y=0, line_color=C["grey"], line_width=0.8)
    return fig


# ================================================================
# STUDIO 4 — Scatter Entropia × Forward Returns
# ================================================================

def build_scatter_grid(result: EntropyResult, entropy_key: str = "shannon_ret") -> go.Figure:
    """
    Studio 4 — 2×2 scatter (1M/3M/6M/12M): entropia corrente vs forward return.

    Su ogni pannello:
      • punti grezzi colorati per decennio (bassa opacità),
      • curva di predittività binnata (media per decile ± SE), in bianco,
      • smoother kernel (ciano) per la forma non-lineare,
      • annotazione con Pearson r, Spearman ρ, p-value HAC e N efficace.
    """
    entropy_col, _ = _MEASURE_COLS[entropy_key]
    label = ENTROPY_MEASURES.get(entropy_key, entropy_key)
    pred = result.predictivity.get(entropy_key, {})
    slope_stats = pred.get("slope_stats", pd.DataFrame())
    binned = pred.get("binned", {})
    smoother = pred.get("smoother", {})
    feat = result.feat
    periodi = list(FORWARD_PERIODS.keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Entropia → Forward Return {p}" for p in periodi],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    decade_colors = {
        1950: "#4FC3F7", 1960: "#4DB6AC", 1970: "#81C784",
        1980: "#FFD54F", 1990: "#FF8A65", 2000: "#F06292",
        2010: "#BA68C8", 2020: "#E0E0E0",
    }
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (periodo, (row, col)) in enumerate(zip(periodi, positions)):
        col_fwd = f"fwd_{periodo}"
        sub = feat[[entropy_col, col_fwd, "anno"]].dropna()
        sub["decade"] = (sub["anno"] // 10 * 10).astype(int)

        # Punti grezzi per decennio (bassa opacità → leggibilità)
        for decade, grp in sub.groupby("decade"):
            color = decade_colors.get(decade, C["grey"])
            fig.add_trace(go.Scatter(
                x=grp[entropy_col], y=grp[col_fwd],
                mode="markers",
                name=str(decade),
                marker=dict(color=color, size=3, opacity=0.28),
                showlegend=(idx == 0),
                legendgroup=str(decade),
                hovertemplate=(
                    "Anno: %{customdata}<br>"
                    "Entropia: %{x:.3f}<br>"
                    f"Fwd {periodo}: %{{y:.2f}}%<extra></extra>"
                ),
                customdata=sub.loc[grp.index, "anno"],
            ), row=row, col=col)

        # Smoother kernel
        if periodo in smoother:
            xs, ys = smoother[periodo]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name="Smoother",
                line=dict(color=C["cyan"], width=2.2),
                showlegend=(idx == 0), legendgroup="smoother",
                hovertemplate="Entropia: %{x:.3f}<br>Fwd medio: %{y:.2f}%<extra></extra>",
            ), row=row, col=col)

        # Curva binnata (media per decile ± SE)
        bdf = binned.get(periodo, pd.DataFrame())
        if not bdf.empty:
            fig.add_trace(go.Scatter(
                x=bdf["bin_center"], y=bdf["mean"],
                mode="markers+lines",
                name="Media per decile ± SE",
                line=dict(color=C["white"], width=1.4),
                marker=dict(color=C["white"], size=6, symbol="diamond"),
                error_y=dict(type="data", array=bdf["se"], color=C["white"],
                             thickness=1.0, width=2),
                showlegend=(idx == 0), legendgroup="binned",
                customdata=np.column_stack([bdf["hit_rate"], bdf["n"]]),
                hovertemplate=(
                    "Entropia≈%{x:.3f}<br>Fwd medio: %{y:.2f}%<br>"
                    "Hit-rate: %{customdata[0]:.1f}%  ·  N: %{customdata[1]:,}<extra></extra>"
                ),
            ), row=row, col=col)

        # Linea dello zero (rendimento nullo)
        fig.add_hline(y=0, line_color=C["grey"], line_width=0.6, line_dash="dot",
                      row=row, col=col)

        # Annotazione statistica (HAC)
        srow = slope_stats[slope_stats["periodo"] == periodo] if not slope_stats.empty else slope_stats
        if not slope_stats.empty and len(srow):
            r_val = float(srow["pearson_r"].iloc[0])
            rho   = float(srow["spearman_rho"].iloc[0])
            p_hac = float(srow["p_hac"].iloc[0])
            n_eff = float(srow["n_eff"].iloc[0])
            p_str = f"{p_hac:.3f}" if p_hac >= 0.001 else "&lt;0.001"
            axn = idx + 1
            xref = "x domain" if axn == 1 else f"x{axn} domain"
            yref = "y domain" if axn == 1 else f"y{axn} domain"
            fig.add_annotation(
                xref=xref, yref=yref,
                x=0.03, y=0.97, xanchor="left", yanchor="top",
                text=(f"r={r_val:.2f}  ρ={rho:.2f}<br>"
                      f"p_HAC={p_str}  N_eff={n_eff:.0f}"),
                showarrow=False, align="left",
                font=dict(size=10, color=C["white"]),
                bgcolor="rgba(0,0,0,0.45)", bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1, borderpad=3,
            )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Studio 4 — {label} vs Forward Returns (curva binnata + smoother, p HAC)",
                font=dict(size=15, color=C["text"]),
            ),
            height=760,
        )
    )

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(_axis("Entropia"), row=r, col=c)
            fig.update_yaxes(_axis("Forward Return (%)"), row=r, col=c)

    for ann in fig.layout.annotations:
        # solo i titoli dei subplot (gli altri hanno già font impostato)
        if ann.text.startswith("Entropia →"):
            ann.font.color = C["grey"]
            ann.font.size = 11

    return fig


# ================================================================
# STUDIO 5 — Percentile storico Entropia
# ================================================================

def build_percentile_chart(result: EntropyResult) -> go.Figure:
    """
    Studio 5 — 2 subplot:
      - Row 1: Percentile storico espandente di Shannon Entropy (0-100)
               con zone di allerta a p=20 (verde) e p=80 (rossa)
      - Row 2: Prezzo (scala log)
    """
    feat = result.feat
    label = result.ticker_label

    pct_series = compute_percentile_series(feat["shannon_ret"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Percentile Storico Entropia Returns",
            "Prezzo (log)",
        ],
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    # ── Percentile ──────────────────────────────────────────────
    # Fill: zona allerta alta (>80) = rosso, zona allerta bassa (<20) = verde
    fig.add_trace(go.Scatter(
        x=feat.index, y=pct_series,
        name="Percentile Entropia",
        line=dict(color=C["cyan"], width=1.1),
        fill="tozeroy",
        fillcolor="rgba(0,188,212,0.06)",
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Percentile: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # Linea di allerta alta
    fig.add_hline(
        y=ALERT_HIGH, line_dash="dash", line_color=C["red"], line_width=1.5,
        annotation_text=f"Allerta Alta (p={ALERT_HIGH})",
        annotation_font_color=C["red"],
        annotation_position="top right",
        row=1, col=1,
    )

    # Linea di allerta bassa
    fig.add_hline(
        y=ALERT_LOW, line_dash="dash", line_color=C["green"], line_width=1.5,
        annotation_text=f"Zona Calma (p={ALERT_LOW})",
        annotation_font_color=C["green"],
        annotation_position="bottom right",
        row=1, col=1,
    )

    # ── Prezzo ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=feat["close"],
        name="Prezzo",
        line=dict(color=C["orange"], width=1.0),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Prezzo: %{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Studio 5 — Percentile storico Entropia: zone di allerta",
                font=dict(size=15, color=C["text"]),
            ),
            height=600,
        )
    )

    fig.update_yaxes(_axis("Percentile (0–100)"), row=1, col=1)
    fig.update_yaxes(_axis("Prezzo", log=True), row=2, col=1)
    fig.update_xaxes(_axis(), row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = C["grey"]
        ann.font.size = 11

    return fig


# ================================================================
# STUDIO 6 — Cross-entropia + Heatmap stagionale
# ================================================================

def build_cross_entropy_heatmap(result: EntropyResult) -> go.Figure:
    """
    Studio 6 — 2 subplot:
      - Row 1: Correlazione rolling (6M = 126g) tra PE e Shannon Returns
      - Row 2: Heatmap della Shannon Entropy media per (anno × mese)
    """
    feat   = result.feat
    hmap   = result.heatmap_data
    xcorr  = result.cross_corr

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Correlazione rolling {REGIME_CORR_WINDOW}g: Perm. Entropy ↔ Shannon Entropy",
            "Heatmap Entropia Shannon Returns (media mensile per anno)",
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.65],
    )

    # ── Cross-correlazione rolling ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=feat.index, y=xcorr,
        name="Corr rolling 6M",
        line=dict(color=C["purple"], width=1.1),
        fill="tozeroy",
        fillcolor="rgba(171,71,188,0.12)",
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Correlazione: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_color=C["grey"], line_width=0.8, row=1, col=1)

    # ── Heatmap anno × mese ─────────────────────────────────────
    years   = list(hmap.index)
    months  = list(hmap.columns)
    z_vals  = hmap.values

    fig.add_trace(go.Heatmap(
        x=months,
        y=years,
        z=z_vals,
        colorscale="RdYlGn_r",   # verde = bassa entropia (più ordine), rosso = alta
        colorbar=dict(
            title=dict(text="H (bit)", font=dict(color=C["grey"], size=10)),
            tickfont=dict(color=C["text"], size=9),
        ),
        hovertemplate="Anno: %{y}<br>Mese: %{x}<br>H̄: %{z:.3f}<extra></extra>",
        name="H Shannon",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Studio 6 — Entropia incrociata + Heatmap stagionale",
                font=dict(size=15, color=C["text"]),
            ),
            height=700,
        )
    )

    fig.update_yaxes(_axis("Correlazione"), row=1, col=1)
    fig.update_xaxes(_axis(), row=1, col=1)
    fig.update_yaxes(_axis("Anno"), row=2, col=1)
    fig.update_xaxes(_axis("Mese"), row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = C["grey"]
        ann.font.size = 11

    return fig
