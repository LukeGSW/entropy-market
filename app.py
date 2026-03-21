"""
app.py — Analisi dell'Entropia dei Mercati Finanziari
======================================================
Dashboard Streamlit per lo studio della complessità informazionale dei mercati.

Autore  : Kriterion Quant
Versione: 1.0.0
Fonte   : EODHD Historical Data API

Deployment:
  streamlit run app.py
  Secrets richiesti: EODHD_API_KEY
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from src.data_fetcher import (
    TICKER_MAP,
    fetch_ohlcv,
    get_label,
    resolve_ticker,
)
from src.calculations import (
    SHANNON_WINDOW,
    PE_ORDER,
    FORWARD_PERIODS,
    ALERT_HIGH,
    ALERT_LOW,
    build_features,
    compute_percentile_series,
)
from src.charts import (
    build_shannon_overview,
    build_perm_entropy_chart,
    build_regime_bar_chart,
    build_scatter_grid,
    build_percentile_chart,
    build_cross_entropy_heatmap,
)
from src.export import build_entropy_export

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="Analisi Entropia | Kriterion Quant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HEADER
# ============================================================
st.title("🔬 Analisi dell'Entropia — Studi Quantitativi")
st.markdown(
    "**Entropia di Shannon · Permutation Entropy · Regime di mercato · "
    "Scatter Entropia/Volatilità · Percentile storico · Heatmap stagionale**"
)

st.markdown("""
L'**entropia** misura il grado di disordine, incertezza e complessità di una serie temporale.
Applicata ai mercati finanziari, permette di quantificare quanto i prezzi siano
**prevedibili** (bassa entropia) o **caotici** (alta entropia) in un determinato periodo.

> **Come si usa:** seleziona un asset, imposta la data di partenza e clicca *Esegui Analisi*.
> Il calcolo di tutte le misure richiede alcuni secondi.
""")
st.divider()

# ============================================================
# SIDEBAR — PARAMETRI
# ============================================================
with st.sidebar:
    st.header("⚙️ Parametri Analisi")

    # Selezione ticker rapida
    st.markdown("**Scorciatoie:**")
    shortcuts = list(TICKER_MAP.keys())
    cols = st.columns(4)
    selected_shortcut: str | None = None
    for i, s in enumerate(shortcuts):
        if cols[i % 4].button(s, key=f"btn_{s}", width="stretch"):
            selected_shortcut = s

    # Input manuale del ticker
    st.markdown("---")
    manual_ticker = st.text_input(
        "Ticker EODHD (es. AAPL.US, GSPC.INDX)",
        value="",
        placeholder="GSPC.INDX",
        help="Formato: SYMBOL.EXCHANGE  —  es. AAPL.US, MSFT.US, GSPC.INDX",
    )

    # Risoluzione ticker finale
    if selected_shortcut:
        eodhd_ticker = resolve_ticker(selected_shortcut)
        st.success(f"Ticker selezionato: **{eodhd_ticker}**")
    elif manual_ticker.strip():
        eodhd_ticker = manual_ticker.strip().upper()
        st.success(f"Ticker: **{eodhd_ticker}**")
    else:
        eodhd_ticker = "GSPC.INDX"
        st.info("Nessun ticker selezionato → default: **GSPC.INDX** (S&P 500)")

    st.divider()

    # Data di partenza
    start_date = st.date_input(
        "Data di partenza",
        value=pd.Timestamp("1950-01-01"),
        min_value=pd.Timestamp("1900-01-01"),
        help="I dati verranno filtrati a partire da questa data.",
    )

    st.divider()

    # Parametri avanzati
    with st.expander("🔧 Parametri avanzati"):
        shannon_window = st.slider(
            "Finestra Shannon Entropy (giorni)",
            min_value=21, max_value=252, value=SHANNON_WINDOW, step=1,
            help="Finestra rolling per il calcolo della Shannon Entropy. "
                 "Default: 63g ≈ 1 trimestre.",
        )
        pe_order = st.select_slider(
            "Ordine Permutation Entropy (m)",
            options=[3, 4, 5, 6],
            value=PE_ORDER,
            help="Embedding dimension per la Permutation Entropy. "
                 "Valori più alti catturano pattern più lunghi ma richiedono più dati.",
        )

    st.divider()
    st.caption("📊 Dati: EODHD Historical Data API")
    st.caption("🏛️ Kriterion Quant — Finanza Quantitativa")

    run_analysis = st.button(
        "▶ Esegui Analisi",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# ESECUZIONE ANALISI
# ============================================================
if not run_analysis:
    st.info(
        "👈 Seleziona un asset nella sidebar e clicca **Esegui Analisi** per avviare lo studio."
    )
    st.stop()

ticker_label = get_label(eodhd_ticker)

# ── Fetch dati ───────────────────────────────────────────────
# EODHD tronca le risposte a ~19.000 record: non passiamo from_date all'API
# ma scarichiamo l'intera storia e filtriamo lato Python.
# to_date è esplicito nella firma (chiave cache) → dati sempre aggiornati.
today_str = pd.Timestamp.today().strftime("%Y-%m-%d")

with st.spinner(f"Scaricamento dati per **{eodhd_ticker}** da EODHD..."):
    try:
        df_full = fetch_ohlcv(
            ticker=eodhd_ticker,
            to_date=today_str,
        )
    except Exception as e:
        st.error(f"❌ Errore nel download dei dati: {e}")
        st.stop()

if df_full.empty:
    st.error(
        f"Nessun dato trovato per **{eodhd_ticker}**. "
        "Verifica il ticker (formato: SYMBOL.EXCHANGE) e le date."
    )
    st.stop()

# Filtro per start_date scelto dall'utente (lato Python, non API)
df_raw = df_full[df_full.index >= pd.Timestamp(start_date)]

if df_raw.empty:
    st.error(
        f"Nessun dato trovato per **{eodhd_ticker}** a partire dal {start_date}. "
        "Prova una data di partenza più recente."
    )
    st.stop()

# ── Calcolo feature ───────────────────────────────────────────
log_placeholder = st.empty()
log_lines: list[str] = []

with st.spinner("Calcolo feature di entropia in corso..."):
    result = build_features(
        df=df_raw,
        ticker=eodhd_ticker,
        ticker_label=ticker_label,
        shannon_window=shannon_window,
        pe_order=pe_order,
        log_lines=log_lines,
    )

# ── Log di esecuzione ────────────────────────────────────────
with log_placeholder.container():
    with st.expander("📋 Log di esecuzione", expanded=False):
        for line in result.log_lines:
            st.markdown(f"- {line}")

st.divider()

# ============================================================
# KPI SECTION
# ============================================================
st.subheader("📊 Metriche chiave")

feat = result.feat
pct_series = compute_percentile_series(feat["shannon_ret"])
current_pct    = float(pct_series.iloc[-1])
current_sh_ret = float(feat["shannon_ret"].iloc[-1])
current_pe     = float(feat["perm_entropy"].iloc[-1])
current_regime = str(feat["regime"].iloc[-1])

regime_color_map = {"Bassa": "🟢", "Media": "🟡", "Alta": "🔴"}
regime_icon = regime_color_map.get(current_regime, "⚪")

r1, r2, r3, r4, r5, r6 = st.columns(6)

r1.metric(
    "Barre analizzate",
    f"{result.n_feat:,}",
    help=f"Righe con feature complete. Totale grezzo: {result.n_raw:,}",
)
r2.metric(
    "Shannon Entropy (attuale)",
    f"{current_sh_ret:.3f} bit",
    help=f"Valore corrente della Shannon Entropy dei log-returns (finestra {shannon_window}g)",
)
r3.metric(
    "Perm. Entropy (attuale)",
    f"{current_pe:.4f}",
    help=f"Permutation Entropy normalizzata [0,1] (ordine m={pe_order})",
)
r4.metric(
    "Percentile storico",
    f"{current_pct:.1f}°",
    help=f"Posizione percentuale dell'entropia corrente nella distribuzione storica espandente",
)
r5.metric(
    "Regime attuale",
    f"{regime_icon} {current_regime}",
    help="Regime determinato dai tertili (P33/P67) della Shannon Entropy storica",
)
r6.metric(
    "Periodo analizzato",
    f"{feat.index[0].year}–{feat.index[-1].year}",
    help=f"Da {feat.index[0].date()} a {feat.index[-1].date()}",
)

st.divider()

# ============================================================
# SEZIONE 1 — STUDIO 1: Shannon Entropy Panoramica
# ============================================================
st.subheader("📈 Studio 1 — Entropia di Shannon: panoramica")

st.markdown("""
La **Shannon Entropy** misura la quantità di informazione (o incertezza) in una distribuzione.
Applicata a una finestra rolling di rendimenti finanziari, risponde alla domanda:
*quanto sono imprevedibili i movimenti di prezzo nel periodo corrente?*

**Come si calcola:**
> In una finestra di N giorni, i log-return vengono discretizzati in bin.
> Per ogni bin si calcola la probabilità empirica p_i.
> L'entropia è H = −Σ p_i · log₂(p_i) (misurata in **bit**).

- **H alta** → rendimenti distribuiti uniformemente → mercato imprevedibile, caotico
- **H bassa** → rendimenti concentrati → mercato con struttura, potenzialmente più prevedibile

I 4 pannelli mostrano l'entropia applicata a tre diverse serie:
**Returns** (distribuzione dei rendimenti), **Volatilità rolling** e **Skewness rolling**.
Queste catturano diverse dimensioni dell'incertezza di mercato.

> 📌 Nota: si usano 3 misure perché la sola entropia dei returns può essere stabile
> mentre la volatilità o l'asimmetria mostrano segnali di regime nascosti.
""")

fig1 = build_shannon_overview(result)
st.plotly_chart(fig1, width="stretch")
st.divider()

# ============================================================
# SEZIONE 2 — STUDIO 2: Permutation Entropy
# ============================================================
st.subheader("🔀 Studio 2 — Permutation Entropy: complessità locale")

st.markdown(f"""
La **Permutation Entropy (PE)**, introdotta da Bandt & Pompe (2002), misura la complessità
di una serie temporale attraverso i **pattern ordinali** — cioè l'ordine relativo dei valori
in una finestra di m osservazioni consecutive.

**Come si calcola (ordine m = {pe_order}):**
> Per ogni finestra di {pe_order} valori consecutivi si identifica il pattern di ordinamento
> (es. "crescente-decrescente-crescente").
> Ci sono {pe_order}! = {__import__("math").factorial(pe_order)} pattern possibili.
> La PE è l'entropia di Shannon di questi pattern, **normalizzata in [0, 1]**.

- **PE ≈ 1** → tutti i pattern sono equiprobabili → alta complessità, mercato efficiente
- **PE << 1** → alcuni pattern dominano → struttura temporale, possibile prevedibilità

Le linee tratteggiate indicano **P33 = {result.pe_p33:.4f}** e **P67 = {result.pe_p67:.4f}**,
che definiscono le soglie dei tre regimi (basso / medio / alto).

> 📌 La Permutation Entropy è robusta al rumore e non richiede assunzioni distributive.
> Per i mercati azionari maturi (es. S&P 500), la PE è tipicamente ≥ 0.96,
> con i minimi corrispondenti a crisi o forti trend direzionali.
""")

fig2 = build_perm_entropy_chart(result)
st.plotly_chart(fig2, width="stretch")
st.divider()

# ============================================================
# SEZIONE 3 — STUDIO 3: Regime → Forward Returns
# ============================================================
st.subheader("📊 Studio 3 — Regime Entropia → Forward Returns medi")

st.markdown(f"""
Questo studio verifica se esiste una **relazione sistematica** tra il regime di entropia
corrente e i rendimenti futuri del mercato.

**Metodologia:**
> Si suddivide la storia in 3 regime usando i **tertili** (P33/P67) della Shannon Entropy:
> - 🟢 **Bassa entropia** (H ≤ {result.regime_p33:.3f}): mercato con struttura, rendimenti concentrati
> - 🟡 **Media entropia** ({result.regime_p33:.3f} < H ≤ {result.regime_p67:.3f}): stato neutro
> - 🔴 **Alta entropia** (H > {result.regime_p67:.3f}): mercato caotico/incerto
>
> Per ciascun regime si calcola il forward return medio a 1M / 3M / 6M / 12M.

**Interpretazione operativa:**
- Se la barra verde (bassa entropia) è sistematicamente più alta delle altre,
  la bassa entropia è un segnale **rialzista** prospettico.
- Questo può essere usato come **filtro di regime** in strategie sistematiche.

> ⚠️ I risultati sono descrittivi e basati sulla storia: le performance passate
> non garantiscono risultati futuri. Sempre verificare la robustezza statistica
> e le condizioni di mercato attuali.
""")

fig3 = build_regime_bar_chart(result)
st.plotly_chart(fig3, width="stretch")

# Tabella riepilogativa
with st.expander("📋 Tabella completa Forward Returns per Regime"):
    pivot = result.regime_fwd.pivot(index="regime", columns="periodo", values="fwd_mean")
    pivot = pivot.reindex(["Bassa", "Media", "Alta"])[["1M", "3M", "6M", "12M"]]

    def _color_val(val):
        if val > 0.5:
            return "background-color: #1A5C1A; color: #C0F0C0"
        elif val > 0:
            return "background-color: #5D7A1A; color: #E0F0B0"
        elif val > -0.5:
            return "background-color: #B35C00; color: #FFE0B0"
        return "background-color: #8B1A1A; color: #FFD0D0"

    st.dataframe(
        pivot.style
        .format("{:.2f}%")
        .map(_color_val),
        use_container_width=False,
    )

st.divider()

# ============================================================
# SEZIONE 4 — STUDIO 4: Scatter Entropia × Forward Returns
# ============================================================
st.subheader("🔍 Studio 4 — Scatter Entropia vs Forward Returns")

st.markdown("""
Lo scatter plot mostra la **relazione lineare** tra il livello di entropia corrente
e il forward return realizzato. Ogni punto rappresenta un giorno di trading,
colorato in base al **decennio** di appartenenza.

**Correlazioni di Pearson (r) e p-value:**
- **r negativo** → alta entropia associata a minori rendimenti futuri (relazione inversa)
- **p < 0.05** → correlazione statisticamente significativa al 95%

""")

corr_cols = st.columns(4)
for i, (periodo, (r, p)) in enumerate(result.scatter_corr.items()):
    p_str = f"{p:.3f}" if p >= 0.001 else "< 0.001"
    delta_color = "inverse" if r < 0 else "normal"
    corr_cols[i].metric(
        label=f"r ({periodo})",
        value=f"{r:.3f}",
        delta=f"p = {p_str}",
        delta_color="off",
        help=f"Correlazione di Pearson tra Shannon Entropy e Forward Return {periodo}",
    )

st.markdown("""
> 📌 Una correlazione negativa sistematica tra entropia e rendimenti futuri
> è coerente con la teoria dell'**efficienza informazionale adattiva** (AMH, Lo 2004):
> nei periodi di alta incertezza (alta entropia), il mercato tende ad avere
> rendimenti attesi inferiori perché il rischio è più difficile da prezzare.

Le **linee di regressione** mostrano il trend lineare per ciascun orizzonte.
I **punti di quadrante** (mediana × mediana) suddividono il piano in 4 zone operative:
- 🟢 Bassa ent / Alto ret — zona favorevole
- 🔴 Alta ent / Basso ret — zona sfavorevole
""")

fig4 = build_scatter_grid(result)
st.plotly_chart(fig4, width="stretch")
st.divider()

# ============================================================
# SEZIONE 5 — STUDIO 5: Percentile storico
# ============================================================
st.subheader("📉 Studio 5 — Percentile storico Entropia: zone di allerta")

st.markdown(f"""
Il **percentile storico espandente** risponde alla domanda:
*rispetto a tutta la storia disponibile fino ad oggi, quanto è alta l'entropia attuale?*

A differenza di una semplice comparazione con la media, il percentile permette di capire
se siamo in una **fase estrema** (coda della distribuzione) o in una situazione normale.

**Zone di allerta:**
- 🔴 **Percentile > {ALERT_HIGH}** → entropia storicamente alta → incertezza elevata,
  cautela nelle posizioni direzionali
- ⚪ **Percentile {ALERT_LOW}–{ALERT_HIGH}** → zona neutrale
- 🟢 **Percentile < {ALERT_LOW}** → entropia storicamente bassa → mercato più ordinato,
  condizioni favorevoli per strategie trend-following

**Percentile attuale: {float(compute_percentile_series(feat["shannon_ret"]).iloc[-1]):.1f}°**

> 📌 Il percentile storico è uno strumento di **risk overlay**: può essere combinato
> con qualsiasi strategia sistematica per aumentare o ridurre l'esposizione al rischio
> in funzione del regime di complessità del mercato.
""")

fig5 = build_percentile_chart(result)
st.plotly_chart(fig5, width="stretch")
st.divider()

# ============================================================
# SEZIONE 6 — STUDIO 6: Cross-entropia + Heatmap stagionale
# ============================================================
st.subheader("🌡️ Studio 6 — Entropia incrociata + Heatmap stagionale")

st.markdown("""
Lo Studio 6 combina due analisi complementari:

**Correlazione rolling tra Permutation Entropy e Shannon Entropy:**
> Una correlazione alta (vicina a 1) indica che le due misure di complessità
> concordano: il mercato è uniformemente prevedibile o imprevedibile.
> Una correlazione bassa o negativa segnala **discordanza** tra le due misure,
> che può indicare transizioni di regime o anomalie strutturali.

**Heatmap stagionale:**
> La heatmap mostra la Shannon Entropy **media mensile** per ogni anno della storia.
> Permette di identificare:
> - **Pattern stagionali**: mesi storicamente più "caotici" (rosso) o più ordinati (verde)
> - **Anomalie storiche**: anni in cui l'intera curva stagionale era atipica
>   (es. 2020, 2008–2009)

> 📌 Tradizionalmente settembre e ottobre sono mesi ad alta incertezza per i mercati
> azionari. La heatmap permette di verificare se questo pattern è presente
> anche nella serie di entropia.
""")

fig6 = build_cross_entropy_heatmap(result)
st.plotly_chart(fig6, width="stretch")
st.divider()

# ============================================================
# SEZIONE 7 — EXPANDER METODOLOGIA
# ============================================================
with st.expander("🔬 Metodologia, Fondamenti Teorici e Riferimenti"):
    st.markdown(f"""
    ## Metodologia

    ### Dati
    - **Fonte:** EODHD Historical Data API (`eodhd.com`)
    - **Ticker usato:** `{eodhd_ticker}` — {ticker_label}
    - **Periodo:** `{feat.index[0].date()}` → `{feat.index[-1].date()}`
    - **Prezzo di riferimento:** `adjusted_close` (corretto per dividendi e split)
    - **Barre grezze:** `{result.n_raw:,}` | **Feature complete:** `{result.n_feat:,}`

    ### Pipeline di calcolo

    | Step | Formula | Note |
    |------|---------|------|
    | Log-return | `r_t = ln(P_t / P_{{t-1}})` | Additivi e simmetrici |
    | Volatilità rolling | `σ_t = std(r_{{t-20}}, ..., r_t)` | Finestra 21g ≈ 1 mese |
    | Skewness rolling | `skew_t` rolling 63g | Asimmetria distribuzione locale |
    | Shannon Entropy | `H(t) = −Σ p_i·log₂(p_i)` | Finestra {shannon_window}g, {10} bin |
    | Permutation Entropy | `PE(t) = H_perm / log₂(m!)` | m={pe_order}, finestra {shannon_window}g, normalizzata [0,1] |
    | Forward return | `fwd_N(t) = Σ r_{{t+1..t+N}}` | N ∈ {{21, 63, 126, 252}} |
    | Percentile | `pct(t) = rank(H_t) / rank_max` | Expanding (solo storia passata) |
    | Regime | Tertili P33={result.regime_p33:.4f}, P67={result.regime_p67:.4f} | Bassa/Media/Alta |

    ### Shannon Entropy

    Introdotta da Claude Shannon (1948), la Shannon Entropy misura il contenuto
    informativo di una variabile casuale discreta:

    **H = −Σᵢ pᵢ · log₂(pᵢ)**

    dove pᵢ è la probabilità dell'i-esimo esito. L'unità di misura è il **bit**.
    Il valore massimo per k bin equamente distribuiti è H_max = log₂(k).

    Applicazione finanziaria: i rendimenti vengono discretizzati in bin,
    e H misura quanto sono "dispersi" — un mercato efficiente mostra H alta.

    ### Permutation Entropy (Bandt & Pompe, 2002)

    La PE considera solo l'**ordine relativo** dei valori consecutivi,
    rendendola robusta al rumore e scale-invariant.

    Per m={pe_order}: ci sono {__import__("math").factorial(pe_order)} pattern ordinali possibili.
    La PE normalizzata è in [0, 1]:
    - **1.0** → tutti i pattern sono equiprobabili → massima complessità
    - **0.0** → un solo pattern domina → serie monotona

    ### Correlazione Scatter (Studio 4)

    | Orizzonte | r di Pearson | p-value | Interpretazione |
    |-----------|-------------|---------|-----------------|
    | 1M  | {result.scatter_corr.get("1M",  (0,0))[0]:.4f} | {result.scatter_corr.get("1M",  (0,1))[1]:.4f} | {'Significativo' if result.scatter_corr.get('1M',(0,1))[1] < 0.05 else 'Non significativo'} al 5% |
    | 3M  | {result.scatter_corr.get("3M",  (0,0))[0]:.4f} | {result.scatter_corr.get("3M",  (0,1))[1]:.4f} | {'Significativo' if result.scatter_corr.get('3M',(0,1))[1] < 0.05 else 'Non significativo'} al 5% |
    | 6M  | {result.scatter_corr.get("6M",  (0,0))[0]:.4f} | {result.scatter_corr.get("6M",  (0,1))[1]:.4f} | {'Significativo' if result.scatter_corr.get('6M',(0,1))[1] < 0.05 else 'Non significativo'} al 5% |
    | 12M | {result.scatter_corr.get("12M", (0,0))[0]:.4f} | {result.scatter_corr.get("12M", (0,1))[1]:.4f} | {'Significativo' if result.scatter_corr.get('12M',(0,1))[1] < 0.05 else 'Non significativo'} al 5% |

    ### Riferimenti Bibliografici

    1. **Shannon, C.E. (1948).** *A Mathematical Theory of Communication.*
       Bell System Technical Journal, 27, 379–423 e 623–656.
    2. **Bandt, C. & Pompe, B. (2002).** *Permutation Entropy: A Natural Complexity
       Measure for Time Series.* Physical Review Letters, 88(17), 174102.
    3. **Lo, A.W. (2004).** *The Adaptive Markets Hypothesis.*
       Journal of Portfolio Management, 30(5), 15–29.
    4. **Peters, O. (2019).** *The ergodicity problem in economics.*
       Nature Physics, 15, 1216–1221.
    5. **Pincus, S.M. (1991).** *Approximate entropy as a measure of system complexity.*
       Proceedings of the National Academy of Sciences, 88(6), 2297–2301.

    ### Limitazioni e avvertenze

    - L'analisi è **descrittiva**, non predittiva: le relazioni osservate nella storia
      possono non replicarsi in futuro.
    - La Shannon Entropy è sensibile alla scelta del numero di bin; la PE è più robusta.
    - Per asset con storia breve (< 5 anni) le stime delle code della distribuzione
      sono poco affidabili.
    - Nessuna delle misure presentate costituisce un segnale operativo autonomo.
      Vanno sempre integrate in un framework di risk management completo.
    """)

# ============================================================
# EXPORT JSON
# ============================================================
st.markdown("---")
st.subheader("⬇ Esporta dati per ricerca alpha")
st.markdown(
    "Scarica il JSON completo con time series, regime alpha, correlazioni, alpha signals e "
    "statistiche stagionali — strutturato per analisi esterne di insight e alpha."
)
_export_payload = build_entropy_export(result)
_json_bytes = json.dumps(_export_payload, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button(
    label="⬇ Scarica JSON Backtest",
    data=_json_bytes,
    file_name=f"entropy_{eodhd_ticker.replace('.', '_')}_{date.today()}.json",
    mime="application/json",
    help="JSON strutturato per ricerca di insight e alpha: time series, regime alpha, segnali, heatmap, statistiche.",
)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "📊 **Kriterion Quant** — Finanza Quantitativa | "
    "Dati: EODHD Historical Data | "
    "Questa analisi è fornita a scopo educativo e non costituisce consulenza finanziaria."
)
