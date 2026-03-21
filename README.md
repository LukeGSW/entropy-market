# 🔬 Analisi dell'Entropia dei Mercati Finanziari

Dashboard Streamlit per lo studio della complessità informazionale dei mercati tramite
**Shannon Entropy**, **Permutation Entropy** e analisi di regime.

> Parte del progetto **[Kriterion Quant](https://kriterionquant.com)** — Finanza Quantitativa

---

## Studi implementati

| Studio | Descrizione |
|--------|------------|
| **1 — Shannon Entropy** | Panoramica su Returns, Volatilità e Skewness |
| **2 — Permutation Entropy** | Complessità locale, normalizzata [0,1] |
| **3 — Regime → Forward Returns** | Forward returns medi per regime Bassa/Media/Alta |
| **4 — Scatter Entropia × Returns** | Correlazione entropia-rendimenti futuri (1M/3M/6M/12M) |
| **5 — Percentile storico** | Zone di allerta su distribuzione storica espandente |
| **6 — Cross-entropy + Heatmap** | Correlazione rolling PE↔Shannon + stagionalità mensile |

---

## Struttura del repository

```
entropy-market/
├── app.py                  # Entry point Streamlit
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme Kriterion Quant
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py     # EODHD API + ticker map
│   ├── calculations.py     # Shannon, PE, regime, forward returns
│   └── charts.py           # 6 funzioni di visualizzazione Plotly
└── README.md
```

---

## Installazione locale

```bash
git clone https://github.com/<tuo-username>/entropy-market.git
cd entropy-market
pip install -r requirements.txt

# Crea il file secrets (NON committare)
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
EODHD_API_KEY = "la_tua_chiave_eodhd"
EOF

streamlit run app.py
```

---

## Deploy su Streamlit Cloud

1. Fork/push su GitHub
2. Vai su [share.streamlit.io](https://share.streamlit.io)
3. Seleziona il repository e come entry point `app.py`
4. In **Settings → Secrets** aggiungi:
   ```toml
   EODHD_API_KEY = "la_tua_chiave_eodhd"
   ```
5. Deploy ✅

---

## API Key EODHD

Registrati su [eodhd.com](https://eodhd.com) per ottenere una chiave API.
Il piano gratuito include dati storici giornalieri per indici e crypto.

---

## Fondamenti teorici

- **Shannon (1948)** — A Mathematical Theory of Communication
- **Bandt & Pompe (2002)** — Permutation Entropy (PRL 88, 174102)
- **Lo (2004)** — Adaptive Markets Hypothesis
- **Peters (2019)** — The ergodicity problem in economics (Nature Physics)

---

## Disclaimer

Questa dashboard è fornita a **solo scopo educativo e di ricerca**.
Non costituisce consulenza finanziaria. Le performance passate non garantiscono
risultati futuri. Sempre verificare qualsiasi strategia con un professionista abilitato.

---

*© Kriterion Quant — [kriterionquant.com](https://kriterionquant.com)*
