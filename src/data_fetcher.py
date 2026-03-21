"""
data_fetcher.py
===============
Scarica dati OHLCV da EODHD e risolve i ticker abbreviati usati nella UI.

Fonte dati: EODHD Historical Data API (https://eodhd.com)
Formato ticker EODHD: SYMBOL.EXCHANGE  (es. GSPC.INDX, BTC-USD.CC)
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st

# ================================================================
# MAPPA SCORCIATOIE → TICKER EODHD
# ================================================================

TICKER_MAP: dict[str, str] = {
    "SPX":    "GSPC.INDX",     # S&P 500
    "NDX":    "NDX.INDX",      # Nasdaq-100
    "DAX":    "GDAXI.INDX",    # DAX 40
    "FTMIB":  "FTSEMIB.INDX",  # FTSE MIB (Italia)
    "BTCUSD": "BTC-USD.CC",    # Bitcoin / USD
    "ETHUSD": "ETH-USD.CC",    # Ethereum / USD
    "NI225":  "N225.INDX",     # Nikkei 225
    "HSI":    "HSI.INDX",      # Hang Seng Index
    "GC1":    "GC.COMM",       # Gold futures
    "CL1":    "CL.COMM",       # Crude Oil WTI futures
    "VIX":    "VIX.INDX",      # CBOE Volatility Index
}

# Etichette leggibili per la UI (titoli dei grafici)
TICKER_LABELS: dict[str, str] = {
    "GSPC.INDX":    "S&P 500 (SPX)",
    "NDX.INDX":     "Nasdaq-100 (NDX)",
    "GDAXI.INDX":   "DAX 40",
    "FTSEMIB.INDX": "FTSE MIB",
    "BTC-USD.CC":   "Bitcoin (BTC/USD)",
    "ETH-USD.CC":   "Ethereum (ETH/USD)",
    "N225.INDX":    "Nikkei 225",
    "HSI.INDX":     "Hang Seng Index",
    "GC.COMM":      "Gold Futures (GC1)",
    "CL.COMM":      "Crude Oil WTI (CL1)",
    "VIX.INDX":     "VIX (CBOE Vol. Index)",
}


def resolve_ticker(shortcut: str) -> str:
    """
    Converte una scorciatoia (es. 'SPX') nel ticker EODHD completo.
    Se non trovato nella mappa, restituisce il valore invariato.
    """
    return TICKER_MAP.get(shortcut.upper(), shortcut)


def get_label(eodhd_ticker: str) -> str:
    """Restituisce un'etichetta leggibile per il ticker (per titoli e KPI)."""
    return TICKER_LABELS.get(eodhd_ticker, eodhd_ticker)


# ================================================================
# FETCH DATI EODHD
# ================================================================

@st.cache_data(ttl=3600)
def fetch_ohlcv(
    ticker: str,
    from_date: str = "1950-01-01",
    to_date: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Scarica i prezzi storici giornalieri da EODHD Historical Data API.

    Parameters
    ----------
    ticker    : Ticker in formato EODHD (es. 'GSPC.INDX', 'BTC-USD.CC')
    from_date : Data di inizio nel formato 'YYYY-MM-DD'
    to_date   : Data di fine ('YYYY-MM-DD'). Se None usa la data di oggi.
    api_key   : Chiave API EODHD. Se None viene letta da st.secrets.

    Returns
    -------
    pd.DataFrame con colonne: open, high, low, close, adjusted_close, volume
    Index: DatetimeIndex
    """
    if api_key is None:
        api_key = st.secrets["EODHD_API_KEY"]

    if to_date is None:
        to_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?from={from_date}&to={to_date}"
        f"&period=d&api_token={api_key}&fmt=json"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Assicura che tutte le colonne numeriche siano float
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open", "high", "low", "close", "adjusted_close", "volume"]]
