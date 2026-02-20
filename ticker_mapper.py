"""
ticker_mapper.py
================
Maps NSE/BSE stock ticker symbols to full company names.
Used to build accurate news search queries for the newsdata.io API.

Author: Upgraded Stock Prediction App
"""

# ──────────────────────────────────────────────────────────────────────────────
# TICKER → COMPANY NAME MAP  (NSE .NS tickers)
# ──────────────────────────────────────────────────────────────────────────────
TICKER_MAP: dict[str, str] = {
    # ── IT & Technology ───────────────────────────────────────────────────────
    "TCS.NS":         "Tata Consultancy Services",
    "INFY.NS":        "Infosys",
    "WIPRO.NS":       "Wipro",
    "HCLTECH.NS":     "HCL Technologies",
    "TECHM.NS":       "Tech Mahindra",
    "LTIM.NS":        "LTIMindtree",
    "MPHASIS.NS":     "Mphasis",
    "PERSISTENT.NS":  "Persistent Systems",
    "COFORGE.NS":     "Coforge",
    "OFSS.NS":        "Oracle Financial Services",

    # ── Banking & Finance ──────────────────────────────────────────────────────
    "HDFCBANK.NS":    "HDFC Bank",
    "ICICIBANK.NS":   "ICICI Bank",
    "KOTAKBANK.NS":   "Kotak Mahindra Bank",
    "AXISBANK.NS":    "Axis Bank",
    "SBIN.NS":        "State Bank of India",
    "BANKBARODA.NS":  "Bank of Baroda",
    "PNB.NS":         "Punjab National Bank",
    "CANBK.NS":       "Canara Bank",
    "FEDERALBNK.NS":  "Federal Bank",
    "INDUSINDBK.NS":  "IndusInd Bank",
    "BAJFINANCE.NS":  "Bajaj Finance",
    "BAJAJFINSV.NS":  "Bajaj Finserv",
    "CHOLAFIN.NS":    "Cholamandalam Investment",
    "MUTHOOTFIN.NS":  "Muthoot Finance",
    "HDFC.NS":        "Housing Development Finance Corporation",

    # ── Energy & Oil ──────────────────────────────────────────────────────────
    "RELIANCE.NS":    "Reliance Industries",
    "ONGC.NS":        "Oil and Natural Gas Corporation",
    "BPCL.NS":        "Bharat Petroleum",
    "IOC.NS":         "Indian Oil Corporation",
    "GAIL.NS":        "GAIL India",
    "POWERGRID.NS":   "Power Grid Corporation of India",
    "NTPC.NS":        "NTPC",
    "ADANIGREEN.NS":  "Adani Green Energy",
    "ADANIPORTS.NS":  "Adani Ports",
    "ADANIENT.NS":    "Adani Enterprises",

    # ── FMCG & Consumer ───────────────────────────────────────────────────────
    "HINDUNILVR.NS":  "Hindustan Unilever",
    "ITC.NS":         "ITC",
    "NESTLEIND.NS":   "Nestle India",
    "BRITANNIA.NS":   "Britannia Industries",
    "DABUR.NS":       "Dabur India",
    "MARICO.NS":      "Marico",
    "COLPAL.NS":      "Colgate-Palmolive India",
    "GODREJCP.NS":    "Godrej Consumer Products",
    "EMAMILTD.NS":    "Emami",

    # ── Pharma & Healthcare ───────────────────────────────────────────────────
    "SUNPHARMA.NS":   "Sun Pharmaceutical Industries",
    "DRREDDY.NS":     "Dr. Reddy's Laboratories",
    "CIPLA.NS":       "Cipla",
    "DIVISLAB.NS":    "Divi's Laboratories",
    "BIOCON.NS":      "Biocon",
    "AUROPHARMA.NS":  "Aurobindo Pharma",
    "TORNTPHARM.NS":  "Torrent Pharmaceuticals",
    "ALKEM.NS":       "Alkem Laboratories",
    "LUPIN.NS":       "Lupin",
    "ABBOTINDIA.NS":  "Abbott India",

    # ── Automobile ────────────────────────────────────────────────────────────
    "MARUTI.NS":      "Maruti Suzuki India",
    "TATAMOTORS.NS":  "Tata Motors",
    "M&M.NS":         "Mahindra and Mahindra",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "EICHERMOT.NS":   "Eicher Motors",
    "HEROMOTOCO.NS":  "Hero MotoCorp",
    "TVSMOTOR.NS":    "TVS Motor Company",
    "ASHOKLEY.NS":    "Ashok Leyland",
    "MOTHERSON.NS":   "Motherson Sumi Systems",

    # ── Infrastructure & Capital Goods ────────────────────────────────────────
    "LT.NS":          "Larsen and Toubro",
    "ULTRACEMCO.NS":  "UltraTech Cement",
    "GRASIM.NS":      "Grasim Industries",
    "ACC.NS":         "ACC",
    "AMBUJACEM.NS":   "Ambuja Cements",
    "SIEMENS.NS":     "Siemens India",
    "ABB.NS":         "ABB India",
    "BHEL.NS":        "Bharat Heavy Electricals",

    # ── Metals & Mining ───────────────────────────────────────────────────────
    "TATASTEEL.NS":   "Tata Steel",
    "JSWSTEEL.NS":    "JSW Steel",
    "HINDALCO.NS":    "Hindalco Industries",
    "VEDL.NS":        "Vedanta",
    "COALINDIA.NS":   "Coal India",
    "NMDC.NS":        "NMDC",
    "SAIL.NS":        "Steel Authority of India",

    # ── Telecom & Media ───────────────────────────────────────────────────────
    "BHARTIARTL.NS":  "Bharti Airtel",
    "IDEA.NS":        "Vodafone Idea",
    "INDIAMART.NS":   "IndiaMART InterMESH",
    "ZOMATO.NS":      "Zomato",
    "NYKAA.NS":       "FSN E-Commerce Ventures Nykaa",
    "PAYTM.NS":       "Paytm One97 Communications",

    # ── Index / ETF (for reference) ────────────────────────────────────────────
    "^NSEI":          "Nifty 50",
    "^BSESN":         "BSE Sensex",

    # ── Global popular tickers ────────────────────────────────────────────────
    "AAPL":           "Apple",
    "MSFT":           "Microsoft",
    "GOOGL":          "Alphabet Google",
    "GOOG":           "Alphabet Google",
    "AMZN":           "Amazon",
    "TSLA":           "Tesla",
    "META":           "Meta Platforms Facebook",
    "NVDA":           "NVIDIA",
    "NFLX":           "Netflix",
}


def get_company_name(ticker: str) -> str:
    """
    Return the full company name for a given ticker symbol.

    Lookup order:
    1. Exact match in TICKER_MAP (case-insensitive).
    2. Fallback: strip exchange suffix (.NS / .BO / .BSE) and return the
       remaining part as a human-readable search term.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "TCS.NS", "RELIANCE.NS", "AAPL".

    Returns
    -------
    str
        Full company name suitable for use in a news search query.
    """
    ticker_upper = ticker.strip().upper()

    # 1. Direct lookup
    if ticker_upper in TICKER_MAP:
        return TICKER_MAP[ticker_upper]

    # 2. Case-insensitive fallback scan
    for key, value in TICKER_MAP.items():
        if key.upper() == ticker_upper:
            return value

    # 3. Strip suffix fallback
    for suffix in (".NS", ".BO", ".BSE", ".NSE"):
        if ticker_upper.endswith(suffix):
            base = ticker_upper.replace(suffix, "")
            # Convert "BAJAJ-AUTO" → "Bajaj Auto"
            name = base.replace("-", " ").replace("&", " and ").title()
            return name

    # 4. Return ticker itself as last resort
    return ticker.strip()


def list_all_tickers() -> list[str]:
    """Return a sorted list of all supported ticker symbols."""
    return sorted(TICKER_MAP.keys())


if __name__ == "__main__":
    # Quick smoke-test
    samples = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "SBIN.NS",
               "ZOMATO.NS", "UNKNWN.NS", "AAPL"]
    for t in samples:
        print(f"{t:25s}  →  {get_company_name(t)}")
