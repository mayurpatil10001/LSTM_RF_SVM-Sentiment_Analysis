"""Test the fixed news API fetch."""
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NEWSDATA_API_KEY", "")
print(f"API Key: {repr(api_key[:12])}...")

from sentiment_utils import fetch_news, _build_search_query
from ticker_mapper import get_company_name

# Test query builder
for t, c in [("TCS.NS", "Tata Consultancy Services"), ("RELIANCE.NS", "Reliance Industries"), ("INFY.NS", "Infosys")]:
    q = _build_search_query(t, c)
    print(f"  Query for {t}: '{q}'")

print()

# Test fetch_news (fixed - no from_date/to_date)
ticker = "TCS.NS"
company = get_company_name(ticker)
news = fetch_news(ticker, company, "2022-01-01", "2025-12-31", api_key)
print(f"Articles fetched for {ticker}: {len(news)}")
if not news.empty:
    print(news[['title', 'pubDate', 'source_id']].head(3).to_string())
    print("\nFinBERT test on first article:")
    from sentiment_utils import run_finbert
    sample = news.head(2)
    scored = run_finbert(sample)
    print(scored[['title', 'sentiment_label', 'sentiment_score', 'confidence']].to_string())
else:
    print("No articles returned")

# Also test for a different ticker
print()
ticker2 = "RELIANCE.NS"
company2 = get_company_name(ticker2)
news2 = fetch_news(ticker2, company2, "2022-01-01", "2025-12-31", api_key)
print(f"Articles fetched for {ticker2}: {len(news2)}")
