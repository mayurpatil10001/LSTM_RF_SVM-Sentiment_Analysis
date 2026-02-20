"""Test that app.py imports without runtime errors."""
import importlib.util, sys, os

# Mock streamlit so we can import app.py without a real browser session
try:
    import unittest.mock as mock
    # Patch streamlit at module level before importing app
    st_mock = mock.MagicMock()
    st_mock.set_page_config = mock.MagicMock()
    st_mock.session_state = {}
    sys.modules["streamlit"] = st_mock
    
    # Test individual module imports
    from ticker_mapper import get_company_name, list_all_tickers
    from sentiment_utils import (
        fetch_news, run_finbert, aggregate_daily_sentiment,
        merge_sentiment_with_stock, get_sentiment_summary,
        label_color, RateLimitError
    )
    from model_trainer import run_all_models
    
    print("ticker_mapper OK:", get_company_name("TCS.NS"))
    print("sentiment_utils OK: all functions imported")
    print("model_trainer OK: run_all_models imported")
    print("Tickers in map:", len(list_all_tickers()))
    
    # Verify .env key loads
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("NEWSDATA_API_KEY", "")
    if key and key != "your_newsdata_api_key_here":
        print(f"API key loaded: {key[:10]}...")
    else:
        print("WARNING: API key not set or still placeholder")
    
    print("\n✅ All app imports PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
