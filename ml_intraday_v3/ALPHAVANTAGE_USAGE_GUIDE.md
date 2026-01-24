# Alpha Vantage MCP Usage Guide

## Quick Reference

### Available via MCP Tools
The Alpha Vantage MCP is now configured in `.mcp.json` and enabled in `.claude/settings.local.json`.

### Rate Limits
- **Free Tier**: 25 requests per day, 1 request per second
- **Premium**: Unlimited daily, higher burst rate

## Common Use Cases

### 1. Check Global Market Status
```python
# Via Claude Code MCP
mcp__alphavantage__TOOL_CALL("MARKET_STATUS", {})

# Returns: Open/closed status for US, UK, Germany, Japan, etc.
```

### 2. Search for International Symbols
```python
# Find FTSE 100 tickers
mcp__alphavantage__TOOL_CALL("SYMBOL_SEARCH", {
    "keywords": "FTSE",
    "datatype": "json"
})

# Find DAX stocks
mcp__alphavantage__TOOL_CALL("SYMBOL_SEARCH", {
    "keywords": "DAX",
    "datatype": "json"
})
```

### 3. Fetch Intraday Stock Data
```python
# Get 5-minute bars for London-listed stock
mcp__alphavantage__TOOL_CALL("TIME_SERIES_INTRADAY", {
    "symbol": "TSCO.LON",  # Tesco (London)
    "interval": "5min",
    "outputsize": "compact",
    "datatype": "json"
})

# Get German stock
mcp__alphavantage__TOOL_CALL("TIME_SERIES_INTRADAY", {
    "symbol": "SAP.DEX",  # SAP (Frankfurt)
    "interval": "5min",
    "outputsize": "full"
})
```

### 4. Get Technical Indicators
```python
# RSI for international stock
mcp__alphavantage__TOOL_CALL("RSI", {
    "symbol": "7203.TYO",  # Toyota (Tokyo)
    "interval": "daily",
    "time_period": 14,
    "series_type": "close"
})

# ATR for volatility analysis
mcp__alphavantage__TOOL_CALL("ATR", {
    "symbol": "TSCO.LON",
    "interval": "5min",
    "time_period": 14
})
```

### 5. Forex Data (For Currency Pairs)
```python
# EUR/USD intraday
mcp__alphavantage__TOOL_CALL("FX_INTRADAY", {
    "from_symbol": "EUR",
    "to_symbol": "USD",
    "interval": "5min",
    "outputsize": "compact"
})
```

## Symbol Formats by Exchange

| Exchange | Format | Example |
|----------|--------|---------|
| **US (NYSE/NASDAQ)** | `SYMBOL` | `IBM`, `AAPL` |
| **London (LSE)** | `SYMBOL.LON` | `TSCO.LON`, `VOD.LON` |
| **Frankfurt/XETRA** | `SYMBOL.DEX` | `SAP.DEX`, `DAI.DEX` |
| **Tokyo (TSE)** | `CODE.TYO` | `7203.TYO`, `9984.TYO` |
| **Hong Kong** | `CODE.HKG` | `0700.HKG` |

**Note**: These are for STOCKS, not FUTURES. For futures (MES, FDAX, NKD), you need CME/exchange-specific data.

## Integration with ml_intraday_v3

### Research Workflow (Stocks → Futures Adaptation)

1. **Study International Market Dynamics**
```python
# Fetch FTSE 100 components to understand UK market
# Analyze volatility patterns, correlation with US
# Identify regime characteristics
```

2. **Correlation Analysis**
```python
# Pull historical daily data for major indices
us_data = fetch_daily("SPY")
uk_data = fetch_daily("VUKE.LON")  # FTSE 100 ETF
de_data = fetch_daily("EXS1.DEX")  # DAX ETF

# Calculate rolling correlations
correlations = calculate_rolling_corr(us_data, uk_data, window=60)
```

3. **Volatility Regime Detection**
```python
# Use Alpha Vantage to fetch ATR for index components
# Understand typical volatility levels per market
# Adapt stop-loss / take-profit for that market
```

### Limitations for Algorithmic Trading

**Alpha Vantage is NOT suitable for:**
- ❌ Live futures trading (provides stock data)
- ❌ High-frequency data collection (rate limits)
- ❌ Real-time execution signals (15-min delay on free tier)
- ❌ CME futures data (MES, FDAX, NKD not available)

**Alpha Vantage IS useful for:**
- ✅ Research & correlation studies
- ✅ Understanding international market dynamics
- ✅ Feature engineering inspiration
- ✅ Regime detection research
- ✅ Macro indicator data (CPI, GDP, etc.)

## Example: Research Script

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "3XBV9VY0KQRJ3LO8"

def fetch_daily_data(symbol: str, outputsize: str = "full") -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol"""
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Parse time series
    ts = data.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    return df.sort_index()

# Example: Compare US, UK, Germany volatility
if __name__ == "__main__":
    # Fetch data (respect rate limits!)
    us_data = fetch_daily_data("SPY")  # S&P 500 ETF
    time.sleep(12)  # Rate limit: 1 req/sec
    uk_data = fetch_daily_data("ISF.LON")  # FTSE 100 ETF
    time.sleep(12)
    de_data = fetch_daily_data("EXS1.DEX")  # DAX ETF

    # Calculate daily returns
    us_returns = us_data['4. close'].pct_change()
    uk_returns = uk_data['4. close'].pct_change()
    de_returns = de_data['4. close'].pct_change()

    # Correlation matrix
    corr_matrix = pd.DataFrame({
        'US': us_returns,
        'UK': uk_returns,
        'DE': de_returns
    }).corr()

    print("Correlation Matrix:")
    print(corr_matrix)

    # Rolling correlation
    rolling_corr = pd.DataFrame({
        'US': us_returns,
        'UK': uk_returns
    }).rolling(60).corr().unstack()['US']['UK']

    print("\nRolling 60-day US-UK Correlation (last 10 days):")
    print(rolling_corr.tail(10))
```

## MCP vs. Direct API

### Using MCP (Recommended in Claude Code)
```python
# Within Claude Code conversation
"Can you use Alpha Vantage to check if London market is open?"

# Claude will call:
mcp__alphavantage__TOOL_CALL("MARKET_STATUS", {})
```

### Direct API (For Scripts)
```python
import requests

url = "https://www.alphavantage.co/query"
params = {
    "function": "MARKET_STATUS",
    "apikey": "3XBV9VY0KQRJ3LO8"
}

response = requests.get(url, params=params)
print(response.json())
```

## Best Practices

1. **Respect Rate Limits**
   - Free tier: 25 requests/day
   - Add 12-second delays between calls
   - Cache results locally

2. **Symbol Validation**
   - Always use `SYMBOL_SEARCH` first to verify format
   - International symbols need exchange suffix

3. **Data Quality**
   - Spot-check data for gaps/anomalies
   - Compare with other sources for validation
   - Some international stocks have thin volume

4. **Use Cases**
   - Research & analysis: Great
   - Live trading signals: NOT suitable (use futures data instead)

## Troubleshooting

### Error: "Rate limit exceeded"
**Solution**: You've hit 25 requests/day limit. Wait until next day or upgrade to premium.

### Error: "Invalid symbol"
**Solution**: Use `SYMBOL_SEARCH` to find correct symbol format with exchange suffix.

### Empty Response
**Solution**: Market might be closed, or symbol doesn't have data for requested timeframe.

## References

- **Alpha Vantage Docs**: https://www.alphavantage.co/documentation/
- **API Key Management**: https://www.alphavantage.co/support/#api-key
- **Premium Plans**: https://www.alphavantage.co/premium/

---

**Created**: 2026-01-23
**MCP Server**: alphavantage (configured in .mcp.json)
