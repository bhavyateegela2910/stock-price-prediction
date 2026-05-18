import ta

def add_indicators(data):

    # Moving Average
    data["MA50"] = ta.trend.sma_indicator(
        data["Close"],
        window=50
    )

    # RSI
    data["RSI"] = ta.momentum.rsi(
        data["Close"],
        window=14
    )

    # MACD
    data["MACD"] = ta.trend.macd(
        data["Close"]
    )

    return data