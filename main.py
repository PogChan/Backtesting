import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

st.title("Pivot Reversal Backtest: Wednesday & Friday")

# ---- Sidebar parameters ----
symbol = st.sidebar.text_input("Symbol", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100_000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 20, 5) / 100.0

@st.cache_data
def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[["Open", "Close"]].copy()
    df["weekday"] = df.index.dayofweek
    return df

df = get_data(symbol, start_date, end_date)
if df.empty:
    st.error("No data loaded. Check ticker and date range.")
    st.stop()

def backtest_pivot(df, pivot_day, initial_capital, risk_per_trade):
    WED, FRI = 2, 4
    entry_day = WED if pivot_day == "Wednesday" else FRI
    exit_day = FRI if pivot_day == "Wednesday" else WED

    df = df.copy()
    df["is_entry"] = df["weekday"] == entry_day

    # --- Improved reference close calculation ---
    if pivot_day == "Wednesday":
        # Get the last Friday's close for every row, ffill handles weekends/holidays
        df["last_fri_close"] = df["Close"].where(df["weekday"] == FRI)
        df["last_fri_close"] = df["last_fri_close"].ffill().shift(1)
        df["ref_close"] = df["last_fri_close"]
    else:
        # Get the last Wednesday's close for every row
        df["last_wed_close"] = df["Close"].where(df["weekday"] == WED)
        df["last_wed_close"] = df["last_wed_close"].ffill().shift(1)
        df["ref_close"] = df["last_wed_close"]

    trades = []
    entries = df[df["is_entry"]]
    for entry_date, entry_row in entries.iterrows():
        def _to_float(val):
            """Force value to scalar float if possible, even if it's a Series."""
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            try:
                return float(val)
            except Exception:
                return float('nan')
            
        ref = _to_float(entry_row["ref_close"])
        entry_price = _to_float(entry_row["Close"])
        if math.isnan(ref) or math.isnan(entry_price):
            continue
        direction = 1 if entry_price < ref else -1

        entry_idx = df.index.get_loc(entry_date)
        future = df.iloc[entry_idx + 1 :]
        future_pivot = future[future["weekday"] == exit_day]
        if future_pivot.empty:
            continue
        exit_date = future_pivot.index[0]
        exit_price = _to_float(future_pivot.iloc[0]["Close"])
        if math.isnan(exit_price):
            continue
        trade_return = direction * (exit_price - entry_price) / entry_price

        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "position": "LONG" if direction > 0 else "SHORT",
            "return": trade_return,
        })

    # Portfolio simulation
    capital = initial_capital
    cap_series = pd.Series(capital, index=df.index)
    for t in trades:
        profit = (capital * risk_per_trade) * t["return"]
        cap_series.loc[t["exit_date"]:] += profit
        capital += profit

    cols = ["entry_date","entry_price","exit_date","exit_price","position","return"]
    trade_df = pd.DataFrame(trades, columns=cols)
    trade_df["return"] = pd.to_numeric(trade_df["return"], errors="coerce").fillna(0.0)
    trade_df["return_pct"] = (trade_df["return"] * 100).round(2)
    win_rate = (trade_df["return"] > 0).mean() if not trade_df.empty else 0.0
    total_return_pct = (cap_series.iloc[-1] / initial_capital - 1) * 100

    return {
        "trades": trade_df,
        "equity_curve": cap_series,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
    }

# Run both pivot strategies
results = {
    "Wednesday": backtest_pivot(df, "Wednesday", initial_capital, risk_per_trade),
    "Friday":    backtest_pivot(df, "Friday",    initial_capital, risk_per_trade),
}

# UI selection and display
pivot_choice = st.radio("Select Pivot Day Strategy", ["Wednesday", "Friday"])
res = results[pivot_choice]

st.subheader(f"Performance: {pivot_choice}")
st.metric("Total Trades", len(res["trades"]))
st.metric("Win Rate", f"{res['win_rate']:.2%}")
st.metric("Total Return", f"{res['total_return_pct']:.2f}%")

st.subheader("Equity Curve")
fig, ax = plt.subplots()
ax.plot(res["equity_curve"].index, res["equity_curve"].values)
ax.set_title(f"{symbol} Equity Curve ({pivot_choice})")
ax.set_xlabel("Date"); ax.set_ylabel("Equity ($)")
st.pyplot(fig)

st.subheader("Trade Log")
if not res["trades"].empty:
    st.dataframe(res["trades"])
else:
    st.write("No trades generated.")

if st.checkbox("Show Daily Equity Table"):
    de = res["equity_curve"].rename("Equity").reset_index()
    de.columns = ["Date", "Equity"]
    st.dataframe(de)