import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay
from datetime import datetime
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay

st.set_page_config(page_title="OPEX Pivot Probabilities", layout="wide")

def get_monthly_opex_dates(start_date, end_date, trading_days):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    opex_dates = []
    current = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date)

    while current <= end:
        fridays = pd.date_range(current, current + pd.offsets.MonthEnd(0), freq='W-FRI')
        if len(fridays) >= 3:
            opex = fridays[2]
            # If OPEX is a trading day, use it
            if opex in trading_days and start_date <= opex <= end_date:
                opex_dates.append(opex)
            else:
                # Snap to previous trading day before opex (for holiday e.g. Good Friday)
                idx = trading_days.searchsorted(opex, side='right') - 1
                if idx >= 0:
                    prev_opex = trading_days[idx]
                    # Only add if it's in the same month and not already added
                    if prev_opex.month == opex.month and prev_opex not in opex_dates and start_date <= prev_opex <= end_date:
                        opex_dates.append(prev_opex)
        current += pd.offsets.MonthBegin(1)
    return opex_dates

def get_trading_day_before(date, days_before, trading_days):
    """
    Return the trading day exactly `days_before` entries before `date`
    in the sorted DatetimeIndex `trading_days`.
    """
    # Ensure `date` matches exactly one of the index values
    try:
        loc = trading_days.get_loc(pd.Timestamp(date))
    except KeyError:
        return None

    target = loc - days_before
    if target < 0:
        return None

    return trading_days[target]


def snap_to_next_trading_day(trading_days, target):
    idx = trading_days.searchsorted(target, side='left')
    if idx >= len(trading_days):
        return None
    return trading_days[idx]

def calc_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def is_exact_pivot(prices, pivot_date, ma_window=5):
    """
    Return (True, direction) if pivot_date is an inflection:
    slope(before) and slope(after) have opposite signs.
    """
    ma = prices.rolling(ma_window).mean()
    # need at least one point before and after
    if pivot_date not in ma.index:
        return False, None
    idx = ma.index.get_loc(pivot_date)
    if idx < 1 or idx >= len(ma) - 1:
        return False, None

    prev_val = ma.iloc[idx - 1]
    curr_val = ma.iloc[idx]
    next_val = ma.iloc[idx + 1]
    slope_before = curr_val - prev_val
    slope_after  = next_val - curr_val

    # ignore flats
    if slope_before == 0 or slope_after == 0:
        return False, None

    if np.sign(slope_before) != np.sign(slope_after):
        direction = 'up' if slope_after > 0 else 'down'
        return True, direction
    return False, None


def analyze_opex_pivots(df, opex_dates, windows, atr_period=14, ma_window=5):
    results = []
    atr = calc_atr(df, period=atr_period)
    trading_days = df.index

    for opex in opex_dates:
        row = {'OPEX': opex}
        quad = opex.month in [3, 6, 9, 12]
        win_list = windows.copy()
        if quad:
            win_list += [30]

        for win in win_list:
            # 1) find the candidate date exactly win BDs before OPEX
            candidate = get_trading_day_before(opex, win, trading_days)
            if candidate is None or candidate not in df.index:
                row[f'{win}d_pivot'] = None
                continue

            # 2) slice a little extra around it for MA calc
            start = trading_days[max(0, trading_days.get_loc(candidate) - ma_window - 1)]
            end   = trading_days[min(len(trading_days)-1,
                                     trading_days.get_loc(candidate) + ma_window + 1)]
            prices = df.loc[start:end, 'Close']

            # 3) test that candidate is a pivot
            hit, direction = is_exact_pivot(prices, candidate, ma_window)
            if not hit:
                row[f'{win}d_pivot'] = None
                continue

            # 4) compute move and consolidation from win-start to OPEX
            win_start = get_trading_day_before(opex, win, trading_days)
            window_prices = df.loc[win_start:opex, 'Close']
            start_price = window_prices.iloc[0]
            end_price   = window_prices.iloc[-1]
            atr_val     = atr.loc[candidate] if candidate in atr.index else np.nan
            move        = abs(end_price - start_price)
            consolidation = (move < 1.3 * atr_val) if not np.isnan(atr_val) else False

            row[f'{win}d_pivot'] = {
                'date': candidate,
                'direction': direction,
                'consolidation': consolidation,
                'move': move,
                'atr': atr_val
            }

        results.append(row)
    return results

def summarize_pivot_probabilities(results, windows):
    stats = []
    for win in windows:
        total = found = cons = 0
        for row in results:
            val = row.get(f'{win}d_pivot')
            if val is not None:
                found += 1
                consolidation = val['consolidation']
                if hasattr(consolidation, 'item'):
                    consolidation = consolidation.item()
                if pd.isna(consolidation):
                    consolidation = False
                if consolidation:
                    cons += 1
            total += 1
        stats.append({
            'Window': f'{win} business days before',
            'Pivot Hit Rate': f"{100 * found / total:.1f}%" if total else "N/A",
            'Consolidation Rate': f"{100 * cons / found:.1f}%" if found else "N/A"
        })
    return pd.DataFrame(stats)

def load_spy_data(start_date, end_date):
    # Download SPY — yfinance sometimes returns a MultiIndex even for a single ticker
    df = yf.download('SPY', start=start_date, end=end_date, auto_adjust=False)

    # If columns are MultiIndex, drop the first level entirely
    if isinstance(df.columns, pd.MultiIndex):
        # e.g. columns like ('SPY','Open'), ('SPY','High'), etc.
        df.columns = df.columns.droplevel(1)
    # Now df.columns should be ['Open','High','Low','Close','Adj Close','Volume']
    df = df[df['Volume'] > 0]
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df

def plot_chart(df, opex_dates, results, win_select, ma_window):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="SPY"
    ))

    # Just plot the OPEX lines as shapes and annotations (with pd.Timestamp)
    for opex in opex_dates:
        fig.add_shape(
            type="line",
            x0=opex, x1=opex,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=opex, y=1.02,
            xref="x", yref="paper",
            text="OPEX",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12, color="red"),
        )

    for row in results:
        val = row.get(f'{win_select}d_pivot')
        if val and not pd.isna(val['date']):
            cons = val['consolidation']
            if hasattr(cons, 'item'):
                cons = cons.item()
            if pd.isna(cons):
                cons = False

            pivot_date = val['date']

            if pivot_date not in df.index:
                st.write("❗️pivot_date not in df.index", {
                    "pivot_date": pivot_date,
                    "pivot_date_type": str(type(pivot_date)),
                    "first_df_index": df.index[0],
                    "first_df_index_type": str(type(df.index[0])),
                    "pivot_date_str": str(pivot_date),
                    "df_index_sample_str": [str(i) for i in df.index[:5]]
                })
                try:
                    st.write("pivot_date strftime:", pivot_date.strftime('%Y-%m-%d'))
                    st.write("df.index strftime:", [i.strftime('%Y-%m-%d') for i in df.index[:5]])
                except Exception as e:
                    st.write("Could not strftime:", e)
                continue  # Skip marker if not found

            yval = df.loc[pivot_date, 'Close']
            # Fix: If yval is a Series (duplicate index), just take the first one
            if isinstance(yval, pd.Series):
                yval = yval.iloc[0]

            if pd.isna(yval):
                st.write(f"⚠️ NaN for {pivot_date}, skipping marker.")
                continue

            fig.add_trace(go.Scatter(
                x=[pivot_date], y=[yval],
                mode='markers',
                marker=dict(
                    size=14,
                    color='green' if not cons else 'orange',
                    symbol='triangle-up' if val['direction']=='up' else 'triangle-down'
                ),
                name=f"{win_select}d Pivot"
            ))

    fig.update_layout(
        title=f"SPY Price Chart with {win_select}d Pivots Highlighted",
        xaxis_rangeslider_visible=False,
        height=700
    )
    return fig
# ----- STREAMLIT UI -----

st.title("SPY Monthly OPEX Pivot Probability App")
st.markdown(
"""
Analyze the historical probability of a pivot around monthly OPEX expiration.
Pivots are trend inflection points (up to down or vice versa).
Consolidations are moves < 1.3 ATR.
"""
)

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start Date", value=datetime(2017,1,1))
    end_date = st.date_input("End Date", value=datetime.now())
    ma_window = st.slider("Pivot Trend MA Window", 3, 10, 5)
    atr_period = st.slider("ATR Period", 5, 30, 14)
    pivot_days_str = st.text_input("Pivot Windows (business days, comma separated)", "14,11,8,7,5,3")
    windows = [int(x.strip()) for x in pivot_days_str.split(",") if x.strip().isdigit()]
    if st.button("Run Analysis"):
        st.session_state['run'] = True

if 'run' in st.session_state and st.session_state['run']:
    with st.spinner("Loading SPY data..."):
        df = load_spy_data(start_date, end_date)
        trading_days = df.index
        opex_dates = get_monthly_opex_dates(start_date, end_date, trading_days)
    st.success(f"Loaded {len(df)} rows. Found {len(opex_dates)} OPEX dates.")
    with st.spinner("Analyzing pivots..."):
        results = analyze_opex_pivots(df, opex_dates, windows, atr_period, ma_window)
        prob_table = summarize_pivot_probabilities(results, windows)
    st.header("Pivot Probability Table")
    st.dataframe(prob_table, use_container_width=True)
    st.header("Detailed Pivot Table")
    detailed = []
    for row in results:
        opex = row['OPEX'].strftime('%Y-%m-%d')
        for win in windows:
            val = row.get(f'{win}d_pivot')
            if val:
                move_val = val['move']
                if hasattr(move_val, "item"):
                    move_val = move_val.item()
                if pd.isna(move_val):
                    move_val = 0.0
                detailed.append({
                    'OPEX': opex,
                    'Window': f"{win}d",
                    'Pivot Date': val['date'].strftime('%Y-%m-%d'),
                    'Direction': val['direction'],
                    'Consolidation': val['consolidation'],
                    'Move': f"${move_val:.2f}",
                    'ATR': f"${val['atr']:.2f}" if val['atr'] else "N/A"
                })
    st.dataframe(pd.DataFrame(detailed), use_container_width=True)
    st.header("Pivot Chart")
    win_select = st.selectbox("Show Pivots For Which Window?", windows)
    fig = plot_chart(df, opex_dates, results, win_select, ma_window)
    st.plotly_chart(fig, use_container_width=True)
