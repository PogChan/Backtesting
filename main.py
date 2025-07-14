import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import Day, BDay

st.set_page_config(layout="wide", page_title="OPEX Pivot Analyzer")

st.title("📈 OPEX Pivot Analyzer")
st.markdown("""
This tool analyzes historical stock data to identify trend pivots in relation to monthly options expiration (OPEX) dates.
- **OPEX Date**: The third Friday of each month. If it's a holiday, the Thursday before is used.
- **Trend**: Determined using a Simple Moving Average (SMA).
  - **Uptrend**: Price is consistently above the SMA.
  - **Downtrend**: Price is consistently below the SMA.
  - **Consolidation**: Price moves within an Average True Range (ATR) band around the SMA.
- **Pivot**: A change in the trend's slope (e.g., from uptrend to downtrend).
The analysis calculates how many trading days before an OPEX date these pivots tend to occur.
""")

# --- Helper Functions ---

@st.cache_data
def get_opex_dates(start_date, end_date):
    """
    Calculates all OPEX dates within a given date range.
    """
    opex_dates = []
    holidays = USFederalHolidayCalendar().holidays()

    # Generate all third Fridays for the date range
    all_fridays = pd.date_range(start_date, end_date, freq='WOM-3FRI')

    for date in all_fridays:
        # If the third Friday is a holiday or a weekend (unlikely but safe), move to the previous business day
        while date in holidays or date.weekday() > 4:
            date -= BDay(1)
        opex_dates.append(date.normalize())

    return sorted(list(set(opex_dates)))

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            # e.g. columns like ('SPY','Open'), ('SPY','High'), etc.
            data.columns = data.columns.droplevel(1)
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

def calculate_atr(data, period=14):
    """
    Calculates the Average True Range (ATR).
    """
    data['high-low'] = data['High'] - data['Low']
    data['high-pc'] = np.abs(data['High'] - data['Close'].shift(1))
    data['low-pc'] = np.abs(data['Low'] - data['Close'].shift(1))
    data['tr'] = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=period).mean()
    return data

def classify_trend(data, sma_period=20, atr_period=14):
    # Work on a copy so we don’t clobber the caller’s DataFrame
    df = data.copy()

    # 1. Compute SMA and ATR
    df['sma'] = df['Close'].rolling(window=sma_period).mean()
    df = calculate_atr(df, period=atr_period)

    # 2. Drop any rows where we can’t compute both SMA and ATR
    df = df.dropna(subset=['sma', 'atr'])

    # 3. Pre‐compute the thresholds
    upper = df['sma'] + df['atr'] * 0.5
    lower = df['sma'] - df['atr'] * 0.5

    # 4. Initialize trend to “Consolidation”
    df['trend'] = 'Consolidation'

    # 5. Assign Uptrend / Downtrend directly
    df.loc[df['Close'] > upper, 'trend'] = 'Uptrend'
    df.loc[df['Close'] < lower, 'trend'] = 'Downtrend'

    return df

def find_pivots(data):
    """
    Identifies pivot points where the trend changes.
    The pivot is marked the day *before* the trend officially changes.
    """
    data['prev_trend'] = data['trend'].shift(1)
    # A pivot occurs if the trend today is different from the trend yesterday
    # And neither trend is 'Consolidation' for a clearer signal
    pivot_conditions = (data['trend'] != data['prev_trend']) & \
                       (data['trend'] != 'Consolidation') & \
                       (data['prev_trend'] != 'Consolidation')

    data['is_pivot'] = pivot_conditions

    # The actual pivot date is the day *before* the change
    pivot_dates = data[data['is_pivot']].index.to_series().shift(-1, freq='B').dropna()

    return list(pivot_dates.dt.normalize())

def get_sma(data, sma_period):
    df = data.copy()
    df['sma'] = df['Close'].rolling(window=sma_period).mean()
    return df.dropna(subset=['sma'])

def find_simple_pivots(data):
    """
    Returns a list of pivot dates defined as the day BEFORE price crosses the SMA.
    """
    df = data.copy()
    # Align yesterday's values
    close_y = df['Close'].shift(1)
    sma_y   = df['sma'].shift(1)

    # Today's vs yesterday's relation to SMA
    down_cross = (close_y > sma_y) & (df['Close'] < df['sma'])
    up_cross   = (close_y < sma_y) & (df['Close'] > df['sma'])

    # On any cross, pivot is 'yesterday'
    pivot_flags = down_cross | up_cross
    pivot_days  = df.index[pivot_flags].to_series().shift(1, freq='B').dropna()

    return list(pivot_days.dt.normalize())

# --- Streamlit UI ---
st.sidebar.header("⚙️ Analysis Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "SPY").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
sma_period = st.sidebar.slider("SMA Period", 5, 100, 5)
atr_period = st.sidebar.slider("ATR Period", 5, 50, 14)

if st.sidebar.button("Analyze Pivots"):
    if ticker:
        with st.spinner(f"Fetching data for {ticker} and analyzing pivots..."):
            # 1. Get Data and OPEX dates
            data = get_stock_data(ticker, start_date, end_date)

            if data is not None:
                opex_dates = get_opex_dates(start_date, end_date)

                # Ensure data index is datetime
                data.index = pd.to_datetime(data.index)

                # 2. Classify Trend and Find Pivots
                data = get_sma(data, sma_period=sma_period)   # or use your `sma_period` slider
                pivot_dates = find_simple_pivots(data)

                if not opex_dates:
                    st.warning("No OPEX dates found in the selected range.")
                elif not pivot_dates:
                    st.warning("No significant pivots found in the selected range with the current settings.")
                else:
                    st.success("Analysis Complete!")

                    # 3. Analyze pivots relative to OPEX
                    days_before_opex_list = []
                    pivot_details = []

                    # Create a trading day calendar from the data index
                    trading_days = data.index.normalize()

                    for pivot_date in pivot_dates:
                        # Find the next OPEX date after the pivot
                        future_opex = [od for od in opex_dates if od > pivot_date]
                        if future_opex:
                            next_opex = future_opex[0]

                            # Find previous opex date to define the window
                            past_opex_list = [od for od in opex_dates if od < pivot_date]
                            prev_opex = past_opex_list[-1] if past_opex_list else start_date

                            # Ensure pivot is within the current OPEX cycle
                            if pivot_date > prev_opex:
                                try:
                                    # Get the index location of both dates in our trading day series
                                    pivot_loc = trading_days.get_loc(pivot_date)
                                    opex_loc = trading_days.get_loc(next_opex)

                                    # The difference in index location is the number of trading days
                                    days_before = opex_loc - pivot_loc
                                    days_before_opex_list.append(days_before)

                                    pivot_details.append({
                                        "Pivot Date": pivot_date.strftime('%Y-%m-%d'),
                                        "Next OPEX Date": next_opex.strftime('%Y-%m-%d'),
                                        "Trading Days Before OPEX": days_before
                                    })
                                except KeyError:
                                    # This can happen if a pivot or opex date falls on a non-trading day not in our index
                                    pass

                    if not days_before_opex_list:
                        st.warning("Could not link any found pivots to an OPEX cycle.")
                    else:
                        # 4. Display Results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Pivot Frequency Analysis")
                            stats_df = pd.Series(days_before_opex_list).value_counts(normalize=True).mul(100).rename_axis('Days Before OPEX').reset_index(name='Percentage')
                            stats_df = stats_df.sort_values(by="Days Before OPEX")

                            st.bar_chart(stats_df.set_index("Days Before OPEX"))
                            st.dataframe(stats_df.style.format({"Percentage": "{:.2f}%"}), use_container_width=True)

                        with col2:
                            st.subheader("Detailed Pivot Log")
                            details_df = pd.DataFrame(pivot_details)
                            st.dataframe(details_df, height=400, use_container_width=True)

                        st.subheader("Trend and Pivot Chart")
                        st.markdown(f"Chart for **{ticker}** showing Close Price, SMA, Trend, and identified Pivot Points.")

                        chart_data = data.copy()
                        chart_data['pivot_marker'] = np.nan
                        valid_pivot_dates = [pd.to_datetime(p['Pivot Date']) for p in pivot_details]
                        chart_data.loc[chart_data.index.isin(valid_pivot_dates), 'pivot_marker'] = chart_data['High'] * 1.02

                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], name='Close', line=dict(color='skyblue')))
                        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['sma'], name=f'{sma_period}-day SMA', line=dict(color='orange', dash='dash')))

                        # Add pivot markers
                        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['pivot_marker'], name='Pivot', mode='markers', marker=dict(color='red', size=10, symbol='triangle-down')))

                        fig.update_layout(
                            title=f'{ticker} Price, SMA, and Pivots',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            legend_title='Legend',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)


    else:
        st.info("Please enter a stock ticker and click 'Analyze Pivots'.")

