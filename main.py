import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configure vectorbt
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']['width'] = 1000

st.set_page_config(page_title="Vectorbt Pivot Strategy", layout="wide")
st.title("🚀 Vectorbt-Powered Pivot Strategy Backtester")

# ---- Sidebar Parameters ----
st.sidebar.header("📊 Strategy Configuration")
symbol = st.sidebar.text_input("Symbol", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("💰 Portfolio Settings")
initial_cash = st.sidebar.number_input("Initial Capital ($)", value=100_000, step=1000)
fees = st.sidebar.slider("Transaction Fees (%)", 0.0, 0.1, 0.01) / 100
size_type = st.sidebar.selectbox("Position Sizing", ["Percent", "Target Percent", "Fixed"])
if size_type == "Percent":
    size = st.sidebar.slider("Position Size (% of Portfolio)", 10, 100, 10) / 100
elif size_type == "Target Percent":
    size = st.sidebar.slider("Target Position (% of Portfolio)", 10, 100, 50) / 100
else:
    size = st.sidebar.number_input("Fixed Position Size ($)", value=10000, step=1000)

st.sidebar.subheader("🎯 Signal Parameters")
min_move_threshold = st.sidebar.slider("Min Move Threshold (%)", 0.1, 5.0, 0.5) / 100
enable_atr_exits = st.sidebar.checkbox("Enable ATR-based Exits", True)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
profit_target_atr = st.sidebar.slider("Profit Target (ATR Multiple)", 0.5, 5.0, 2.0)
stop_loss_atr = st.sidebar.slider("Stop Loss (ATR Multiple)", 0.5, 3.0, 1.5)

st.sidebar.subheader("🎲 Monte Carlo & Optimization")
mc_runs = st.sidebar.slider("Monte Carlo Runs", 100, 2000, 1000)
enable_optimization = st.sidebar.checkbox("Enable Parameter Optimization", False)

@st.cache_data
def load_data(symbol, start, end, atr_period=14):
    """Load and prepare data with technical indicators"""
    try:
        # 1) Download data
        data = yf.download(symbol, start=start, end=end)

        # 2) If yfinance gave you a MultiIndex (e.g. (ticker, field)), drop that top level
        if isinstance(data.columns, pd.MultiIndex):
            # this will turn e.g. ('SPY','Close') → 'Close'
            data.columns = data.columns.droplevel(1)
        st.write(data)
        # 4) Clean & rename to exactly what you expect
        data = data.dropna()
        data = data.rename(columns={
            'Open':   'Open',
            'High':   'High',
            'Low':    'Low',
            'Close':  'Close',
            'Volume': 'Volume'
        })  # adjust if your download ever has slightly different labels

        # 5) Add weekday
        data['weekday'] = data.index.dayofweek

        # 6) ATR
        high_low        = data['High'] - data['Low']
        high_prev_close = (data['High'] - data['Close'].shift()).abs()
        low_prev_close  = (data['Low']  - data['Close'].shift()).abs()
        true_range      = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        data['ATR']     = true_range.rolling(window=atr_period, min_periods=1).mean()

        # 7) Returns
        data['returns'] = data['Close'].pct_change()

        return data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

class VectorbtPivotStrategy:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.close = data['Close']
        self.high = data['High']
        self.low = data['Low']
        self.atr = data['ATR']

    def generate_pivot_signals(self):
        """Generate pivot reversal signals for an arbitrary sequence of pivot weekdays."""
        pivots    = self.params.get('pivot_sequence', [2, 4])  # e.g. [Wed=2, Fri=4]
        threshold = self.params['min_move_threshold']
        use_atr   = self.params['enable_atr_exits']

        closes = self.data['Close']
        wdays  = self.data['weekday']

        # 1) Get all pivot-date indices
        pivot_dates = {
            d: self.data.index[wdays == d]
            for d in pivots
        }

        # 2) Precompute “previous pivot close” via forward‐fill
        prev_close = {}

        for i, d in enumerate(pivots):
            prev_d = pivots[i - 1]           # wraps around
            prev_close[d] = closes.where(wdays == prev_d).ffill()

        # 3) Build empty series
        entries   = pd.Series(False, index=self.data.index)
        exits     = pd.Series(False, index=self.data.index)
        direction = pd.Series(0,     index=self.data.index)
        signal_details = []
        position_open = False
        current_dir = 0

        # 4) Vectorized moves + mask + direction + scheduled exits
        for i, d in enumerate(pivots):
            dates = pivot_dates[d]
            # previous pivot closes for these dates
            pc = prev_close[d].loc[dates]

            # 1) compute raw moves, then immediately sanitize infinities
            moves = (closes.loc[dates] - pc) / pc
            moves = moves.replace([np.inf, -np.inf], np.nan)

            # 2) mask significant moves
            mask = moves.abs() >= threshold
            if not mask.any():
                continue

            # 3) mark entries
            entries.loc[dates[mask]] = True

            # 4) build a direction series defaulting to 0
            dir_vec = pd.Series(0, index=dates)

            # only on masked dates do we assign +1 / –1
            # note: np.sign of a non-nan is always ±1 or 0
            signed = -np.sign(moves.loc[mask])          # negative sign flips move→direction
            dir_vec.loc[mask] = signed.astype(int)     # now safe, since no NaNs here

            # stick that back into the master direction Series
            direction.loc[dates] = dir_vec


            # exits: next pivot day (unless ATR exits)
            if not use_atr:
                next_dates = pivot_dates[pivots[(i + 1) % len(pivots)]].values
                # for each entry-date find its index in next_dates
                pos = np.searchsorted(next_dates, dates.values)
                valid = (pos < len(next_dates)) & mask.values
                exit_dates = next_dates[pos[valid]]
                exits.loc[exit_dates] = True
            else:
                exit_dates = np.array([pd.NaT] * mask.sum())

            for dt, mv in zip(dates[mask], moves[mask]):
                # Always reverse direction
                new_dir = -np.sign(mv)

                previous_dt = self.data.index[self.data.index.get_loc(dt) - 1] if self.data.index.get_loc(dt) > 0 else dt
                if position_open and previous_dt in exits.index:
                    exits.loc[previous_dt] = True

                # Open new trade
                entries.loc[dt] = True
                direction.loc[dt] = new_dir
                current_dir = new_dir
                position_open = True

                signal_details.append({
                    'entry_date': dt,
                    'pivot_type': d,
                    'trend_move': mv,
                    'direction': new_dir,
                    'scheduled_exit': None  # we'll use ATR logic later
                })


        return entries, exits, direction, signal_details

    def generate_atr_exits(self, entries, direction):
        """Generate ATR-based profit targets and stop losses"""
        if not self.params['enable_atr_exits']:
            return pd.Series(False, index=self.data.index), pd.Series(False, index=self.data.index)

        current_entry_date = None

        # Initialize exit signals
        profit_exits = pd.Series(False, index=self.data.index)
        stop_exits = pd.Series(False, index=self.data.index)

        # Track open positions
        position = 0
        entry_price = 0
        entry_atr = 0

        for i, date in enumerate(self.data.index):
            if entries[date]:
                position = direction[date]
                entry_price = self.close[date]
                entry_atr = self.atr[date]
                current_entry_date = date
                continue

            if position != 0 and pd.notna(entry_atr):
                profit_target = entry_price + position * entry_atr * self.params['profit_target_atr']
                stop_loss     = entry_price - position * entry_atr * self.params['stop_loss_atr']

                if position > 0:
                    if self.high[date] >= profit_target:
                        profit_exits[date] = True
                        position = 0
                        continue
                    elif self.low[date] <= stop_loss:
                        stop_exits[date] = True
                        position = 0
                        continue
                else:
                    if self.low[date] <= profit_target:
                        profit_exits[date] = True
                        position = 0
                        continue
                    elif self.high[date] >= stop_loss:
                        stop_exits[date] = True
                        position = 0
                        continue
        return profit_exits, stop_exits

    def run_backtest(self):
        """Run vectorbt backtest"""
        # Generate signals
        entries, exits, direction, signal_details = self.generate_pivot_signals()

        # Generate ATR exits if enabled
        if self.params['enable_atr_exits']:
            profit_exits, stop_exits = self.generate_atr_exits(entries, direction)
            # Combine exits
            exits = exits | profit_exits | stop_exits

        # Create size array based on direction
        size_array = direction.copy()

        # 1. Split entry signals
        long_entries = entries & (direction == 1)
        short_entries = entries & (direction == -1)
        # long_entries = long_entries.shift(1, fill_value=False)
        # short_entries = short_entries.shift(1, fill_value=False)

        size_param = self.params['size'] * self.params['initial_cash']
        size_type_vbt = 'value'

        portfolio = vbt.Portfolio.from_signals(
            close=self.close,
            entries=long_entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=exits,
            size=size_param,
            size_type=size_type_vbt,
            init_cash=self.params['initial_cash'],
            fees=self.params['fees'],
            freq='D',
        )

        return portfolio, signal_details

def run_monte_carlo(portfolio, runs=1000):
    """Run Monte Carlo simulation by shuffling trade returns"""
    try:
        # Get trade returns
        trades = portfolio.trades.records_readable
        if len(trades) == 0:
            return None

        returns = trades['Return'].values

        # Run Monte Carlo
        final_values = []
        max_drawdowns = []

        for _ in range(runs):
            # Shuffle returns
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate cumulative performance
            cumulative = np.cumprod(1 + shuffled_returns / 100)  # Assuming returns are in percentage
            final_value = cumulative[-1] * portfolio.init_cash

            # Calculate max drawdown
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            max_dd = np.max(drawdown)

            final_values.append(final_value)
            max_drawdowns.append(max_dd)

        return {
            'final_values': final_values,
            'max_drawdowns': max_drawdowns,
            'percentiles': {
                p: np.percentile(final_values, p) for p in [5, 10, 25, 50, 75, 90, 95]
            }
        }
    except Exception as e:
        st.error(f"Monte Carlo simulation failed: {e}")
        return None

def optimize_parameters(data, param_ranges):
    """Optimize strategy parameters using vectorbt"""
    try:
        # Create parameter combinations
        min_moves = np.arange(param_ranges['min_move'][0], param_ranges['min_move'][1], 0.001)
        atr_profits = np.arange(param_ranges['atr_profit'][0], param_ranges['atr_profit'][1], 0.5)
        atr_stops = np.arange(param_ranges['atr_stop'][0], param_ranges['atr_stop'][1], 0.25)

        best_sharpe = -np.inf
        best_params = None
        results = []

        # Limited optimization due to computation time
        sample_combinations = 50  # Limit combinations for Streamlit

        for i, min_move in enumerate(np.random.choice(min_moves, min(len(min_moves), 10))):
            for j, atr_profit in enumerate(np.random.choice(atr_profits, min(len(atr_profits), 5))):
                for k, atr_stop in enumerate(np.random.choice(atr_stops, min(len(atr_stops), 5))):
                    if i * len(atr_profits) * len(atr_stops) + j * len(atr_stops) + k >= sample_combinations:
                        break

                    # Test parameters
                    test_params = {
                        'min_move_threshold': min_move,
                        'profit_target_atr': atr_profit,
                        'stop_loss_atr': atr_stop,
                        'enable_atr_exits': True,
                        'initial_cash': 100000,
                        'fees': 0.0001,
                        'size': 0.25,
                        'size_type': 'Percent'
                    }

                    try:
                        strategy = VectorbtPivotStrategy(data, test_params)
                        portfolio, _ = strategy.run_backtest()

                        if portfolio.trades.count > 5:  # Minimum trades for valid test
                            sharpe = portfolio.sharpe_ratio
                            total_return = portfolio.total_return

                            results.append({
                                'min_move': min_move,
                                'atr_profit': atr_profit,
                                'atr_stop': atr_stop,
                                'sharpe': sharpe,
                                'total_return': total_return,
                                'trades': portfolio.trades.count
                            })

                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_params = test_params.copy()
                    except:
                        continue

        return pd.DataFrame(results), best_params
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None, None

# Load data
data = load_data(symbol, start_date, end_date)
if data is None:
    st.stop()

st.success(f"✅ Loaded {len(data)} days of {symbol} data")

# Prepare parameters
params = {
    'min_move_threshold': min_move_threshold,
    'enable_atr_exits': enable_atr_exits,
    'profit_target_atr': profit_target_atr,
    'stop_loss_atr': stop_loss_atr,
    'initial_cash': initial_cash,
    'fees': fees,
    'size': size,
    'size_type': size_type
}

# Create strategy
strategy = VectorbtPivotStrategy(data, params)

# Run backtest
with st.spinner("🚀 Running vectorbt backtest..."):
    portfolio, signal_details = strategy.run_backtest()

# Display results
st.header("📊 Backtest Results")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Return", f"{portfolio.total_return():.1%}")
with col2:
    st.metric("Sharpe Ratio", f"{portfolio.sharpe_ratio():.2f}")
with col3:
    st.metric("Max Drawdown", f"{portfolio.max_drawdown():.1%}")
with col4:
    st.metric("Total Trades", portfolio.trades.count())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Win Rate", f"{portfolio.trades.win_rate():.1%}")
with col2:
    st.metric("Profit Factor", f"{portfolio.trades.profit_factor():.2f}")
with col3:
    st.metric("Avg Return/Trade", f"{portfolio.trades.returns.mean():.2%}")
with col4:
    st.metric("Final Value", f"${portfolio.value().iloc[-1]:,.0f}")

# Plot portfolio performance
st.subheader("📈 Portfolio Performance")
fig = portfolio.plot()
st.plotly_chart(fig, use_container_width=True)

# Plot drawdown
st.subheader("📉 Drawdown Analysis")
dd_fig = portfolio.drawdowns.plot()
st.plotly_chart(dd_fig, use_container_width=True)

# Signal analysis
st.subheader("🎯 Signal Analysis")
signal_df = pd.DataFrame(signal_details)
if not signal_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        pivot_counts = signal_df['pivot_type'].value_counts()
        st.write("**Signals by Pivot Type:**")
        for pivot_type, count in pivot_counts.items():
            st.write(f"- {pivot_type}: {count}")

    with col2:
        direction_counts = signal_df['direction'].value_counts()
        st.write("**Signals by Direction:**")
        long_signals = len(signal_df[signal_df['direction'] == 1])
        short_signals = len(signal_df[signal_df['direction'] == -1])
        st.write(f"- Long: {long_signals}")
        st.write(f"- Short: {short_signals}")

# Monte Carlo simulation
if st.button("🎲 Run Monte Carlo Simulation"):
    with st.spinner(f"Running {mc_runs} Monte Carlo iterations..."):
        mc_results = run_monte_carlo(portfolio, mc_runs)

        if mc_results:
            st.subheader("🎲 Monte Carlo Results")

            # Display percentiles
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("5th Percentile", f"${mc_results['percentiles'][5]:,.0f}")
            with col2:
                st.metric("50th Percentile", f"${mc_results['percentiles'][50]:,.0f}")
            with col3:
                st.metric("95th Percentile", f"${mc_results['percentiles'][95]:,.0f}")

            # Plot distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=mc_results['final_values'],
                nbinsx=50,
                name="Final Portfolio Values"
            ))
            fig.add_vline(x=initial_cash, line_dash="dash", line_color="red",
                        annotation_text="Initial Capital")
            fig.add_vline(x=portfolio.value.iloc[-1], line_dash="dash", line_color="green",
                        annotation_text="Actual Result")
            fig.update_layout(
                title="Monte Carlo Distribution of Final Portfolio Values",
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

# Parameter optimization
if enable_optimization and st.button("🔧 Optimize Parameters"):
    with st.spinner("Optimizing parameters..."):
        param_ranges = {
            'min_move': (0.002, 0.02),
            'atr_profit': (1.0, 4.0),
            'atr_stop': (0.5, 2.5)
        }

        opt_results, best_params = optimize_parameters(data, param_ranges)

        if opt_results is not None and not opt_results.empty:
            st.subheader("🔧 Optimization Results")

            # Show best parameters
            if best_params:
                st.write("**Best Parameters:**")
                st.write(f"- Min Move Threshold: {best_params['min_move_threshold']:.3f}")
                st.write(f"- Profit Target ATR: {best_params['profit_target_atr']:.1f}")
                st.write(f"- Stop Loss ATR: {best_params['stop_loss_atr']:.1f}")

            # Show optimization results
            st.write("**Top 10 Parameter Combinations:**")
            top_results = opt_results.nlargest(10, 'sharpe')
            st.dataframe(top_results.round(4))

# Detailed trade analysis
st.subheader("📋 Trade Details")
trades_df = portfolio.trades.records_readable.copy()

if len(trades_df) > 0:
    # Format trades dataframe

    display_trades = trades_df[[
        'Entry Timestamp',
        'Exit Timestamp',
        'Direction',
        'Size',
        'Avg Entry Price',
        'Avg Exit Price',
        'Return',
        'Status'
    ]].copy()

    display_trades['Return'] = display_trades['Return'].round(2)
    display_trades['Avg Entry Price'] = display_trades['Avg Entry Price'].round(2)
    display_trades['Avg Exit Price'] = display_trades['Avg Exit Price'].round(2)

    st.dataframe(display_trades, use_container_width=True)

    # Trade statistics
    st.write("**Trade Statistics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Average Trade Duration: {portfolio.trades.duration.mean():.1f} days")
    with col2:
        st.write(f"Best Trade: {portfolio.trades.returns.max():.2%}")
    with col3:
        st.write(f"Worst Trade: {portfolio.trades.returns.min():.2%}")

# Strategy information
with st.expander("📖 Strategy Information"):
    st.markdown("""
    ## Vectorbt Pivot Strategy Features

    ### Core Strategy:
    - **Wednesday Pivots**: Reverse Friday→Wednesday trends
    - **Friday Pivots**: Reverse Wednesday→Friday trends
    - **Signal Filter**: Only trade moves above minimum threshold

    ### Vectorbt Advantages:
    - ⚡ **Fast Execution**: Vectorized operations for speed
    - 📊 **Rich Analytics**: Built-in performance metrics
    - 🎲 **Monte Carlo**: Statistical confidence analysis
    - 🔧 **Optimization**: Parameter tuning capabilities
    - 📈 **Visualization**: Professional plotting tools

    ### Risk Management:
    - **ATR-based Exits**: Volatility-adjusted targets and stops
    - **Position Sizing**: Flexible sizing methods
    - **Transaction Costs**: Realistic fee modeling

    ### Recommended Workflow:
    1. Start with default parameters
    2. Analyze signal quality and trade statistics
    3. Run Monte Carlo to understand risk
    4. Optimize parameters if needed
    5. Test on out-of-sample data
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Vectorbt Pro**: This implementation leverages vectorbt's speed and analytics for professional-grade backtesting.")
