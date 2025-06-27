import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.title("Pivot Reversal Analysis: Wed & Fri Pivots")

# ---- Sidebar parameters ----
symbol = st.sidebar.text_input("Symbol", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100_000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 20, 5) / 100.0
min_move_threshold = st.sidebar.slider("Min Move for Signal (%)", 0.1, 3.0, 0.5) / 100.0

@st.cache_data
def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["weekday"] = df.index.dayofweek  # Monday=0, ..., Friday=4
    return df

df = get_data(symbol, start_date, end_date)
if df.empty:
    st.error("No data loaded. Check ticker and date range.")
    st.stop()

def analyze_wednesday_pivots(df, min_move_threshold=0.005):
    """
    Analyze Wednesday pivots: Does Wed reverse the Fri->Wed trend?
    """
    WED, FRI = 2, 4
    
    wednesdays = df[df["weekday"] == WED].copy()
    fridays = df[df["weekday"] == FRI].copy()
    
    pivot_analysis = []
    
    for wed_idx in range(len(wednesdays)):
        wed_date = wednesdays.index[wed_idx]
        wed_close = float(wednesdays.iloc[wed_idx]["Close"])
        
        # Find the previous Friday
        previous_fridays = fridays[fridays.index < wed_date]
        if len(previous_fridays) == 0:
            continue
        
        prev_fri_date = previous_fridays.index[-1]  # Most recent Friday before this Wed
        prev_fri_close = float(previous_fridays.iloc[-1]["Close"])
        
        # Calculate the trend from Fri->Wed
        fri_to_wed_move = (wed_close - prev_fri_close) / prev_fri_close
        fri_to_wed_direction = 1 if fri_to_wed_move > 0 else -1
        
        # Only analyze if the Fri->Wed move was significant
        if abs(fri_to_wed_move) < min_move_threshold:
            continue
            
        # Find the next Friday to see if Wed was a pivot
        future_fridays = fridays[fridays.index > wed_date]
        if len(future_fridays) == 0:
            continue
            
        next_fri_date = future_fridays.index[0]
        next_fri_close = float(future_fridays.iloc[0]["Close"])
        
        # Calculate the move from Wed->Fri
        wed_to_fri_move = (next_fri_close - wed_close) / wed_close
        wed_to_fri_direction = 1 if wed_to_fri_move > 0 else -1
        
        # Check if Wednesday was a reversal pivot
        is_reversal = (abs(wed_to_fri_move) > min_move_threshold and 
                      fri_to_wed_direction != wed_to_fri_direction)
        
        is_continuation = (abs(wed_to_fri_move) > min_move_threshold and 
                          fri_to_wed_direction == wed_to_fri_direction)
        
        # Calculate potential trade return if we traded the reversal
        # Enter on Wed close, exit on Fri close, direction opposite to Fri->Wed trend
        trade_direction = -fri_to_wed_direction  # Opposite to the trend we're reversing
        trade_return = trade_direction * wed_to_fri_move
        
        # Debug: Add actual trade calculation
        if trade_direction > 0:  # Long trade
            actual_return = (next_fri_close - wed_close) / wed_close
        else:  # Short trade
            actual_return = (wed_close - next_fri_close) / wed_close
        
        trade_return = actual_return
        
        pivot_analysis.append({
            'pivot_date': wed_date,
            'prev_fri_date': prev_fri_date,
            'next_fri_date': next_fri_date,
            'prev_fri_close': prev_fri_close,
            'wed_close': wed_close,
            'next_fri_close': next_fri_close,
            'fri_to_wed_move': fri_to_wed_move,
            'wed_to_fri_move': wed_to_fri_move,
            'fri_to_wed_direction': 'UP' if fri_to_wed_direction > 0 else 'DOWN',
            'wed_to_fri_direction': 'UP' if wed_to_fri_direction > 0 else 'DOWN',
            'is_reversal': is_reversal,
            'is_continuation': is_continuation,
            'pattern_type': 'Reversal' if is_reversal else ('Continuation' if is_continuation else 'Weak_Signal'),
            'trade_return': trade_return,
            'trade_direction': 'LONG' if trade_direction > 0 else 'SHORT'
        })
    
    return pd.DataFrame(pivot_analysis)

def analyze_friday_pivots(df, min_move_threshold=0.005):
    """
    Analyze Friday pivots: Does Fri reverse the Wed->Fri trend?
    """
    WED, FRI = 2, 4
    
    wednesdays = df[df["weekday"] == WED].copy()
    fridays = df[df["weekday"] == FRI].copy()
    
    pivot_analysis = []
    
    for fri_idx in range(len(fridays)):
        fri_date = fridays.index[fri_idx]
        fri_close = float(fridays.iloc[fri_idx]["Close"])
        
        # Find the previous Wednesday
        previous_wednesdays = wednesdays[wednesdays.index < fri_date]
        if len(previous_wednesdays) == 0:
            continue
        
        prev_wed_date = previous_wednesdays.index[-1]  # Most recent Wed before this Fri
        prev_wed_close = float(previous_wednesdays.iloc[-1]["Close"])
        
        # Calculate the trend from Wed->Fri
        wed_to_fri_move = (fri_close - prev_wed_close) / prev_wed_close
        wed_to_fri_direction = 1 if wed_to_fri_move > 0 else -1
        
        # Only analyze if the Wed->Fri move was significant
        if abs(wed_to_fri_move) < min_move_threshold:
            continue
            
        # Find the next Wednesday to see if Fri was a pivot
        future_wednesdays = wednesdays[wednesdays.index > fri_date]
        if len(future_wednesdays) == 0:
            continue
            
        next_wed_date = future_wednesdays.index[0]
        next_wed_close = float(future_wednesdays.iloc[0]["Close"])
        
        # Calculate the move from Fri->Wed
        fri_to_wed_move = (next_wed_close - fri_close) / fri_close
        fri_to_wed_direction = 1 if fri_to_wed_move > 0 else -1
        
        # Check if Friday was a reversal pivot
        is_reversal = (abs(fri_to_wed_move) > min_move_threshold and 
                      wed_to_fri_direction != fri_to_wed_direction)
        
        is_continuation = (abs(fri_to_wed_move) > min_move_threshold and 
                          wed_to_fri_direction == fri_to_wed_direction)
        
        # Calculate potential trade return if we traded the reversal
        # Enter on Fri close, exit on Wed close, direction opposite to Wed->Fri trend
        trade_direction = -wed_to_fri_direction  # Opposite to the trend we're reversing
        
        # Debug: Add actual trade calculation
        if trade_direction > 0:  # Long trade
            actual_return = (next_wed_close - fri_close) / fri_close
        else:  # Short trade
            actual_return = (fri_close - next_wed_close) / fri_close
        
        trade_return = actual_return
        
        pivot_analysis.append({
            'pivot_date': fri_date,
            'prev_wed_date': prev_wed_date,
            'next_wed_date': next_wed_date,
            'prev_wed_close': prev_wed_close,
            'fri_close': fri_close,
            'next_wed_close': next_wed_close,
            'wed_to_fri_move': wed_to_fri_move,
            'fri_to_wed_move': fri_to_wed_move,
            'wed_to_fri_direction': 'UP' if wed_to_fri_direction > 0 else 'DOWN',
            'fri_to_wed_direction': 'UP' if fri_to_wed_direction > 0 else 'DOWN',
            'is_reversal': is_reversal,
            'is_continuation': is_continuation,
            'pattern_type': 'Reversal' if is_reversal else ('Continuation' if is_continuation else 'Weak_Signal'),
            'trade_return': trade_return,
            'trade_direction': 'LONG' if trade_direction > 0 else 'SHORT'
        })
    
    return pd.DataFrame(pivot_analysis)

def backtest_strategy(pivot_df, initial_capital, risk_per_trade):
    """
    Backtest the pivot reversal strategy
    """
    if pivot_df.empty:
        return pd.DataFrame(), pd.Series([initial_capital])
    
    # Filter for reversal trades only
    reversal_trades = pivot_df[pivot_df['is_reversal']].copy()
    
    if reversal_trades.empty:
        return pd.DataFrame(), pd.Series([initial_capital])
    
    # Calculate position sizes and P&L
    capital = initial_capital
    equity_curve = []
    trade_details = []
    
    for _, trade in reversal_trades.iterrows():
        position_size = capital * risk_per_trade
        trade_pnl = position_size * trade['trade_return']
        capital += trade_pnl
        
        trade_details.append({
            'entry_date': trade['pivot_date'],
            'exit_date': trade.get('next_fri_date', trade.get('next_wed_date')),
            'direction': trade['trade_direction'],
            'return_pct': trade['trade_return'] * 100,
            'position_size': position_size,
            'pnl': trade_pnl,
            'equity': capital
        })
        
        equity_curve.append(capital)
    
    trades_df = pd.DataFrame(trade_details)
    equity_series = pd.Series(equity_curve)
    
    return trades_df, equity_series

# Run the analysis
st.subheader("Pivot Analysis Results")

tab1, tab2, tab3 = st.tabs(["Wednesday Pivots", "Friday Pivots", "Combined Strategy"])

with tab1:
    st.write("### Wednesday Pivot Analysis")
    st.write("*Analyzing if Wednesday reverses the Friday→Wednesday trend*")
    
    wed_patterns = analyze_wednesday_pivots(df, min_move_threshold)
    
    if not wed_patterns.empty:
        # Summary statistics
        total_patterns = len(wed_patterns)
        reversals = int(wed_patterns['is_reversal'].sum())
        continuations = int(wed_patterns['is_continuation'].sum())
        weak_signals = total_patterns - reversals - continuations
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", total_patterns)
        with col2:
            st.metric("Reversals", f"{reversals} ({reversals/total_patterns:.1%})")
        with col3:
            st.metric("Continuations", f"{continuations} ({continuations/total_patterns:.1%})")
        with col4:
            st.metric("Weak Signals", f"{weak_signals} ({weak_signals/total_patterns:.1%})")
        
        # Reversal trading performance
        if reversals > 0:
            reversal_trades = wed_patterns[wed_patterns['is_reversal']]
            avg_return = reversal_trades['trade_return'].mean()
            win_rate = (reversal_trades['trade_return'] > 0).mean()
            
            st.write("#### Wednesday Reversal Trade Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Return per Trade", f"{avg_return:.2%}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1%}")
        
        # Show detailed data
        st.write("#### Detailed Wednesday Pivot Data")
        display_df = wed_patterns[['pivot_date', 'fri_to_wed_move', 'wed_to_fri_move', 
                                  'fri_to_wed_direction', 'wed_to_fri_direction', 
                                  'pattern_type', 'trade_return']].copy()
        display_df['fri_to_wed_move'] = (display_df['fri_to_wed_move'] * 100).round(2)
        display_df['wed_to_fri_move'] = (display_df['wed_to_fri_move'] * 100).round(2)
        display_df['trade_return'] = (display_df['trade_return'] * 100).round(2)
        st.dataframe(display_df)
    else:
        st.write("No significant Wednesday patterns found with current parameters.")

with tab2:
    st.write("### Friday Pivot Analysis")
    st.write("*Analyzing if Friday reverses the Wednesday→Friday trend*")
    
    fri_patterns = analyze_friday_pivots(df, min_move_threshold)
    
    if not fri_patterns.empty:
        # Summary statistics
        total_patterns = len(fri_patterns)
        reversals = int(fri_patterns['is_reversal'].sum())
        continuations = int(fri_patterns['is_continuation'].sum())
        weak_signals = total_patterns - reversals - continuations
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", total_patterns)
        with col2:
            st.metric("Reversals", f"{reversals} ({reversals/total_patterns:.1%})")
        with col3:
            st.metric("Continuations", f"{continuations} ({continuations/total_patterns:.1%})")
        with col4:
            st.metric("Weak Signals", f"{weak_signals} ({weak_signals/total_patterns:.1%})")
        
        # Reversal trading performance
        if reversals > 0:
            reversal_trades = fri_patterns[fri_patterns['is_reversal']]
            avg_return = reversal_trades['trade_return'].mean()
            win_rate = (reversal_trades['trade_return'] > 0).mean()
            
            st.write("#### Friday Reversal Trade Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Return per Trade", f"{avg_return:.2%}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1%}")
        
        # Show detailed data
        st.write("#### Detailed Friday Pivot Data")
        display_df = fri_patterns[['pivot_date', 'wed_to_fri_move', 'fri_to_wed_move', 
                                  'wed_to_fri_direction', 'fri_to_wed_direction', 
                                  'pattern_type', 'trade_return']].copy()
        display_df['wed_to_fri_move'] = (display_df['wed_to_fri_move'] * 100).round(2)
        display_df['fri_to_wed_move'] = (display_df['fri_to_wed_move'] * 100).round(2)
        display_df['trade_return'] = (display_df['trade_return'] * 100).round(2)
        st.dataframe(display_df)
    else:
        st.write("No significant Friday patterns found with current parameters.")

with tab3:
    st.write("### Combined Pivot Strategy")
    
    # Combine both strategies
    all_trades = []
    
    if 'wed_patterns' in locals() and not wed_patterns.empty:
        wed_reversal_trades = wed_patterns[wed_patterns['is_reversal']].copy()
        if not wed_reversal_trades.empty:
            st.write(f"Found {len(wed_reversal_trades)} Wednesday reversal trades")
            for _, trade in wed_reversal_trades.iterrows():
                all_trades.append({
                    'entry_date': trade['pivot_date'],
                    'exit_date': trade['next_fri_date'],
                    'pivot_type': 'Wednesday',
                    'direction': trade['trade_direction'],
                    'return': trade['trade_return'],
                    'entry_price': trade['wed_close'],
                    'exit_price': trade['next_fri_close']
                })
    
    if 'fri_patterns' in locals() and not fri_patterns.empty:
        fri_reversal_trades = fri_patterns[fri_patterns['is_reversal']].copy()
        if not fri_reversal_trades.empty:
            st.write(f"Found {len(fri_reversal_trades)} Friday reversal trades")
            for _, trade in fri_reversal_trades.iterrows():
                all_trades.append({
                    'entry_date': trade['pivot_date'],
                    'exit_date': trade['next_wed_date'],
                    'pivot_type': 'Friday',
                    'direction': trade['trade_direction'],
                    'return': trade['trade_return'],
                    'entry_price': trade['fri_close'],
                    'exit_price': trade['next_wed_close']
                })
    
    if all_trades:
        combined_df = pd.DataFrame(all_trades)
        combined_df = combined_df.sort_values('entry_date')
        
        # Debug: Show some sample trades
        st.write("#### Sample Trades (First 10)")
        sample_trades = combined_df.head(10).copy()
        sample_trades['return_pct'] = (sample_trades['return'] * 100).round(3)
        st.dataframe(sample_trades[['entry_date', 'exit_date', 'pivot_type', 'direction', 
                                   'entry_price', 'exit_price', 'return_pct']])
        
        # Check for anomalies
        st.write("#### Trade Return Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram of returns
        ax1.hist(combined_df['return'] * 100, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Trade Returns (%)')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(combined_df['return'] * 100)
        ax2.set_title('Trade Returns Box Plot')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Portfolio simulation
        capital = initial_capital
        equity_curve = [capital]
        
        for _, trade in combined_df.iterrows():
            position_size = capital * risk_per_trade
            trade_pnl = position_size * trade['return']
            capital += trade_pnl
            equity_curve.append(capital)
        
        # Performance metrics
        total_trades = len(combined_df)
        winning_trades = (combined_df['return'] > 0).sum()
        losing_trades = (combined_df['return'] < 0).sum()
        breakeven_trades = (combined_df['return'] == 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = (capital - initial_capital) / initial_capital
        avg_return = combined_df['return'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1%}")
        with col3:
            st.metric("Total Return", f"{total_return:.1%}")
        with col4:
            st.metric("Avg Return/Trade", f"{avg_return:.2%}")
            
        # Additional metrics
        st.write("#### Detailed Performance Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Winning Trades", winning_trades)
        with col2:
            st.metric("Losing Trades", losing_trades)
        with col3:
            st.metric("Break-even Trades", breakeven_trades)
        
        if winning_trades > 0:
            avg_win = combined_df[combined_df['return'] > 0]['return'].mean()
            st.metric("Average Win", f"{avg_win:.2%}")
        
        if losing_trades > 0:
            avg_loss = combined_df[combined_df['return'] < 0]['return'].mean()
            st.metric("Average Loss", f"{avg_loss:.2%}")
        
        # Equity curve
        st.write("#### Equity Curve")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(equity_curve)), equity_curve)
        ax.set_title(f'{symbol} - Combined Pivot Reversal Strategy')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Trade details
        st.write("#### All Trades")
        display_combined = combined_df.copy()
        display_combined['return_pct'] = (display_combined['return'] * 100).round(3)
        st.dataframe(display_combined[['entry_date', 'exit_date', 'pivot_type', 'direction', 
                                     'entry_price', 'exit_price', 'return_pct']])
        
    else:
        st.write("No reversal trades found with current parameters.")

# Strategy Summary
st.subheader("Strategy Setup Summary")

st.write(f"""
**Current Analysis Parameters:**
- **Symbol**: {symbol}
- **Minimum Move Threshold**: {min_move_threshold:.1%}
- **Risk per Trade**: {risk_per_trade:.1%}

**Wednesday Pivot Strategy:**
- **Setup**: Look for significant Friday→Wednesday move
- **Entry**: Wednesday close (betting on reversal)
- **Exit**: Following Friday close
- **Direction**: Opposite to the Friday→Wednesday trend

**Friday Pivot Strategy:**
- **Setup**: Look for significant Wednesday→Friday move  
- **Entry**: Friday close (betting on reversal)
- **Exit**: Following Wednesday close
- **Direction**: Opposite to the Wednesday→Friday trend

**Key Metrics to Monitor:**
- Reversal percentage vs continuation percentage
- Average magnitude of moves before/after pivots
- Win rate of reversal trades
- Optimal minimum move threshold for signal quality
""")

st.sidebar.markdown("---")
st.sidebar.write("**Tips:**")
st.sidebar.write("- Increase min move threshold to filter out noise")
st.sidebar.write("- Lower threshold captures more signals but may reduce quality")
st.sidebar.write("- Monitor both pivot types for different market conditions")