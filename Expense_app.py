import streamlit as st
import pandas as pd
import plotly.express as px
import os
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 1. CONFIG & DATA LOAD ---
st.set_page_config(page_title="Personal Finance Pro", layout="wide")
FILE_NAME = "expenses.csv"

if not os.path.exists(FILE_NAME):
    df_init = pd.DataFrame(columns=["Date", "Category", "Amount", "Notes"])
    df_init.to_csv(FILE_NAME, index=False)

def load_data():
    data = pd.read_csv(FILE_NAME)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# --- 2. SIDEBAR: INPUT & BUDGET ---
st.sidebar.header("📝 Add Transaction")
date = st.sidebar.date_input("Date", datetime.date.today())
category = st.sidebar.selectbox("Category", ["Food", "Travel", "Rent", "Shopping", "Bills", "Health", "Other"])
amount = st.sidebar.number_input("Amount (₹)", min_value=0)
notes = st.sidebar.text_input("Short Note")

if st.sidebar.button("Save Expense"):
    new_entry = pd.DataFrame([[date, category, amount, notes]], columns=["Date", "Category", "Amount", "Notes"])
    new_entry.to_csv(FILE_NAME, mode='a', header=False, index=False)
    st.sidebar.success("Saved successfully!")
    st.rerun()

st.sidebar.divider()
st.sidebar.header("🎯 Budget Setting")
budget_limit = st.sidebar.number_input("Monthly Budget Goal (₹)", min_value=0, value=10000)

# --- 3. MAIN DASHBOARD ---
st.title("💰 Smart Expense Analytics")
df = load_data()

if not df.empty:
    total_spend = df['Amount'].sum()
    avg_spend = df['Amount'].mean()
    
    # Budget Alert
    if total_spend > budget_limit:
        st.error(f"🚨 Budget Exceeded! Total: ₹{total_spend} (Limit: ₹{budget_limit})")
    elif total_spend > (budget_limit * 0.8):
        st.warning(f"⚠️ Warning: you have reached 80% of your budget")

    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"₹{total_spend:,}")
    col2.metric("Transactions", len(df))
    col3.metric("Avg. per Expense", f"₹{int(avg_spend):,}")

    st.divider()

    # --- 4. CHARTS SECTION ---
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Category-wise Split")
        fig_pie = px.pie(df, values='Amount', names='Category', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("Daily Spending Trend")
        daily_df = df.groupby('Date')['Amount'].sum().reset_index()
        fig_bar = px.bar(daily_df, x='Date', y='Amount', color_discrete_sequence=['#febd69'])
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 5. LEVEL 3: AI FORECASTING ---
    st.divider()
    st.header("🤖 AI Spending Forecast")

    if len(df) >= 3:
        # Preprocessing for ML
        df['DayNum'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['DayNum']].values
        y = df['Amount'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediction for next 30 days
        last_day = df['DayNum'].max()
        future_day = np.array([[last_day + 30]])
        prediction = model.predict(future_day)[0]
        
        f_col1, f_col2 = st.columns(2)
        f_col1.metric("Predicted Next Big Expense", f"₹{int(prediction):,}")
        
        if prediction > avg_spend * 1.5:
            st.error("⚠️ AI Alert: chances of increased spending in upcoming days!")
        else:
            st.success("✅ AI Insight: Spending trend is currently stable.")

        # Trend Chart
        trend_line = model.predict(X)
        df['Trend'] = trend_line
        fig_trend = px.line(df, x='Date', y=['Amount', 'Trend'], title="Actual vs AI Trend Line")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("write three different dates for AI prediction.")

    # Raw Data Table
    with st.expander("See All Transactions"):
        st.table(df.sort_values(by="Date", ascending=False))
else:
    st.warning("No data available yet. Please add entries from the sidebar!")