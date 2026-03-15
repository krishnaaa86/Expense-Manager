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

def load_data():
    if not os.path.exists(FILE_NAME) or os.stat(FILE_NAME).st_size == 0:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Notes"])
    data = pd.read_csv(FILE_NAME)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def save_data(df):
    df.to_csv(FILE_NAME, index=False)

df = load_data()

# --- 2. SIDEBAR: INPUT ---
st.sidebar.header("📝 Add Transaction")
date = st.sidebar.date_input("Date", datetime.date.today())
category = st.sidebar.selectbox("Category", ["Food", "Travel", "Rent", "Shopping", "Bills", "Health", "Other"])
amount = st.sidebar.number_input("Amount (₹)", min_value=0)
notes = st.sidebar.text_input("Short Note")

if st.sidebar.button("Save Expense"):
    new_entry = pd.DataFrame([[pd.to_datetime(date), category, amount, notes]], columns=["Date", "Category", "Amount", "Notes"])
    df = pd.concat([df, new_entry], ignore_index=True)
    save_data(df)
    st.sidebar.success("Saved successfully!")
    st.rerun()

st.sidebar.divider()
budget_limit = st.sidebar.number_input("Monthly Budget Goal (₹)", min_value=0, value=10000)

# --- 3. MAIN DASHBOARD ---
st.title("💰 Smart Expense Analytics")

if not df.empty:
    total_spend = df['Amount'].sum()
    remaining = budget_limit - total_spend
    
    # --- ENHANCED BUDGET ALERTS ---
    if total_spend > budget_limit:
        over_by = total_spend - budget_limit
        st.error(f"🚨 Budget Exceeded by ₹{over_by:,}! (Total Spend: ₹{total_spend:,})")
    elif total_spend > (budget_limit * 0.8):
        st.warning(f"⚠️ 80% Budget Reached! You have only ₹{remaining:,} left before hitting the limit.")
    else:
        st.success(f"✅ You are within budget! ₹{remaining:,} remaining.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"₹{total_spend:,}")
    col2.metric("Transactions", len(df))
    col3.metric("Avg. Spend", f"₹{int(df['Amount'].mean()):,}")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, values='Amount', names='Category', hole=0.4, title="Category Split"), use_container_width=True)
    with c2:
        daily = df.groupby('Date')['Amount'].sum().reset_index()
        st.plotly_chart(px.bar(daily, x='Date', y='Amount', title="Daily Trend"), use_container_width=True)

    # --- 4. AI FORECASTING ---
    st.divider()
    st.header("🤖 AI Spending Forecast")
    if len(df) >= 3:
        df['DayNum'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['DayNum']].values
        y = df['Amount'].values
        model = LinearRegression().fit(X, y)
        prediction = model.predict([[df['DayNum'].max() + 30]])[0]
        
        f_col1, f_col2 = st.columns(2)
        f_col1.metric("Predicted Next Month's Expense", f"₹{max(0, int(prediction)):,}")
        st.info("AI Analysis: Based on your current habits, this is your estimated future spend.")
    else:
        st.info("Add at least 3 transactions for AI insights.")

    # --- 5. DATA TABLE & MANAGEMENT ---
    st.divider()
    st.header("⚙️ History & Management")
    
    with st.expander("📄 View Full Transaction History", expanded=True):
        st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)

    tab1, tab2 = st.tabs(["Edit/Delete Entry", "Danger Zone"])
    with tab1:
        edit_idx = st.selectbox("Select Transaction to Edit/Delete", options=df.index, 
                               format_func=lambda x: f"{df.iloc[x]['Date'].date()} - {df.iloc[x]['Category']} - ₹{df.iloc[x]['Amount']}")
        row = df.iloc[edit_idx]
        col_ed1, col_ed2 = st.columns(2)
        new_amt = col_ed1.number_input("Update Amount", value=int(row['Amount']), key="upd_amt")
        new_cat = col_ed2.selectbox("Update Category", ["Food", "Travel", "Rent", "Shopping", "Bills", "Health", "Other"], 
                                   index=["Food", "Travel", "Rent", "Shopping", "Bills", "Health", "Other"].index(row['Category']), key="upd_cat")
        
        b1, b2 = st.columns(2)
        if b1.button("Update Entry", type="primary"):
            df.at[edit_idx, 'Amount'] = new_amt
            df.at[edit_idx, 'Category'] = new_cat
            save_data(df)
            st.success("Updated!")
            st.rerun()
        if b2.button("Delete Selected Entry"):
            df = df.drop(edit_idx)
            save_data(df)
            st.warning("Deleted!")
            st.rerun()

    with tab2:
        if st.button("🚨 CLEAR ALL DATA"):
            if os.path.exists(FILE_NAME):
                os.remove(FILE_NAME)
                st.rerun()
else:
    st.info("No data found. Add some expenses! ✨")
