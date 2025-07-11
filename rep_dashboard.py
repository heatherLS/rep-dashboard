import streamlit as st
import pandas as pd
import math
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="Rep Dashboard", layout="wide")
st.title("🌟 Rep Performance Dashboard")

# --- TABS ---
tabs = st.tabs(["Calculator", "Leaderboard"])

# --- LOAD DATA ---
sheet_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=171451260"

@st.cache_data
def load_data():
    df = pd.read_csv(sheet_url, header=1)
    return df

df = load_data()

# --- CLEAN DATA ---
rep_col = 'Rep'
conversion_col = 'Conversion'
df = df[df[rep_col].notna() & df[conversion_col].notna()].copy()
df[conversion_col] = df[conversion_col].astype(str).str.replace('%', '').str.strip()
df = df[df[conversion_col] != '']
df[conversion_col] = df[conversion_col].astype(float)
df['First_Name'] = df['First_Name'].astype(str).str.strip()
df['Last_Name'] = df['Last_Name'].astype(str).str.strip()
df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']

# --- CALCULATOR TAB ---
with tabs[0]:
    st.header("🌿 Attach Rate Calculator")
    default_targets = {
        "All In Attach": 0.25,
        "Lawn Treatment": 0.055,
        "Bush Trimming": 0.07,
        "Leaf Removal": 0.025,
        "Flower Bed Weeding": 0.015,
        "Mosquito": 0.02,
    }

    metric = st.selectbox("Choose a service", list(default_targets.keys()) + ["Other"], key="metric_select")
    if metric == "Other":
        metric = st.text_input("Enter custom service name", key="custom_metric")

    goal_rate = st.number_input(f"Target Attach Rate (%) for {metric}", min_value=0.0, max_value=100.0,
                                value=round(default_targets.get(metric, 0.05)*100, 2)) / 100

    wins = st.number_input("Total Closed Won Deals", min_value=0, value=30, key="won_input")
    current_attach = st.number_input(f"Current {metric} Sales", min_value=0, value=1)

    current_attach_rate = (current_attach / wins) if wins > 0 else 0
    needed_attach_total = math.ceil(goal_rate * wins)
    remaining_attach = max(0, needed_attach_total - current_attach)
    projected_attach_rate = (needed_attach_total / wins) if wins > 0 else 0

    st.subheader("📍 Attach Rate Progress")
    st.metric(label="Current Attach Rate", value=f"{current_attach_rate*100:.2f}%")
    st.metric(label=f"Projected Attach Rate (If Goal Met)", value=f"{projected_attach_rate*100:.2f}%")
    st.metric(label=f"{metric} Sales Needed for Goal", value=needed_attach_total)
    st.metric(label=f"More {metric} Sales Needed", value=remaining_attach)

    if remaining_attach == 0:
        st.success(f"You're crushing your {metric} goal! 🎉")
    else:
        st.warning(f"You need {remaining_attach} more {metric} sale(s) to hit your target.")

    st.markdown("---")

    st.header("📞 Conversion Rate Calculator")
    current_calls = st.number_input("Current Calls Made", min_value=0, value=100, key="calls_now")
    current_wins = st.number_input("Current Wins", min_value=0, value=20, key="wins_now")
    target_conversion = st.number_input("Target Conversion Rate (%)", min_value=0.0, max_value=100.0, value=25.0) / 100
    projected_calls = st.number_input("Future Total Calls (Optional - for projection)", min_value=0, value=current_calls, key="future_calls")

    current_conversion_rate = (current_wins / current_calls) if current_calls > 0 else 0
    projected_wins_needed = math.ceil(projected_calls * target_conversion)
    projected_conversion_rate = (projected_wins_needed / projected_calls) if projected_calls > 0 else 0
    projected_remaining = max(0, projected_wins_needed - current_wins)

    st.subheader("📍 Current Performance")
    st.metric(label="Current Conversion Rate", value=f"{current_conversion_rate*100:.2f}%")

    st.subheader("🔮 Projection Based on Future Calls")
    st.metric(label=f"Projected Conversion (If Goal Met)", value=f"{projected_conversion_rate*100:.2f}%")
    st.metric(label=f"Wins Needed for {projected_calls} Calls", value=projected_wins_needed)
    st.metric(label="More Wins Needed", value=projected_remaining)

    if projected_remaining == 0:
        st.success("You're on pace to hit your conversion target! 🚀")
    else:
        st.info(f"You'll need {projected_remaining} more win(s) to hit {target_conversion*100:.1f}% conversion.")

    st.markdown("---")
    st.caption("Built with 💚 by Heather & ChatGPT")

# --- LEADERBOARD TAB ---
with tabs[1]:
    # 🎉 Double Digits Shoutout
    double_digit_celebs = df[pd.to_numeric(df['Wins'], errors='coerce').fillna(0) >= 10]
    if not double_digit_celebs.empty:
        names = ", ".join(double_digit_celebs['First_Name'].tolist())
        st.markdown(f"""
        <div style='text-align: center; color: purple; font-size: 22px; font-weight: bold; margin-top: 20px;'>
            🎉 DOUBLE DIGITS CLUB: {names} {'has' if len(double_digit_celebs)==1 else 'have'} crushed 10+ wins today!
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Built with 💚 by Heather & ChatGPT")
