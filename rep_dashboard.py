import streamlit as st
import math

st.set_page_config(page_title="Rep Performance Dashboard", layout="centered")

st.title("📊 Sales Rep Performance Dashboard")

st.markdown("Use this tool to track your Attach Rate and Conversion goals in real-time.")

# ---- Attach Rate Section ----
st.header("🌿 Attach Rate Calculator")

# Default attach rate targets
default_targets = {
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

# Attach Calculations
current_attach_rate = (current_attach / wins) if wins > 0 else 0
needed_attach_total = math.ceil(goal_rate * wins)
remaining_attach = max(0, needed_attach_total - current_attach)
projected_attach_rate = (needed_attach_total / wins) if wins > 0 else 0

# Output for Attach Rate
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

# ---- Conversion Rate Section ----
st.header("📞 Conversion Rate Calculator")

current_calls = st.number_input("Current Calls Made", min_value=0, value=100, key="calls_now")
current_wins = st.number_input("Current Wins", min_value=0, value=20, key="wins_now")
target_conversion = st.number_input("Target Conversion Rate (%)", min_value=0.0, max_value=100.0, value=25.0) / 100

# Projected Calls Input
projected_calls = st.number_input("Future Total Calls (Optional - for projection)", min_value=0, value=current_calls, key="future_calls")

# Conversion Calculations
current_conversion_rate = (current_wins / current_calls) if current_calls > 0 else 0
projected_wins_needed = math.ceil(projected_calls * target_conversion)
projected_conversion_rate = (projected_wins_needed / projected_calls) if projected_calls > 0 else 0
projected_remaining = max(0, projected_wins_needed - current_wins)

# Output for Conversion
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
