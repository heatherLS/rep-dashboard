import streamlit as st
import pandas as pd
import math
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Rep Dashboard", layout="centered")
st.title("🌟 Sales Rep Performance Dashboard")

# 🔁 Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="datarefresh")

tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🧮 Calculator", "Bonus & History"])

# ---- Shared Config ----
sheet_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=171451260"

@st.cache_data
def load_data():
    df = pd.read_csv(sheet_url, header=1)
    return df

with tab1:
    st.markdown("<h1 style='text-align: center;'>📈 Conversion Rate Leaderboard</h1>", unsafe_allow_html=True)
    df = load_data()
    
    # 🎂 Birthday & Anniversary shoutouts
    today = datetime.now()

    if 'Birthday' in df.columns:
        df['Birthday'] = pd.to_datetime(df['Birthday'], errors='coerce')
        bdays_today = df[df['Birthday'].dt.strftime('%m-%d') == today.strftime('%m-%d')]
        for _, row in bdays_today.iterrows():
            st.markdown(f"<div style='text-align: center; color: orange; font-size: 20px;'>🧁🎉 Happy Birthday, {row['First_Name']}! 🎉🧁</div>", unsafe_allow_html=True)

    if 'Start Date' in df.columns:
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        anniv_today = df[df['Start Date'].dt.strftime('%m-%d') == today.strftime('%m-%d')]
        for _, row in anniv_today.iterrows():
            years = today.year - row['Start Date'].year
            st.markdown(f"<div style='text-align: center; color: teal; font-size: 20px;'>🥳🎉 Happy {years}-year Anniversary, {row['First_Name']}! 🎉🥳</div>", unsafe_allow_html=True)

    df['Calls'] = pd.to_numeric(df['Calls'], errors='coerce').fillna(0)
    df = df[df['Calls'] >= 1]

    rep_col = 'Rep'
    conversion_col = 'Conversion'

    df['First_Name'] = df['First_Name'].astype(str).str.strip()
    df['Last_Name'] = df['Last_Name'].astype(str).str.strip()
    df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']

    df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce').fillna(0)
    double_digit_celebs = df[df['Wins'] >= 10]
    if not double_digit_celebs.empty:
        names = ", ".join(double_digit_celebs['First_Name'].tolist())
        st.markdown(f"<div style='text-align: center; color: purple; font-size: 22px; font-weight: bold;'>🎉 DOUBLE DIGITS CLUB: {names} {'has' if len(double_digit_celebs)==1 else 'have'} crushed 10+ wins today!</div>", unsafe_allow_html=True)

    all_reps = sorted(df[rep_col].dropna().unique())
    user = st.selectbox("👤 Who's using this app right now?", all_reps, key="selected_rep")

    user_data = df[df[rep_col] == user]
    first_name = user_data['First_Name'].values[0] if not user_data.empty else "Rep"

    try:
        personal_conversion = float(user_data[conversion_col].astype(str).str.replace('%', '').str.strip().values[0]) if not user_data.empty else 0.0
    except:
        personal_conversion = 0.0

    st.markdown("<h2 style='text-align: center;'>📊 Your Conversion Rate</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{personal_conversion:.2f}%</h3>", unsafe_allow_html=True)

    # 👥 Top Team
if 'Team Name' in df.columns:
    df_team = df[df['Wins'].notna() & df['Calls'].notna()].copy()
    df_team['Wins'] = pd.to_numeric(df_team['Wins'], errors='coerce').fillna(0)
    df_team['Calls'] = pd.to_numeric(df_team['Calls'], errors='coerce').replace(0, pd.NA)
    df_team = df_team.dropna(subset=['Calls'])

    team_stats = df_team.groupby('Team Name').agg(
        Total_Calls=('Calls', 'sum'),
        Total_Wins=('Wins', 'sum')
    ).reset_index()
    team_stats['Conversion'] = (team_stats['Total_Wins'] / team_stats['Total_Calls']) * 100
    team_stats = team_stats.sort_values(by='Conversion', ascending=False)

    if not team_stats.empty:
        st.markdown("<h2 style='text-align: center;'>👥 Top 3 Teams</h2>", unsafe_allow_html=True)
        medals = ['🥇', '🥈', '🥉']
        for i, (_, team_row) in enumerate(team_stats.head(3).iterrows()):
            medal = medals[i] if i < len(medals) else ''
            logo_filename = team_row['Team Name'].replace(' ', '_').lower() + '.png'
            logo_url = f"https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{logo_filename}"
            st.markdown(f"""
                <div style="text-align: center; font-size: 22px; font-weight: bold;">
                    {medal} {team_row['Team Name']}<br>
                    <img src="{logo_url}" width="90"><br>
                    {team_row['Conversion']:.2f}% — {int(team_row['Total_Wins'])} wins / {int(team_row['Total_Calls'])} calls
                </div><br>
            """, unsafe_allow_html=True)



    # 🏅 Top 3 Reps
    df[conversion_col] = df[conversion_col].astype(str).str.replace('%', '').str.strip()
    df[conversion_col] = pd.to_numeric(df[conversion_col], errors='coerce').fillna(0)
    leaderboard = df[['Full_Name', conversion_col]].sort_values(by=conversion_col, ascending=False).reset_index(drop=True)
    leaderboard['Rank'] = leaderboard.index + 1

    st.markdown("<h2 style='text-align: center;'>🏅 Top 3 Reps</h2>", unsafe_allow_html=True)
    for _, row in leaderboard.head(3).iterrows():
        st.markdown(f"""
            <div style='text-align: center; font-size: 22px; font-weight: bold;'>
                {row['Rank']}. {row['Full_Name']} — {row[conversion_col]:.2f}%
            </div>
        """, unsafe_allow_html=True)

    # 🏆 Full Leaderboard with logos and green highlight (NOW CORRECTLY INDENTED)
    if 'Team Name' in df.columns:
        df['Team Name'] = df['Team Name'].astype(str)
        df['Team_Logo'] = df['Team Name'].apply(
            lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='40'>" if pd.notna(name) else ""
        )
        leaderboard = leaderboard.merge(df[['Full_Name', 'Team_Logo']], on='Full_Name', how='left')
        leaderboard_display = leaderboard[['Rank', 'Full_Name', conversion_col, 'Team_Logo']].copy()
        leaderboard_display.columns = ['Rank', 'Rep Name', 'Conversion (%)', 'Team Logo']

        def highlight_steve_green(row):
            try:
                if float(row['Conversion (%)']) >= 26:
                    row['Rep Name'] = f"<span style='color: green; font-weight: bold;'>{row['Rep Name']}</span>"
            except:
                pass
            return row

        leaderboard_display = leaderboard_display.apply(highlight_steve_green, axis=1)

        st.markdown("<h2 style='text-align: center;'>🏆 Full Leaderboard</h2>", unsafe_allow_html=True)
        st.write(
            leaderboard_display.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        st.markdown("""
        <style>
        table td:first-child, table th:first-child {
            text-align: center !important;
        }
        table {
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)


    # 🍃 LT Leaderboard
    if 'Lawn Treatment' in df.columns:
        df['Lawn Treatment'] = pd.to_numeric(df['Lawn Treatment'], errors='coerce').fillna(0)
        if 'Team_Logo' not in df.columns:
            df['Team_Logo'] = df['Team Name'].astype(str).apply(
                lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='30'>" if pd.notna(name) else ""
            )

        lt_leaderboard = df[['Full_Name', 'Lawn Treatment', 'Team_Logo']].sort_values(by='Lawn Treatment', ascending=False).reset_index(drop=True)
        lt_leaderboard['Rank'] = lt_leaderboard.index + 1
        medals = ['🥇', '🥈', '🥉']

        st.markdown("<h2 style='text-align: center;'>🍃 Top 3 Lawn Treatment Sellers</h2>", unsafe_allow_html=True)
        for i, row in lt_leaderboard.head(3).iterrows():
            logo_img = row['Team_Logo']
            medal = medals[i] if i < len(medals) else ''
            st.markdown(f"""
                <div style='text-align: center; font-size: 24px; font-weight: bold;'>
                    {medal} {row['Full_Name']} {logo_img} — {int(row['Lawn Treatment'])} LT
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center;'>🍃 Full LT Leaderboard</h2>", unsafe_allow_html=True)
        lt_display = lt_leaderboard[['Rank', 'Full_Name', 'Lawn Treatment', 'Team_Logo']]
        lt_display.columns = ['Rank', 'Rep Name', 'Lawn Treatment', 'Team Logo']

        st.write(
            lt_display.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        st.markdown("""
        <style>
        table td:first-child, table th:first-child {
            text-align: center !important;
        }
        table {
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)



# --------------------------------------------
# 🧮 TAB 2: Calculator
# --------------------------------------------
with tab2:
    # ... (calculator code remains unchanged)
    pass

st.markdown("---")
st.caption("Built with 💚 by Heather & ChatGPT")


# --------------------------------------------
# 🧮 TAB 2: Calculator
# --------------------------------------------
with tab2:
    st.header("🌿 Attach Rate Calculator")

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

    current_attach_rate = (current_attach / wins) if wins > 0 else 0
    needed_attach_total = math.ceil(goal_rate * wins)
    remaining_attach = max(0, needed_attach_total - current_attach)
    projected_attach_rate = (needed_attach_total / wins) if wins > 0 else 0

    st.subheader("📍 Attach Rate Progress")
    st.metric("Current Attach Rate", f"{current_attach_rate*100:.2f}%")
    st.metric("Projected Attach Rate", f"{projected_attach_rate*100:.2f}%")
    st.metric(f"{metric} Needed for Goal", needed_attach_total)
    st.metric(f"More {metric} Needed", remaining_attach)

    if remaining_attach == 0:
        st.success(f"You're crushing your {metric} goal! 🎉")
    else:
        st.warning(f"You need {remaining_attach} more {metric} sale(s) to hit your target.")

    st.markdown("---")
    st.header("📞 Conversion Rate Calculator")

    current_calls = st.number_input("Current Calls Made", min_value=0, value=100, key="calls_now")
    current_wins = st.number_input("Current Wins", min_value=0, value=20, key="wins_now")
    target_conversion = st.number_input("Target Conversion Rate (%)", min_value=0.0, max_value=100.0, value=25.0) / 100

    projected_calls = st.number_input("Future Total Calls", min_value=0, value=current_calls, key="future_calls")

    current_conversion_rate = (current_wins / current_calls) if current_calls > 0 else 0
    projected_wins_needed = math.ceil(projected_calls * target_conversion)
    projected_conversion_rate = (projected_wins_needed / projected_calls) if projected_calls > 0 else 0
    projected_remaining = max(0, projected_wins_needed - current_wins)

    st.subheader("📍 Current Performance")
    st.metric("Current Conversion Rate", f"{current_conversion_rate*100:.2f}%")
    st.subheader("🔮 Future Projection")
    st.metric("Projected Conversion Rate", f"{projected_conversion_rate*100:.2f}%")
    st.metric("Wins Needed", projected_wins_needed)
    st.metric("More Wins Needed", projected_remaining)

    if projected_remaining == 0:
        st.success("You're on pace to hit your conversion target! 🚀")
    else:
        st.info(f"You'll need {projected_remaining} more win(s) to hit {target_conversion*100:.1f}%.")


# ---------------------------------------------------
# 🔥 TAB 3: Bonus Dashboard
# ---------------------------------------------------
with tab3:
    st.header("🔥 Bonus & History")

    # Bonus metric setup
    def get_points(val, tiers):
        for threshold, pts in tiers:
            if val >= threshold:
                return pts
        return 0

    conversion_tiers = [(26, 5), (24, 4), (23, 3), (21, 2), (20, 1), (19, 0)]
    attach_tiers = [(27, 2), (26, 1), (25, 0)]
    lt_tiers = [(8.25, 3), (7.5, 2), (6.5, 1), (5.5, 0)]
    qa_tiers = [(100, 2), (92, 1), (80, 0)]

    # ✅ Grab selected rep from session_state
    email = st.session_state.get("selected_rep")

    if not email:
        st.warning("Please select a rep on the Leaderboard tab first.")
        st.stop()

    match = df[df['Rep'].astype(str).str.strip() == email.strip()]
    if match.empty:
        st.error("Could not find this rep in the data.")
        st.stop()

    row = match.iloc[0]


    def percent(val):
        try:
            return float(str(val).replace('%', '').strip())
        except:
            return 0

    metrics = {
        'Conversion': percent(row.get('BonusConversion', 0)),
        'All-In Attach': percent(row.get('BonusAllinAttach', 0)),
        'Lawn Treatment': percent(row.get('BonusLT', 0)),
        'QA': percent(row.get('BonusQA', 0))
    }

    points = {
        'Conversion': get_points(metrics['Conversion'], conversion_tiers),
        'All-In Attach': get_points(metrics['All-In Attach'], attach_tiers),
        'Lawn Treatment': get_points(metrics['Lawn Treatment'], lt_tiers),
        'QA': get_points(metrics['QA'], qa_tiers)
    }

    st.subheader(f"🧑‍🌾 Growth Stats for {row['First_Name']}")

    for k in metrics:
        st.markdown(f"**{k}**: {metrics[k]:.2f}% — Points: `{points[k]}`")

    st.subheader("🌱 Focus Patch")
    focus = min(points, key=points.get)
    st.info(f"Your area of growth: **{focus}** — currently {metrics[focus]:.2f}%")

    total_points = sum(points.values())
    payout_levels = {8: "$5.00", 7: "$4.00", 6: "$3.00", 5: "$2.00", 3: "$1.00"}
    eligible = all(p >= 0 for p in points.values())
    hourly = payout_levels.get(total_points, "$0.00") if eligible else "$0.00"

    st.markdown(f"**🌼 Points Earned:** {total_points} — **Hourly Forecast:** {hourly}")
    st.caption("You’ll earn this rate *only* if all 4 base qualifiers are met.")

    # 🏅 Personal Bests Section
    st.markdown("### 🏅 Personal Bests")

    # Load personal bests from history sheet (adjust URL or GID if needed)
    history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
    history_df = pd.read_csv(history_url, header=1)  # Start at row 2
    history_df.columns = history_df.columns.str.strip()  # Clean up any spaces in headers


    # Filter by selected rep
    rep_history = history_df[history_df['Rep'].astype(str).str.strip() == row['Rep']]

    # Compute personal bests
    pb_wins = rep_history['Wins'].max()
    pb_lawn = rep_history['Lawn Treatment'].max()
    pb_bush = rep_history['Bush Trimming'].max() if 'Bush Trimming' in rep_history.columns else 0
    pb_flower = rep_history['Flower Bed Weeding'].max() if 'Flower Bed Weeding' in rep_history.columns else 0
    pb_mosquito = rep_history['Mosquito'].max() if 'Mosquito' in rep_history.columns else 0

    # Personal Best Challenge Formatter
    def challenge_line(label, pb_val, emoji):
        if pd.isna(pb_val) or pb_val == 0:
            return f"{emoji} **{label} PB:** 0 — Let’s set a new record today! 💪"
        else:
            return f"{emoji} **{label} PB:** {int(pb_val)} — Can you hit {int(pb_val) + 1} today?"

    # Display all
    st.markdown(f"""
    <div style='font-size:18px'>
    🏆 **Wins PB:** {int(pb_wins) if not pd.isna(pb_wins) else 0} — Can you close {int(pb_wins)+1 if not pd.isna(pb_wins) else 1} today? 💥<br>
    {challenge_line('LT', pb_lawn, '🌿')}<br>
    {challenge_line('Bush', pb_bush, '🌳')}<br>
    {challenge_line('Flower', pb_flower, '🌸')}<br>
    {challenge_line('Mosquito', pb_mosquito, '🦟')}
    </div>
    """, unsafe_allow_html=True)



    st.markdown("### 🌾 Bonus Tiers")
    st.markdown("""
    | Metric           | Base  | Green | Super Green | Super Duper | Steve Green |
    |------------------|-------|-------|--------------|--------------|-------------|
    | Conversion       | 19% (0) | 20% (1) | 21% (2)       | 23% (3)       | 26% (5)      |
    | All-In Attach    | 25% (0) | 26% (1) | 27% (2)       | —            | —           |
    | Lawn Treatment   | 5.5% (0) | 6.5% (1) | 7.5% (2)     | 8.25% (3)     | —           |
    | QA               | 80% (0) | 92% (1) | 100% (2)     | —            | —           |
    """)

    chart = pd.DataFrame({"Metric": list(points.keys()), "Points": list(points.values())})
    st.bar_chart(chart.set_index("Metric"))
