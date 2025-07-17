import streamlit as st
import pandas as pd
import math
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from datetime import datetime, timedelta


st.set_page_config(page_title="Rep Dashboard", layout="wide")
st.title("🌟 Sales Rep Performance Dashboard")

# 🔁 Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="datarefresh")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboard", "🧮 Calculator", "💰Bonus & History", "📅 Yesterday"])

# ---- Shared Config ----
sheet_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=171451260"

history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
@st.cache_data(ttl=86400)  # cache for 24 hours (86400 seconds)
def load_history():
    return pd.read_csv(history_url, header=1)


def load_data():
    return pd.read_csv(sheet_url, header=1)

with tab1:
    st.markdown("<h1 style='text-align: center;'>📊 Conversion Rate Leaderboard</h1>", unsafe_allow_html=True)
    df = load_data()

    # 🎂 Birthday & Anniversary shoutouts
    today = datetime.now()

    if 'Birthday' in df.columns:
        df['Birthday'] = pd.to_datetime(df['Birthday'], errors='coerce')
        bdays_today = df[df['Birthday'].dt.strftime('%m-%d') == today.strftime('%m-%d')]
        for _, row in bdays_today.iterrows():
            st.markdown(f"<div style='text-align: center; color: orange; font-size: 20px;'>🌼🎉 Happy Birthday, {row['First_Name']}! 🎉🌼</div>", unsafe_allow_html=True)

    if 'Start Date' in df.columns:
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        anniv_today = df[df['Start Date'].dt.strftime('%m-%d') == today.strftime('%m-%d')]
        for _, row in anniv_today.iterrows():
            years = today.year - row['Start Date'].year
            st.markdown(f"<div style='text-align: center; color: teal; font-size: 20px;'>🥳🎉 Happy {years}-year Anniversary, {row['First_Name']}! 🎉🥳</div>", unsafe_allow_html=True)

    # ✅ Convert Calls to numeric and keep everyone for rep selection
    df['Calls'] = pd.to_numeric(df['Calls'], errors='coerce').fillna(0)

    rep_col = 'Rep'
    conversion_col = 'Conversion'


    df['First_Name'] = df['First_Name'].astype(str).str.strip()
    df['Last_Name'] = df['Last_Name'].astype(str).str.strip()
    df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']

    df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce').fillna(0)
    double_digit_celebs = df[df['Wins'] >= 10]
    if not double_digit_celebs.empty:
        names = ", ".join(double_digit_celebs['First_Name'].tolist())
        st.markdown(
            f"<div style='text-align: center; color: purple; font-size: 22px; font-weight: bold;'>🎉 DOUBLE DIGITS CLUB: {names} {'has' if len(double_digit_celebs)==1 else 'have'} crushed 10+ wins today!</div>",
            unsafe_allow_html=True
        )

    all_reps = sorted(df[rep_col].dropna().unique())
    all_reps.insert(0, "🔍 Select your name")
    user = st.selectbox("👤 Who's using this app right now?", all_reps, key="selected_rep")

    if user == "🔍 Select your name":
        st.warning("Please select your name from the list to continue.")
        st.stop()

    active_df = df[df['Calls'] >= 1]
    user_data = df[df[rep_col] == user]
    first_name = user_data['First_Name'].values[0] if not user_data.empty else "Rep"

with tab1:
    # Your current leaderboard, shoutouts, and top 3 service blocks here...

    selected_rep = st.session_state.get("selected_rep", None)

    # Clean up percent columns safely
    percent_columns = ['Conversion', 'All-In Attach', 'Lawn Treatment Attach']
    for col in percent_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')


    selected_rep = st.session_state.get("selected_rep", None)

    if selected_rep:
        match = df[df['Rep'] == selected_rep]
    
        if not match.empty:
            rep_row = match.iloc[0]
            team_name = rep_row.get("Team Name", "Unknown")

        # 🧢 Team logo
        team_logo = f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{team_name.replace(' ', '_').lower()}.png' width='80'>" if pd.notna(team_name) else ""

        # ✅ Correctly calculate conversion and team rank
        team_stats = df[df['Calls'] > 0].copy()
        team_stats['Calls'] = pd.to_numeric(team_stats['Calls'], errors='coerce')
        team_stats['Wins'] = pd.to_numeric(team_stats['Wins'], errors='coerce')

        team_totals = team_stats.groupby("Team Name").agg(
            Total_Calls=("Calls", "sum"),
            Total_Wins=("Wins", "sum")
        ).reset_index()

        team_totals['Conversion'] = (team_totals['Total_Wins'] / team_totals['Total_Calls']) * 100
        team_totals['Rank'] = team_totals['Conversion'].rank(ascending=False, method='min').astype(int)

        if team_name in team_totals['Team Name'].values:
            team_rank = int(team_totals.loc[team_totals['Team Name'] == team_name, 'Rank'].values[0])
        else:
            team_rank = "N/A"


        # 🖼️ Display logo and rank
        st.markdown(f"""
        <div style='text-align: center;'>
            {team_logo}<br>
            <div style='font-size: 18px; color: #00cccc;'>🏅 Team Rank: {team_rank}</div>
        </div>
        """, unsafe_allow_html=True)

        # 📊 Team stats
        team_df = df[df['Team Name'] == team_name].copy()
        team_df['Conversion'] = pd.to_numeric(team_df['Conversion'], errors='coerce')
        team_df['All-In Attach'] = pd.to_numeric(team_df['All-In Attach'], errors='coerce')
        team_df['Lawn Treatment'] = pd.to_numeric(team_df['Lawn Treatment'], errors='coerce')

        team_total_calls = team_df['Calls'].sum()
        team_total_wins = team_df['Wins'].sum()
        team_conversion_rate = (team_total_wins / team_total_calls) * 100 if team_total_calls > 0 else 0

        attach_services = ['Lawn Treatment', 'Leaf Removal', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
        for svc in attach_services:
            team_df[svc] = pd.to_numeric(team_df.get(svc, 0), errors='coerce').fillna(0)

        team_total_attaches = team_df[attach_services].sum(axis=1).sum()
        team_attach_rate = (team_total_attaches / team_total_wins) * 100 if team_total_wins > 0 else 0

        team_lawn_treatment = team_df['Lawn Treatment'].sum()
        team_lt_attach = (team_lawn_treatment / team_total_wins) * 100 if team_total_wins > 0 else 0


        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px;'>
            <b>{team_name} Team Averages:</b><br>
            🧮 Conversion: {team_conversion_rate:.2f}%<br>
            🧩 All-In Attach: {team_attach_rate:.2f}%<br>
            🍃 Lawn Treatment Attach: {team_lt_attach:.2f}%

        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Selected rep not found in the dataset.")

    # Now continue on to show personal performance metrics
    

    # 👀 If user has 0 calls today, show message
    user_calls = user_data['Calls'].sum() if not user_data.empty else 0
    if user_calls == 0:
        st.warning("📞 No calls logged for you today yet! Let’s change that. 💪")

    try:
        personal_conversion = float(user_data[conversion_col].astype(str).str.replace('%', '').str.strip().values[0]) if not user_data.empty else 0.0
    except:
        personal_conversion = 0.0



    st.markdown("<h2 style='text-align: center;'>📊 Your Conversion Rate</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{personal_conversion:.2f}%</h3>", unsafe_allow_html=True)

    # 🌝 Motivational Blurb
    if personal_conversion >= 26:
        st.success("🌟 Steve Green Level! You're an elite closer!")
        st.balloons()
    elif personal_conversion >= 23:
        st.success("🚀 Super Duper Green! You're on fire!")
    elif personal_conversion >= 21:
        st.info("🌿 Super Green! You're doing awesome!")
    elif personal_conversion >= 20:
        st.info("📈 Green Zone! Keep pushing and you'll level up!")
    elif personal_conversion >= 19:
        st.warning("🚫 Almost There! Just 1% more for payout.")
    else:
        st.error("❌ Below Base. Let’s lock in and close the gap!")

    # 🍃 LT Motivation if none sold
    if 'Lawn Treatment' in user_data.columns and not user_data.empty:
        user_lt = pd.to_numeric(user_data['Lawn Treatment'], errors='coerce').fillna(0).values[0]
        if user_lt == 0:
            st.warning("🍃 You haven’t landed any Lawn Treatments today... Just one gets you in the race for bonus pay!")




    # 🔥 Win Streak + Motivation
    user_wins = user_data['Wins'].values[0] if not user_data.empty else 0

    if user_wins >= 7:
        st.markdown(f"<div style='text-align: center; font-size: 20px; color: red;'>🔥 {user_wins}-Win Streak! You're on fire!</div>", unsafe_allow_html=True)

    if user_wins < 10:
        remaining = 10 - user_wins
        if remaining > 0:
            st.markdown(f"<div style='text-align: center; font-size: 18px; color: orange;'>💡 Just {remaining} more to join the Double Digits Club!</div>", unsafe_allow_html=True)
    
# 🎯 Conversion Milestones (Centered and Readable)
st.markdown("<h3 style='text-align:center; color:#ffffff;'>🎯 Conversion Milestones</h3>", unsafe_allow_html=True)

conversion_targets = [19, 20, 21, 23, 26]
call_count = int(user_calls)
win_count = int(user_wins)

rows = []
for target in conversion_targets:
    needed_wins = math.ceil((target / 100) * call_count)
    remaining = max(0, needed_wins - win_count)
    hit = remaining == 0

    status = "✅ Hit!" if hit else f"🎯 {remaining} more win(s)"
    rows.append(f"{target}% → Need {needed_wins} wins ({status})")

st.markdown(
    "<div style='text-align:center; font-size: 16px; line-height: 1.8; font-weight: 500; color: #f8f8f8;'>"
    + "<br>".join(rows) +
    "</div>",
    unsafe_allow_html=True
)



def show_service_leaderboard(df, column_name, emoji, title):
    if column_name not in df.columns:
        return

    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    leaderboard = (
        df[['Full_Name', column_name, 'Team_Logo']]
        .sort_values(by=column_name, ascending=False)
        .reset_index(drop=True)
    )
    leaderboard['Rank'] = leaderboard.index + 1

    # ←—— ADD THIS: if the top‑3 are all zero, just skip rendering
    if leaderboard.head(3)[column_name].sum() == 0:
        return

    medals = ['🥇', '🥈', '🥉']

    st.markdown(
        f"<h3 style='text-align: center; font-size:20px;'>{emoji} Top 3 {title}</h3>",
        unsafe_allow_html=True
    )
    for i, row in leaderboard.head(3).iterrows():
        logo_img = row['Team_Logo']
        medal   = medals[i] if i < len(medals) else ''
        st.markdown(f"""
            <div style='text-align: center; font-size: 16px; font-weight: bold;'>
                {medal} {logo_img} {row['Full_Name']} — {int(row[column_name])}
            </div>
        """, unsafe_allow_html=True)


    # 🧑‍🤝‍🧑 Top Team Section — continue from here...

    # 👥 Top Team
if 'Team Name' in df.columns:
    df_team = df[df['Wins'].notna() & df['Calls'].notna()].copy()
    df_team['Wins'] = pd.to_numeric(df_team['Wins'], errors='coerce').fillna(0)
    df_team['Calls'] = pd.to_numeric(df_team['Calls'], errors='coerce').replace(0, pd.NA)
    df_team = df_team.dropna(subset=['Calls'])

    # ✅ Properly calculate team totals and conversion rank
    team_stats = df[df['Calls'] > 0].copy()
    team_stats['Calls'] = pd.to_numeric(team_stats['Calls'], errors='coerce')
    team_stats['Wins'] = pd.to_numeric(team_stats['Wins'], errors='coerce')

    team_totals = team_stats.groupby("Team Name").agg(
        Total_Calls=("Calls", "sum"),
        Total_Wins=("Wins", "sum")
    ).reset_index()

    team_totals['Conversion'] = (team_totals['Total_Wins'] / team_totals['Total_Calls']) * 100
    team_totals['Rank'] = team_totals['Conversion'].rank(ascending=False, method='min').astype(int)

    # 🎖️ Display Top 3 Teams (Side by Side)
    if not team_totals.empty:
        st.markdown("<h2 style='text-align: center;'>👥 Top 3 Teams</h2>", unsafe_allow_html=True)
        medals = ['🥇 1st Place', '🥈 2nd Place', '🥉 3rd Place']
        cols = st.columns(3)
    
        top_3 = team_totals.sort_values(by="Conversion", ascending=False).head(3)
    
        for i, (_, team_row) in enumerate(top_3.iterrows()):
            with cols[i]:
                logo_filename = team_row['Team Name'].replace(' ', '_').lower() + '.png'
                logo_url = f"https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{logo_filename}"
            
                st.markdown(f"""
                    <div style='text-align: center; font-size: 18px; font-weight: bold;'>
                        {medals[i]}<br>
                        {team_row['Team Name']}<br>
                        <img src="{logo_url}" width="80"><br><br>
                        <span style='font-size: 16px;'>
                            {team_row['Conversion']:.2f}%<br>
                            {int(team_row['Total_Wins'])} wins / {int(team_row['Total_Calls'])} calls
                        </span>
                    </div>
                """, unsafe_allow_html=True)



    # 🥇 Get top team stats
    top_team_row = team_totals.sort_values(by="Conversion", ascending=False).iloc[0]
    top_team_name = top_team_row['Team Name']
    top_team_wins = top_team_row['Total_Wins']
    top_team_attaches = df[df['Team Name'] == top_team_name][['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']].sum().sum()
    top_team_lt = df[df['Team Name'] == top_team_name]['Lawn Treatment'].sum()

    # 🧮 Your team stats (already defined as team_name)
    your_team_wins = team_totals.loc[team_totals['Team Name'] == team_name, 'Total_Wins'].values[0]
    your_team_attaches = df[df['Team Name'] == team_name][['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']].sum().sum()
    your_team_lt = df[df['Team Name'] == team_name]['Lawn Treatment'].sum()

    # 📈 Calculate required additional wins to beat top team's conversion rate
    top_team = team_totals.sort_values(by="Conversion", ascending=False).iloc[0]
    your_team_row_df = team_totals[team_totals["Team Name"] == team_name]

    # 📈 Your team info
your_team_row_df = team_totals[team_totals["Team Name"] == team_name]

if not your_team_row_df.empty:
    your_team_row = your_team_row_df.iloc[0]

    if team_name == top_team["Team Name"]:
        # 🎉 Your team is already #1!
        st.markdown(f"""
        <div style='text-align: center; font-size: 20px; margin-top: 10px; padding: 12px; border-radius: 10px;
                    background-color: rgba(0, 128, 0, 0.1); color: #d1ffd1; border: 2px solid #0f0;'>
            🎉 <b>Congrats, {team_name} is currently the top team!</b><br>
            Keep crushing it to stay on top! 💪🔥
        </div>
        """, unsafe_allow_html=True)
    else:
        # Calculate how many more wins your team needs to take the top spot
        your_wins = your_team_row["Total_Wins"]
        your_calls = your_team_row["Total_Calls"]
        top_conversion_rate = top_team["Conversion"] / 100

        denominator = 1 - top_conversion_rate
        if denominator <= 0:
            needed_wins = 0
        else:
            needed_wins = max(0, math.ceil((top_conversion_rate * your_calls - your_wins) / denominator))

        # 📉 Calculate attaches and LT
        needed_attaches = max(0, int(top_team_attaches - your_team_attaches))
        needed_lt = max(0, int(top_team_lt - your_team_lt))

        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px; padding: 10px; border-radius: 8px;
                    background-color: rgba(255, 255, 255, 0.05); color: #f9f9f9; border: 1px solid #444;'>
            <b>Can your team take the top spot?</b><br><br>
            🏆 Top Team: <b>{top_team["Team Name"]}</b><br>
            💪 Your Team: <b>{team_name}</b><br><br>
            Your team needs:<br>
            • <b>{needed_wins} more wins</b><br>
            • <b>{needed_attaches} more attaches</b><br>
            • <b>{needed_lt} more Lawn Treatments</b><br>
            to surpass the top team.
        </div>
        """, unsafe_allow_html=True)





   # 🏊 Top 3 Reps
    active_df[conversion_col] = active_df[conversion_col].astype(str).str.replace('%', '').str.strip()
    active_df[conversion_col] = pd.to_numeric(active_df[conversion_col], errors='coerce').fillna(0)
    leaderboard = active_df[['Full_Name', conversion_col]].sort_values(by=conversion_col, ascending=False).reset_index(drop=True)
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

# Show additional service leaderboards side-by-side
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    show_service_leaderboard(df[df['Calls'] >= 1], 'Lawn Treatment', '🌱', 'Lawn Treatment')

with col2:
    show_service_leaderboard(df[df['Calls'] >= 1], 'Bush Trimming', '🌳', 'Bush Trim')

with col3:
    show_service_leaderboard(df[df['Calls'] >= 1], 'Mosquito', '🦟', 'Mosquito')

with col4:
    show_service_leaderboard(df[df['Calls'] >= 1], 'Flower Bed Weeding', '🌸', 'Flower Bed Weeding')

with col5:
    show_service_leaderboard(df[df['Calls'] >= 1], 'Leaf Removal', '🍂', 'Leaf Removal')

# 🍃 Full LT Leaderboard with additional services
if 'Lawn Treatment' in df.columns:
    df['Lawn Treatment'] = pd.to_numeric(df['Lawn Treatment'], errors='coerce').fillna(0)
    df['Bush Trimming'] = pd.to_numeric(df.get('Bush Trimming', 0), errors='coerce').fillna(0)
    df['Mosquito'] = pd.to_numeric(df.get('Mosquito', 0), errors='coerce').fillna(0)
    df['Flower Bed Weeding'] = pd.to_numeric(df.get('Flower Bed Weeding', 0), errors='coerce').fillna(0)
    df['Leaf Removal'] = pd.to_numeric(df.get('Leaf Removal', 0), errors='coerce').fillna(0)

    if 'Team_Logo' not in df.columns:
        df['Team_Logo'] = df['Team Name'].astype(str).apply(
            lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='30'>" if pd.notna(name) else ""
        )

    lt_leaderboard = df[df['Calls'] >= 1].copy()
    lt_leaderboard = lt_leaderboard.sort_values(by='Lawn Treatment', ascending=False).reset_index(drop=True)
    lt_leaderboard['Rank'] = lt_leaderboard.index + 1

    # Use First and Last Name instead of email
    lt_leaderboard['Rep Name'] = lt_leaderboard.apply(
        lambda row: f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
        if row.get('First Name') or row.get('Last Name') else row.get('Rep', 'Unknown'), axis=1
    )

    # Select and rename columns for display
    lt_display = lt_leaderboard[['Rank', 'Rep Name', 'Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal', 'Team_Logo']]
    lt_display.columns = ['Rank', 'Rep Name', 'Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal', 'Team Logo']

    st.markdown("<h2 style='text-align: center;'>🌱 Full LT Leaderboard</h2>", unsafe_allow_html=True)
    st.markdown(lt_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Center the table visually
    st.markdown("""
    <style>
    table td, table th {
        text-align: center !important;
        vertical-align: middle;
    }
    table {
        margin-left: auto;
        margin-right: auto;
        border-collapse: collapse;
        width: 90%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    th {
        background-color: #333;
        color: white;
        font-weight: bold;
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
    
    import random
    dummy = random.randint(1, 99999)
    st.markdown(f"<div style='display:none'>{dummy}</div>", unsafe_allow_html=True)

    components.html("""
    <div id="emoji-container" style="position: relative; height: 1px;"></div>

    <style>
    .money-emoji {
        position: absolute;
        top: 0;
        font-size: 2rem;
        animation: drop 4s linear infinite;
    }

    @keyframes drop {
        to {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
    </style>

    <script>
    const container = document.getElementById("emoji-container");
    if (container) {
        for (let i = 0; i < 30; i++) {
            const emoji = document.createElement("div");
            emoji.className = "money-emoji";
            emoji.innerText = "💸";
            emoji.style.left = Math.random() * 100 + "vw";
            emoji.style.animationDelay = Math.random() * 2 + "s";
            emoji.style.fontSize = (Math.random() * 20 + 20) + "px";
            container.appendChild(emoji);
        }
    }
    </script>
    """, height=1)


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

    # Load personal bests from history sheet
    history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
    history_df = pd.read_csv(history_url, header=1)  # Start at row 2
    history_df.columns = history_df.columns.str.strip()  # Clean up any spaces in headers

    # Filter all rows for the selected rep
    rep_history = history_df[history_df['Rep'].astype(str).str.strip() == user]

    # Safely convert all relevant columns to numeric
    rep_history['Wins'] = pd.to_numeric(rep_history['Wins'], errors='coerce').fillna(0)
    rep_history['Lawn Treatment'] = pd.to_numeric(rep_history['Lawn Treatment'], errors='coerce').fillna(0)
    rep_history['Bush Trimming'] = pd.to_numeric(rep_history.get('Bush Trimming', 0), errors='coerce').fillna(0)
    rep_history['Flower Bed Weeding'] = pd.to_numeric(rep_history.get('Flower Bed Weeding', 0), errors='coerce').fillna(0)
    rep_history['Mosquito'] = pd.to_numeric(rep_history.get('Mosquito', 0), errors='coerce').fillna(0)

    # Compute personal bests across all days
    pb_wins = rep_history['Wins'].max()
    pb_lawn = rep_history['Lawn Treatment'].max()
    pb_bush = rep_history['Bush Trimming'].max()
    pb_flower = rep_history['Flower Bed Weeding'].max()
    pb_mosquito = rep_history['Mosquito'].max()

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


# --------------------------------------------
# 📅 TAB 4: Yesterday’s Snapshot
# --------------------------------------------
with tab4:
    st.markdown("<h1 style='text-align: center;'>🗓️ Coming SOON: Incorrect below Yesterday's Leaderboard</h1>", unsafe_allow_html=True)

   
