import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# ⛔️ TEMP: neutralize st.stop() so other tabs can't kill the run
if "SOFT_STOP_PATCHED" not in st.session_state:
    def _soft_stop():
        # show a minimal warning but DO NOT stop the app
        st.caption("⚠️ Another tab requested stop; ignoring so the app keeps running (demo mode).")
        return
    try:
        import streamlit as _st  # local alias just in case
        _st.stop = _soft_stop
    except Exception:
        pass
    st.session_state["SOFT_STOP_PATCHED"] = True



st.set_page_config(page_title="Rep Dashboard", layout="wide")
st.title("🌟 Sales Rep Performance Dashboard")

# 🔁 Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="datarefresh")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Leaderboard", "🧮 Calculator", "💰Bonus & History", "📅 Yesterday", "👩‍💻 Team Lead Dashboard", "Senior Manager View", "Game Hub"])

# ---- Shared Config ----
sheet_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=171451260"

history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
@st.cache_data(show_spinner=False)
def load_history(_cache_bust_key: str):
    df = pd.read_csv(history_url, header=1)
    # 🧼 Remove mid-sheet header rows
    df = df[df['Date'].astype(str).str.lower() != 'date']
    # ✅ Keep only rows that have valid reps
    df = df[df['Rep'].notna()]
    # 📅 Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df





def load_data():
    return pd.read_csv(sheet_url, header=1)

def show_yesterday_service_top(df, column_name, emoji, title):
    if column_name not in df.columns:
        return
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    leaderboard = df[['Full_Name', column_name]].sort_values(by=column_name, ascending=False)
    leaderboard = leaderboard[leaderboard[column_name] > 0].head(3)
    if leaderboard.empty:
        return
    st.markdown(f"<h4 style='text-align: center;'>{emoji} Top 3 {title}</h4>", unsafe_allow_html=True)
    for _, row in leaderboard.iterrows():
        st.markdown(f"<div style='text-align: center;'>{row['Full_Name']} — {int(row[column_name])}</div>", unsafe_allow_html=True)

# --- Live tiers loader (re-usable for any section) ---
BONUS_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    "/export?format=csv&gid=374383792"
)

@st.cache_data(show_spinner=False)
def load_section_tiers(url: str, section_label: str):
    """
    Returns list of (threshold_percent, points) for a given section label, e.g. "Conversion".
    Reads column B (threshold %) and column D (points) until the next section/break.
    """
    import pandas as pd
    def pct(x):
        s = str(x).replace('%','').strip()
        try: return float(s)
        except: return None

    df = pd.read_csv(url, header=None).fillna("")
    colB = df.iloc[:,1].astype(str).str.strip()  # thresholds + section headers in col B

    # find the section header row with exact label in col B
    rows = colB[colB == section_label].index.tolist()
    if not rows: 
        return []

    start = rows[0] + 1
    out = []
    for r in range(start, len(df)):
        b = str(df.iat[r,1]).strip()       # threshold %
        c = str(df.iat[r,2]).strip()       # label text (Base/Green/etc) - we don't need it for logic
        # stop when we hit blank or the start of another section
        if b == "" or b in ("Goals","Points","Current Cycle","All-In Attach Rate","LT","Conversion","QA"):
            break
        thr = pct(b)
        pts_raw = df.iat[r,3] if df.shape[1] > 3 else None
        try:
            pts = int(str(pts_raw).strip())
        except:
            continue
        if thr is not None:
            out.append((thr, pts))
    # sort highest→lowest so your >= checks work as expected
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def get_point_threshold(tiers, points_value, fallback):
    """Return the threshold % for a given points bucket (0,1,2,3,5), else fallback."""
    matches = [thr for thr, pts in tiers if pts == points_value]
    if matches:
        # if multiple rows share same points bucket, use the lowest threshold for that bucket
        return float(min(matches))
    return fallback

# ================================
# LIVE GOALS / TIERS (shared)
# ================================
BONUS_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    "/export?format=csv&gid=374383792"
)

@st.cache_data(show_spinner=False)
def load_section_tiers(url: str, section_label: str):
    import pandas as pd
    def pct(x):
        s = str(x).replace('%', '').strip()
        try: return float(s)
        except: return None
    try:
        df = pd.read_csv(url, header=None).fillna("")
    except Exception:
        return []
    colB = df.iloc[:, 1].astype(str).str.strip()
    rows = colB[colB == section_label].index.tolist()
    if not rows: return []
    start = rows[0] + 1
    out = []
    for r in range(start, len(df)):
        b = str(df.iat[r, 1]).strip()
        if b == "" or b in ("Goals","Points","Current Cycle","All-In Attach Rate","LT","Conversion","QA"):
            break
        thr = pct(b)
        pts_raw = df.iat[r,3] if df.shape[1] > 3 else None
        try: pts = int(str(pts_raw).strip())
        except: continue
        if thr is not None:
            out.append((thr, pts))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def get_point_threshold(tiers, points_value, fallback):
    matches = [thr for thr, pts in tiers if pts == points_value]
    if matches: return float(min(matches))
    return fallback

@st.cache_data(show_spinner=False)
def load_all_tiers(url: str):
    metrics = {
        "Conversion": "Conversion",
        "Attach": "All-In Attach Rate",
        "LT": "LT",
        "QA": "QA",
    }
    fallbacks = {
        "Conversion": {0:20.0,1:21.0,2:22.0,3:24.0,5:27.0},
        "Attach": {0:25.0,1:26.0,2:27.0},
        "LT": {0:5.5,1:6.5,2:7.5,3:8.25},
        "QA": {0:80.0,1:92.0,2:100.0},
    }
    out = {}
    for key, section in metrics.items():
        tiers = load_section_tiers(url, section)
        mapped = {}
        for p, fb in fallbacks[key].items():
            mapped[p] = get_point_threshold(tiers, p, fb)
        out[key] = mapped
    return out

TIERS = load_all_tiers(BONUS_SHEET_URL)

# Convenience base thresholds
BASE_CONV   = TIERS["Conversion"].get(0, 20.0)
BASE_ATTACH = TIERS["Attach"].get(0, 25.0)
BASE_LT     = TIERS["LT"].get(0, 5.5)
BASE_QA     = TIERS["QA"].get(0, 80.0)

# ----------------------------
# 🛠️ Name normalization helper
# ----------------------------
def normalize_name(x: str) -> str:
    """
    Normalize names for matching/grouping:
    - Convert to string
    - Strip leading/trailing spaces
    - Lowercase
    - Collapse multiple spaces
    """
    import re
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)  # collapse multiple spaces
    return x




# ---------- RECORDS & WEEK CHAMPIONS HELPERS ----------
from pytz import timezone
eastern = timezone('US/Eastern')

def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

def get_records_to_beat(history_df):
    """
    Highest single-day totals across the entire history for:
    Wins + each attach service.
    Returns a list of dicts with metric, value, full_name, team, date.
    """
    metrics = ['Wins', 'Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal']
    history_df = history_df.copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'], errors='coerce')
    history_df = clean_numeric(history_df, metrics)

    out = []
    for m in metrics:
        if m not in history_df.columns or history_df[m].max() == 0:
            continue
        idx = history_df[m].idxmax()
        row = history_df.loc[idx]
        full_name = f"{str(row.get('First_Name','')).strip()} {str(row.get('Last_Name','')).strip()}".strip() or str(row.get('Rep','Unknown'))
        out.append({
            "metric": m,
            "value": int(row[m]),
            "full_name": full_name,
            "team": str(row.get('Team Name', 'Unknown')),
            "date": row['Date'].date() if pd.notna(row['Date']) else None
        })
    return out

def get_last_week_range():
    """
    Returns the last fully completed Sunday–Saturday week.
    Example if today is 2025-08-11 (Mon):
      -> start = 2025-08-03 (Sun), end = 2025-08-09 (Sat)
    """
    today = datetime.now(eastern).date()
    yesterday = today - timedelta(days=1)

    # Saturday is 5 where Monday=0..Sunday=6
    days_back_to_sat = (yesterday.weekday() - 5) % 7
    end = yesterday - timedelta(days=days_back_to_sat)   # most recent Saturday strictly before today
    start = end - timedelta(days=6)                      # previous Sunday
    return start, end



def get_last_week_champions(history_df, min_calls_team=50, min_calls_rep=10):
    """
    Returns:
      - top_team_by_conversion: (team_name, conv_pct, wins, calls) or None
      - top_reps_by_wins: DataFrame of top 3 reps by conversion last week
    NOTE: min call thresholds are intentionally ignored for now.
    """
    start, end = get_last_week_range()
    df = history_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[(df['Date'].dt.date >= start) & (df['Date'].dt.date <= end)].copy()

    # Clean numerics we aggregate on
    df = clean_numeric(df, ['Calls','Wins','Lawn Treatment','Bush Trimming','Mosquito','Flower Bed Weeding','Leaf Removal'])

    # ======================
    # Teams (no min filter)
    # ======================
    team_totals = df.groupby('Team Name', dropna=False).agg(
        Calls=('Calls','sum'),
        Wins=('Wins','sum')
    ).reset_index()

    top_team = None
    if not team_totals.empty:
        # Safe conversion; 0 calls -> 0% instead of NaN/inf
        denom = team_totals['Calls'].replace(0, pd.NA)
        team_totals['Conversion'] = (team_totals['Wins'] / denom) * 100
        team_totals['Conversion'] = team_totals['Conversion'].fillna(0.0)

        top_row = team_totals.sort_values(['Conversion','Wins'], ascending=[False, False]).iloc[0]
        top_team = (
            top_row['Team Name'],
            float(top_row['Conversion']),
            int(top_row['Wins']),
            int(top_row['Calls'])
        )

    # ==========================================
    # Reps — rank by CONVERSION (no min filter)
    # ==========================================
    rep_totals = df.groupby(['Rep','First_Name','Last_Name','Team Name'], dropna=False).agg(
        Wins=('Wins','sum'),
        Calls=('Calls','sum')
    ).reset_index()

    if not rep_totals.empty:
        denom = rep_totals['Calls'].replace(0, pd.NA)
        rep_totals['Conversion'] = (rep_totals['Wins'] / denom) * 100
        rep_totals['Conversion'] = rep_totals['Conversion'].fillna(0.0)

        # Build display name, fallback to Rep if First/Last missing
        rep_totals['Full_Name'] = (
            rep_totals['First_Name'].astype(str).str.strip().fillna('') + ' ' +
            rep_totals['Last_Name'].astype(str).str.strip().fillna('')
        ).str.strip()
        rep_totals.loc[rep_totals['Full_Name'].eq('') | rep_totals['Full_Name'].isna(), 'Full_Name'] = rep_totals['Rep'].astype(str)

        top_reps = rep_totals.sort_values(['Conversion','Wins'], ascending=[False, False]).head(3).copy()
    else:
        top_reps = pd.DataFrame(columns=['Full_Name','Conversion','Wins','Calls','Team Name'])

    return top_team, top_reps[['Full_Name','Conversion','Wins','Calls','Team Name']]

# =========================
# IQ PANELS + POOL BANNER
# =========================
import pandas as pd
from datetime import timezone

# Map "display name" → (sheet column name, emoji)
IQ_MAP = [
    ("Lawn Treatment",                 "Lawn Treatment",                 "🌱"),
    ("Bushes",                         "Bush Trimming",                  "🌳"),  # sheet uses Bush Trimming
    ("Mosquito",                       "Mosquito",                       "🦟"),
    ("Flower Bed Weeding",             "Flower Bed Weeding",             "🌸"),
    ("Leaf Removal",                   "Leaf Removal",                   "🍂"),
    ("Overseeding and Aeration IQ",    "Overseeding and Aeration IQ",    "🌾"),
    ("Lime Treatment",                 "Lime Treatment",                 "🍋"),
    ("Disease Fungicide",              "Disease Fungicide",              "🧫"),
    ("Pool",                           "Pool",                           "🏊"),
]

CORE_IQS = [
    "Lawn Treatment", "Bushes", "Mosquito", "Flower Bed Weeding", "Leaf Removal"
]
SPECIALTY_IQS = [
    "Overseeding and Aeration IQ", "Lime Treatment", "Disease Fungicide", "Pool"
]

TOP_N_PER_IQ = 5  # show top N reps per card

def _nice_card(title: str, icon: str, body_md: str):
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(0,0,0,0.08);
            padding:14px;
            border-radius:16px;
            box-shadow:0 2px 10px rgba(0,0,0,0.10);
            min-height:140px;
        ">
            <div style="font-weight:700;font-size:16px;margin-bottom:8px;">{icon} {title}</div>
            <div style="line-height:1.6;">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _render_single_iq_card(df: pd.DataFrame, display_name: str, col_name: str, icon: str):
    """Show top sellers for one IQ (hide if col missing or all zeros)."""
    if col_name not in df.columns:
        return
    d = df[['Full_Name', col_name]].copy()
    d[col_name] = pd.to_numeric(d[col_name], errors='coerce').fillna(0)
    d = d[d[col_name] > 0].sort_values(col_name, ascending=False)
    if d.empty:
        # For Pool specifically, show a nudge if no sales yet
        if display_name == "Pool":
            _nice_card(display_name, icon, "_No sales yet — be the first to make a splash!_")
        return
    lines = [f"• <b>{r['Full_Name']}</b> — {int(r[col_name])}" for _, r in d.head(TOP_N_PER_IQ).iterrows()]
    _nice_card(display_name, icon, "<br>".join(lines))

def _render_iq_row(df: pd.DataFrame, title: str, which_list: list[str]):
    st.markdown(f"#### {title}")
    cols = st.columns(4)
    # keep order defined in which_list but skip missing/zero panels automatically
    name_to_meta = {n: (c, e) for n, c, e in IQ_MAP}
    i = 0
    for name in which_list:
        if name not in name_to_meta: 
            continue
        col_name, emoji = name_to_meta[name]
        with cols[i % 4]:
            _render_single_iq_card(df, name, col_name, emoji)
        i += 1

def render_all_iq_panels(df: pd.DataFrame, collapsible_specialty: bool = False):
    """Two tidy rows of IQ cards, auto-hiding zeroes."""
    st.markdown("### 🧩 Instant Quotes (IQ) Wins")
    _render_iq_row(df, "🔥 Top IQ Sellers", CORE_IQS)
    if collapsible_specialty:
        with st.expander("🌱 Specialty IQs (click to expand)", expanded=False):
            _render_iq_row(df, "", SPECIALTY_IQS)
    else:
        _render_iq_row(df, "🌱 Specialty IQs", SPECIALTY_IQS)
    st.caption("Only reps with >0 per IQ are shown. Panels auto-hide if no sales.")

def render_pool_splash_banners(df: pd.DataFrame):
    """Banner every time someone has ≥1 Pool sale (simple, no events stream needed)."""
    if "Pool" not in df.columns:
        return
    tmp = df[['Full_Name', 'Pool']].copy()
    tmp['Pool'] = pd.to_numeric(tmp['Pool'], errors='coerce').fillna(0)
    tmp = tmp[tmp['Pool'] > 0].sort_values('Pool', ascending=False)
    if tmp.empty:
        return
    for _, r in tmp.iterrows():
        st.markdown(
            f"""
            <div style="
                border-left:6px solid #06b6d4;
                background:rgba(6,182,212,0.10);
                padding:12px 14px;
                border-radius:12px;
                margin-bottom:10px;
                font-size:16px;
            ">
                🏊 <b>SPLASH ALERT!</b> <b>{r['Full_Name']}</b> sold a <b>Pool</b> IQ!
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()


with tab1:
    st.markdown("<h1 style='text-align: center;'>📊 Conversion Rate Leaderboard</h1>", unsafe_allow_html=True)
    df = load_data()

    from pytz import timezone
    eastern = timezone('US/Eastern')
    today = datetime.now(eastern).date()

    # 🧹 Clean headers and strip spaces
    df.columns = df.columns.str.strip()

    # Pull fresh history for records/champions
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H')
    history_df = load_history(cache_bust_key)
    history_df.columns = history_df.columns.str.strip()


    # 🎂 Handle Birthdays
    if 'Birthday' in df.columns:
        def clean_birthday(date_str):
            if pd.isna(date_str):
                return pd.NaT
            try:
                # Remove extra spaces
                date_str = " ".join(str(date_str).split()).strip()
                return pd.to_datetime(date_str + ' 2000', format='%B %d %Y', errors='coerce')
            except:
                return pd.NaT

    df['Birthday_MD'] = df['Birthday'].apply(clean_birthday)
    today_md = today.strftime('%m-%d')
    bdays_today = df[df['Birthday_MD'].dt.strftime('%m-%d') == today_md]

    for _, row in bdays_today.iterrows():
        full_name = f"{row['First_Name']} {row['Last_Name']}".strip()
        st.markdown(
            f"<div style='text-align: center; color: orange; font-size: 20px;'>🌼🎉 Happy Birthday, {full_name}! 🎉🌼</div>",
            unsafe_allow_html=True
        )


    # 🗓 Handle Anniversaries
    if 'Start Date' in df.columns:
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        anniv_today = df[df['Start Date'].dt.strftime('%m-%d') == today.strftime('%m-%d')]

        for _, row in anniv_today.iterrows():
            years = today.year - row['Start Date'].year
            full_name = f"{row['First_Name']} {row['Last_Name']}".strip()
            st.markdown(
                f"<div style='text-align: center; color: teal; font-size: 20px;'>🥳🎉 Happy {years}-year Anniversary, {full_name}! 🎉🥳</div>",
                unsafe_allow_html=True
            )



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
        names = ", ".join(
            (double_digit_celebs['First_Name'] + " " + double_digit_celebs['Last_Name']).str.strip().tolist()
)
        st.markdown(
            f"<div style='text-align: center; color: purple; font-size: 22px; font-weight: bold;'>🎉 DOUBLE DIGITS CLUB: {names} {'has' if len(double_digit_celebs)==1 else 'have'} crushed 10+ wins today!</div>",
            unsafe_allow_html=True
        )

    # 🏊 Pool splash shoutouts at the very top
    render_pool_splash_banners(df)

    all_reps = sorted(df[rep_col].dropna().unique())
    all_reps.insert(0, "🔍 Select your name")
    user = st.selectbox("👤 Who's using this app right now?", all_reps, key="selected_rep")

    if user == "🔍 Select your name":
        st.warning("Please select your name from the list to continue.")
        st.stop()

    active_df = df[df['Calls'] >= 1]
    user_data = df[df[rep_col] == user]
    first_name = user_data['First_Name'].values[0] if not user_data.empty else "Rep"

    # --------------------------------------------
    # 🏆 Records to Beat (All-Time Single-Day Highs)
    # --------------------------------------------
    records = get_records_to_beat(history_df)

    if records:
        st.markdown("<h2 style='text-align:center;'>🏆 Records to Beat (All-Time Single-Day Highs)</h2>", unsafe_allow_html=True)
        cols = st.columns(min(3, len(records)))
        for i, rec in enumerate(records):
            with cols[i % len(cols)]:
                date_str = rec['date'].strftime('%b %d, %Y') if rec['date'] else '—'
                st.markdown(f"""
                <div style='text-align:center; padding:14px; border:1px solid #ddd; border-radius:12px; margin-bottom:12px;'>
                    <div style='font-size:18px; font-weight:700;'>{rec['metric']}</div>
                    <div style='font-size:36px; font-weight:800; margin:6px 0;'>{rec['value']}</div>
                    <div style='font-size:14px; opacity:0.9;'>{rec['full_name']}</div>
                    <div style='font-size:12px; opacity:0.7;'>{rec['team']}</div>
                    <div style='font-size:12px; opacity:0.7;'>on {date_str}</div>
                    <div style='margin-top:8px; font-size:13px;'>🔥 Number to beat!</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No historical records found yet — go set the bar! 🔥")

    # --------------------------------------------
    # 👑 Last Week Champions (Sun–Sat ending yesterday)
    # --------------------------------------------
    top_team, top_reps = get_last_week_champions(history_df)

    st.markdown("<h2 style='text-align:center;'>👑 Last Week Champions</h2>", unsafe_allow_html=True)

    # Team crown
    if top_team:
        team_name, conv, wins, calls = top_team
        st.markdown(f"""
        <div style='text-align:center; padding:14px; border:2px solid #222; border-radius:12px; margin-bottom:14px;'>
            <div style='font-size:18px; font-weight:700;'>🥇 Top Team by Conversion</div>
            <div style='font-size:24px; font-weight:800; margin:6px 0;'>{team_name}</div>
            <div style='font-size:14px;'>🎯 {conv:.2f}% — {wins} wins / {calls} calls</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No team activity recorded last week.")


    # Top reps last week by CONVERSION (with wins/calls shown)
    if not top_reps.empty:
        st.markdown("<div style='text-align:center; font-weight:700; font-size:18px;'>🏅 Top Reps by Conversion (Last Week)</div>", unsafe_allow_html=True)
        cols = st.columns(min(3, len(top_reps)))
        for i, (_, r) in enumerate(top_reps.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style='text-align:center; padding:12px; border:1px solid #ddd; border-radius:12px; margin-top:8px;'>
                    <div style='font-size:18px; font-weight:700;'>{r['Full_Name']}</div>
                    <div style='font-size:14px; opacity:0.9;'>{r['Team Name']}</div>
                    <div style='font-size:20px; font-weight:800; margin-top:6px;'>🎯 {r['Conversion']:.2f}% conversion</div>
                    <div style='font-size:12px; opacity:0.7;'>🏆 {int(r['Wins'])} wins • {int(r['Calls'])} calls</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No rep activity recorded last week.")



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

        attach_services = ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
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



    # =========================
    # 📊 Your Conversion Rate (LIVE tiers)
    # =========================
    st.markdown("<h2 style='text-align: center;'>📊 Your Conversion Rate</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{personal_conversion:.2f}%</h3>", unsafe_allow_html=True)

    # Pull ALL Conversion tiers from the sheet
    # (Requires BONUS_SHEET_URL + load_section_tiers + get_point_threshold to be defined once at top)
    conv_tiers = load_section_tiers(BONUS_SHEET_URL, "Conversion")

    # Map to your buckets using the Points column (with safe fallbacks)
    thr_base        = get_point_threshold(conv_tiers, 0, 20.0)  # Base
    thr_green       = get_point_threshold(conv_tiers, 1, 21.0)  # Green
    thr_super_green = get_point_threshold(conv_tiers, 2, 22.0)  # Super Green
    thr_super_duper = get_point_threshold(conv_tiers, 3, 24.0)  # Super Duper
    thr_steve       = get_point_threshold(conv_tiers, 5, 27.0)  # Steve Green

    # 🌝 Motivational Blurb — now fully dynamic
    if personal_conversion >= thr_steve:
        st.success("🌟 Steve Green Level! You're an elite closer!")
        st.balloons()
    elif personal_conversion >= thr_super_duper:
        st.success("🚀 Super Duper Green! You're on fire!")
    elif personal_conversion >= thr_super_green:
        st.info("🌿 Super Green! You're doing awesome!")
    elif personal_conversion >= thr_green:
        st.info("📈 Green Zone! Keep pushing and you'll level up!")
    elif personal_conversion >= thr_base:
        gap = max(0.0, thr_green - personal_conversion) if thr_green > thr_base else max(0.0, thr_base - personal_conversion)
        st.warning(f"🚫 Almost There! Just {gap:.2f}% more for payout.")
    else:
        st.error(f"❌ Below Base ({thr_base:.2f}%). Let’s lock in and close the gap!")



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
    
    # =========================
    # 🎯 Conversion Milestones (LIVE tiers)
    # =========================
    import math

    # 1) Get live conversion tiers from the sheet (requires the helpers defined once at top)
    conv_tiers = load_section_tiers(BONUS_SHEET_URL, "Conversion")

    # Map point buckets → thresholds (with safe fallbacks if sheet not reachable)
    thr_base        = get_point_threshold(conv_tiers, 0, 20.0)  # Base
    thr_green       = get_point_threshold(conv_tiers, 1, 21.0)  # Green
    thr_super_green = get_point_threshold(conv_tiers, 2, 22.0)  # Super Green
    thr_super_duper = get_point_threshold(conv_tiers, 3, 24.0)  # Super Duper
    thr_steve       = get_point_threshold(conv_tiers, 5, 27.0)  # Steve Green

    # Build the ladder in ascending order, removing duplicates/None
    ladder = [thr_base, thr_green, thr_super_green, thr_super_duper, thr_steve]
    ladder = sorted({t for t in ladder if t is not None})

    # 2) Safely get this rep's Calls/Wins for the math
    def safe_int(val, default=0):
        try:
            v = pd.to_numeric(val, errors="coerce")
            return default if pd.isna(v) else int(v)
        except Exception:
            return default

    # Try to use existing vars if you already computed them; otherwise fall back to row/user_data
    if 'user_calls' in locals():
        calls = safe_int(user_calls)
        wins  = safe_int(user_wins if 'user_wins' in locals() else 0)
    elif 'row' in locals():
        calls = safe_int(row.get('Calls', 0))
        wins  = safe_int(row.get('Wins', 0))
    elif 'user_data' in locals() and not user_data.empty:
        calls = safe_int(user_data['Calls'].values[0])
        wins  = safe_int(user_data['Wins'].values[0])
    else:
        calls, wins = 0, 0  # last resort

    # 3) Render milestones from live thresholds
    st.markdown(
        "<h3 style='text-align: center;'>🎯 Conversion Milestones</h3>",
        unsafe_allow_html=True
    )
    lines = []
    for thr in ladder:
        required_wins = math.ceil((thr / 100.0) * calls) if calls > 0 else 0
        more_wins = max(0, required_wins - wins)
        status = "✅ Hit!" if more_wins == 0 else f"Need {more_wins} wins"
        lines.append(f"{thr:.2f}% → {status}")

    st.markdown(
        f"<div style='text-align: center; line-height: 1.8em;'>{'<br>'.join(lines)}</div>",
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



   # 🥇 Get top team stats (used internally, not displayed) — SAFE VERSION
    top_team_attaches = 0
    top_team_lt = 0
    top_team_name = None

    if not team_totals.empty:
        # Determine the current #1 team by conversion
        top_team_row = team_totals.sort_values(by="Conversion", ascending=False).iloc[0]
        top_team_name = top_team_row['Team Name']

        # Make sure service columns are numeric before summing
        df_numeric = df.copy()
        for col in ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']:
            df_numeric[col] = pd.to_numeric(df_numeric.get(col, 0), errors='coerce').fillna(0)

        # Totals for the top team
        top_team_attaches = df_numeric[df_numeric['Team Name'] == top_team_name][
            ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
        ].sum().sum()
        top_team_lt = df_numeric[df_numeric['Team Name'] == top_team_name]['Lawn Treatment'].sum()

    # 🧮 Your team stats (already defined as team_name)
    if team_name in team_totals['Team Name'].values:
        your_team_wins = team_totals.loc[team_totals['Team Name'] == team_name, 'Total_Wins'].values[0]
        your_team_attaches = df_numeric[df_numeric['Team Name'] == team_name][
            ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
        ].sum().sum()
        your_team_lt = df_numeric[df_numeric['Team Name'] == team_name]['Lawn Treatment'].sum()
    else:
        your_team_wins = 0
        your_team_attaches = 0
        your_team_lt = 0



    # 📈 Calculate required additional wins to beat top team's conversion rate
    top_team = None
    if (
        not team_totals.empty and 
        "Conversion" in team_totals.columns and 
        not team_totals["Conversion"].isna().all()
    ):
        sorted_teams = team_totals.sort_values(by="Conversion", ascending=False)
        if not sorted_teams.empty:
            top_team = sorted_teams.iloc[0]


    # 📈 Your team info
    needed_wins = 0
    needed_attaches = 0
    needed_lt = 0


    your_team_row_df = team_totals[team_totals["Team Name"] == team_name]

    if not your_team_row_df.empty:
        your_team_row = your_team_row_df.iloc[0]

        if top_team is not None and team_name == top_team["Team Name"]:
            # 🎉 Your team is already #1!
           st.markdown(f"""
           <div style='text-align: center; font-size: 20px; margin-top: 10px; padding: 12px; border-radius: 10px;
                       background-color: #e6ffe6; color: #004d00; border: 2px solid #00cc00;'>
               🎉 <b>Congrats, {team_name} is currently the top team!</b><br>
               Keep crushing it to stay on top! 💪🔥
           </div>
           """, unsafe_allow_html=True)


        elif top_team is not None:
            # Calculate how many more wins your team needs to take the top spot
            your_wins = your_team_row["Total_Wins"]
            your_calls = your_team_row["Total_Calls"]
            top_conversion_rate = top_team["Conversion"] / 100

            denominator = 1 - top_conversion_rate
            if denominator <= 0:
                needed_wins = 0
            else:
                 needed_wins = max(0, math.ceil((top_conversion_rate * your_calls - your_wins) / denominator))

        

            needed_attaches = max(0, int(top_team_attaches - your_team_attaches))
            needed_lt = max(0, int(top_team_lt - your_team_lt))

            st.markdown(f"""
            <div style='text-align: center; font-size: 18px; margin-top: 10px; padding: 10px; border-radius: 8px;
                        background-color: #f0f0f0; color: #333; border: 1px solid #ccc;'>
                <b>Can your team take the top spot?</b><br><br>
                🏆 Top Team: <b>{getattr(top_team, "Team Name", getattr(top_team, "Team_Name", "Unknown"))}</b><br>
                💪 Your Team: <b>{team_name}</b><br><br>
                Your team needs:<br>
                • <b>{needed_wins} more wins</b><br>
                • <b>{needed_attaches} more attaches</b><br>
                • <b>{needed_lt} more Lawn Treatments</b><br>
                to surpass the top team.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Could not calculate your team's comparison — team data was not found.")

        # 📉 Calculate attaches and LT
        needed_attaches = max(0, int(top_team_attaches - your_team_attaches))
        needed_lt = max(0, int(top_team_lt - your_team_lt))

        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px; padding: 10px; border-radius: 8px;
                    background-color: rgba(255, 255, 255, 0.05); color: #222; border: 1px solid #444;'>
            <b>Can your team take the top spot?</b><br><br>
            🏆 Top Team: <b>{getattr(top_team, "Team Name", getattr(top_team, "Team_Name", "Unknown"))}</b><br>
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

    # 🧩 Instant Quote Leaderboards (replacing the old service leaderboards section)
    render_all_iq_panels(df, collapsible_specialty=False)  # set True if you want the second row collapsed


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

    # -----------------------------
    # Bonus tiers: auto-load from your Sheet
    # -----------------------------
    BONUS_SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=374383792"
    )

    @st.cache_data(show_spinner=False)
    def load_bonus_tiers_from_formatted_sheet(url: str):
        import pandas as pd

        def parse_percent(val):
            s = str(val).strip().replace('%', '')
            if s == '' or s.lower() == 'nan':
                return None
            try:
                return float(s)
            except Exception:
                return None

        try:
            df = pd.read_csv(url, header=None)
        except Exception:
            return {}

        # Column B = thresholds, Column D = points (per your formatted tab)
        colB = df.iloc[:, 1].astype(str).str.strip()
        colD = df.iloc[:, 3] if df.shape[1] > 3 else pd.Series([None] * len(df))

        section_names = {
            "All-In Attach": "All-In Attach Rate",
            "Lawn Treatment": "LT",
            "Conversion": "Conversion",
            "QA": "QA",
        }

        def collect(section_label):
            idx = colB[colB.str.fullmatch(section_label, na=False)].index.tolist()
            if not idx:
                return []
            start = idx[0] + 1
            rows = []
            for r in range(start, len(df)):
                b = str(df.iat[r, 1]).strip()
                # stop on blank or next header/heading words inside the frame
                if b == '' or b in section_names.values() or b in ('Goals', 'Points', 'Current Cycle'):
                    break
                thr = parse_percent(b)
                pts_raw = df.iat[r, 3] if df.shape[1] > 3 else None
                try:
                    pts = int(str(pts_raw).strip())
                except Exception:
                    continue
                if thr is not None:
                    rows.append((thr, pts))
            rows.sort(key=lambda x: x[0], reverse=True)
            return rows

        return {
            "All-In Attach": collect(section_names["All-In Attach"]),
            "Lawn Treatment": collect(section_names["Lawn Treatment"]),
            "Conversion": collect(section_names["Conversion"]),
            "QA": collect(section_names["QA"]),
        }

    def get_tiers_for(metric_name: str, tiers_dict: dict, default: list[tuple[float, int]]):
        tiers = (tiers_dict or {}).get(metric_name, [])
        return tiers if tiers else default

    # Bonus metric setup
    def get_points(val, tiers):
        for threshold, pts in tiers:
            if val >= threshold:
                return pts
        return 0

    # Load tiers (fallback to your previous hard-coded values if the sheet is inaccessible)
    _tiers = load_bonus_tiers_from_formatted_sheet(BONUS_SHEET_URL)

    conversion_tiers = get_tiers_for("Conversion", _tiers, [(27, 5), (25, 4), (24, 3), (22, 2), (21, 1), (20, 0)])
    attach_tiers     = get_tiers_for("All-In Attach", _tiers, [(27, 2), (26, 1), (25, 0)])
    lt_tiers         = get_tiers_for("Lawn Treatment", _tiers, [(8.25, 3), (7.5, 2), (6.5, 1), (5.5, 0)])
    qa_tiers         = get_tiers_for("QA", _tiers, [(100, 2), (92, 1), (80, 0)])

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
    raw_bonus = row.get('Bonus Pay', 0)
    hourly = f"${float(raw_bonus):.2f}" if raw_bonus and str(raw_bonus).strip().replace("$", "") != "0" else "$0.00"
    st.markdown(f"**🌼 Points Earned:** {total_points} — **Hourly Forecast:** {hourly}")
    st.caption("You’ll earn this rate *only* if all 4 base qualifiers are met.")

    # 🏅 Personal Bests Section
    st.markdown("### 🏅 Personal Bests")

    # Load personal bests from history sheet
    history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
    history_df = pd.read_csv(history_url, header=1)  # Start at row 2
    history_df.columns = history_df.columns.str.strip()

    # Filter all rows for the selected rep
    rep_history = history_df[history_df['Rep'].astype(str).str.strip() == email]

    # Safely convert relevant columns to numeric
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

    def challenge_line(label, pb_val, emoji):
        if pd.isna(pb_val) or pb_val == 0:
            return f"{emoji} **{label} PB:** 0 — Let’s set a new record today! 💪"
        else:
            return f"{emoji} **{label} PB:** {int(pb_val)} — Can you hit {int(pb_val) + 1} today?"

    st.markdown(f"""
    <div style='font-size:18px'>
    🏆 **Wins PB:** {int(pb_wins) if not pd.isna(pb_wins) else 0} — Can you close {int(pb_wins)+1 if not pd.isna(pb_wins) else 1} today? 💥<br>
    {challenge_line('LT', pb_lawn, '🌿')}<br>
    {challenge_line('Bush', pb_bush, '🌳')}<br>
    {challenge_line('Flower', pb_flower, '🌸')}<br>
    {challenge_line('Mosquito', pb_mosquito, '🦟')}
    </div>
    """, unsafe_allow_html=True)

    # --------------- Dynamic Bonus Tiers table (from Sheet) ---------------
    st.markdown("### 🌾 Bonus Tiers")

    TIER_LABELS = {
        0: "Base",
        1: "Green",
        2: "Super Green",
        3: "Super Duper",
        4: "Super D-Duper",
        5: "Steve Green",
    }

    def fmt_pct(x: float) -> str:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return f"{s}%"

    tiers_by_metric = {
        "Conversion": conversion_tiers,
        "All-In Attach": attach_tiers,
        "Lawn Treatment": lt_tiers,
        "QA": qa_tiers,
    }

    # union of point levels present across all metrics
    used_points = sorted(set(p for tiers in tiers_by_metric.values() for _, p in tiers))
    columns = ["Metric"] + [TIER_LABELS.get(p, f"Tier {p}") for p in used_points]

    rows = []
    for metric, tiers in tiers_by_metric.items():
        # map point -> threshold (pick the threshold associated with that point)
        by_points = {}
        for thr, pts in tiers:
            # keep the threshold for this points bucket (higher threshold usually later in list; any is fine)
            by_points[pts] = thr
        row = [metric]
        for pts in used_points:
            row.append(f"{fmt_pct(by_points[pts])} ({pts})" if pts in by_points else "—")
        rows.append(row)

    tiers_df = pd.DataFrame(rows, columns=columns)
    st.dataframe(tiers_df, use_container_width=True, hide_index=True)

    # Points bar chart
    chart = pd.DataFrame({"Metric": list(points.keys()), "Points": list(points.values())})
    st.bar_chart(chart.set_index("Metric"))


# --------------------------------------------
# 📅 TAB 4: Yesterday’s Snapshot
# --------------------------------------------
with tab4:
    st.markdown("<h1 style='text-align: center;'>📅 Yesterday's Leaderboard</h1>", unsafe_allow_html=True)

    from pytz import timezone
    eastern = timezone('US/Eastern')
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H')  # refreshes hourly

    history_df = load_history(cache_bust_key)
    history_df.columns = history_df.columns.str.strip()


   
    # 🧠 Yesterday = actual performance day
    from pytz import timezone
    eastern = timezone('US/Eastern')
    yesterday = datetime.now(eastern).date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")  # ← compare using this

    # 🧼 Clean and parse
    history_df['Date'] = history_df['Date'].astype(str).str.strip()
    history_df['Date'] = pd.to_datetime(history_df['Date'], errors='coerce')
    history_df['Date_str'] = history_df['Date'].dt.date.astype(str)  # ← compare from this

    available_dates = history_df['Date_str'].dropna().unique().tolist()
    

    # 🧠 Pure string comparison now
    if yesterday_str in available_dates:
        snapshot_date = pd.to_datetime(yesterday_str).date()
        st.success(f"✅ Showing performance snapshot for {snapshot_date}")
    else:
        snapshot_date = pd.to_datetime(max(available_dates)).date() if available_dates else None
        st.info(f"⚠️ No performance data found for {yesterday_str}. Showing most recent available data from {snapshot_date} instead.")

    # ✅ Filter with actual Date column (not string version)
    yesterday_df = history_df[history_df['Date'].dt.date == snapshot_date]






    # ✅ Get selected rep from session state
    selected_rep = st.session_state.get("selected_rep")
    if not selected_rep:
        st.warning("Please select your name on the Leaderboard tab first.")
        st.stop()

    user_data = yesterday_df[yesterday_df['Rep'] == selected_rep]
    if user_data.empty:
        st.error(f"No performance data for {selected_rep} on {snapshot_date}.")
        st.stop()

    first_name = user_data['First_Name'].values[0] if 'First_Name' in user_data.columns else selected_rep
    st.markdown(
        f"<h3 style='text-align: center;'>🕰️ Snapshot for {first_name} — {yesterday.strftime('%B %d, %Y')}</h3>",
        unsafe_allow_html=True
    )

    # ---- Your Stats
    user_calls = int(user_data['Calls'].values[0])
    user_wins = int(user_data['Wins'].values[0])
    personal_conversion = (user_wins / user_calls * 100) if user_calls > 0 else 0

    st.markdown(f"<h3 style='text-align: center;'>📞 {user_calls} Calls | 🏆 {user_wins} Wins | 🎯 {personal_conversion:.2f}% Conversion</h3>", unsafe_allow_html=True)

    # ---- Team Info
    team_name = user_data['Team Name'].values[0] if 'Team Name' in user_data.columns else "Unknown"
    team_logo = f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{team_name.replace(' ', '_').lower()}.png' width='80'>" if pd.notna(team_name) else ""

    st.markdown(f"<div style='text-align: center;'>{team_logo}<br><b>{team_name} (Yesterday)</b></div>", unsafe_allow_html=True)

    # 🔧 Fix string columns so we can safely compare and calculate
    numeric_cols = ['Calls', 'Wins', 'Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
    for col in numeric_cols:
        if col in yesterday_df.columns:
            yesterday_df[col] = pd.to_numeric(yesterday_df[col], errors='coerce').fillna(0)


    # ---- Team Rank
    team_stats = yesterday_df[yesterday_df['Calls'] > 0].copy()
    team_stats['Calls'] = pd.to_numeric(team_stats['Calls'], errors='coerce')
    team_stats['Wins'] = pd.to_numeric(team_stats['Wins'], errors='coerce')

    team_totals = team_stats.groupby("Team Name").agg(
        Total_Calls=("Calls", "sum"),
        Total_Wins=("Wins", "sum")
    ).reset_index()
    team_totals['Conversion'] = (team_totals['Total_Wins'] / team_totals['Total_Calls']) * 100
    team_totals['Rank'] = team_totals['Conversion'].rank(ascending=False, method='min').astype(int)

    team_rank = team_totals.loc[team_totals['Team Name'] == team_name, 'Rank'].values[0] if team_name in team_totals['Team Name'].values else "N/A"
    st.markdown(f"<div style='text-align: center; font-size: 18px;'>🏅 Team Rank Yesterday: <b>{team_rank}</b></div>", unsafe_allow_html=True)

    # ---- Team Averages
    team_df = yesterday_df[yesterday_df['Team Name'] == team_name].copy()
    services = ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
    for svc in services:
        team_df[svc] = pd.to_numeric(team_df.get(svc, 0), errors='coerce').fillna(0)

    team_total_calls = team_df['Calls'].sum()
    team_total_wins = team_df['Wins'].sum()
    team_conversion_rate = (team_total_wins / team_total_calls * 100) if team_total_calls > 0 else 0
    team_attach_total = team_df[services].sum(axis=1).sum()
    team_attach_rate = (team_attach_total / team_total_wins * 100) if team_total_wins > 0 else 0
    team_lt = team_df['Lawn Treatment'].sum()
    team_lt_attach = (team_lt / team_total_wins * 100) if team_total_wins > 0 else 0

    st.markdown(f"""
    <div style='text-align:center; font-size: 18px; margin-top: 10px;'>
        <b>Team Averages (Yesterday)</b><br>
        🎯 Conversion: {team_conversion_rate:.2f}%<br>
        🧩 All-In Attach: {team_attach_rate:.2f}%<br>
        🌱 Lawn Treatment Attach: {team_lt_attach:.2f}%
    </div>
    """, unsafe_allow_html=True)




    # ---- Top 3 Reps (Conversion)
    active_df = yesterday_df[yesterday_df['Calls'] > 0].copy()
    active_df['Conversion'] = (active_df['Wins'] / active_df['Calls']) * 100
    active_df['Full_Name'] = active_df['First_Name'].astype(str).str.strip() + ' ' + active_df['Last_Name'].astype(str).str.strip()
    active_df = active_df.sort_values(by='Conversion', ascending=False)

    st.markdown("<h3 style='text-align: center;'>🏅 Top 3 Reps (Yesterday)</h3>", unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(active_df.head(3).iterrows()):
        logo_url = f"https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{row['Team Name'].replace(' ', '_').lower()}.png" if pd.notna(row['Team Name']) else ""
        st.markdown(f"""
            <div style='text-align: center; font-size: 18px;'>
                <img src="{logo_url}" width="40"><br>
                {medals[i]} {row['Full_Name']} — {row['Conversion']:.2f}%
            </div>
        """, unsafe_allow_html=True)



    # ---- Top 3 Teams
    st.markdown("<h3 style='text-align: center;'>👥 Top 3 Teams (Yesterday)</h3>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (_, team_row) in enumerate(team_totals.sort_values(by="Conversion", ascending=False).head(3).iterrows()):
        with cols[i]:
            logo_url = f"https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{team_row['Team Name'].replace(' ', '_').lower()}.png"
            st.markdown(f"""
            <div style='text-align: center; font-size: 16px;'>
                <b>{team_row['Team Name']}</b><br>
                <img src="{logo_url}" width="60"><br>
                {team_row['Conversion']:.2f}%<br>
                {int(team_row['Total_Wins'])} wins / {int(team_row['Total_Calls'])} calls
            </div>
            """, unsafe_allow_html=True)

    

    # ---- Full Conversion Leaderboard (with logos)
    active_df = active_df[['Full_Name', 'Team Name', 'Conversion']].reset_index(drop=True)
    active_df['Rank'] = active_df.index + 1
    active_df['Conversion'] = active_df['Conversion'].map('{:.2f}%'.format)


    # Build Team Logo column
    active_df['Team_Logo'] = active_df['Team Name'].astype(str).apply(
        lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='30'>" if pd.notna(name) else ""
    )

    # Display Conversion leaderboard with logo
    st.markdown("<h3 style='text-align: center;'>🏆 Full Conversion Leaderboard (Yesterday)</h3>", unsafe_allow_html=True)
    st.markdown(
        active_df[['Rank', 'Full_Name', 'Conversion', 'Team_Logo']].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )


    # ✅ Build Full_Name field for attach shoutouts
    if 'Full_Name' not in yesterday_df.columns:
        if 'First_Name' in yesterday_df.columns and 'Last_Name' in yesterday_df.columns:
            yesterday_df['Full_Name'] = yesterday_df['First_Name'].astype(str).str.strip() + ' ' + yesterday_df['Last_Name'].astype(str).str.strip()
        elif 'Name_Proper' in yesterday_df.columns:
            yesterday_df['Full_Name'] = yesterday_df['Name_Proper']
        else:
            yesterday_df['Full_Name'] = yesterday_df['Rep']

    # 🧩 Top 3 Attach Reps for each service
    st.markdown("<hr><h3 style='text-align: center;'>🧩 Top 3 Attach Leaders (Yesterday)</h3>", unsafe_allow_html=True)
    attach_cols = ['Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal']
    emojis = ['🌿', '🌳', '🦟', '🌸', '🍂']
    titles = ['Lawn Treatment', 'Bush Trim', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal']

    attach_cols_zipped = zip(attach_cols, emojis, titles)
    col_blocks = st.columns(len(attach_cols))
    for col, (svc, emoji, title) in zip(col_blocks, attach_cols_zipped):
        with col:
            show_yesterday_service_top(yesterday_df.copy(), svc, emoji, title)


    # 🧾 Full Attach Leaderboard
    st.markdown("<h3 style='text-align: center;'>📋 Full Attach Service Leaderboard (Yesterday)</h3>", unsafe_allow_html=True)

    # Make sure service columns are numeric
    for col in attach_cols:
        yesterday_df[col] = pd.to_numeric(yesterday_df.get(col, 0), errors='coerce').fillna(0)

    # Build Team Logo column
    if 'Team_Logo' not in yesterday_df.columns and 'Team Name' in yesterday_df.columns:
        yesterday_df['Team_Logo'] = yesterday_df['Team Name'].astype(str).apply(
            lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='30'>" if pd.notna(name) else ""
        )

    lt_display = yesterday_df[yesterday_df['Calls'] > 0].copy()  # 👈 Only reps with >0 calls
    lt_display['Rank'] = lt_display['Lawn Treatment'].rank(ascending=False, method='min').astype(int)
    lt_display = lt_display.sort_values(by='Lawn Treatment', ascending=False)


    # Display columns
    display_cols = ['Rank', 'Full_Name'] + attach_cols + ['Team_Logo']
    lt_display = lt_display[display_cols]
    lt_display.columns = ['Rank', 'Rep Name', 'Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal', 'Team Logo']

    st.markdown(lt_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Style
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
        width: 95%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    th {
        background-color: #333;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# 👩‍💻 TAB 5:  Team Lead Dashboard
# --------------------------------------------

with tab5:

    df = load_data()

    from pytz import timezone
    eastern = timezone('US/Eastern')
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H')  # refreshes hourly

    history_df = load_history(cache_bust_key)
    # --- Live Base Goals from Avatar Leaderboard sheet ---
    BONUS_SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=374383792"
    )

    @st.cache_data(show_spinner=False)
    def load_base_goals(url: str):
        df = pd.read_csv(url, header=None)
        df = df.fillna("")

        # Helper: find first row where "Base" appears
        def find_base(section_label):
            section_row = df.index[df.iloc[:,1].astype(str).str.strip() == section_label].tolist()
            if not section_row: 
                return None
            start = section_row[0] + 1
            for r in range(start, len(df)):
                label = str(df.iat[r,2]).strip()
                if label.lower() == "base":
                    val = str(df.iat[r,1]).replace("%","").strip()
                    try: return float(val)
                    except: return None
            return None

        return {
            "Conversion": find_base("Conversion"),
            "Attach": find_base("All-In Attach Rate"),
            "LT": find_base("LT"),
            "QA": find_base("QA")
        }

    base_goals = load_base_goals(BONUS_SHEET_URL)
    def coalesce_num(x, fallback):
        try:
            if x is None:
                return fallback
            if isinstance(x, float) and np.isnan(x):
                return fallback
            return float(x)
        except Exception:
        r    eturn fallback
    BASE_CONV   = coalesce_num(base_goals.get("Conversion"), 20.0)
    BASE_ATTACH = coalesce_num(base_goals.get("Attach"), 25.0)
    BASE_LT     = coalesce_num(base_goals.get("LT"), 5.5)
    BASE_QA     = coalesce_num(base_goals.get("QA"), 80.0)

    # ---- SELECT TEAM LEAD ----
    manager_directs = df['Manager_Direct'].dropna().unique()
    selected_lead = st.selectbox("Select Your Name (Team Lead):", sorted(manager_directs))

    # Filter reps under selected team lead
    team_df = df[df['Manager_Direct'] == selected_lead].copy()

    # Determine team name from the lead's reps
    team_name = team_df['Team Name'].dropna().astype(str).unique()[0] if not team_df.empty else "Unknown"

    # Calculate team rank across all teams
    team_stats = df[df['Calls'] > 0].copy()
    team_stats['Calls'] = pd.to_numeric(team_stats['Calls'], errors='coerce')
    team_stats['Wins'] = pd.to_numeric(team_stats['Wins'], errors='coerce')

    team_totals = team_stats.groupby("Team Name").agg(
        Total_Calls=("Calls", "sum"),
        Total_Wins=("Wins", "sum")
    ).reset_index()
    team_totals['Conversion'] = (team_totals['Total_Wins'] / team_totals['Total_Calls']) * 100
    team_totals['Rank'] = team_totals['Conversion'].rank(ascending=False, method='min').astype(int)

    # Extract this team’s rank
    team_rank = int(team_totals.loc[team_totals['Team Name'] == team_name, 'Rank'].values[0]) if team_name in team_totals['Team Name'].values else "N/A"

    # ---- TEAM STATS ----
    st.subheader(f"📊 Team Stats for {selected_lead}")
    if team_df.empty:
        st.warning("No reps found for this Team Lead.")
    else:
        attach_cols = ['Lawn Treatment', 'Leaf Removal', 'Mosquito', 'Flower Bed Weeding', 'Bush Trimming', 'Wins', 'Calls']
        for col in attach_cols:
            team_df[col] = pd.to_numeric(team_df[col], errors='coerce').fillna(0)

        team_df['All-In Attach %'] = (
            (team_df['Lawn Treatment'] +
            team_df['Leaf Removal'] +
            team_df['Mosquito'] +
            team_df['Flower Bed Weeding'] +
            team_df['Bush Trimming']) /
            team_df['Wins'].replace(0, np.nan)
        ) * 100
        team_df['All-In Attach %'] = team_df['All-In Attach %'].fillna(0)

        cols_to_show = ['Name_Proper', 'Conversion', 'LT Attach', 'All-In Attach %', 'BonusQA']
        display_df = team_df[cols_to_show].copy()
        display_df.columns = ['Rep', 'Conversion %', 'LT Attach %', 'All-In Attach %', 'QA %']

        for col in ['Conversion %', 'LT Attach %', 'QA %']:
            display_df[col] = (
                display_df[col]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.strip()
                .replace(['', 'nan'], '0')
                .str.extract(r'([0-9.]+)', expand=False)
                .fillna('0')
                .astype(float)
            )

        total_calls = team_df['Calls'].sum()
        total_wins = team_df['Wins'].sum()
        total_lawn_treatments = team_df['Lawn Treatment'].sum()
        total_attaches = (
            team_df['Lawn Treatment'] +
            team_df['Leaf Removal'] +
            team_df['Mosquito'] +
            team_df['Flower Bed Weeding'] +
            team_df['Bush Trimming']
        ).sum()

        avg_conversion = (total_wins / total_calls) * 100 if total_calls > 0 else 0
        avg_attach = (total_attaches / total_wins) * 100 if total_wins > 0 else 0
        avg_lt = (total_lawn_treatments / total_wins) * 100 if total_wins > 0 else 0

        qa_values = team_df['BonusQA'].dropna().astype(str).str.replace('%', '', regex=False).str.extract(r'([0-9.]+)', expand=False).astype(float)
        qa_display = qa_values.iloc[0] if not qa_values.empty else 0

        # ---------- SAFE TOP TEAM / YOUR TEAM METRICS ----------
        # Make sure numeric attach columns exist & are numeric
        df_numeric = df.copy()
        for col in ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']:
            if col not in df_numeric.columns:
                df_numeric[col] = 0
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

        # Defaults so we never crash if data is missing
        top_team_name = "—"
        top_team_attaches = 0
        top_team_lt = 0
        your_team_attaches = 0
        your_team_lt = 0
        needed_wins = 0
        needed_attaches = 0
        needed_lt = 0

        # Compute only when we have data
        if not team_df.empty and not team_totals.empty and ('Team Name' in df_numeric.columns):
            # Top team by conversion
            top_row = team_totals.sort_values(by="Conversion", ascending=False).iloc[0]
            top_team_name = str(top_row.get('Team Name', '—'))
            top_total_wins = float(top_row.get('Total_Wins', 0) or 0.0)

            # Your team totals already computed above: total_wins, team_name, etc.
            your_total_wins = float(total_wins or 0.0)

            # Sum attaches for top team & your team
            attach_cols_sum = ['Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']

            top_team_attaches = df_numeric.loc[df_numeric['Team Name'] == top_team_name, attach_cols_sum].sum().sum()
            top_team_lt       = df_numeric.loc[df_numeric['Team Name'] == top_team_name, 'Lawn Treatment'].sum()

            your_team_attaches = df_numeric.loc[df_numeric['Team Name'] == team_name, attach_cols_sum].sum().sum()
            your_team_lt       = df_numeric.loc[df_numeric['Team Name'] == team_name, 'Lawn Treatment'].sum()

            # Needed deltas (never negative)
            needed_wins      = max(0, int(round(top_total_wins - your_total_wins)))
            needed_attaches  = max(0, int(round(top_team_attaches - your_team_attaches)))
            needed_lt        = max(0, int(round(top_team_lt - your_team_lt)))

        # ---------- RENDER TOP TEAM CARD ----------
        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px; padding: 10px; border-radius: 8px;
                    background-color: #f0f0f0; color: #333; border: 1px solid #ccc;'>
            <b>Can your team take the top spot?</b><br><br>
            🏆 Top Team: <b>{top_team_name}</b><br>
            💪 Your Team: <b>{team_name}</b><br><br>
            Your team needs:<br>
            • <b>{needed_wins} more wins</b><br>
            • <b>{needed_attaches} more attaches</b><br>
            • <b>{needed_lt} more Lawn Treatments</b><br>
            to surpass the top team.
        </div>
        """, unsafe_allow_html=True)

        valid_qa = display_df['QA %']
        valid_qa = valid_qa[valid_qa > 0]
        avg_qa = valid_qa.mean() if not valid_qa.empty else 0

        # 🧢 Team Logo + Rank
        team_logo_url = f"https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{team_name.replace(' ', '_').lower()}.png"
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src="{team_logo_url}" width="100"><br>
            <div style='font-size: 20px; color: teal; font-weight: bold;'>🏅 Team Rank: {team_rank}</div>
        </div>
        """, unsafe_allow_html=True)

        # 🌱 Today’s Team Averages — Styled Block
        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px;'>
            <b>{team_name} — Today’s Team Averages:</b><br>
            🧮 Conversion: {avg_conversion:.2f}%<br>
            🧩 All-In Attach: {avg_attach:.2f}%<br>
            🍃 Lawn Treatment Attach: {avg_lt:.2f}%
        </div>
        """, unsafe_allow_html=True)

        display_df.set_index('Rep', inplace=True)

        def highlight_top_nonzero(s):
            is_max = s == s[s > 0].max()
            return ['background-color: #FFD700; color: black' if v else '' for v in is_max]

        highlight_style = display_df.style.apply(highlight_top_nonzero, axis=0)
        st.write("### Rep Breakdown")
        st.dataframe(highlight_style.format("{:.2f}"), use_container_width=True)

        # ----------------------------
        # 💸 TEAM LEAD BONUS TRACKER
        # ----------------------------
        st.subheader("💸 Team Lead Bonus Tracker")

        from datetime import datetime, timedelta
        import math

        # --- PAY CYCLE FUNCTION ---
        def get_current_pay_cycle():
            today = datetime.today().date()
            base_date = datetime(2025, 7, 20).date()  # Start of the first pay cycle
            days_since = (today - base_date).days
            cycle_start = base_date + timedelta(days=(days_since // 14) * 14)
            cycle_end = cycle_start + timedelta(days=13)
            return cycle_start, cycle_end

        cycle_start, cycle_end = get_current_pay_cycle()

        # --- LOAD BONUS SHEET ---
        tl_bonus_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=1302605632"
        tl_bonus_df = pd.read_csv(tl_bonus_url)
        tl_bonus_df.columns = tl_bonus_df.columns.str.strip()

        # --- CLEAN + SELECT ---
        tl_bonus_df['TL_clean'] = tl_bonus_df['Team Lead'].astype(str).str.strip().str.lower()
        selected_lead_clean = selected_lead.strip().lower()
        lead_bonus = tl_bonus_df[tl_bonus_df['TL_clean'] == selected_lead_clean]

        if lead_bonus.empty:
            st.warning(f"No bonus data found for: {selected_lead}")
        else:
            bonus_row = lead_bonus.iloc[0]

            def parse_pct(val):
                try:
                    return float(str(val).replace('%', '').strip())
                except:
                    return 0

            conversion = parse_pct(bonus_row['Conversion'])
            attach = parse_pct(bonus_row['Attach'])
            lt = parse_pct(bonus_row['LT'])
            qa = parse_pct(bonus_row['QA'])
            total_points = int(bonus_row['Total Points']) if str(bonus_row['Total Points']).isdigit() else 0
            tier = str(bonus_row['Bonus Tier']).strip()
            increase = str(bonus_row['$ Increase']).strip()

            def check(val, threshold):
                return "✅" if val >= threshold else "❌"

            st.markdown(f"""
            ### 📈 {selected_lead}'s Bonus Snapshot  
            - **Conversion:** {conversion:.2f}% {check(conversion, BASE_CONV)}  
            - **All-In Attach:** {attach:.2f}% {check(attach, BASE_ATTACH)}  
            - **Lawn Treatment:** {lt:.2f}% {check(lt, BASE_LT)}  
            - **QA:** {qa:.2f}% {check(qa, BASE_QA)}  
            - **Total Points:** {total_points}  
            - **Bonus Tier:** `{tier}`  
            - **Hourly Increase:** **{increase}**
            """)

            # --- BONUS BLURB ---
            try:
                hourly_float = float(increase.replace("$", "").strip())
            except:
                hourly_float = 0

            if hourly_float > 0:
                estimated_hours = float(bonus_row.get("Hours Worked", 80))
                bonus_total = hourly_float * estimated_hours
                import math
                hours_display = int(estimated_hours) if isinstance(estimated_hours, (int, float)) and not math.isnan(estimated_hours) else 80
                bonus_total_safe = hourly_float * hours_display


                st.markdown("### 🌟 Bonus ")
                st.success(f"You're currently earning **${hourly_float:.2f}/hour**!")
                st.markdown(f"With an estimated **{hours_display} hours worked**, that's about **${bonus_total_safe:.2f} extra this cycle!** 💸")
                st.markdown("What will you spend your bonus on — a new mower or margarita pitcher? 😎")

            else:
                st.warning("No bonus just yet — but you’re not far! Let's mow down those goals:")

            # --- HISTORY SHEET CALC ---
            st.markdown("### 🧠 What You Need to Hit Bonus Goals based on google form attach entries")

            history_df['Date'] = pd.to_datetime(history_df['Date'], errors='coerce')
            cycle_df = history_df[
                (history_df['Date'].dt.date >= cycle_start) &
                (history_df['Date'].dt.date <= cycle_end) &
                (history_df['Manager_Direct'].astype(str).str.strip().str.lower() == selected_lead_clean)
            ].copy()

            # Clean numeric columns
            for col in ['Wins', 'Lawn Treatment', 'Leaf Removal', 'Mosquito', 'Flower Bed Weeding', 'Bush Trimming']:
                cycle_df[col] = pd.to_numeric(cycle_df[col], errors='coerce').fillna(0)

            # Totals
            cycle_wins = cycle_df['Wins'].sum()
            cycle_lt = cycle_df['Lawn Treatment'].sum()
            cycle_attaches = (
                cycle_df['Lawn Treatment'] +
                cycle_df['Leaf Removal'] +
                cycle_df['Mosquito'] +
                cycle_df['Flower Bed Weeding'] +
                cycle_df['Bush Trimming']
            ).sum()

            lt_pct = (cycle_lt / cycle_wins) * 100 if cycle_wins > 0 else 0
            attach_pct = (cycle_attaches / cycle_wins) * 100 if cycle_wins > 0 else 0

            st.markdown(f"""
            - 📦 **All-In Attach Rate:** `{attach_pct:.2f}%` {check(attach_pct, 25)}
            - 🍃 **Lawn Treatment Rate:** `{lt_pct:.2f}%` {check(lt_pct, 5.5)}
            """)

            thresholds = {"Conversion": BASE_CONV, "Attach": BASE_ATTACH, "LT": BASE_LT, "QA": BASE_QA}
            team_calls = float(bonus_row.get("Call #", 0))
            team_wins = float(bonus_row.get("Win #", 0))
            team_qa_scores = display_df['QA %']

            needs = []

            # 🏎️ Conversion
            actual_conversion = (team_wins / team_calls) * 100 if team_calls > 0 else 0
            if actual_conversion < thresholds["Conversion"] - 0.001:
                required_wins = (thresholds["Conversion"] / 100) * team_calls
                more_wins_needed = math.ceil(required_wins - team_wins)
                if more_wins_needed > 0:
                    needs.append(
                        f"🏎️ **{more_wins_needed} more Wins** to reach {thresholds['Conversion']:.2f}% Conversion"
                    )

            # 📦 Attach
            if attach_pct < thresholds["Attach"]:
                needed_attaches = math.ceil((thresholds["Attach"] / 100) * cycle_wins)
                more_attaches_needed = max(0, needed_attaches - cycle_attaches)
                needs.append(
                    f"📦 **{more_attaches_needed} more Attaches** to hit {thresholds['Attach']:.2f}% All-In Attach"
                )

            # 🍃 LT
            if lt_pct < thresholds["LT"]:
                needed_lt = math.ceil((thresholds["LT"] / 100) * cycle_wins)
                more_lt_needed = max(0, needed_lt - cycle_lt)
                needs.append(
                    f"🌱 **{more_lt_needed} more Lawn Treatments** to reach {thresholds['LT']:.2f}% LT"
                )
            # ✅ QA
            current_qa_avg = team_qa_scores[team_qa_scores > 0].mean()
            if current_qa_avg < thresholds["QA"]:
                num_agents = team_qa_scores.shape[0]
                needed_qa_total = thresholds["QA"] * num_agents
                current_qa_total = team_qa_scores.sum()
                more_100s = math.ceil((needed_qa_total - current_qa_total) / (100 - thresholds["QA"]))
                more_100s = max(1, more_100s)
                needs.append(
                    f"🎯 **{more_100s} more 100 QA scores** to average {thresholds['QA']:.0f}%"
                )

            if needs:
                st.warning("You're not far! Here's what your team needs to meet **all 4 base goals** and cash in:")
                for line in needs:
                    st.markdown(f"- {line}")
            else:
                st.success("You’re crushing it — your team is currently hitting **all 4 base goals** 💪 Time to rake in that bonus! 🌿💸")

            # --- POINT CHART ---
            st.caption("Team Leads earn bonus pay based on their team's performance in 4 metrics. All base thresholds must be met to qualify.")

        # ---- SHOUTOUT GENERATOR ----
        st.subheader("📣 Shoutout Generator")
        fun_phrases = {
            "Conversion %": "Now that's how you mow down objections!",
            "LT Attach %": "Sprinkling in those extras like a true lawn care artist!",
            "All-In Attach %": "Pulled out all the weeds and sealed the deal!",
            "QA %": "Precision cuts and perfect scripts — QA on point!"
        }

        top_performers = {}
        for col in display_df.columns:
            non_zero_vals = display_df[col][display_df[col] > 0]
            if not non_zero_vals.empty:
                top_value = non_zero_vals.max()
                tied_reps = non_zero_vals[non_zero_vals == top_value].index.tolist()
                top_performers[col] = (tied_reps, top_value)

        for metric, (reps, value) in top_performers.items():
            rep_names = ", ".join([f"**{rep}**" for rep in reps])
            shoutout = f"🌟 Big shoutout to {rep_names} for leading the team in **{metric}** at **{value:.1f}%**! {fun_phrases.get(metric, 'You raked in results!')}"
            st.code(shoutout, language='markdown')

        # ---- MOST IMPROVED ----
        st.subheader("🔄 Most Improved")

        available_dates = sorted(history_df['Date'].dropna().dt.date.unique())

        st.write("### Select Timeframes to Compare")
        col_a, col_b = st.columns(2)

        with col_a:
            start_a = st.selectbox("Start of Period A", available_dates, index=0, key='start_a')
            end_a = st.selectbox("End of Period A", available_dates, index=len(available_dates)//2, key='end_a')

        with col_b:
            start_b = st.selectbox("Start of Period B", available_dates, index=len(available_dates)//2 + 1, key='start_b')
            end_b = st.selectbox("End of Period B", available_dates, index=len(available_dates)-1, key='end_b')

        def get_period_df(start, end):
            period_df = history_df[(history_df['Date'].dt.date >= start) &
                                   (history_df['Date'].dt.date <= end) &
                                   (history_df['Manager_Direct'] == selected_lead)].copy()

            # Explicit list of needed numeric columns
            cols_to_sum = [
                'Calls', 'Wins', 'Lawn Treatment', 'Bush Trimming',
                'Flower Bed Weeding', 'Mosquito', 'Leaf Removal'
            ]

            # Ensure all numeric columns are properly cleaned
            for col in cols_to_sum:
                period_df[col] = (
                    period_df[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.extract(r'([0-9.]+)', expand=False)
                    .astype(float)
                    .fillna(0)
                )

            # Group by rep and sum raw numbers
            grouped = period_df.groupby('Name_Proper')[cols_to_sum].sum().reset_index()

            # Recalculate performance metrics
            grouped['Conversion'] = (grouped['Wins'] / grouped['Calls'].replace(0, np.nan)) * 100
            grouped['LT Attach'] = (grouped['Lawn Treatment'] / grouped['Wins'].replace(0, np.nan)) * 100
            grouped['All-In Attach %'] = (
                 (grouped['Lawn Treatment'] +
                 grouped['Bush Trimming'] +
                 grouped['Flower Bed Weeding'] +
                 grouped['Mosquito'] +
                 grouped['Leaf Removal']) /
                grouped['Wins'].replace(0, np.nan)
            ) * 100

            grouped.fillna(0, inplace=True)

            return grouped

        df_a = get_period_df(start_a, end_a)
        df_b = get_period_df(start_b, end_b)

        improvements = []
        for _, row in display_df.reset_index().iterrows():
            rep_name = row['Rep']
            a = df_a[df_a['Name_Proper'] == rep_name]
            b = df_b[df_b['Name_Proper'] == rep_name]
            if not a.empty and not b.empty:
                prev_conversion = a['Conversion'].iloc[0]
                prev_lt = a['LT Attach'].iloc[0]
                prev_attach = a['All-In Attach %'].iloc[0]

                current_conversion = b['Conversion'].iloc[0]
                current_lt = b['LT Attach'].iloc[0]
                current_attach = b['All-In Attach %'].iloc[0]

                improvements.append({
                    'Rep': rep_name,
                    'Conversion Before': prev_conversion,
                    'Conversion Now': current_conversion,
                    'Conversion Change': current_conversion - prev_conversion,
                    'LT Attach Before': prev_lt,
                    'LT Attach Now': current_lt,
                    'LT Attach Change': current_lt - prev_lt,
                    'All-In Attach Before': prev_attach,
                    'All-In Attach Now': current_attach,
                    'All-In Attach Change': current_attach - prev_attach
                })

        if improvements:
            imp_df = pd.DataFrame(improvements)
            for col in ['Conversion Change', 'LT Attach Change', 'All-In Attach Change']:
                imp_df[col] = imp_df[col].map(lambda x: f"⬆️ {x:.1f}%" if x > 0 else (f"⬇️ {abs(x):.1f}%" if x < 0 else "—"))

            st.dataframe(imp_df, use_container_width=True)

        # 🎯 Most Improved Shoutout + Honorable Mentions
        shout_metrics = ['Conversion Change', 'LT Attach Change', 'All-In Attach Change']
        improvement_scores = []

        if improvements:
            for _, row in imp_df.iterrows():
                score = 0
                for metric in shout_metrics:
                    if '⬆️' in row[metric]:
                        score += float(row[metric].replace('⬆️','').replace('%','').strip())
                improvement_scores.append((row['Rep'], score))

            if improvement_scores:
                sorted_improvers = sorted(improvement_scores, key=lambda x: x[1], reverse=True)
                most_improved_rep, top_score = sorted_improvers[0]

                st.markdown(
                    f"""### 🌟 **Most Improved Agent**  
                    Massive congrats to **{most_improved_rep}**, who made the biggest leap in performance — you're leveling up like a legend! 🚀"""
                )

                # Honorable Mentions
                honorable_mentions = sorted_improvers[1:4]  # up to 3
                if honorable_mentions:
                    shout_lines = [f"**{rep}** (Total Gain: {score:.1f}%)" for rep, score in honorable_mentions]
                    shout_text = " • ".join(shout_lines)
                    st.markdown(f"🏅 **Honorable Mentions:** {shout_text}")

        # Placeholder Sections
        st.subheader("🏆 Personal Bests (Coming Soon)")
        st.subheader("⚔️ Compare With Another Team (Coming Soon)")

# ----------------------------
# 🏆 TAB 6: Senior Manager View
# ----------------------------
with tab6:
    from math import ceil
    from datetime import datetime, timedelta
    from pytz import timezone
    import pandas as pd
    import numpy as np

    eastern = timezone('US/Eastern')

    # ---------- helpers ----------
    def pick_col(df, candidates, required=True, label_hint=""):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            st.error(f"Missing required column for {label_hint or candidates}")
        return None

    def coerce_date(df, date_col):
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df

    def add_week_start(df, date_col):
        if date_col and date_col in df.columns:
            df["Week Start"] = df[date_col].dt.to_period("W-MON").apply(lambda p: p.start_time.date())
        return df

    def safe_to_num(s):
        """Convert to numeric safely, works for Series/DataFrame/scalars."""
        if isinstance(s, (int, float, np.number)):
            return 0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else float(s)
        try:
            return pd.to_numeric(s, errors="coerce").fillna(0.0)
        except Exception:
            return 0.0

    def safe_sum(df, col):
        if col and col in df.columns:
            return safe_to_num(df[col]).sum()
        return 0.0

    def compute_rollup(df, group_cols, cols):
        """Aggregate true totals by group (no averaging of averages)."""
        out = []
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame(columns=group_cols + ["wins","calls","attaches","lt_hits","qa_num","qa_den"])

        for keys, chunk in df.groupby(group_cols, dropna=False):
            wins     = safe_sum(chunk, cols.get("wins"))
            calls    = safe_sum(chunk, cols.get("calls"))
            attaches = safe_sum(chunk, cols.get("attaches"))
            lt_hits  = safe_sum(chunk, cols.get("lt_hits")) if cols.get("lt_hits") else 0.0
            qa_num   = safe_sum(chunk, cols.get("qa_num")) if cols.get("qa_num") else 0.0
            qa_den   = safe_sum(chunk, cols.get("qa_den")) if cols.get("qa_den") else 0.0

            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row.update({
                "wins": wins, "calls": calls, "attaches": attaches, "lt_hits": lt_hits,
                "qa_num": qa_num, "qa_den": qa_den
            })
            out.append(row)
        return pd.DataFrame(out) if out else pd.DataFrame(columns=group_cols+["wins","calls","attaches","lt_hits","qa_num","qa_den"])

    def add_rates(df):
        """Add ratio columns (0..1)."""
        df = df.copy()
        for col in ("wins","calls","attaches","lt_hits","qa_num","qa_den"):
            if col not in df.columns:
                df[col] = 0.0

        df["Conversion %"]   = (df["wins"] / df["calls"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["All-In Attach %"] = (df["attaches"] / df["wins"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["LT %"]           = (df["lt_hits"] / df["wins"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["QA %"]           = (df["qa_num"] / df["qa_den"]).replace([np.inf, -np.inf], 0).fillna(0)
        return df

    def format_pct(x):
        try:
            return f"{100*float(x):.1f}%"
        except Exception:
            return "-"

    def normalize_name(s):
        s = str(s or "").strip()
        if "," in s:
            last, first = [p.strip() for p in s.split(",", 1)]
            s = f"{first} {last}"
        return " ".join(s.split())

    # --- Load frames from your existing functions ---
    df = load_data()
    cache_bust_key = datetime.now(eastern).strftime("%Y-%m-%d-%H")
    history_df = load_history(cache_bust_key)

    # --- Candidate mappings (includes component attach cols so we can derive) ---
    CANDIDATE = {
        "date":       ["Date", "date"],
        "name":       ["Name_Proper", "Name", "Rep"],
        "role":       ["Current_Role", "Role"],
        "mgr_direct": ["Manager_Direct", "Manager", "Direct_Manager", "Reports_To"],
        "wins":       ["Wins", "Sales", "Sold"],
        "calls":      ["Total Calls", "Calls", "Contacts", "Leads", "Total Leads"],
        "attaches":   ["All-In Attaches", "Attaches", "All-In Attach Count"],
        "lt_hits":    ["Lawn Treatment", "LT Hits", "Lawn Treatment Wins", "LT Count"],
        "leaf":       ["Leaf Removal", "Leaf_Removal"],
        "mosq":       ["Mosquito", "Mosquito Treatment", "Mosquito_Treatment"],
        "flower":     ["Flower Bed Weeding", "Flower_Bed_Weeding", "Bed Weeding"],
        "bush":       ["Bush Trimming", "Bush_Trimming"],
        "qa_num":     ["QA_Points_Passed", "QA_Numerator"],
        "qa_den":     ["QA_Points_Possible", "QA_Denominator"],
    }

    # --- Resolve columns in daily df ---
    DATE_COL   = pick_col(df, CANDIDATE["date"], required=False)
    NAME_COL   = pick_col(df, CANDIDATE["name"], label_hint="Name")
    ROLE_COL   = pick_col(df, CANDIDATE["role"], label_hint="Role", required=False)
    MGR_COL    = pick_col(df, CANDIDATE["mgr_direct"], label_hint="Manager_Direct")
    WINS_COL   = pick_col(df, CANDIDATE["wins"], label_hint="Wins")
    CALLS_COL  = pick_col(df, CANDIDATE["calls"], label_hint="Calls")
    ATTACH_COL = pick_col(df, CANDIDATE["attaches"], required=False)
    LT_COL     = pick_col(df, CANDIDATE["lt_hits"], required=False)
    LEAF_COL   = pick_col(df, CANDIDATE["leaf"], required=False)
    MOSQ_COL   = pick_col(df, CANDIDATE["mosq"], required=False)
    FLOWER_COL = pick_col(df, CANDIDATE["flower"], required=False)
    BUSH_COL   = pick_col(df, CANDIDATE["bush"], required=False)
    QA_N_COL   = pick_col(df, CANDIDATE["qa_num"], required=False)
    QA_D_COL   = pick_col(df, CANDIDATE["qa_den"], required=False)

    # --- Resolve columns in history df ---
    HIST_DATE  = pick_col(history_df, CANDIDATE["date"], required=False)
    H_NAME_COL = pick_col(history_df, CANDIDATE["name"], label_hint="Name (history)")
    H_ROLE_COL = pick_col(history_df, CANDIDATE["role"], required=False)
    H_MGR_COL  = pick_col(history_df, CANDIDATE["mgr_direct"], label_hint="Manager_Direct (history)")
    WINS_H     = pick_col(history_df, CANDIDATE["wins"], label_hint="Wins (history)")
    CALLS_H    = pick_col(history_df, CANDIDATE["calls"], label_hint="Calls (history)")
    ATT_H      = pick_col(history_df, CANDIDATE["attaches"], required=False)
    LT_H       = pick_col(history_df, CANDIDATE["lt_hits"], required=False)
    LEAF_H     = pick_col(history_df, CANDIDATE["leaf"], required=False)
    MOSQ_H     = pick_col(history_df, CANDIDATE["mosq"], required=False)
    FLOWER_H   = pick_col(history_df, CANDIDATE["flower"], required=False)
    BUSH_H     = pick_col(history_df, CANDIDATE["bush"], required=False)

    # --- Dates & normalize names ---
    df = coerce_date(df, DATE_COL)
    history_df = coerce_date(history_df, HIST_DATE)

    if NAME_COL in df.columns: df[NAME_COL] = df[NAME_COL].map(normalize_name)
    if MGR_COL in df.columns:  df[MGR_COL]  = df[MGR_COL].map(normalize_name)
    if H_NAME_COL in history_df.columns: history_df[H_NAME_COL] = history_df[H_NAME_COL].map(normalize_name)
    if H_MGR_COL in history_df.columns:  history_df[H_MGR_COL]  = history_df[H_MGR_COL].map(normalize_name)

    # 🔹 Explicit TL → SM mapping you provided
    TL_TO_SM = {
        "Amber Knoten": "Heather Painter",
        "Mona Lapuz": "Heather Painter",
        "Christian Umali": "Bernadeth Cotino",
        "Kyla Sangalang": "Bernadeth Cotino",
        "Faye Andrade": "Bernadeth Cotino",
        "Marc Solis": "Bernadeth Cotino",
        "Christine Ebriega": "Jason Roberts",
        "Katherine Militar": "Jason Roberts",
    }

    def map_sm_for_row(row):
        nm  = str(row.get(NAME_COL, "")).strip()
        rl  = str(row.get(ROLE_COL, "")).lower() if ROLE_COL else ""
        mgr = str(row.get(MGR_COL,  "")).strip()
        # If row is a Senior Manager, SM is themself
        if "senior manager" in rl:
            return nm or "Unassigned"
        # If row is a Team Lead, use explicit mapping on *their name*
        if nm in TL_TO_SM:
            return TL_TO_SM[nm]
        # Else try mapping via their manager (TL)
        if mgr in TL_TO_SM:
            return TL_TO_SM[mgr]
        return "Unassigned"

    if NAME_COL:
        df["SM_ROLLUP"] = df.apply(map_sm_for_row, axis=1)
    if H_NAME_COL:
        history_df["SM_ROLLUP"] = history_df.apply(map_sm_for_row, axis=1)

    # ---------- Build today view ----------
    if DATE_COL and pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        today = datetime.now(eastern).date()
        today_df = df[df[DATE_COL].dt.date == today].copy()
        if today_df.empty:
            today_df = df.copy()
    else:
        today_df = df.copy()

    # Derive attaches if no single attaches column
    if ATTACH_COL:
        today_df["__attaches"] = safe_to_num(today_df[ATTACH_COL])
    else:
        parts = []
        for c in [LT_COL, LEAF_COL, MOSQ_COL, FLOWER_COL, BUSH_COL]:
            if c and c in today_df.columns:
                parts.append(safe_to_num(today_df[c]))
        today_df["__attaches"] = sum(parts) if parts else 0.0

    today_df["__wins"]   = safe_to_num(today_df[WINS_COL])  if WINS_COL  else 0.0
    today_df["__calls"]  = safe_to_num(today_df[CALLS_COL]) if CALLS_COL else 0.0
    today_df["__lt_hits"]= safe_to_num(today_df[LT_COL])    if LT_COL    else 0.0
    today_df["__qa_num"] = safe_to_num(today_df.get(QA_N_COL, 0))
    today_df["__qa_den"] = safe_to_num(today_df.get(QA_D_COL, 0))

    sm_daily = compute_rollup(
        today_df.assign(
            wins=today_df["__wins"], calls=today_df["__calls"],
            attaches=today_df["__attaches"], lt_hits=today_df["__lt_hits"],
            qa_num=today_df["__qa_num"], qa_den=today_df["__qa_den"],
        ),
        group_cols=["SM_ROLLUP"],
        cols={"wins":"wins","calls":"calls","attaches":"attaches","lt_hits":"lt_hits","qa_num":"qa_num","qa_den":"qa_den"},
    )
    sm_daily = add_rates(sm_daily)

    # ---------- UI: header & controls ----------
    st.subheader("🏆 Senior Manager View")
    metric_choice = st.radio("Rank by:", ["Conversion %", "All-In Attach %"], horizontal=True)

    # Filter out Unassigned for ranking
    rank_df = sm_daily[sm_daily["SM_ROLLUP"] != "Unassigned"].copy()
    if not rank_df.empty:
        sort_col = metric_choice if metric_choice in rank_df.columns else "Conversion %"
        rank_df = rank_df.sort_values(by=sort_col, ascending=False, ignore_index=True)

    # ---------- Top SM card & “Beat the leader” ----------
    if not rank_df.empty:
        top_sm = rank_df.iloc[0]
        st.markdown(
            f"""
            <div style="text-align:center; padding:10px; border-radius:16px; box-shadow:0 2px 10px rgba(0,0,0,0.08);">
              <div style="font-size:22px;">🏆 <b>Top Senior Manager Today:</b> {top_sm.get("SM_ROLLUP","—")}</div>
              <div style="font-size:16px; margin-top:6px;">
                Conversion: <b>{format_pct(top_sm.get("Conversion %",0))}</b> •
                Attach: <b>{format_pct(top_sm.get("All-In Attach %",0))}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🧮 How to Beat the Current Leader")
        sm_to_chase = st.selectbox("Select Senior Manager to analyze:", rank_df["SM_ROLLUP"].tolist())
        leader_val  = float(top_sm.get(metric_choice, 0.0))
        target_val  = leader_val + 1e-6  # tiny epsilon to take the lead

        row = rank_df[rank_df["SM_ROLLUP"] == sm_to_chase].iloc[0]
        cur_wins  = float(row.get("wins", 0))
        cur_calls = float(row.get("calls", 0))
        cur_att   = float(row.get("attaches", 0))

        if metric_choice == "Conversion %":
            if cur_calls > 0:
                needed = max(0, ceil(target_val * cur_calls - cur_wins))
                st.write(f"- Current: {format_pct(row.get('Conversion %',0))}  •  Wins/Calls: {int(cur_wins)}/{int(cur_calls)}")
                st.success(f"- Add **{needed}** more wins (at current call volume) to take the lead.")
            else:
                st.warning("No calls yet today for this SM.")
        else:
            if cur_wins > 0:
                needed = max(0, ceil(target_val * cur_wins - cur_att))
                st.write(f"- Current: {format_pct(row.get('All-In Attach %',0))}  •  Attaches/Wins: {int(cur_att)}/{int(cur_wins)}")
                st.success(f"- Add **{needed}** more attaches (at current wins) to take the lead.")
            else:
                st.warning("No wins yet today for this SM.")

        # Optional totals table
        show_tbl = st.checkbox("Show Senior Manager totals table", value=False)
        if show_tbl:
            cols = ["SM_ROLLUP","wins","calls","attaches","Conversion %","All-In Attach %","LT %","QA %"]
            tmp = rank_df[cols].copy()
            for c in ("Conversion %","All-In Attach %","LT %","QA %"):
                if c in tmp.columns:
                    tmp[c] = tmp[c].apply(format_pct)
            st.dataframe(tmp.rename(columns={"SM_ROLLUP": "Senior Manager"}), use_container_width=True)
    else:
        st.info("Not enough assigned data yet to display the Senior Manager leaderboard.")

    st.markdown("---")
    st.subheader("📈 Week-over-Week Team Lead Performance")

    # ---------- Weekly TL rollup from history_df ----------
    history_df = add_week_start(history_df, HIST_DATE)

    # Use Manager_Direct as TL, normalized
    TL_H_COL = "TL_ROLLUP"
    if H_MGR_COL:
        history_df[TL_H_COL] = history_df[H_MGR_COL].map(normalize_name)
    else:
        history_df[TL_H_COL] = "Unassigned"

    # Derive attaches in history if needed
    if ATT_H:
        history_df["__attaches_h"] = safe_to_num(history_df[ATT_H])
    else:
        parts_h = []
        for c in [LT_H, LEAF_H, MOSQ_H, FLOWER_H, BUSH_H]:
            if c and c in history_df.columns:
                parts_h.append(safe_to_num(history_df[c]))
        history_df["__attaches_h"] = sum(parts_h) if parts_h else 0.0

    history_df["__wins_h"]  = safe_to_num(history_df[WINS_H])   if WINS_H   else 0.0
    history_df["__calls_h"] = safe_to_num(history_df[CALLS_H])  if CALLS_H  else 0.0

    today = datetime.now(eastern).date()
    history_df = add_week_start(history_df, HIST_DATE)

    # Derive TL field (already present earlier, just ensure it's there)
    TL_H_COL = "TL_ROLLUP" if "TL_ROLLUP" in history_df.columns else "TL_ROLLUP"
    if H_MGR_COL and "TL_ROLLUP" not in history_df.columns:
        history_df[TL_H_COL] = history_df[H_MGR_COL].map(normalize_name)
    elif "TL_ROLLUP" not in history_df.columns:
        history_df[TL_H_COL] = "Unassigned"

    # Pick the most recent two weeks that actually exist in history
    week_vals = pd.Series(history_df.get("Week Start", pd.Series(dtype="object"))).dropna().sort_values().unique()
    if len(week_vals) == 0:
        st.info("Not enough week-over-week data yet for TLs.")
    else:
        current_week_start = week_vals[-1]
        prior_week_start   = week_vals[-2] if len(week_vals) >= 2 else None

    # Derive attaches in history if needed
    if ATT_H:
        history_df["__attaches_h"] = safe_to_num(history_df[ATT_H])
    else:
        parts_h = []
        for c in [LT_H, LEAF_H, MOSQ_H, FLOWER_H, BUSH_H]:
            if c and c in history_df.columns:
                parts_h.append(safe_to_num(history_df[c]))
        history_df["__attaches_h"] = sum(parts_h) if parts_h else 0.0

    history_df["__wins_h"]  = safe_to_num(history_df[WINS_H])  if WINS_H  else 0.0
    history_df["__calls_h"] = safe_to_num(history_df[CALLS_H]) if CALLS_H else 0.0

    if all(x is not None for x in [TL_H_COL, WINS_H, CALLS_H]) and prior_week_start is not None:
        tl_week = compute_rollup(
            history_df.assign(
                wins     = history_df["__wins_h"],
                calls    = history_df["__calls_h"],
                attaches = history_df["__attaches_h"],
                lt_hits  = safe_to_num(history_df.get(LT_H, 0)),
            ),
            group_cols=[TL_H_COL, "Week Start"],
            cols={"wins":"wins","calls":"calls","attaches":"attaches","lt_hits":"lt_hits"}
        )
        tl_week = add_rates(tl_week)

        cur = tl_week[tl_week["Week Start"] == current_week_start]
        prv = tl_week[tl_week["Week Start"] == prior_week_start]
        wow = pd.merge(prv, cur, on=TL_H_COL, how="outer", suffixes=("_prv", "_cur")).fillna(0)

        def delta_str(curv, prvv):
            d = float(curv) - float(prvv)
            arrow = "🔺" if d > 0 else ("🔻" if d < 0 else "⏸️")
            return f"{format_pct(curv)} {arrow}{abs(d)*100:.1f}"

        if not wow.empty:
            improved_cnt = (wow["Conversion %_cur"] > wow["Conversion %_prv"]).sum()
            total_tls = len(wow.index)
            st.markdown(f"**Summary:** {improved_cnt}/{total_tls} TLs improved in Conversion this week.")

            view = pd.DataFrame({
                "Team Lead": wow[TL_H_COL],
                "Conv% (Δ)": [delta_str(r["Conversion %_cur"], r["Conversion %_prv"]) for _, r in wow.iterrows()],
                "Attach% (Δ)": [delta_str(r["All-In Attach %_cur"], r["All-In Attach %_prv"]) for _, r in wow.iterrows()],
                "LT% (Δ)": [delta_str(r.get("LT %_cur", 0), r.get("LT %_prv", 0)) for _, r in wow.iterrows()],
            })
            st.dataframe(view, use_container_width=True)
        else:
            st.info("Not enough week-over-week data yet for TLs.")
    else:
        st.info("Need at least two distinct weeks in history to show week-over-week.")

    st.markdown("---")
    st.subheader("🧭 Leadership Accountability Roll-Up (WoW Improvement %)")

    if all(x is not None for x in [H_MGR_COL, WINS_H, CALLS_H]):
        # Build weekly TL rollups with SM_ROLLUP to evaluate % of TLs improving under each SM
        hist_w = add_week_start(history_df.copy(), HIST_DATE)

        tl_week_all = compute_rollup(
            hist_w.assign(
                wins     = safe_to_num(hist_w[WINS_H])  if WINS_H  else 0.0,
                calls    = safe_to_num(hist_w[CALLS_H]) if CALLS_H else 0.0,
                attaches = (safe_to_num(hist_w[ATT_H]) if ATT_H else
                            (safe_to_num(hist_w.get(LT_H, 0)) +
                             safe_to_num(hist_w.get(LEAF_H, 0)) +
                             safe_to_num(hist_w.get(MOSQ_H, 0)) +
                             safe_to_num(hist_w.get(FLOWER_H, 0)) +
                             safe_to_num(hist_w.get(BUSH_H, 0)))),
                lt_hits  = safe_to_num(hist_w.get(LT_H, 0)),
            ),
            group_cols=["SM_ROLLUP", TL_H_COL, "Week Start"],
            cols={"wins":"wins","calls":"calls","attaches":"attaches","lt_hits":"lt_hits"}
        )
        tl_week_all = add_rates(tl_week_all)

        cur_all = tl_week_all[tl_week_all["Week Start"] == current_week_start]
        prv_all = tl_week_all[tl_week_all["Week Start"] == prior_week_start]
        wow_all = pd.merge(
            prv_all, cur_all,
            on=["SM_ROLLUP", TL_H_COL],
            how="outer",
            suffixes=("_prv", "_cur")
        ).fillna(0)

        def pct_improving(df, metric_cur, metric_prv):
            if df.empty: return 0.0
            total = len(df)
            if total == 0: return 0.0
            improved = (df[metric_cur] > df[metric_prv]).sum()
            return improved / total

        if not wow_all.empty:
            accountability = (
                wow_all.groupby("SM_ROLLUP", dropna=False)
                .apply(lambda g: pd.Series({
                    "% TLs Improving (Conv)": pct_improving(g, "Conversion %_cur", "Conversion %_prv"),
                    "% TLs Improving (Attach)": pct_improving(g, "All-In Attach %_cur", "All-In Attach %_prv"),
                }))
                .reset_index()
            )
            accountability = accountability[accountability["SM_ROLLUP"] != "Unassigned"]
            if not accountability.empty:
                show_acc = accountability.copy()
                show_acc["% TLs Improving (Conv)"] = show_acc["% TLs Improving (Conv)"].apply(format_pct)
                show_acc["% TLs Improving (Attach)"] = show_acc["% TLs Improving (Attach)"].apply(format_pct)
                st.dataframe(show_acc.rename(columns={"SM_ROLLUP": "Senior Manager"}), use_container_width=True)
            else:
                st.info("No accountable TL data yet for SM roll-up.")
        else:
            st.info("Not enough data yet to compute accountability roll-up.")
    else:
        st.info("Missing columns in history to compute accountability.")


# --------------------------------------------
# 🎮 TAB 7: Game Hub – Training Arcade (context + branching discovery)
# --------------------------------------------
with tab7:
    import streamlit as st
    import random
    import pandas as pd

    st.header("🎮 Game Hub — Sales Training Arcade")
    st.caption("Practice with real-world context. Get faster, dig deeper, close smarter.")

    # -----------------------
    # Player identity (safe)
    # -----------------------
    player = st.session_state.get("selected_rep")
    if not player:
        player = st.text_input("Who’s playing? (type your name)", key="gamehub_player").strip()
        if player:
            st.session_state["selected_rep"] = player
    if not player:
        st.info("Enter your name to start playing.")
        st.stop()

    # -----------------------
    # Shared session state
    # -----------------------
    st.session_state.setdefault("arcade_scores", {})           # per-game cumulative score {game_name: int}
    st.session_state.setdefault("weekly_scores", {})           # overall per-rep {rep: int}
    st.session_state.setdefault("ob_qid", 0)                   # Objection Battles round id (unique radio key)
    st.session_state.setdefault("ob_question", None)           # current objection dict
    st.session_state.setdefault("ob_locked", False)            # lock after answer
    st.session_state.setdefault("last_mode", None)             # remember last game mode
    st.session_state.setdefault("mystery_state", None)         # branching state for Mystery Customer
    st.session_state.setdefault("mow_qid", 0)                  # MowRadar round id
    st.session_state.setdefault("mow_case", None)              # current MowRadar case

    def add_points(game: str, pts: int):
        st.session_state["arcade_scores"][game] = st.session_state["arcade_scores"].get(game, 0) + pts
        st.session_state["weekly_scores"][player] = st.session_state["weekly_scores"].get(player, 0) + pts

    # =================================
    # CONTENT (enriched with context)
    # =================================
    # (1) Objection Battles — with mini transcript context
    OBJECTION_BANK = [
        {
            "objection": "I just want one cut, not 3.",
            "context": [
                "Customer: I just need one cut to try you out, not 3.",
            ],
            "options": [
                # A (BEST)
                "I completely understand wanting to test us out first. While we do maintain a 3-cut minimum, think of it this way — the first cut is only $19 with our promo, so you're getting a great trial rate. The 3-cut minimum lets the lawn adjust to our cutting pattern and height, which actually improves how your grass grows. Plus it's our service guarantee — if you're not happy with cut #1, we have cuts 2 and 3 to make it right. After those 3 cuts, you can cancel anytime with just 48 hours notice — no contracts, no hassle.",
                # B
                "I hear you on wanting to try us first — that makes total sense. Unfortunately, our system requires the 3-cut minimum for scheduling, but here's what I can do: I'll note on your account that you might only want one service, and after we complete that first cut, you can call our support team to discuss options. They might be able to work something out for you. The good news is that first cut is only $19, so you're not risking much, and most people end up loving the service anyway.",
                # C
                "That's a fair request, and I appreciate you being upfront about it. We actually can accommodate a one-time cut, but because it requires special routing and doesn't allow us to maintain your lawn properly over time, we have to charge the full rate without the promo — so it would be 76 instead of the $19 first-cut special. Most customers find it makes more sense to do the 3 cuts at the promo rate since you're getting three services for basically the price of one regular cut. What do you think?"
            ],
            "answer": 0,
            "why": "Educates on benefits of the 3-cut minimum, maintains value proposition, and removes risk perception."
        },
        {
            "objection": "The last company had a 3-cut minimum too but wouldn't cut if grass was wet and canceled on me 3 times!",
            "context": [
                "Customer: The last company kept canceling when it was damp — super frustrating."
            ],
            "options": [
                # A (BEST)
                "That sounds incredibly frustrating — I can understand why you'd be hesitant. We work through light rain and damp grass — we only avoid heavy downpours or soaking wet conditions that would damage your lawn or leave ruts. Our crews are local and know Florida weather, so they're used to working around it. If we ever do need to reschedule, you'll get a text immediately with the new time, not be left wondering. Plus, with our 3-day quality check, if you're not happy with how we handle that first service, you won't be charged.",
                # B
                "Oh wow, that's terrible service! We're definitely not like those other companies — we're much more professional and reliable. We understand Florida weather is unpredictable, but we make it work. Our crews are trained to handle various conditions and we pride ourselves on not canceling unless absolutely necessary. We've been in business for years and have thousands of happy customers who switched from other services just like you're doing. I promise you'll see the difference from that very first cut.",
                # C
                "I'm really sorry you had that experience — weather can definitely be challenging in this business. What we do differently is give you a two-day service window instead of a specific day, which gives us flexibility to work around weather. So if it's raining Wednesday, we can still service you Thursday without it being considered a cancellation. We also have multiple crews, so if one can't make it, we can often send another. The 3-cut minimum helps us learn your property's specific needs so we can serve you better even in tricky weather."
            ],
            "answer": 0,
            "why": "Directly addresses the exact pain (rain cancellations) with specific policy + quality guarantee."
        },
        {
            "objection": "My grass is really high — can someone come today?",
            "context": [
                "Customer: It's overgrown — feels urgent!"
            ],
            "options": [
                # A
                "I understand it feels urgent when the grass gets that high. Unfortunately, our crews' routes are set in advance and we can't accommodate same-day service. Our scheduling system works on a two-day window basis, and the absolute earliest we can get someone out is Wednesday or Thursday. However, if you sign up now, you'll be first on the route for that window, and the crew will definitely take care of that overgrowth. With grass that high, they'll likely need extra time anyway, so advance scheduling ensures they allocate enough time for your property.",
                # B
                "I know that feeling when the grass gets away from you! Same-day service isn't something we offer because our crews need time to plan their routes and load the right equipment. However, since you mentioned it's really high, that actually works in your favor — our crews love these kinds of transformations and will make sure your lawn looks amazing. Wednesday is only two days away, and trust me, after seeing hundreds of overgrown lawns, a couple more days won't make a difference in the difficulty or price.",
                # C (BEST)
                "I completely understand the urgency — nobody wants their lawn looking jungle-like! While we can't guarantee same-day service, here's what I can do: I'm putting you on our hot list, which gives you about a 70% chance of getting serviced tomorrow. Our crews check this list when they finish routes early. You're also locked in for Wednesday/Thursday as your guaranteed window. The crew will text you as soon as they can squeeze you in. If the grass is over 9 inches, there might be an additional fee, but they'll let you know before starting."
            ],
            "answer": 2,
            "why": "Shows maximum effort, manages expectations, provides both hope (hot list) and certainty (48-hour window)."
        },
        {
            "objection": "I'm old and don't want to deal with apps — too many passwords!",
            "context": [
                "Customer: I'd rather not download or manage an app."
            ],
            "options": [
                # A
                "I completely understand — technology can be overwhelming with all these apps and passwords these days. The good news is our app is really user-friendly and designed for all ages. We'll send you a simple link, you create one password, and that's it. You can save the password in your phone so you never have to remember it. Many of our senior customers actually end up loving it because they can see exactly when we're coming and can message the crew directly if needed. But if you really don't want to use it, you can always call our support line.",
                # B (BEST)
                "No problem at all! You don't need to use the app if you don't want to. We'll handle everything through text messages, which you mentioned you're comfortable with. The crew will text you when they're on the way, and you can text back if you need anything. The app is just an extra option that's there if you ever want to try it, but plenty of our customers never use it and do everything by phone and text. We'll make sure you're taken care of either way.",
                # C
                "That's perfectly understandable, and you're not alone in feeling that way! The app is optional — think of it as a convenience for those who want it. What most customers in your situation do is have a family member or neighbor help them set it up initially, and then they rarely need to log in again. We can handle most things over the phone, though some features like instantly skipping a service or changing your schedule are much faster through the app. But don't worry, our customer service team is always available by phone during business hours to help with anything you need."
            ],
            "answer": 1,
            "why": "Completely removes the friction by meeting the customer where they are (SMS/phone), no pushiness."
        },
    ]

    # (2) Discovery Flows — UPDATED to Basic Service + add-ons focus
    DISCOVERY_FLOWS = [
        {
            "name": "The Overwhelmed Homeowner",
            "customer_opening": "I just need someone to cut my grass - I'm too busy.",
            "hidden_need": "Multiple yard tasks piling up, wants comprehensive solution",
            "questions": [
                {
                    "text": "How often are you currently cutting it yourself?",
                    "strength": "weak",
                    "customer_response": "Maybe once a month when I remember.",
                    "reveals": ["infrequent_maintenance"]
                },
                {
                    "text": "What's taking up most of your time these days?",
                    "strength": "strong",
                    "customer_response": "Work is crazy and I travel constantly.",
                    "reveals": ["time_constrained", "travel_schedule"]
                },
                {
                    "text": "Is it just the grass or are other yard tasks piling up too?",
                    "strength": "strong",
                    "customer_response": "Honestly, everything's getting away from me. The bushes by my front door are blocking the mailbox now.",
                    "reveals": ["bush_trimming_need", "multiple_issues", "curb_appeal_concern"]
                },
                {
                    "text": "Any flower beds that are getting overrun with weeds?",
                    "strength": "strong",
                    "customer_response": "Oh god, the beds are a disaster. I haven't touched them all summer.",
                    "reveals": ["bed_weeding_need", "seasonal_neglect"]
                },
                {
                    "text": "What's your budget range?",
                    "strength": "weak",
                    "customer_response": "I don't know, just reasonable I guess.",
                    "reveals": []
                },
                {
                    "text": "Do you have any pets?",
                    "strength": "weak",
                    "customer_response": "No, why does that matter?",
                    "reveals": []
                }
            ],
            "ideal_bundle": ["Bush Trimming", "Flower Bed Weeding"]  # Basic service is always included
        },
        {
            "name": "The Lawn Perfectionist",
            "customer_opening": "My neighbor's lawn looks amazing and mine looks terrible.",
            "hidden_need": "Wants thick, green lawn like neighbor's but has spreading problems",
            "questions": [
                {
                    "text": "When did you first notice yours wasn't looking good?",
                    "strength": "weak",
                    "customer_response": "A few months ago I guess.",
                    "reveals": ["timeline_vague"]
                },
                {
                    "text": "What specifically do you notice about their lawn?",
                    "strength": "strong",
                    "customer_response": "Theirs is so thick and green. Mine has thin spots and some yellowing patches.",
                    "reveals": ["thin_lawn", "color_issues", "comparison_motivation"]
                },
                {
                    "text": "Are those thin spots spreading or staying the same size?",
                    "strength": "strong",
                    "customer_response": "They're definitely getting bigger. Started with just one spot, now there's like five.",
                    "reveals": ["spreading_problem", "treatment_needed"]
                },
                {
                    "text": "Have you tried any store-bought treatments yourself?",
                    "strength": "strong",
                    "customer_response": "I bought some stuff but I don't know when or how to apply it. It's confusing.",
                    "reveals": ["diy_confusion", "needs_professional"]
                },
                {
                    "text": "What type of grass do you have?",
                    "strength": "weak",
                    "customer_response": "I have no idea. Grass is grass, right?",
                    "reveals": []
                },
                {
                    "text": "How often do you water?",
                    "strength": "weak",
                    "customer_response": "Whenever I remember.",
                    "reveals": []
                }
            ],
            "ideal_bundle": ["Lawn Treatment Program"]  # Only lawn treatment needed - no bush/bed issues mentioned
        },
        {
            "name": "The New Homeowner Discovery",
            "customer_opening": "We just bought this house and the yard is completely overgrown.",
            "hidden_need": "Needs initial reset plus ongoing maintenance, has deadline pressure",
            "questions": [
                {
                    "text": "Congratulations! How long has the property been vacant?",
                    "strength": "strong",
                    "customer_response": "Two months. Everything's completely overgrown - you can't even see what's supposed to be there.",
                    "reveals": ["major_overgrowth", "property_discovery_needed"]
                },
                {
                    "text": "What's your biggest priority to tackle first?",
                    "strength": "strong",
                    "customer_response": "Getting it presentable for the neighbors! Right now you can't even see the flower beds - they're covered in weeds.",
                    "reveals": ["curb_appeal_urgent", "bed_weeding_need", "neighbor_pressure"]
                },
                {
                    "text": "Are the bushes overgrown too?",
                    "strength": "strong",
                    "customer_response": "The ones by the front windows are way too tall - you can barely see out of the house.",
                    "reveals": ["bush_trimming_need", "visibility_blocked"]
                },
                {
                    "text": "How quickly do you need this done?",
                    "strength": "strong",
                    "customer_response": "Pretty soon - my in-laws are visiting next month and I'm embarrassed by how it looks.",
                    "reveals": ["deadline_pressure", "family_visit"]
                },
                {
                    "text": "What's your budget?",
                    "strength": "weak",
                    "customer_response": "We just bought a house, so we're watching spending.",
                    "reveals": ["budget_conscious"]
                },
                {
                    "text": "Do you plan to do any landscaping yourself?",
                    "strength": "weak",
                    "customer_response": "Maybe eventually, but not right now.",
                    "reveals": []
                }
            ],
            "ideal_bundle": ["Flower Bed Weeding", "Bush Trimming"]  # Addressing the specific visibility/curb appeal issues
        },
        {
            "name": "The Rental Property Manager",
            "customer_opening": "I need lawn service for a rental property - just basic maintenance.",
            "hidden_need": "Tenant satisfaction issues that could affect renewals",
            "questions": [
                {
                    "text": "Is this for one property or multiple?",
                    "strength": "weak",
                    "customer_response": "Just this one for now.",
                    "reveals": ["single_property"]
                },
                {
                    "text": "What does your lease require tenants to maintain?",
                    "strength": "strong",
                    "customer_response": "They're supposed to handle basic upkeep, but they're not doing a good job.",
                    "reveals": ["tenant_failure", "lease_requirements"]
                },
                {
                    "text": "Any specific issues the tenants have mentioned?",
                    "strength": "strong",
                    "customer_response": "Actually yes - they keep complaining about mosquitoes on the back patio. They can't use it in the evenings.",
                    "reveals": ["mosquito_problem", "tenant_complaints", "outdoor_space_unusable"]
                },
                {
                    "text": "How important is tenant satisfaction for renewals?",
                    "strength": "strong",
                    "customer_response": "Very important. Happy tenants renew leases. Unhappy tenants cost me money finding replacements.",
                    "reveals": ["tenant_retention_focus", "cost_avoidance"]
                },
                {
                    "text": "What's the monthly rent?",
                    "strength": "weak",
                    "customer_response": "Why does that matter for lawn service?",
                    "reveals": []
                },
                {
                    "text": "How often do you inspect the property?",
                    "strength": "weak",
                    "customer_response": "Quarterly, usually.",
                    "reveals": ["inspection_schedule"]
                }
            ],
            "ideal_bundle": ["Mosquito Treatment"]  # Solves the specific tenant complaint issue
        },
        {
            "name": "The Time-Crunched Professional",
            "customer_opening": "I travel constantly for work and my yard is embarrassing.",
            "hidden_need": "Wants consistent, comprehensive maintenance during frequent absences",
            "questions": [
                {
                    "text": "How often are you home to maintain it yourself?",
                    "strength": "strong",
                    "customer_response": "Maybe one weekend a month. I'm on the road constantly.",
                    "reveals": ["frequent_travel", "minimal_availability"]
                },
                {
                    "text": "What specific issues are you noticing when you get back?",
                    "strength": "strong",
                    "customer_response": "Every time I come home, there are more weeds in the flower beds and the bushes look more overgrown.",
                    "reveals": ["bed_weeding_need", "bush_trimming_need", "progressive_deterioration"]
                },
                {
                    "text": "How do you think the neighbors feel about it?",
                    "strength": "strong",
                    "customer_response": "I'm sure they're annoyed. One neighbor even mentioned the bushes are hanging over the sidewalk.",
                    "reveals": ["neighbor_complaints", "property_boundaries", "community_standards"]
                },
                {
                    "text": "Would you prefer we handle everything so it stays neat while you're gone?",
                    "strength": "strong",
                    "customer_response": "Yes! I need everything maintained consistently. I can't deal with it being a mess every time I return.",
                    "reveals": ["comprehensive_solution_wanted", "consistency_priority"]
                },
                {
                    "text": "What's your monthly travel schedule like?",
                    "strength": "weak",
                    "customer_response": "It varies, but I'm gone a lot.",
                    "reveals": []
                },
                {
                    "text": "Do you have an HOA?",
                    "strength": "weak",
                    "customer_response": "No, just trying to be a good neighbor.",
                    "reveals": []
                }
            ],
            "ideal_bundle": ["Flower Bed Weeding", "Bush Trimming"]  # Addressing the specific maintenance issues during travel
        }
    ]

    # (3) MowRadar Scenarios — recommend ideal bundle from cues
    MOWRADAR_CASES = [
        {
            "cues": ["High humidity", "Near standing water", "Neighbors had mosquito complaints"],
            "services": ["Mowing", "Lawn Treatment", "Mosquito Treatment", "Bush Trimming", "Flower Bed Weeding", "Leaf Removal"],
            "best": {"Mosquito Treatment", "Mowing", "Flower Bed Weeding"}
        },
        {
            "cues": ["Heavy shade", "Fallen sticks on lawn", "Thick plant beds"],
            "services": ["Mowing", "Lawn Treatment", "Mosquito Treatment", "Bush Trimming", "Flower Bed Weeding", "Mulch Refresh"],
            "best": {"Bush Trimming", "Flower Bed Weeding", "Lawn Treatment"}
        },
    ]

    # =========================
    # Mode selector (3 games)
    # =========================
    mode = st.radio(
        "Choose a game:",
        ["Objection Battles", "Mystery Customer", "MowRadar Scenarios"],
        horizontal=True,
        key="gamehub_mode",
    )

    if st.session_state["last_mode"] != mode:
        st.session_state["ob_locked"] = False
        st.session_state["last_mode"] = mode

    # ==========================================
    # GAME 1 — Objection Battles (no-repeat)
    # ==========================================
    if mode == "Objection Battles":
        st.subheader("🧠 Pick the best rebuttal")
        GAME = "Objection Battles"

        # hold a question; ensure no immediate repeat
        if not st.session_state.get("ob_question"):
            st.session_state["ob_question"] = random.choice(OBJECTION_BANK)

        q = st.session_state["ob_question"]
        qid = st.session_state["ob_qid"]
        radio_key = f"ob_pick_{qid}"

        # context block
        with st.container():
            st.markdown(
                "<div style='background:#f6f6f8;border-radius:12px;padding:10px;margin-bottom:8px;'>"
                + "<br/>".join([f"<span style='color:#666;'>{line}</span>" for line in q["context"]])
                + "</div>",
                unsafe_allow_html=True
            )

        pick = st.radio(
            f"Objection: *{q['objection']}*",
            q["options"],
            index=None,
            key=radio_key,
            disabled=st.session_state["ob_locked"]
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Lock In", disabled=st.session_state["ob_locked"]):
                if pick is None:
                    st.warning("Pick an option first.")
                else:
                    idx = q["options"].index(pick)
                    if idx == q["answer"]:
                        st.success("✅ Correct! +10 pts")
                        add_points(GAME, 10)
                    else:
                        st.error("❌ Not quite. +2 pts for trying!")
                        add_points(GAME, 2)
                    with st.expander("Why this works"):
                        st.write(q["why"])
                    st.session_state["ob_locked"] = True

        with c2:
            if st.button("Next Round"):
                new_q = random.choice(OBJECTION_BANK)
                while new_q == st.session_state["ob_question"] and len(OBJECTION_BANK) > 1:
                    new_q = random.choice(OBJECTION_BANK)
                st.session_state["ob_question"] = new_q
                st.session_state["ob_qid"] += 1
                st.session_state["ob_locked"] = False
                st.rerun()

        with c3:
            if st.button("Reset Game Score"):
                st.session_state["arcade_scores"][GAME] = 0
                st.info("Objection Battles score reset.")

        st.metric("🎯 Your Objection Battles Score", st.session_state["arcade_scores"].get(GAME, 0))

    # ===================================================
    # GAME 2 — Mystery Customer (branching discovery)
    # ===================================================
    elif mode == "Mystery Customer":
        st.subheader("🕵️ Ask up to 3 discovery questions — each reveals a customer reply — then recommend a bundle")
        GAME = "Mystery Customer"

        # --- UPDATED helpers (add-on focus) ---
        SERVICE_EXPLAINS = {
            "Bush Trimming": "Keeps bushes from blocking windows/walkways and maintains clean property lines.",
            "Flower Bed Weeding": "Prevents the 'still looks messy after mowing' problem - huge curb appeal impact.",
            "Leaf Removal": "Prevents matting, dead spots, and saves you weekends raking in fall.",
            "Lawn Treatment Program": "Thickens grass, evens color, and prevents weeds long-term.",
            "Mosquito Treatment": "Makes outdoor spaces usable during peak mosquito season.",
            "Initial Cleanup": "Resets severely overgrown properties so regular maintenance actually looks good.",
        }
        TAG_TO_SERVICE = {
            "bed_weeding_need": "Flower Bed Weeding",
            "bush_trimming_need": "Bush Trimming",
            "leaf_removal_need": "Leaf Removal",
            "treatment_needed": "Lawn Treatment Program",
            "mosquito_problem": "Mosquito Treatment",
            "needs_professional": "Lawn Treatment Program",
            "major_overgrowth": "Initial Cleanup",  # For extreme cases needing reset before regular service
        }

        def coaching_for_misses(target_bundle: set, chosen_set: set) -> list[str]:
            """Explain why missed target services matter."""
            notes = []
            for svc in sorted(target_bundle - chosen_set):
                why = SERVICE_EXPLAINS.get(svc, "")
                notes.append(f"- **{svc}** — {why}")
            return notes

        def suggest_strong_questions(case: dict, asked_idx: list[int]) -> list[tuple[str, list[str]]]:
            """Return up to two strong questions you didn't ask, with the services they'd unlock."""
            suggestions = []
            for i, q in enumerate(case["questions"]):
                if q.get("strength") != "strong" or i in asked_idx:
                    continue
                reveals = q.get("reveals", [])
                tie_ins = sorted({TAG_TO_SERVICE[t] for t in reveals if t in TAG_TO_SERVICE})
                if tie_ins:
                    suggestions.append((q["text"], tie_ins))
            return suggestions[:2]

        # init state
        if not st.session_state["mystery_state"]:
            case = random.choice(DISCOVERY_FLOWS)
            st.session_state["mystery_state"] = {
                "case": case,
                "asked": [],              # question indices asked
                "good_hits": 0,           # count of strong questions
                "revealed": set(),        # tags we've learned
                "replies": [],            # list of (Q, customer_response)
                "locked": False           # after scoring
            }
        # sticky feedback container across reruns
        st.session_state.setdefault("mc_feedback", None)

        ms = st.session_state["mystery_state"]
        case = ms["case"]
        st.write(f"**Customer:** {case['customer_opening']}")

        # Show chat-style thread of what we've asked + replies
        if ms["replies"]:
            st.markdown("<div style='border-left:3px solid #e0e0e0;padding-left:10px;margin:8px 0;'>", unsafe_allow_html=True)
            for qtext, resp in ms["replies"]:
                st.markdown(f"**You:** {qtext}")
                st.markdown(f"<span style='color:#555;'>Customer:</span> {resp}", unsafe_allow_html=True)
                st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

        remaining = [(i, q) for i, q in enumerate(case["questions"]) if i not in ms["asked"]]
        can_ask = len(ms["asked"]) < 3 and not ms["locked"]

        # pick a discovery question to ask next
        if can_ask and remaining:
            label_map = {f"{i}: {q['text']}": i for i, q in remaining}
            choice = st.selectbox(
                "Choose your next discovery question:",
                list(label_map.keys()),
                index=None,
                key=f"mc_qpick_{len(ms['asked'])}"
            )
            if st.button("Ask this question"):
                if choice is None:
                    st.warning("Pick a question first.")
                else:
                    i = label_map[choice]
                    qinfo = case["questions"][i]
                    ms["asked"].append(i)
                    if qinfo["strength"] == "strong":
                        ms["good_hits"] += 1
                    # record reply & reveals
                    ms["replies"].append((qinfo["text"], qinfo["customer_response"]))
                    for tag in qinfo.get("reveals", []):
                        ms["revealed"].add(tag)
                    # clear old feedback when new info arrives
                    st.session_state["mc_feedback"] = None
                    st.rerun()

        asked_count = len(ms["asked"])
        st.caption(f"Questions asked: **{asked_count}/3**  •  Strong hits: **{ms['good_hits']}**")

        # guidance based on reveals (subtle coaching)
        if ms["revealed"]:
            hints = []
            tag_hints = {
                "bush_trimming_need": "Bush trimming is relevant.",
                "bed_weeding_need": "Flower bed weeding likely adds value.",
                "overgrown": "Over 9\" may need extra time — set expectations.",
                "major_overgrowth": "Consider an Initial Cleanup before recurring mowing.",
                "leaf_removal_need": "Leaf removal is a time-saver and improves curb appeal.",
                "treatment_needed": "Lawn treatment can address thin/yellow patches.",
                "needs_professional": "Position our program over DIY confusion.",
                "mosquito_problem": "Mosquito treatment can make the patio usable again.",
            }
            for t in sorted(ms["revealed"]):
                if t in tag_hints:
                    hints.append(tag_hints[t])
            if hints:
                st.info("What you’ve uncovered: " + " • ".join(hints))

        # After at least 1 question, allow recommendation
        if asked_count >= 1:
            all_services = sorted({
                "Initial Cleanup", "Mowing", "Bush Trimming", "Flower Bed Weeding",
                "Leaf Removal", "Lawn Treatment Program", "Weed Control", "Full Service Package", "Mosquito Treatment"
            })
            chosen = st.multiselect(
                "Recommend a bundle (what would you sell?):",
                options=all_services,
                key=f"mc_bundle_{asked_count}",
                disabled=ms["locked"]
            )

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("Submit Recommendation", disabled=ms["locked"]):
                    if not chosen:
                        st.warning("Pick at least one service.")
                    else:
                        target = set(case["ideal_bundle"])
                        chosen_set = set(chosen)
                        correct_hits = len(chosen_set & target)

                        # scoring
                        pts = ms["good_hits"] * 5
                        if correct_hits >= max(1, len(target) - 1):
                            pts += 10
                        if chosen_set == target:
                            pts += 5
                        # alignment bonus via revealed tags
                        aligned = {TAG_TO_SERVICE[t] for t in ms["revealed"] if t in TAG_TO_SERVICE}
                        pts += 3 * len(chosen_set & aligned)

                        add_points(GAME, pts)
                        ms["locked"] = True

                        # Build sticky feedback payload so it persists after rerun
                        feedback = {
                            "score": pts,
                            "target": sorted(target),
                            "chosen": sorted(chosen_set),
                            "aligned": sorted(chosen_set & aligned),
                            "coaching": [],
                            "miss_qs": []
                        }

                        # If pick is weak, add why it's not best + how to open up the sale
                        weak_pick = correct_hits == 0 or correct_hits < min(2, len(target))
                        if weak_pick:
                            feedback["coaching"].append("Why your pick missed:")
                            feedback["coaching"].extend(coaching_for_misses(target, chosen_set))
                            # Suggest strong questions not asked that tie to missed services
                            miss_qs = suggest_strong_questions(case, ms["asked"])
                            if miss_qs:
                                feedback["miss_qs"].append("Try asking one of these next time to unlock the need:")
                                for qtext, ties in miss_qs:
                                    feedback["miss_qs"].append(f"- **{qtext}** → surfaces: {', '.join(ties)}")
                        else:
                            feedback["coaching"].append("Nice! You targeted the core need. See details below.")

                        st.session_state["mc_feedback"] = feedback
                        st.rerun()

            with colB:
                if st.button("Next Case"):
                    new_case = random.choice(DISCOVERY_FLOWS)
                    while new_case is case and len(DISCOVERY_FLOWS) > 1:
                        new_case = random.choice(DISCOVERY_FLOWS)
                    st.session_state["mystery_state"] = {
                        "case": new_case, "asked": [], "good_hits": 0,
                        "revealed": set(), "replies": [], "locked": False
                    }
                    st.session_state["mc_feedback"] = None
                    st.rerun()

        # --- Sticky Results & Coaching panel (persists until Next Case) ---
        if st.session_state["mc_feedback"]:
            fb = st.session_state["mc_feedback"]
            st.markdown("### ✅ Results & Coaching")
            st.success(f"Scored **+{fb['score']}** pts")
            with st.expander("What we looked for (opens by default)", expanded=True):
                st.write(f"- **Hidden need:** {case['hidden_need']}")
                st.write(f"- **Ideal bundle:** {', '.join(fb['target'])}")
                st.write(f"- **Your pick:** {', '.join(fb['chosen']) if fb['chosen'] else '—'}")
                if fb["aligned"]:
                    st.write(f"- **Nice alignment:** {', '.join(fb['aligned'])}")
                if fb["coaching"]:
                    st.markdown("**Coaching:**")
                    for line in fb["coaching"]:
                        st.write(line)
                if fb["miss_qs"]:
                    st.markdown("**Better questions next time:**")
                    for line in fb["miss_qs"]:
                        st.write(line)

        st.metric("🎯 Your Mystery Customer Score", st.session_state["arcade_scores"].get(GAME, 0))

    # ==========================================
    # GAME 3 — MowRadar Scenarios (bundle pick)
    # ==========================================
    elif mode == "MowRadar Scenarios":
        st.subheader("🌱 Read the cues, recommend the right bundle")
        GAME = "MowRadar Scenarios"

        if not st.session_state["mow_case"]:
            st.session_state["mow_case"] = random.choice(MOWRADAR_CASES)
        case = st.session_state["mow_case"]
        qid = st.session_state["mow_qid"]

        st.write("**Cues:** " + " • ".join(case["cues"]))
        pick = st.multiselect("Select the best 2–3 services:", case["services"], key=f"mow_pick_{qid}")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Lock In Bundle"):
                if not pick:
                    st.warning("Pick at least two services.")
                else:
                    chosen = set(pick)
                    best = case["best"]
                    hits = len(chosen & best)
                    pts = hits * 10 - max(0, len(chosen - best)) * 3
                    if chosen == best:
                        pts += 5
                    pts = max(0, pts)
                    add_points(GAME, pts)
                    st.success(f"✅ Scored **+{pts}** pts")
                    with st.expander("Why this works"):
                        st.write(f"- **Ideal bundle:** {', '.join(sorted(best))}\n- **Your pick:** {', '.join(sorted(chosen))}")

        with c2:
            if st.button("Next Scenario"):
                new_case = random.choice(MOWRADAR_CASES)
                while new_case is case and len(MOWRADAR_CASES) > 1:
                    new_case = random.choice(MOWRADAR_CASES)
                st.session_state["mow_case"] = new_case
                st.session_state["mow_qid"] += 1
                st.rerun()

        st.metric("🎯 Your MowRadar Score", st.session_state["arcade_scores"].get(GAME, 0))

    # =========================
    # Weekly Top 3 Leaderboard
    # =========================
    st.markdown("---")
    st.subheader("🏆 Weekly Leaders (Top 3)")
    if st.session_state["weekly_scores"]:
        board = pd.DataFrame(
            sorted(st.session_state["weekly_scores"].items(), key=lambda x: x[1], reverse=True),
            columns=["Rep", "Score"]
        ).head(3)
        st.dataframe(board, use_container_width=True, hide_index=True)
    else:
        st.caption("No scores yet — play a round to appear here!")

    # Optional: quick reset (local/session only)
    with st.expander("Admin / Debug"):
        if st.button("Reset ALL Arcade Scores (this browser)"):
            st.session_state["arcade_scores"] = {}
            st.info("Cleared per-game scores for this session.")
        if st.button("Reset Weekly Leaderboard (this browser)"):
            st.session_state["weekly_scores"] = {}
            st.info("Cleared weekly leaderboard (local session).")
