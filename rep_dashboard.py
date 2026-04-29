import os
import io
import re
import base64
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from datetime import datetime, timedelta

st.set_page_config(page_title="Rep Dashboard", layout="wide")

# -----------------------------------------------------------------------
# 🔐 GOOGLE SSO AUTH
# -----------------------------------------------------------------------
ALLOWED_DOMAIN = "lawnstarter.com"

# Roles that can use "View As" to see other reps
TL_ROLES    = {"Team Lead"}
SM_ROLES    = {"Senior Manager", "Director", "VP"}

# Cache auth in session_state so it survives re-runs and component refreshes
if st.user.is_logged_in:
    st.session_state['_auth_email']      = (st.user.email      or "").strip()
    st.session_state['_auth_given_name'] = (st.user.given_name or "").strip()
    st.session_state['_auth_full_name']  = (st.user.name       or "").strip()

_is_authed = st.session_state.get('_auth_email', '')

if not _is_authed:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 20px;'>
            <div style='font-size:48px;'>🌿</div>
            <h1 style='font-size:36px; font-weight:900;'>Sales Rep Dashboard</h1>
            <p style='font-size:18px; color:#aaa; margin-bottom:32px;'>
                Sign in with your LawnStarter Google account to continue.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.login("google")
    st.stop()

user_email      = st.session_state['_auth_email']
user_given_name = st.session_state['_auth_given_name']
user_full_name  = st.session_state['_auth_full_name']

if not user_email.lower().endswith(f"@{ALLOWED_DOMAIN}"):
    st.error(f"❌ Access restricted to @{ALLOWED_DOMAIN} accounts. You're signed in as **{user_email}**.")
    st.session_state.pop('_auth_email', None)
    st.logout()
    st.stop()

# -----------------------------------------------------------------------
# Detect role from roster (cached so it doesn't re-fetch every render)
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _load_roster_for_auth():
    ROSTER_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=171451260"
    )
    try:
        raw = pd.read_csv(ROSTER_URL, header=None)
        raw.columns = raw.iloc[1]
        raw = raw.iloc[2:].reset_index(drop=True)
        raw.columns = raw.columns.str.strip()
        raw = raw[raw['Rep'].notna() & (raw['Rep'].astype(str).str.strip() != '')].copy()
        raw['rep_key'] = raw['Rep'].str.lower().str.strip()
        return raw[['rep_key', 'First_Name', 'Last_Name', 'Current_Role',
                    'Manager_Direct', 'Team Name']].drop_duplicates('rep_key')
    except Exception:
        return pd.DataFrame()

roster_auth = _load_roster_for_auth()
auth_key    = user_email.lower().strip()
auth_row    = roster_auth[roster_auth['rep_key'] == auth_key]

def _safe(val, fallback=""):
    return str(val).strip() if val is not None and str(val).strip() not in ("", "nan") else fallback

# Use Google's given_name as primary, fall back to roster, then email prefix
auth_first  = user_given_name or (_safe(auth_row['First_Name'].values[0]) if not auth_row.empty else user_email.split('@')[0].split('.')[0].title())
auth_role   = _safe(auth_row['Current_Role'].values[0])   if not auth_row.empty else ""
auth_mgr    = _safe(auth_row['Manager_Direct'].values[0]) if not auth_row.empty else ""
auth_team   = _safe(auth_row['Team Name'].values[0])      if not auth_row.empty else ""

is_tl = any(auth_role.startswith(r) for r in TL_ROLES)
is_sm = any(auth_role.startswith(r) for r in SM_ROLES)

# -----------------------------------------------------------------------
# "View As" — TLs see their team, SMs see everyone, reps see only themselves
# -----------------------------------------------------------------------
if is_sm:
    all_reps = sorted(roster_auth['rep_key'].tolist())
    view_options = ["— View as... —"] + all_reps
    view_label   = "👁 Senior Manager: View as rep"
elif is_tl:
    auth_last = auth_row['Last_Name'].values[0].strip() if not auth_row.empty else ""
    # Manager_Direct is stored as "Last, First" — build all likely formats to match against
    _tl_variants = {
        auth_first.lower(),                                          # "mona"
        f"{auth_first} {auth_last}".lower(),                        # "mona lapuz"
        f"{auth_last}, {auth_first}".lower(),                       # "lapuz, mona"
        f"{auth_last},{auth_first}".lower(),                        # "lapuz,mona"
    }
    team_reps = roster_auth[
        roster_auth['Manager_Direct'].str.strip().str.lower().isin(_tl_variants)
    ]['rep_key'].tolist()
    view_options = [auth_key] + sorted(set(team_reps) - {auth_key})
    view_label   = "👁 Team Lead: View as rep"
else:
    view_options = [auth_key]
    view_label   = None

# Sidebar: logout + view-as
with st.sidebar:
    _pic = getattr(st.user, 'picture', None)
    if _pic:
        st.markdown(f"<img src='{_pic}' width='48' style='border-radius:50%;display:block;margin-bottom:6px;'>", unsafe_allow_html=True)
    st.markdown(f"**{user_full_name or auth_first}**  \n`{user_email}`")
    if auth_team:
        st.caption(f"🏷 {auth_team}")
    st.markdown("---")
    if len(view_options) > 1:
        viewed_email = st.selectbox(view_label, view_options, key="view_as_email")
        if viewed_email == "— View as... —":
            viewed_email = auth_key
    else:
        viewed_email = auth_key
    st.markdown("---")
    if st.button("🚪 Sign out", use_container_width=True):
        st.session_state.clear()
        st.logout()

# Store viewed rep in session state so tabs can read it
st.session_state["selected_rep"] = viewed_email

# Viewed rep's display name
viewed_row   = roster_auth[roster_auth['rep_key'] == viewed_email]
viewed_first = _safe(viewed_row['First_Name'].values[0], viewed_email.split('@')[0].split('.')[0].title()) if not viewed_row.empty else viewed_email.split('@')[0].split('.')[0].title()
viewed_team  = _safe(viewed_row['Team Name'].values[0])  if not viewed_row.empty else ""

# -----------------------------------------------------------------------
# ✅ Personalized welcome banner (shown at top of every page)
# -----------------------------------------------------------------------
_viewing_as_self = (viewed_email == auth_key)
_banner_name     = auth_first if _viewing_as_self else f"{viewed_first} (via {auth_first})"

st.markdown(
    f"<div style='text-align:right; font-size:13px; color:#888; margin-bottom:-10px;'>"
    f"Signed in as <b>{user_email}</b> &nbsp;|&nbsp; "
    f"{'Viewing own dashboard' if _viewing_as_self else f'Viewing: <b>{viewed_email}</b>'}"
    f"</div>",
    unsafe_allow_html=True,
)


st.title("🌟 Sales Rep Performance Dashboard")

# 🔁 Auto-refresh every 5 minutes to match Five9 report cadence
st_autorefresh(interval=300000, key="datarefresh")

page = st.selectbox(
    "Choose a page",
    [
        "📊 Leaderboard",
        "🧮 Calculator",
        "💰Bonus & History",
        "📅 Yesterday",
        "👩‍💻 Team Lead Dashboard",
        "Senior Manager View",
        "📋 My QA",
    ]
)

# ---- Shared Config ----
sheet_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=171451260"

history_url = "https://docs.google.com/spreadsheets/d/1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU/export?format=csv&gid=303010891"
# Try __file__-relative first, then fall back to cwd-relative (works on Streamlit Cloud)
# Primary source: Google Sheet written by the hourly cron job
_HISTORY_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    "/export?format=csv&gid=1441799869"
)
# Local CSV path (fallback when running locally before sheet is populated)
_HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rep_history.csv")
if not os.path.exists(_HISTORY_CSV):
    _HISTORY_CSV = os.path.join(os.getcwd(), "data", "rep_history.csv")

# ---------------------------------------------------------------------------
# Five9 disposition mappings (sourced from dw_silver.fct_five9_calls)
# ---------------------------------------------------------------------------
_WIN_DISPOSITIONS = {
    "Closed Won", "Closed Won- SALT", "PF Closed Won", "Pool Closed Won",
    "EC MQ", "EC Mow", "EC Leaf", "EC LT", "EC Bush", "EC Flower Bed",
    "EC Multiple services", "EC Disease Fungicide", "EC Aeration",
    "EC Aeration Overseeding", "SA Leaf", "SA Bush", "SA Flower Bed",
    "Additional Property",
}
_EXCLUDE_FROM_CALLS = {
    "Support Call", "Transferred To 3rd Party", "Provider Inquiry",
    "Call in Support", "Test", "Internal Call",
}

# ---------------------------------------------------------------------------
# Manual pool win corrections
# Use when a rep closes a pool sale but uses the wrong disposition in Five9.
# Format: { "agent.email@lawnstarter.com": extra_pool_wins }
# Clear entries once the rep re-dispositions or the day rolls over.
# ---------------------------------------------------------------------------
_POOL_WIN_CORRECTIONS = {}

# ---------------------------------------------------------------------------
# Manual pool record override
# Use when the all-time daily pool record was set today (Redshift won't have it yet).
# Set to None to fall back to history automatically.
# ---------------------------------------------------------------------------
_POOL_RECORD_OVERRIDE = {
    "full_name": "Ashley Sagun",
    "team": "The Cutting Edge",
    "value": 2,
    "date": "Apr 20, 2026",
}

@st.cache_data(show_spinner=False, ttl=300)
def fetch_five9_gmail(_cache_bust_key: str) -> dict:
    """
    Fetch the latest Five9 call log CSV from Gmail.
    Returns {agent_email_lower -> {'calls': int, 'wins': int, 'pool_wins': int, 'agent_name': str}}
    Always includes '__status__' key with a diagnostic string (never shown to reps, used for debug).
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        cfg = st.secrets.get("gmail", {})
        if not cfg:
            return {"__status__": "NO_SECRETS: [gmail] section missing from Streamlit secrets"}

        creds = Credentials(
            token=None,
            refresh_token=cfg.get("refresh_token"),
            client_id=cfg.get("client_id"),
            client_secret=cfg.get("client_secret"),
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        # Let the API client handle token refresh automatically — no explicit refresh needed

        service = build("gmail", "v1", credentials=creds)

        results = service.users().messages().list(
            userId="me",
            q='from:reports-noreply@five9.com subject:"Conversion Tracker Call Log" newer_than:1d',
            maxResults=1,
        ).execute()

        msgs = results.get("messages", [])
        if not msgs:
            return {"__status__": "NO_EMAIL: No Five9 report found in inbox (last 24h)"}

        msg = service.users().messages().get(
            userId="me", id=msgs[0]["id"], format="full"
        ).execute()

        csv_bytes = None
        parts = msg["payload"].get("parts", [])
        # Also check nested parts (multipart/mixed inside multipart/related etc.)
        all_parts = list(parts)
        for part in parts:
            all_parts.extend(part.get("parts", []))

        for part in all_parts:
            fname = part.get("filename", "")
            if fname.lower().endswith(".csv"):
                att_id = part["body"].get("attachmentId")
                if att_id:
                    att = service.users().messages().attachments().get(
                        userId="me", messageId=msg["id"], id=att_id
                    ).execute()
                    csv_bytes = base64.urlsafe_b64decode(att["data"])
                elif part["body"].get("data"):
                    csv_bytes = base64.urlsafe_b64decode(part["body"]["data"])
                break

        if csv_bytes is None:
            # Log part names for diagnosis
            part_names = [p.get("filename", p.get("mimeType", "?")) for p in all_parts]
            return {"__status__": f"NO_CSV: Email found but no .csv attachment. Parts: {part_names}"}

        df = pd.read_csv(io.BytesIO(csv_bytes))
        df.columns = df.columns.str.strip()

        if "AGENT EMAIL" not in df.columns:
            return {"__status__": f"BAD_COLS: CSV columns are {list(df.columns)[:10]}"}

        # Strip whitespace from disposition values so " Pool Closed Won " still matches
        df["DISPOSITION"] = df["DISPOSITION"].astype(str).str.strip()
        df["AGENT EMAIL"] = df["AGENT EMAIL"].astype(str).str.strip()

        # Report is already scoped to "today" — no date filter needed
        df = df[~df["DISPOSITION"].isin(_EXCLUDE_FROM_CALLS)]

        # All pool-related win dispositions (case-insensitive "pool" catch-all + known variants)
        _pool_disps = df["DISPOSITION"].dropna().unique()
        # Pool wins = any disposition with "pool" in the name only (PF Closed Won = Payment First, not pool)
        _pool_win_set = {d for d in _pool_disps if isinstance(d, str) and "pool" in d.lower()}

        pool_wins_today = df["DISPOSITION"].isin(_pool_win_set).sum()
        result = {"__status__": f"OK: {len(df)} rows, {df['AGENT EMAIL'].nunique()} reps, {pool_wins_today} pool wins"}
        for email, grp in df.groupby("AGENT EMAIL"):
            agent_name = grp["AGENT NAME"].iloc[0] if "AGENT NAME" in grp.columns else str(email)
            email_key = str(email).strip().lower()
            pool_wins = int(grp["DISPOSITION"].isin(_pool_win_set).sum())
            # Apply manual corrections for mis-dispositioned calls
            pool_wins += _POOL_WIN_CORRECTIONS.get(email_key, 0)
            result[email_key] = {
                "calls":      int(len(grp)),
                "wins":       int(grp["DISPOSITION"].isin(_WIN_DISPOSITIONS).sum()),
                "pool_wins":  pool_wins,
                "agent_name": str(agent_name).strip(),
            }
        return result

    except Exception as _e:
        import traceback
        tb = traceback.format_exc()
        print(f"[fetch_five9_gmail] failed: {_e}\n{tb}")
        return {"__status__": f"EXCEPTION: {type(_e).__name__}: {str(_e)[:200]}"}

@st.cache_data(show_spinner=False, ttl=1800)
def load_history(_cache_bust_key: str):
    try:
        df = pd.read_csv(_HISTORY_SHEET_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Rep'].notna()]
    except Exception:
        # Fallback: local CSV (works when running locally)
        import os as _os
        try:
            if _os.path.exists(_HISTORY_CSV) and _os.path.getsize(_HISTORY_CSV) > 0:
                df = pd.read_csv(_HISTORY_CSV)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df[df['Rep'].notna()]
            else:
                raise FileNotFoundError
        except Exception:
            df = pd.read_csv(history_url, header=1)
            df = df[df['Date'].astype(str).str.lower() != 'date']
            df = df[df['Rep'].notna()]
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure columns expected by the dashboard always exist
    if 'Team Name' not in df.columns:
        df['Team Name'] = 'Unknown'
    if 'First_Name' not in df.columns:
        df['First_Name'] = df['Rep'].str.split('@').str[0].str.split('.').str[0].str.title()
    if 'Last_Name' not in df.columns:
        df['Last_Name'] = ''
    return df




# Live same-day call counts from Five9 (syncs hourly)
_TODAY_CONVERSION_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    "/export?format=csv&gid=356398595"
)

@st.cache_data(show_spinner=False, ttl=300)
def load_data(_cache_bust_key: str):
    df = pd.read_csv(sheet_url, header=1)

    # Enrich service attach metrics with Redshift data (hourly Google Sheet sync)
    # Wins + Calls come from Five9 Gmail CSV below — do NOT include them here
    # Keeps from Sheet: QA, Bonus, rep metadata
    ATTACH_COLS = ['Wins', 'Lawn Treatment', 'Mosquito', 'Bush Trimming', 'Flower Bed Weeding', 'Leaf Removal', 'Pool']

    # Always zero out attach cols — Google Forms data is never used for these
    for _col in ATTACH_COLS:
        if _col in df.columns:
            df[_col] = 0

    try:
        from datetime import date as _date
        hist = pd.read_csv(_HISTORY_SHEET_URL)
        hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
        today_rs = hist[hist['Date'].dt.date == _date.today()].copy()

        if not today_rs.empty:
            df['_rep_key'] = df['Rep'].astype(str).str.lower().str.strip()
            today_rs['_rep_key'] = today_rs['Rep'].astype(str).str.lower().str.strip()

            # Only keep the columns we want to overwrite
            rs_today = today_rs[['_rep_key'] + [c for c in ATTACH_COLS if c in today_rs.columns]]

            df = df.merge(rs_today, on='_rep_key', how='left', suffixes=('', '_rs'))

            # Fill from Redshift (already zeroed above, so NaN → stays 0)
            for col in ATTACH_COLS:
                if f'{col}_rs' in df.columns:
                    df[col] = df[f'{col}_rs'].fillna(0)
                    df.drop(columns=[f'{col}_rs'], inplace=True)

            df.drop(columns=['_rep_key'], inplace=True)
    except Exception as _e:
        import traceback
        print(f"[load_data] RepHistory sheet enrichment failed: {_e}\n{traceback.format_exc()}")

    # Pull real-time Calls + Wins from Five9 Gmail CSV (updates every 5 min)
    _five9_raw = fetch_five9_gmail(_cache_bust_key)
    # Strip the diagnostic key before treating dict as rep data
    _gmail_status = _five9_raw.pop("__status__", "NO_STATUS")
    five9 = _five9_raw  # now clean rep-data only
    print(f"[load_data] Gmail status: {_gmail_status}")
    if five9:
        # Gmail succeeded — override both Calls and Wins with real-time Five9 data
        df['_rep_key'] = df['Rep'].astype(str).str.lower().str.strip()
        df['Calls'] = df['_rep_key'].map({k: v['calls'] for k, v in five9.items()}).fillna(0)
        df['Wins']  = df['_rep_key'].map({k: v['wins']  for k, v in five9.items()}).fillna(0)
        df.drop(columns=['_rep_key'], inplace=True)
    else:
        # Gmail failed — keep Wins from RepHistory (already set via ATTACH_COLS above),
        # just zero Calls so conversion doesn't show phantom data
        df['Calls'] = 0
        print(f"[load_data] Five9 Gmail fetch failed ({_gmail_status}) — Wins from RepHistory, Calls=0")

    # Recalculate Conversion from Redshift Wins + live Five9 Calls
    _wins  = pd.to_numeric(df.get('Wins',  0), errors='coerce').fillna(0)
    _calls = pd.to_numeric(df.get('Calls', 0), errors='coerce').fillna(0)
    df['Conversion'] = (_wins / _calls.replace(0, pd.NA) * 100).fillna(0).round(2)

    return df

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

@st.cache_data(show_spinner=False, ttl=86400)
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

@st.cache_data(show_spinner=False, ttl=86400)
def load_section_tiers(url: str, section_label: str):
    import pandas as pd
    SECTION_HEADERS = {"Goals","Points","Current Cycle","All-In Attach Rate","LT","Conversion","QA"}
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
    consecutive_empty = 0
    for r in range(start, len(df)):
        b = str(df.iat[r, 1]).strip()
        if b in SECTION_HEADERS:
            break                           # hit next section header — stop
        if b == "":
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break                       # 2+ blank rows in a row = end of section
            continue                        # single blank row (e.g. Base row) — skip it
        consecutive_empty = 0
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

@st.cache_data(show_spinner=False, ttl=86400)
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
    metrics = ['Pool', 'Wins', 'Lawn Treatment', 'Bush Trimming', 'Mosquito', 'Flower Bed Weeding', 'Leaf Removal']
    history_df = history_df.copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'], errors='coerce')
    history_df = clean_numeric(history_df, metrics)

    out = []
    for m in metrics:
        if m not in history_df.columns or history_df[m].max() == 0:
            # Still include Pool if there's a manual override
            if m == 'Pool' and _POOL_RECORD_OVERRIDE:
                out.append({
                    "metric": "Pool",
                    "value": _POOL_RECORD_OVERRIDE["value"],
                    "full_name": _POOL_RECORD_OVERRIDE["full_name"],
                    "team": _POOL_RECORD_OVERRIDE["team"],
                    "date": _POOL_RECORD_OVERRIDE["date"],
                })
            continue
        idx = history_df[m].idxmax()
        row = history_df.loc[idx]
        full_name = f"{str(row.get('First_Name','')).strip()} {str(row.get('Last_Name','')).strip()}".strip() or str(row.get('Rep','Unknown'))
        record = {
            "metric": m,
            "value": int(row[m]),
            "full_name": full_name,
            "team": str(row.get('Team Name', 'Unknown')),
            "date": row['Date'].date() if pd.notna(row['Date']) else None
        }
        # If there's a manual pool override and it's higher, use it
        if m == 'Pool' and _POOL_RECORD_OVERRIDE and _POOL_RECORD_OVERRIDE["value"] >= record["value"]:
            record = {
                "metric": "Pool",
                "value": _POOL_RECORD_OVERRIDE["value"],
                "full_name": _POOL_RECORD_OVERRIDE["full_name"],
                "team": _POOL_RECORD_OVERRIDE["team"],
                "date": _POOL_RECORD_OVERRIDE["date"],
            }
        out.append(record)
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
    # Teams
    # ======================
    df_teams = df[
        df['Team Name'].notna() &
        (df['Team Name'].astype(str).str.strip() != '') &
        (df['Team Name'].astype(str).str.strip().str.lower() != 'unknown')
    ].copy()

    team_totals = df_teams.groupby('Team Name').agg(
        Calls=('Calls','sum'),
        Wins=('Wins','sum')
    ).reset_index()

    top_team = None
    if not team_totals.empty:
        team_totals = team_totals[team_totals['Calls'] >= min_calls_team]
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
    # Reps — rank by CONVERSION (min 10 calls)
    # ==========================================
    rep_totals = df.groupby(['Rep','First_Name','Last_Name','Team Name'], dropna=False).agg(
        Wins=('Wins','sum'),
        Calls=('Calls','sum')
    ).reset_index()

    if not rep_totals.empty:
        # Require at least 10 calls to qualify — prevents tiny-sample outliers
        rep_totals = rep_totals[rep_totals['Calls'] >= min_calls_rep]

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
    "Pool", "Lawn Treatment", "Bushes", "Mosquito", "Flower Bed Weeding", "Leaf Removal"
]
SPECIALTY_IQS = [
    "Overseeding and Aeration IQ", "Lime Treatment", "Disease Fungicide"
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
        # Pool column not in sheet yet — show nudge so the slot isn't invisible
        if display_name == "Pool":
            _nice_card(display_name, icon, "_No sales yet — be the first to make a splash! 🌊_")
        return
    d = df[['Full_Name', col_name]].copy()
    d[col_name] = pd.to_numeric(d[col_name], errors='coerce').fillna(0)
    d = d[d[col_name] > 0].sort_values(col_name, ascending=False)
    if d.empty:
        _nice_card(display_name, icon, "_No sales yet — go be the first!_")
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

def render_pool_realtime_banner(five9_data: dict):
    """
    Real-time pool sales banner — driven by Five9 Gmail CSV (updates every 5 min).
    Shows every rep with a Pool Closed Won disposition today.
    """
    pool_sellers = [
        (v["agent_name"], v["pool_wins"])
        for v in five9_data.values()
        if v.get("pool_wins", 0) > 0
    ]
    if not pool_sellers:
        return

    pool_sellers.sort(key=lambda x: -x[1])
    total = sum(n for _, n in pool_sellers)

    names_html = " &nbsp;·&nbsp; ".join(
        f"<b>{name}</b> ({n} {'sale' if n == 1 else 'sales'})"
        for name, n in pool_sellers
    )

    emoji_row = "🏊 " * min(total, 10)

    st.markdown(
        f"""
        <div style="
            border:3px solid #06b6d4;
            background:linear-gradient(135deg,rgba(6,182,212,0.22),rgba(6,182,212,0.06));
            padding:20px 24px;
            border-radius:16px;
            margin-bottom:16px;
            box-shadow:0 4px 24px rgba(6,182,212,0.35);
            text-align:center;
        ">
            <div style="font-size:13px;font-weight:800;color:#06b6d4;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">
                🏊 #1 PRIORITY — POOL SALES TODAY
            </div>
            <div style="font-size:42px;font-weight:900;color:#06b6d4;line-height:1.1;margin:6px 0;">
                {total} Pool {'Sale' if total == 1 else 'Sales'}!
            </div>
            <div style="font-size:22px;margin:6px 0;">{emoji_row}</div>
            <div style="font-size:16px;margin-top:10px;line-height:1.8;">
                {names_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pool_splash_banners(df: pd.DataFrame):
    """Secondary pool banner from Redshift (hourly) — shows individual splash alerts."""
    if "Pool" not in df.columns:
        return
    tmp = df[['Full_Name', 'Pool']].copy()
    tmp['Pool'] = pd.to_numeric(tmp['Pool'], errors='coerce').fillna(0)
    tmp = tmp[tmp['Pool'] > 0].sort_values('Pool', ascending=False)
    if tmp.empty:
        return
    # Individual splash alerts only (summary now handled by render_pool_realtime_banner)
    for _, r in tmp.iterrows():
        st.markdown(
            f"""
            <div style="
                border-left:6px solid #06b6d4;
                background:rgba(6,182,212,0.12);
                padding:14px 18px;
                border-radius:12px;
                margin-bottom:8px;
                font-size:17px;
            ">
                🌊 <b>SPLASH ALERT!</b> <b>{r['Full_Name']}</b> sold
                <b>{int(r['Pool'])} Pool {'IQ' if r['Pool']==1 else 'IQs'}</b> today!
                {'🎉🎉🎉' if r['Pool'] >= 3 else '🎉🎉' if r['Pool'] >= 2 else '🎉'}
            </div>
            """,
            unsafe_allow_html=True
        )


if page == "📊 Leaderboard":
    st.markdown("<h1 style='text-align: center;'>📊 Conversion Rate Leaderboard</h1>", unsafe_allow_html=True)

    from pytz import timezone
    eastern = timezone('US/Eastern')
    today = datetime.now(eastern).date()
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H-') + str(datetime.now(eastern).minute // 5 * 5).zfill(2)

    df = load_data(cache_bust_key).copy()

    # 🧹 Clean headers and strip spaces
    df.columns = df.columns.str.strip()

    # Pull fresh history for records/champions
    history_df = load_history(cache_bust_key).copy()
    history_df.columns = history_df.columns.str.strip()

    # -----------------------------------------------------------------------
    # 👋 Personalized Welcome Banner
    # -----------------------------------------------------------------------
    try:
        from pytz import timezone as _ctz
        _cst = _ctz('America/Chicago')
        _today_cst = datetime.now(_cst).date()
        _yesterday_cst = _today_cst - timedelta(days=1)

        def _pct(val):
            try: return float(str(val).replace('%','').strip())
            except: return 0.0

        # Today's stats from live df
        _me_today = df[df['Rep'].astype(str).str.lower().str.strip() == viewed_email]
        _conv_today  = _pct(_me_today['Conversion'].values[0])  if not _me_today.empty else 0.0
        _wins_today  = int(pd.to_numeric(_me_today['Wins'].values[0],  errors='coerce') or 0) if not _me_today.empty else 0
        _calls_today = int(pd.to_numeric(_me_today['Calls'].values[0], errors='coerce') or 0) if not _me_today.empty else 0
        if _conv_today < 1.0 and _conv_today > 0:
            _conv_today *= 100

        # Yesterday's stats from history
        _hist_me = history_df[
            (history_df['Rep'].astype(str).str.lower().str.strip() == viewed_email) &
            (history_df['Date'].dt.date == _yesterday_cst)
        ]
        _conv_yest = _pct(_hist_me['Conversion'].values[0]) if not _hist_me.empty else None
        if _conv_yest is not None and _conv_yest < 1.0 and _conv_yest > 0:
            _conv_yest *= 100

        # Attach rate today
        _attach_cols = ['Lawn Treatment','Bush Trimming','Flower Bed Weeding','Mosquito','Leaf Removal']
        _attaches = sum(int(pd.to_numeric(_me_today[c].values[0], errors='coerce')) for c in _attach_cols if c in _me_today.columns and not _me_today.empty)
        _attach_rate = (_attaches / _wins_today * 100) if _wins_today > 0 else 0.0
        _lt = int(pd.to_numeric(_me_today['Lawn Treatment'].values[0], errors='coerce')) if 'Lawn Treatment' in _me_today.columns and not _me_today.empty else 0
        _lt_rate = (_lt / _wins_today * 100) if _wins_today > 0 else 0.0

        # Conversion delta vs yesterday
        if _conv_yest is not None and _conv_yest > 0:
            _delta = _conv_today - _conv_yest
            _delta_str = f"({'▲' if _delta >= 0 else '▼'} {abs(_delta):.1f}% vs yesterday)"
            _delta_color = "#22c55e" if _delta >= 0 else "#ef4444"
        else:
            _delta_str, _delta_color = "", "#888"

        # How many wins to pass the next team
        _team_totals_wb = df[df['Calls'].apply(lambda x: pd.to_numeric(x, errors='coerce') or 0) > 0].copy()
        _team_totals_wb['Calls'] = pd.to_numeric(_team_totals_wb['Calls'], errors='coerce')
        _team_totals_wb['Wins']  = pd.to_numeric(_team_totals_wb['Wins'],  errors='coerce')
        _tg = _team_totals_wb.groupby('Team Name').agg(W=('Wins','sum'), C=('Calls','sum')).reset_index()
        _tg = _tg[_tg['C'] >= 10]
        _tg['Conv'] = _tg['W'] / _tg['C'] * 100
        _my_team_row = _tg[_tg['Team Name'].str.strip() == viewed_team] if viewed_team else pd.DataFrame()
        _gap_msg = ""
        if not _my_team_row.empty:
            _my_conv = _my_team_row['Conv'].values[0]
            _above   = _tg[_tg['Conv'] > _my_conv].sort_values('Conv')
            if not _above.empty:
                _next_team     = _above.iloc[0]['Team Name']
                _next_conv     = _above.iloc[0]['Conv']
                _my_calls      = _my_team_row['C'].values[0]
                _my_wins       = _my_team_row['W'].values[0]
                _wins_needed   = max(0, math.ceil(_next_conv / 100 * _my_calls - _my_wins) + 1)
                _gap_msg = f"🏁 <b>{_wins_needed} more win{'s' if _wins_needed != 1 else ''}</b> to pass <b>{_next_team}</b>"

        # Greeting based on time
        _hour = datetime.now(_ctz('America/Chicago')).hour
        _greeting = "Good morning" if _hour < 12 else ("Good afternoon" if _hour < 17 else "Good evening")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(6,182,212,0.08));
            border: 1.5px solid rgba(34,197,94,0.35);
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 18px;
        ">
            <div style="font-size:22px; font-weight:900; margin-bottom:10px;">
                {_greeting}, {viewed_first} 👋
            </div>
            <div style="display:flex; gap:40px; flex-wrap:wrap; font-size:15px;">
                <div>
                    📞 <b>{_calls_today} calls</b> &nbsp;·&nbsp;
                    🏆 <b>{_wins_today} wins</b>
                </div>
                <div>
                    🎯 Conversion: <b>{_conv_today:.1f}%</b>
                    <span style="color:{_delta_color}; font-size:13px;"> {_delta_str}</span>
                </div>
                <div>🧩 Attach Rate: <b>{_attach_rate:.0f}%</b></div>
                <div>🌿 LT Attach: <b>{_lt_rate:.0f}%</b></div>
            </div>
            {"<div style='margin-top:10px; font-size:14px;'>" + _gap_msg + "</div>" if _gap_msg else ""}
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

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

    # 🔄 Force refresh button — clears all caches and re-fetches immediately
    _col_refresh, _col_status = st.columns([1, 5])
    with _col_refresh:
        if st.button("🔄 Force Refresh", key="force_refresh_conv"):
            st.cache_data.clear()
            st.rerun()

    # 🏊 Real-time pool sales banner (Five9 CSV — updates every 5 min)
    _five9_raw_live = fetch_five9_gmail(cache_bust_key)
    _gmail_diag = _five9_raw_live.pop("__status__", "")
    _five9_live = _five9_raw_live  # clean rep data only
    # Show Gmail connection status
    with _col_status:
        if _five9_live:
            st.caption(f"📡 Live Five9 data: {len(_five9_live)} reps · {_gmail_diag}")
        else:
            st.caption(f"⚠️ Live Five9 data unavailable — {_gmail_diag}")
    render_pool_realtime_banner(_five9_live)

    # 🏊 Individual pool splash alerts (Redshift — hourly)
    render_pool_splash_banners(df)

    # Identity comes from SSO auth (set in session_state by the View As sidebar block)
    user = st.session_state.get("selected_rep", viewed_email)

    # Use reps with calls as primary filter; fall back to reps with wins when
    # call data isn't available yet (TodayONLYConversion sheet not populated).
    _calls_available = pd.to_numeric(df['Calls'], errors='coerce').fillna(0).sum() > 0
    active_df = df[pd.to_numeric(df['Calls'], errors='coerce').fillna(0) >= 1].copy() if _calls_available \
        else df[pd.to_numeric(df['Wins'], errors='coerce').fillna(0) > 0].copy()
    user_data = df[df[rep_col] == user]
    if user_data is None or user_data.empty:
        first_name = user_given_name or (user.split()[0] if user else "Rep")
    else:
        if "First_Name" in user_data.columns and user_data["First_Name"].notna().any():
            first_name = str(user_data["First_Name"].dropna().iloc[0]).strip()
        else:
            first_name = user_given_name or (user.split()[0] if user else "Rep")

    # --------------------------------------------
    # 🏆 Records to Beat (All-Time Single-Day Highs)
    # --------------------------------------------
    records = get_records_to_beat(history_df)

    if records:
        st.markdown("<h2 style='text-align:center;'>🏆 Records to Beat (All-Time Single-Day Highs)</h2>", unsafe_allow_html=True)
        # Pool is always rendered first as the hero card
        pool_rec = next((r for r in records if r['metric'] == 'Pool'), None)
        other_recs = [r for r in records if r['metric'] != 'Pool']
        if pool_rec:
            _d = pool_rec['date']
            date_str = _d if isinstance(_d, str) else (_d.strftime('%b %d, %Y') if _d else '—')
            st.markdown(f"""
            <div style='text-align:center; padding:24px; border:3px solid #06b6d4;
                 border-radius:18px;
                 background:linear-gradient(135deg,rgba(6,182,212,0.18),rgba(6,182,212,0.04));
                 margin-bottom:18px; box-shadow:0 4px 24px rgba(6,182,212,0.28);'>
                <div style='font-size:13px;font-weight:800;color:#06b6d4;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;'>
                    🏊 #1 PRIORITY — POOL RECORD
                </div>
                <div style='font-size:20px; font-weight:700; margin:6px 0 2px;'>Pool Cleans Sold in a Single Day</div>
                <div style='font-size:60px; font-weight:900; color:#06b6d4; line-height:1.1; margin:6px 0;'>{pool_rec['value']}</div>
                <div style='font-size:15px; opacity:0.9;'>{pool_rec['full_name']}</div>
                <div style='font-size:13px; opacity:0.7;'>{pool_rec['team']}</div>
                <div style='font-size:13px; opacity:0.7;'>on {date_str}</div>
                <div style='margin-top:12px; font-size:16px; color:#06b6d4; font-weight:800;'>🌊 CAN YOU MAKE A BIGGER SPLASH?</div>
            </div>
            """, unsafe_allow_html=True)
        if other_recs:
            cols = st.columns(min(3, len(other_recs)))
            for i, rec in enumerate(other_recs):
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
        st.warning("📞 No calls logged for you today yet! Let's change that. 💪")

    try:
        personal_conversion = float(user_data[conversion_col].astype(str).str.replace('%', '').str.strip().values[0]) if not user_data.empty else 0.0
        # Sheet may store as decimal (0.28) instead of percent (28) — normalise
        if personal_conversion < 1.0 and personal_conversion > 0:
            personal_conversion *= 100
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
        st.error(f"❌ Below Base ({thr_base:.2f}%). Let's lock in and close the gap!")







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

    # Require a minimum number of calls so a team with 1 win / 1 call doesn't top the board
    MIN_TEAM_CALLS = 5
    team_totals = team_totals[team_totals['Total_Calls'] >= MIN_TEAM_CALLS].copy()

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
                        background-color: rgba(128,128,128,0.15); color: inherit; border: 1px solid rgba(128,128,128,0.4);'>
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
                    background-color: rgba(128,128,128,0.15); color: inherit; border: 1px solid rgba(128,128,128,0.4);'>
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
    # Deduplicate: reps in both Team ABC and an assigned team have duplicate rows.
    # Prefer non-Team ABC row so they show under their real team.
    _lb = active_df.copy()
    _lb['_is_abc'] = _lb['Team Name'].astype(str).str.strip().str.lower() == 'team abc'
    _lb = _lb.sort_values(by=[conversion_col, '_is_abc'], ascending=[False, True])
    _lb = _lb.drop_duplicates(subset='Full_Name', keep='first')
    leaderboard = _lb[['Full_Name', conversion_col]].reset_index(drop=True)
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
        # Deduplicate logo lookup — one row per rep, preferring non-Team ABC
        _logo = df[['Full_Name', 'Team Name', 'Team_Logo']].copy()
        _logo['_is_abc'] = _logo['Team Name'].astype(str).str.strip().str.lower() == 'team abc'
        _logo = _logo.sort_values('_is_abc').drop_duplicates(subset='Full_Name', keep='first')[['Full_Name', 'Team_Logo']]
        leaderboard = leaderboard.merge(_logo, on='Full_Name', how='left')
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

    # 📋 Full Today Attach Leaderboard — all reps with any attach today
    st.markdown("<hr><h3 style='text-align: center;'>📋 Today's Full Attach Leaderboard</h3>", unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="refresh_attach"):
        st.cache_data.clear()
        st.rerun()

    today_attach_cols = ['Pool', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito', 'Leaf Removal', 'Lawn Treatment']
    for c in today_attach_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Only reps with at least one attach today
    attach_df = df[df[today_attach_cols].sum(axis=1) > 0].copy()

    if attach_df.empty:
        st.info("No attach services sold yet today — check back soon!")
    else:
        # Sort: total attaches descending, then Pool (priority) descending
        attach_df['_total'] = attach_df[today_attach_cols].sum(axis=1)
        attach_df = attach_df.sort_values(['_total', 'Pool'], ascending=[False, False]).reset_index(drop=True)
        attach_df['Rank'] = attach_df.index + 1

        # Ensure Team_Logo exists
        if 'Team_Logo' not in attach_df.columns and 'Team Name' in attach_df.columns:
            attach_df['Team_Logo'] = attach_df['Team Name'].astype(str).apply(
                lambda name: f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{name.replace(' ', '_').lower()}.png' width='30'>" if name not in ('', 'nan', 'Unknown') else ""
            )
        elif 'Team_Logo' not in attach_df.columns:
            attach_df['Team_Logo'] = ""

        display_cols = ['Rank', 'Full_Name'] + today_attach_cols + ['Team_Logo']
        attach_display = attach_df[display_cols].copy()
        attach_display.columns = ['Rank', 'Rep Name', '🏊 Pool', '🌳 Bush Trim', '🌸 Flower Bed', '🦟 Mosquito', '🍂 Leaf Removal', '🌿 Lawn Treatment', 'Team Logo']

        # Highlight Pool cells > 0 in cyan
        def style_pool(val):
            try:
                if int(val) > 0:
                    return f"<span style='color:#06b6d4;font-weight:900;font-size:16px;'>{int(val)}</span>"
            except:
                pass
            return str(int(val)) if str(val) not in ('', 'nan') else '0'

        attach_display['🏊 Pool'] = attach_display['🏊 Pool'].apply(style_pool)

        st.markdown(attach_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("""
        <style>
        table td, table th { text-align: center !important; vertical-align: middle; }
        table { margin-left: auto; margin-right: auto; border-collapse: collapse; width: 95%; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
        th { background-color: #333; color: white; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)


# --------------------------------------------
# 🧮 TAB 2: Calculator
# --------------------------------------------
if page == "🧮 Calculator":

    # ── Personalized Bonus Calculator ──────────────────────────────────────
    _CC_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=1572505417"
    )

    @st.cache_data(show_spinner=False, ttl=300)
    def load_current_cycle(url: str) -> pd.DataFrame:
        cc = pd.read_csv(url, header=0)
        cc.columns = cc.columns.str.strip()
        return cc

    def _pct_val(v):
        try: return float(str(v).replace('%', '').strip())
        except: return 0.0

    def _int_val(v):
        try: return int(float(str(v).replace('%', '').strip()))
        except: return 0

    def _next_tier(current_pct, tiers):
        """Return (next_threshold, points) or None if already at top."""
        above = sorted([(t, p) for t, p in tiers if t > current_pct], key=lambda x: x[0])
        return above[0] if above else None

    st.markdown("## 🎯 Your Personalized Bonus Calculator")

    _cc_df = load_current_cycle(_CC_URL)

    # Look up the viewed rep by Name_Proper
    _vname = ""
    if not viewed_row.empty:
        _fn = _safe(viewed_row['First_Name'].values[0])
        _ln = _safe(viewed_row['Last_Name'].values[0])
        _vname = f"{_fn} {_ln}".strip()

    _cc_match = _cc_df[_cc_df['Agent'].str.strip().str.lower() == _vname.lower()] if _vname else pd.DataFrame()

    if _cc_match.empty:
        st.info(f"No current cycle data found for {viewed_first} yet — check back after the next cycle update.")
    else:
        _cc = _cc_match.iloc[0]

        cycle_calls  = _int_val(_cc.get('Call #', 0))
        cycle_wins   = _int_val(_cc.get('Win #', 0))
        cycle_attach = _int_val(_cc.get('Attach #', 0))
        cycle_conv   = _pct_val(_cc.get('Conversion', 0))
        cycle_att    = _pct_val(_cc.get('Attach', 0))
        cycle_lt     = _pct_val(_cc.get('LT', 0))
        cycle_qa     = _pct_val(_cc.get('QA', 0))
        cycle_tier   = str(_cc.get('Bonus Tier', '—')).strip()
        cycle_pts    = str(_cc.get('Total Points', '0')).strip()

        # Load live cycle tiers
        _conv_tiers   = load_section_tiers(BONUS_SHEET_URL, "Conversion")
        _attach_tiers = load_section_tiers(BONUS_SHEET_URL, "All-In Attach Rate")
        _qa_tiers     = load_section_tiers(BONUS_SHEET_URL, "QA")

        # ── Current standing ───────────────────────────────────────────────
        st.caption(f"Cycle data for **{_vname}** — updates every 5 min from CurrentCycle tab.")
        st.markdown(f"**Current Tier:** {cycle_tier} &nbsp;|&nbsp; **Total Points:** {cycle_pts}")
        st.markdown("<br>", unsafe_allow_html=True)

        _c1, _c2, _c3, _c4 = st.columns(4)
        def _tier_label(pct, tiers):
            met = [t for t, _ in tiers if pct >= t]
            return f"✅ {max(met):.0f}% met" if met else "❌ base not met"

        with _c1:
            st.metric("Calls", cycle_calls)
            st.metric("Wins", cycle_wins)
        with _c2:
            st.metric("Conversion", f"{cycle_conv:.1f}%")
            st.caption(_tier_label(cycle_conv, _conv_tiers))
        with _c3:
            st.metric("All-In Attach", f"{cycle_att:.1f}%")
            st.caption(_tier_label(cycle_att, _attach_tiers))
        with _c4:
            st.metric("QA", f"{cycle_qa:.1f}%")
            st.caption(_tier_label(cycle_qa, _qa_tiers))

        # ── What-If Projector ──────────────────────────────────────────────
        st.markdown("### 🔮 What-If Projector")
        st.caption("Slide to see how additional activity would move your metrics and tier.")

        _wc1, _wc2, _wc3 = st.columns(3)
        with _wc1:
            extra_wins    = st.slider("Additional Wins",     0, 60, 0, key="calc_xwins")
        with _wc2:
            extra_calls   = st.slider("Additional Calls",    0, 300, 0, key="calc_xcalls")
        with _wc3:
            extra_attaches = st.slider("Additional Attaches", 0, 60, 0, key="calc_xattach")

        proj_calls  = cycle_calls  + extra_calls
        proj_wins   = cycle_wins   + extra_wins
        proj_attach = cycle_attach + extra_attaches
        proj_conv   = (proj_wins / proj_calls * 100)   if proj_calls > 0 else 0.0
        proj_att    = (proj_attach / proj_wins * 100)  if proj_wins  > 0 else 0.0

        _pc1, _pc2, _pc3 = st.columns(3)
        with _pc1:
            st.metric("Projected Conversion",   f"{proj_conv:.1f}%",
                      delta=f"{proj_conv - cycle_conv:+.1f}%")
        with _pc2:
            st.metric("Projected All-In Attach", f"{proj_att:.1f}%",
                      delta=f"{proj_att - cycle_att:+.1f}%")
        with _pc3:
            st.metric("Projected Calls / Wins",  f"{proj_calls} / {proj_wins}")

        # ── "What you need" callouts ───────────────────────────────────────
        st.markdown("#### 🏁 To reach your next tier:")
        _n1, _n2, _n3 = st.columns(3)

        with _n1:
            nt = _next_tier(proj_conv, _conv_tiers)
            if nt:
                needed_wins = math.ceil(nt[0] / 100 * proj_calls) - proj_wins
                st.info(f"**Conversion → {nt[0]:.0f}%**\n\n"
                        f"Need **{max(0, needed_wins)} more win(s)** at your projected call count.")
            else:
                st.success("🏆 Conversion: top tier reached!")

        with _n2:
            nt = _next_tier(proj_att, _attach_tiers)
            if nt:
                needed_att = math.ceil(nt[0] / 100 * proj_wins) - proj_attach
                st.info(f"**All-In Attach → {nt[0]:.0f}%**\n\n"
                        f"Need **{max(0, needed_att)} more attach(es)** at your projected win count.")
            else:
                st.success("🏆 All-In Attach: top tier reached!")

        with _n3:
            _qa_base = min((t for t, _ in _qa_tiers), default=86.0)
            if cycle_qa >= _qa_base:
                st.success(f"**QA: Base met!**\n\nCurrent: {cycle_qa:.1f}% — above the {_qa_base:.0f}% base. Keep it up!")
            else:
                _qa_gap = _qa_base - cycle_qa
                st.info(f"**QA → {_qa_base:.0f}% base**\n\nCurrent: {cycle_qa:.1f}% — {_qa_gap:.1f}% away from base.")

    st.markdown("---")
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
if page == "💰Bonus & History":
    st.header("🔥 Bonus & History")

    from pytz import timezone
    eastern = timezone('US/Eastern')
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H-') + str(datetime.now(eastern).minute // 5 * 5).zfill(2)
    df = load_data(cache_bust_key).copy()
    df.columns = df.columns.str.strip()

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

    # ✅ Grab selected rep from session_state (always set from auth)
    email = st.session_state.get("selected_rep", viewed_email)

    match = df[df['Rep'].astype(str).str.strip() == email.strip()]
    _bonus_has_data = not match.empty
    if not _bonus_has_data:
        st.info("No sales rep data found for your account on the bonus tracker.")

    def percent(val):
        try:
            return float(str(val).replace('%', '').strip())
        except:
            return 0

    if _bonus_has_data:
        row = match.iloc[0]

        metrics = {
            'Conversion': percent(row.get('BonusConversion', 0)),
            'All-In Attach': percent(row.get('BonusAllinAttach', 0)),
            'QA': percent(row.get('BonusQA', 0))
        }

        points = {
            'Conversion': get_points(metrics['Conversion'], conversion_tiers),
            'All-In Attach': get_points(metrics['All-In Attach'], attach_tiers),
            'QA': get_points(metrics['QA'], qa_tiers)
        }

        st.subheader(f"🧑‍🌾 Growth Stats for {row.get('First_Name', viewed_first)}")
        for k in metrics:
            st.markdown(f"**{k}**: {metrics[k]:.2f}% — Points: `{points[k]}`")

        st.subheader("🌱 Focus Patch")
        focus = min(points, key=points.get)
        st.info(f"Your area of growth: **{focus}** — currently {metrics[focus]:.2f}%")

        total_points = sum(points.values())
        raw_bonus = row.get('Bonus Pay', 0)
        hourly = f"${float(raw_bonus):.2f}" if raw_bonus and str(raw_bonus).strip().replace("$", "") != "0" else "$0.00"
        st.markdown(f"**🌼 Points Earned:** {total_points} — **Hourly Forecast:** {hourly}")

        # ✅ Bonus Qualifier Status — all 3 base thresholds must be met to earn any bonus
        st.markdown("### 🎯 Bonus Qualifier Status")
        st.caption("You'll earn this rate **only if all 3 minimum qualifiers are met** — Conversion, All-In Attach, and QA.")

        def _get_base(tiers, fallback):
            """Return the lowest (base) threshold from a tier list."""
            if tiers:
                return tiers[-1][0]
            return fallback

        base_conv_val   = _get_base(conversion_tiers, 20.0)
        base_attach_val = _get_base(attach_tiers, 25.0)
        base_qa_val     = _get_base(qa_tiers, 80.0)

        def _safe_metric(v):
            try:
                f = float(v)
                return 0.0 if (f != f) else f  # f != f is True only for NaN
            except (TypeError, ValueError):
                return 0.0

        qualifiers = {
            'Conversion':    (_safe_metric(metrics['Conversion']),    base_conv_val,   f"{base_conv_val:.0f}%"),
            'All-In Attach': (_safe_metric(metrics['All-In Attach']), base_attach_val, f"{base_attach_val:.0f}%"),
            'QA':            (_safe_metric(metrics['QA']),            base_qa_val,     f"{base_qa_val:.0f}%"),
        }

        met_list     = [k for k, (val, base, _) in qualifiers.items() if val >= base]
        not_met_list = [k for k, (val, base, _) in qualifiers.items() if val < base]

        # Visual qualifier checklist
        q_cols = st.columns(3)
        labels = ['Conversion', 'All-In Attach', 'QA']
        for i, k in enumerate(labels):
            val, base, need_str = qualifiers[k]
            met = val >= base
            gap = base - val
            with q_cols[i]:
                if met:
                    st.markdown(
                        f"<div style='background:rgba(34,197,94,0.15);border:2px solid #22c55e;border-radius:12px;"
                        f"padding:12px;text-align:center;'>"
                        f"<div style='font-size:24px;'>✅</div>"
                        f"<div style='font-weight:800;'>{k}</div>"
                        f"<div style='font-size:13px;color:#22c55e;'>{val:.1f}% — Base met!</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:rgba(239,68,68,0.12);border:2px solid #ef4444;border-radius:12px;"
                        f"padding:12px;text-align:center;'>"
                        f"<div style='font-size:24px;'>❌</div>"
                        f"<div style='font-weight:800;'>{k}</div>"
                        f"<div style='font-size:13px;color:#ef4444;'>{val:.1f}% — Need {need_str} ({gap:.1f}% away)</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Coaching shoutout
        first = row.get('First_Name', viewed_first)
        if not not_met_list:
            st.success(f"🎉 {first}, you've hit all 3 qualifiers — you're bonus eligible! Keep pushing those points up! 🔥")
        elif len(not_met_list) == 1:
            k = not_met_list[0]
            val, base, need_str = qualifiers[k]
            gap = base - val
            already_met = [m for m in met_list]
            met_str = " and ".join(already_met) if already_met else "nothing yet"
            st.warning(
                f"💪 {first}, you've already met base for **{met_str}** — "
                f"you just need to get your **{k}** to **{need_str}** "
                f"({gap:.1f}% away) to unlock your bonus!"
            )
        elif len(not_met_list) == 2:
            k1, k2 = not_met_list
            _, base1, need1 = qualifiers[k1]
            _, base2, need2 = qualifiers[k2]
            met_str = met_list[0] if met_list else None
            prefix = f"You've got **{met_str}** locked in — " if met_str else ""
            st.error(
                f"📈 {first}, {prefix}focus on **{k1}** (need {need1}) "
                f"and **{k2}** (need {need2}) to qualify for bonus this cycle."
            )
        else:
            st.error(
                f"🚨 {first}, none of the 3 qualifiers are met yet — "
                f"let's focus on Conversion first, then All-In Attach and QA. You've got this!"
            )

    # 🏅 Personal Bests Section
    st.markdown("### 🏅 Personal Bests")

    # Load personal bests using shared load_history (CSV preferred, Sheet fallback)
    from pytz import timezone as _tz
    _eastern = _tz('US/Eastern')
    _pb_cache_key = datetime.now(_eastern).strftime('%Y-%m-%d-%H-') + ('30' if datetime.now(_eastern).minute >= 30 else '00')
    history_df = load_history(_pb_cache_key).copy()
    history_df.columns = history_df.columns.str.strip()

    # Filter all rows for the selected rep
    rep_history = history_df[history_df['Rep'].astype(str).str.strip() == email].copy()

    # Safely convert relevant columns to numeric
    for _col in ['Wins', 'Lawn Treatment', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito', 'Pool']:
        if _col in rep_history.columns:
            rep_history[_col] = pd.to_numeric(rep_history[_col], errors='coerce').fillna(0)
        else:
            rep_history[_col] = 0

    # Compute personal bests across all days
    pb_pool = rep_history['Pool'].max()
    pb_wins = rep_history['Wins'].max()
    pb_lawn = rep_history['Lawn Treatment'].max()
    pb_bush = rep_history['Bush Trimming'].max()
    pb_flower = rep_history['Flower Bed Weeding'].max()
    pb_mosquito = rep_history['Mosquito'].max()

    def challenge_line(label, pb_val, emoji):
        if pd.isna(pb_val) or pb_val == 0:
            return f"{emoji} **{label} PB:** 0 — Let's set a new record today! 💪"
        else:
            return f"{emoji} **{label} PB:** {int(pb_val)} — Can you hit {int(pb_val) + 1} today?"

    # Pool personal best — hero card (Pool is #1 priority!)
    pool_pb_val = int(pb_pool) if not pd.isna(pb_pool) else 0
    pool_challenge = f"Can you sell {pool_pb_val + 1} today?" if pool_pb_val > 0 else "Let's set your first Pool record today! 🌊"
    st.markdown(f"""
    <div style="
        border:3px solid #06b6d4;
        background:linear-gradient(135deg,rgba(6,182,212,0.18),rgba(6,182,212,0.04));
        padding:18px 20px;
        border-radius:16px;
        margin-bottom:14px;
        box-shadow:0 4px 20px rgba(6,182,212,0.25);
        text-align:center;
    ">
        <div style="font-size:12px;font-weight:800;color:#06b6d4;text-transform:uppercase;letter-spacing:2px;">
            🏊 #1 PRIORITY — POOL
        </div>
        <div style="font-size:15px;font-weight:700;margin:4px 0 2px;">Your Pool Personal Best</div>
        <div style="font-size:52px;font-weight:900;color:#06b6d4;line-height:1.1;">{pool_pb_val}</div>
        <div style="font-size:15px;color:#06b6d4;font-weight:700;margin-top:6px;">🌊 {pool_challenge}</div>
    </div>
    """, unsafe_allow_html=True)

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

    # Points bar chart (only when rep data is available)
    if _bonus_has_data:
        chart = pd.DataFrame({"Metric": list(points.keys()), "Points": list(points.values())})
        st.bar_chart(chart.set_index("Metric"))


# --------------------------------------------
# 📅 TAB 4: Yesterday's Snapshot
# --------------------------------------------
if page == "📅 Yesterday":
    st.markdown("<h1 style='text-align: center;'>📅 Yesterday's Leaderboard</h1>", unsafe_allow_html=True)

    from pytz import timezone
    eastern = timezone('US/Eastern')
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H-') + str(datetime.now(eastern).minute // 5 * 5).zfill(2)

    history_df = load_history(cache_bust_key).copy()
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






    # ✅ Enrich yesterday_df with calls + wins from Five9 Google Sheet (source of truth for conversion)
    FIVE9_URL = "https://docs.google.com/spreadsheets/d/1sO4ZDe-n8-ugc-OsoDisHPXXEAap6b2iINcjsWXz5gU/export?format=csv&gid=396976453"
    try:
        _raw = pd.read_csv(FIVE9_URL, header=None)
        # Col 1=Email, Col 9=All in Calls (YESTERDAY), Col 15=Total Wins (YESTERDAY)
        _five9_cv = _raw.iloc[2:, [1, 9, 15]].copy()
        _five9_cv.columns = ['_rep_key', 'five9_calls', 'five9_wins']
        _five9_cv['_rep_key']    = _five9_cv['_rep_key'].astype(str).str.lower().str.strip()
        _five9_cv['five9_calls'] = pd.to_numeric(_five9_cv['five9_calls'], errors='coerce').fillna(0)
        _five9_cv['five9_wins']  = pd.to_numeric(_five9_cv['five9_wins'],  errors='coerce').fillna(0)
        _five9_cv = _five9_cv[_five9_cv['_rep_key'].str.contains('@lawnstarter.com', na=False)]

        yesterday_df = yesterday_df.copy()
        yesterday_df['_rep_key'] = yesterday_df['Rep'].astype(str).str.lower().str.strip()
        yesterday_df = yesterday_df.merge(_five9_cv, on='_rep_key', how='left')
        # Overwrite Calls + Wins with Five9 values where available (source of truth for conversion %)
        yesterday_df['Calls'] = yesterday_df['five9_calls'].combine_first(
            pd.to_numeric(yesterday_df['Calls'], errors='coerce').fillna(0)
        )
        yesterday_df['Wins'] = yesterday_df['five9_wins'].combine_first(
            pd.to_numeric(yesterday_df['Wins'], errors='coerce').fillna(0)
        )
        yesterday_df.drop(columns=['_rep_key', 'five9_calls', 'five9_wins'], inplace=True)
    except Exception:
        pass  # silently fall back to RepHistory calls/wins if sheet unavailable

    # ✅ Get selected rep from session state (always set from auth now)
    selected_rep = st.session_state.get("selected_rep", viewed_email)

    user_data = yesterday_df[yesterday_df['Rep'] == selected_rep]

    # ---- Personal Stats (only shown if rep has data)
    if not user_data.empty:
        first_name = user_data['First_Name'].values[0] if ('First_Name' in user_data.columns) else selected_rep.split('@')[0].title()
        st.markdown(
            f"<h3 style='text-align: center;'>🕰️ Snapshot for {first_name} — {yesterday.strftime('%B %d, %Y')}</h3>",
            unsafe_allow_html=True
        )
        user_calls = int(user_data['Calls'].values[0])
        user_wins = int(user_data['Wins'].values[0])
        personal_conversion = (user_wins / user_calls * 100) if user_calls > 0 else 0
        st.markdown(f"<h3 style='text-align: center;'>📞 {user_calls} Calls | 🏆 {user_wins} Wins | 🎯 {personal_conversion:.2f}% Conversion</h3>", unsafe_allow_html=True)

        # ---- Team Info
        team_name = user_data['Team Name'].values[0] if ('Team Name' in user_data.columns) else "Unknown"
        team_logo = f"<img src='https://raw.githubusercontent.com/heatherLS/rep-dashboard/main/logos/{team_name.replace(' ', '_').lower()}.png' width='80'>" if pd.notna(team_name) else ""
        st.markdown(f"<div style='text-align: center;'>{team_logo}<br><b>{team_name} (Yesterday)</b></div>", unsafe_allow_html=True)
    else:
        team_name = None

    # 🔧 Fix string columns so we can safely compare and calculate
    numeric_cols = ['Calls', 'Wins', 'Lawn Treatment', 'Leaf Removal', 'Bush Trimming', 'Flower Bed Weeding', 'Mosquito']
    for col in numeric_cols:
        if col in yesterday_df.columns:
            yesterday_df[col] = pd.to_numeric(yesterday_df[col], errors='coerce').fillna(0)


    # ---- Team Rank (fall back to Wins if Calls unavailable)
    team_stats = yesterday_df[yesterday_df['Calls'] > 0].copy()
    if team_stats.empty:
        team_stats = yesterday_df[yesterday_df['Wins'] > 0].copy()
    team_stats['Calls'] = pd.to_numeric(team_stats['Calls'], errors='coerce').fillna(0)
    team_stats['Wins'] = pd.to_numeric(team_stats['Wins'], errors='coerce').fillna(0)

    team_totals = team_stats.groupby("Team Name").agg(
        Total_Calls=("Calls", "sum"),
        Total_Wins=("Wins", "sum")
    ).reset_index()
    team_totals['Conversion'] = (team_totals['Total_Wins'] / team_totals['Total_Calls']) * 100
    team_totals['Rank'] = team_totals['Conversion'].rank(ascending=False, method='min').astype(int)

    if team_name:
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




    # ---- Top 3 Reps (Conversion, or Wins if Calls unavailable)
    active_df = yesterday_df[yesterday_df['Calls'] > 0].copy()
    if active_df.empty:
        # Calls not yet available (T+1) — fall back to anyone with wins
        active_df = yesterday_df[yesterday_df['Wins'] > 0].copy()
        active_df['Conversion'] = 0.0
    else:
        active_df['Conversion'] = (active_df['Wins'] / active_df['Calls']) * 100
    active_df['Full_Name'] = active_df['First_Name'].astype(str).str.strip() + ' ' + active_df['Last_Name'].astype(str).str.strip()
    active_df = active_df.sort_values(by=['Conversion', 'Wins'], ascending=False)

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

    lt_display = yesterday_df[yesterday_df[attach_cols].sum(axis=1) > 0].copy()
    # Sort by total attaches descending
    lt_display['_total_attaches'] = lt_display[attach_cols].sum(axis=1)
    lt_display = lt_display.sort_values(by='_total_attaches', ascending=False)
    lt_display['Rank'] = range(1, len(lt_display) + 1)

    # Flag nan nan rows so we can identify them
    lt_display['_name_missing'] = (
        lt_display['First_Name'].astype(str).str.strip().isin(['', 'nan']) |
        lt_display['Last_Name'].astype(str).str.strip().isin(['', 'nan'])
    ) if 'First_Name' in lt_display.columns else False
    lt_display['Full_Name'] = lt_display.apply(
        lambda r: f"⚠️ {r['Rep']}" if r['_name_missing'] else r['Full_Name'], axis=1
    )

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

if page == "👩‍💻 Team Lead Dashboard":

    from pytz import timezone
    eastern = timezone('US/Eastern')
    cache_bust_key = datetime.now(eastern).strftime('%Y-%m-%d-%H-') + str(datetime.now(eastern).minute // 5 * 5).zfill(2)

    df = load_data(cache_bust_key).copy()
    history_df = load_history(cache_bust_key).copy()
    if 'Name_Proper' not in history_df.columns:
        history_df['Name_Proper'] = (
            history_df['First_Name'].astype(str).str.strip()
            + ' '
            + history_df['Last_Name'].astype(str).str.strip()
        ).str.strip()
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
            return fallback
    BASE_CONV   = coalesce_num(base_goals.get("Conversion"), 20.0)
    BASE_ATTACH = coalesce_num(base_goals.get("Attach"), 25.0)
    BASE_LT     = coalesce_num(base_goals.get("LT"), 5.5)
    BASE_QA     = coalesce_num(base_goals.get("QA"), 80.0)

    # ---- SELECT TEAM LEAD ----
    manager_directs = sorted(df['Manager_Direct'].dropna().unique())
    # Pre-select the logged-in TL's name if it matches a Manager_Direct entry
    _tl_default = 0
    if is_tl and not auth_row.empty:
        _auth_last = auth_row['Last_Name'].values[0].strip()
        _auth_first_name = auth_row['First_Name'].values[0].strip()
        _tl_name_fmt = f"{_auth_last}, {_auth_first_name}"  # "Lapuz, Mona"
        if _tl_name_fmt in manager_directs:
            _tl_default = manager_directs.index(_tl_name_fmt)
    selected_lead = st.selectbox("Select Your Name (Team Lead):", manager_directs, index=_tl_default)

    # Filter reps under selected team lead + always show Team ABC (new hires in training)
    _in_training = df['Team Name'].astype(str).str.strip().str.lower() == 'team abc'
    team_df = df[(df['Manager_Direct'] == selected_lead) | _in_training].copy()
    if 'Name_Proper' not in team_df.columns:
        if 'Full_Name' in team_df.columns:
            team_df['Name_Proper'] = team_df['Full_Name']
        elif 'First_Name' in team_df.columns and 'Last_Name' in team_df.columns:
            team_df['Name_Proper'] = (
                team_df['First_Name'].astype(str).str.strip()
                + ' '
                + team_df['Last_Name'].astype(str).str.strip()
            ).str.strip()
        else:
            team_df['Name_Proper'] = team_df['Rep'].astype(str)

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

    # Extract this team's rank
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
                    background-color: rgba(128,128,128,0.15); color: inherit; border: 1px solid rgba(128,128,128,0.4);'>
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

        # 🌱 Today's Team Averages — Styled Block
        st.markdown(f"""
        <div style='text-align: center; font-size: 18px; margin-top: 10px;'>
            <b>{team_name} — Today's Team Averages:</b><br>
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
                st.warning("No bonus just yet — but you're not far! Let's mow down those goals:")

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
                st.success("You're crushing it — your team is currently hitting **all 4 base goals** 💪 Time to rake in that bonus! 🌿💸")

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

        # ──────────────────────────────────────────────
        # 📋 TEAM QA BREAKDOWN
        # ──────────────────────────────────────────────
        st.subheader("📋 Team QA Breakdown")

        _tl_cache_bust = datetime.now(eastern).strftime("%Y-%m-%d-%H")
        with st.spinner("Loading QA data..."):
            _tl_qa_raw, _tl_qa_err = load_qa_data(_tl_cache_bust)

        if _tl_qa_raw.empty:
            st.warning("QA data unavailable — please try again shortly.")
            if _tl_qa_err:
                st.error(f"Error: {_tl_qa_err}")
        else:
            # Build set of canonical agent names for reps on this team
            _tl_teams_names = load_teams_new_names(_tl_cache_bust)
            _team_rep_names = set()
            for _, _tr in team_df.iterrows():
                _fn = str(_tr.get('First_Name', '')).strip()
                _ln = str(_tr.get('Last_Name', '')).strip()
                _rn = f"{_fn} {_ln}".strip()
                _canonical = _tl_teams_names.get(_rn.lower(), _rn)
                _team_rep_names.add(_canonical.lower())

            _tl_qa_df = _tl_qa_raw[
                _tl_qa_raw['Agent'].str.strip().str.lower().isin(_team_rep_names)
            ].copy()

            if _tl_qa_df.empty:
                st.info("No QA observations found for this team yet.")
            else:
                _score_col_tl = 'New Score' if 'New Score' in _tl_qa_df.columns else 'Score'
                _tl_qa_df['_month'] = _tl_qa_df['_scoring_week'].dt.to_period('M')
                _tl_avail = sorted(_tl_qa_df['_month'].dropna().unique(), reverse=True)

                _tl_sel_month_str = st.selectbox(
                    "📅 Select month",
                    [str(m) for m in _tl_avail],
                    key="tl_qa_month_select",
                )
                _tl_sel_period = pd.Period(_tl_sel_month_str, freq='M')
                _tl_month_df = _tl_qa_df[_tl_qa_df['_month'] == _tl_sel_period].copy()

                # Team-level score cards for selected month
                _tl_m_avg = _tl_month_df[_score_col_tl].dropna().mean()
                _tl_h1 = _tl_month_df[_tl_month_df['_scoring_week'].dt.day <= 15][_score_col_tl].dropna()
                _tl_h2 = _tl_month_df[_tl_month_df['_scoring_week'].dt.day > 15][_score_col_tl].dropna()

                _tc1, _tc2, _tc3 = st.columns(3)
                _tc1.metric("Team Monthly Avg",  f"{_tl_m_avg:.1f}%" if not pd.isna(_tl_m_avg) else "—")
                _tc2.metric("1st–15th Avg",      f"{_tl_h1.mean():.1f}%" if len(_tl_h1) > 0 else "—",
                            help="PIP EOR first-half period")
                _tc3.metric("16th–End Avg",      f"{_tl_h2.mean():.1f}%" if len(_tl_h2) > 0 else "—",
                            help="PIP EOR second-half period")

                # 3-month bi-monthly breakdown — team average
                st.markdown("##### 📊 Last 3 Months — Team Average")
                _tl_recent_3 = sorted(_tl_avail, reverse=True)[:3]
                _tl_breakdown = []
                for _tp in _tl_recent_3:
                    _tmd = _tl_qa_df[_tl_qa_df['_month'] == _tp]
                    _tbh1 = _tmd[_tmd['_scoring_week'].dt.day <= 15][_score_col_tl].dropna().mean()
                    _tbh2 = _tmd[_tmd['_scoring_week'].dt.day > 15][_score_col_tl].dropna().mean()
                    _tbma = _tmd[_score_col_tl].dropna().mean()
                    _tl_breakdown.append({
                        "Month":        str(_tp),
                        "1st–15th Avg": f"{_tbh1:.1f}%" if not pd.isna(_tbh1) else "—",
                        "16th–End Avg": f"{_tbh2:.1f}%" if not pd.isna(_tbh2) else "—",
                        "Monthly Avg":  f"{_tbma:.1f}%" if not pd.isna(_tbma) else "—",
                        "# Obs":        int(_tmd[_score_col_tl].notna().sum()),
                    })
                st.dataframe(pd.DataFrame(_tl_breakdown), use_container_width=True, hide_index=True)

                # Per-rep breakdown for selected month
                st.markdown("##### 👤 Per-Rep Breakdown")
                _rep_rows = []
                for _agent_name in sorted(_tl_month_df['Agent'].dropna().unique()):
                    _rep_df = _tl_month_df[_tl_month_df['Agent'] == _agent_name]
                    _rh1 = _rep_df[_rep_df['_scoring_week'].dt.day <= 15][_score_col_tl].dropna()
                    _rh2 = _rep_df[_rep_df['_scoring_week'].dt.day > 15][_score_col_tl].dropna()
                    _rma = _rep_df[_score_col_tl].dropna().mean()
                    _rep_rows.append({
                        "Agent":        _agent_name,
                        "1st–15th Avg": f"{_rh1.mean():.1f}%" if len(_rh1) > 0 else "—",
                        "16th–End Avg": f"{_rh2.mean():.1f}%" if len(_rh2) > 0 else "—",
                        "Monthly Avg":  f"{_rma:.1f}%" if not pd.isna(_rma) else "—",
                        "# Obs":        len(_rep_df),
                    })
                if _rep_rows:
                    st.dataframe(pd.DataFrame(_rep_rows), use_container_width=True, hide_index=True)

        # Placeholder Sections
        st.subheader("🏆 Personal Bests (Coming Soon)")
        st.subheader("⚔️ Compare With Another Team (Coming Soon)")

# ----------------------------
# 🏆 TAB 6: Senior Manager View
# ----------------------------
if page == "Senior Manager View":
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
    cache_bust_key = datetime.now(eastern).strftime("%Y-%m-%d-%H-") + str(datetime.now(eastern).minute // 5 * 5).zfill(2)
    df = load_data(cache_bust_key).copy()
    history_df = load_history(cache_bust_key).copy()
    if 'Name_Proper' not in history_df.columns:
        history_df['Name_Proper'] = (
            history_df['First_Name'].astype(str).str.strip()
            + ' '
            + history_df['Last_Name'].astype(str).str.strip()
        ).str.strip()

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

# ---------------------------------------------------------------------------
# 📋 MY QA PAGE
# ---------------------------------------------------------------------------
_QA_SHEET_ID  = "1Dt7D2nJLmmWyVc39Vss-2nySewiC3pcObRHPDxQ4ads"
_QA_SHEET_TAB = "Sales"

_QA_RUBRIC_TO_COMMENT = {
    "Greeting/Closing":     "Call Flow Comments",
    "Acknowledgement":      "Call Flow Comments",
    "Intro Probing":        "Call Flow Comments",
    "Address Verification": "Call Flow Comments",
    "Protocol":             "Call Flow Comments",
    "Rapport":              "Communication Comments",
    "Professionalism":      "Communication Comments",
    "Presentation":         "Communication Comments",
    "Dead Air":             "Communication Comments",
    "3 Cut Minimum":        "Policy Comments",
    "Long Grass Fee":       "Policy Comments",
    "48 Hour Policy":       "Policy Comments",
    "Proper Rebuttal":      "Objections Comments",
    "Address Accuracy":     "Setup Comments",
    "Contact Info":         "Setup Comments",
    "Service Set Up":       "Setup Comments",
    "Customer Name":        "Setup Comments",
}

@st.cache_data(show_spinner=False, ttl=3600)
def _get_coaching_tip(_cache_key: str, rubric_label: str, comments: tuple) -> str:
    try:
        import anthropic as _anthropic
        _comment_lines = "\n".join(f"- {c}" for c in comments if str(c).strip() not in ("", "nan"))
        if not _comment_lines:
            return ""
        _api_key = (
            st.secrets.get("ANTHROPIC_API_KEY")
            or st.secrets.get("anthropic", {}).get("ANTHROPIC_API_KEY")
            or st.secrets.get("gmail", {}).get("ANTHROPIC_API_KEY")
        )
        if not _api_key:
            return "[error: ANTHROPIC_API_KEY not found in secrets — add it at the TOP of secrets.toml, before any [section] headers]"
        _client = _anthropic.Anthropic(api_key=_api_key)
        _msg = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=220,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a high-performance sales coach for LawnStarter, a residential lawn care company. "
                    f"A sales rep is consistently missing the \"{rubric_label}\" rubric item on QA evaluations.\n\n"
                    f"QA comments from their failed observations:\n{_comment_lines}\n\n"
                    f"Write 2-3 sentences of direct coaching. Requirements:\n"
                    f"- Open by acknowledging what they're already doing in the right direction — never start negative\n"
                    f"- Then pivot to the ONE specific thing that will take them to the next level on this item\n"
                    f"- Assumptive selling frame: the sale is already happening — coach on HOW to close, never IF\n"
                    f"- Pull specific patterns from the comments above, don't be generic\n"
                    f"- Give at least one concrete example phrase or technique they can use on their very next call\n"
                    f"- Tone: encouraging and motivating, like a coach who genuinely believes in them and wants them to win\n"
                    f"No bullet points. One coaching paragraph only."
                ),
            }],
        )
        return _msg.content[0].text.strip()
    except Exception as _e:
        return f"[error: {_e}]"

_QA_RUBRIC_COLS = {
    "Greeting/Closing":     "Call Flow - 10 Pts [Greeting/Closing]",
    "Acknowledgement":      "Call Flow - 10 Pts [Acknowledgement]",
    "Intro Probing":        "Call Flow - 10 Pts [Intro Probing]",
    "Address Verification": "Call Flow - 10 Pts [Address Verification]",
    "Protocol":             "Call Flow - 10 Pts [Protocol]",
    "Rapport":              "Communication - 25 [Rapport]",
    "Professionalism":      "Communication - 25 [Professionalism]",
    "Presentation":         "Communication - 25 [Presentation]",
    "Dead Air":             "Communication - 25 [Dead Air]",
    "3 Cut Minimum":        "Policy - 25 Pts [3 Cut Minimum]",
    "Long Grass Fee":       "Policy - 25 Pts [Long Grass Fee]",
    "48 Hour Policy":       "Policy - 25 Pts [48 Hour Policy]",
    "Proper Rebuttal":      "Handling Objections (Loss Only) - 20 [Proper Rebuttal]",
    "Address Accuracy":     "Admin Setup (Win Only) - 20 Pts [Address Accuracy]",
    "Contact Info":         "Admin Setup (Win Only) - 20 Pts [Contact Info]",
    "Service Set Up":       "Admin Setup (Win Only) - 20 Pts [Service Set Up]",
    "Customer Name":        "Admin Setup (Win Only) - 20 Pts [Customer Name]",
}

_QA_DISPLAY_COLS = [
    "Timestamp", "Scoring Week", "Call ID", "Team Lead", "Type", "Win/Loss",
    "Call Flow Score", "Call Flow Comments",
    "Communication Score", "Communication Comments",
    "Policy Score", "Policy Comments",
    "Objections Score", "Objections Comments",
    "Sales Score", "Sales Comments",
    "Setup Score", "Setup Comments",
    "QA Observation", "Overall Experience", "Overall Experience Comments",
    "Critical Deductions", "Critical Comments",
    "New Score",
]

_TEAMS_NEW_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1sO4ZDe-n8-ugc-OsoDisHPXXEAap6b2iINcjsWXz5gU"
    "/export?format=csv&gid=4048477"
)

@st.cache_data(show_spinner=False, ttl=300)
def load_teams_new_names(_cache_bust_key: str) -> dict:
    try:
        _tn = pd.read_csv(_TEAMS_NEW_URL, header=0)
        _tn.columns = _tn.columns.str.strip()
        _lookup = {}
        for _, _r in _tn.iterrows():
            _fn = str(_r.get('First_Name', '')).strip()
            _ln = str(_r.get('Last_Name', '')).strip()
            _agent = str(_r.get('Agent', '')).strip()
            if _fn and _ln and _agent:
                _lookup[f"{_fn} {_ln}".lower()] = _agent
        return _lookup
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=300)
def load_qa_data(_cache_bust_key: str) -> tuple:
    try:
        from google.oauth2.service_account import Credentials as SACredentials
        from googleapiclient.discovery import build as gapi_build

        _sa = dict(st.secrets["gcp_service_account"])
        _creds = SACredentials.from_service_account_info(
            _sa,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        _svc = gapi_build("sheets", "v4", credentials=_creds, cache_discovery=False)
        _result = _svc.spreadsheets().values().get(
            spreadsheetId=_QA_SHEET_ID,
            range=_QA_SHEET_TAB,
        ).execute()
        _values = _result.get("values", [])
        if len(_values) < 2:
            return pd.DataFrame(), "Sheet returned no data rows"
        _headers = _values[0]
        _rows = [r + [""] * (len(_headers) - len(r)) for r in _values[1:]]
        df = pd.DataFrame(_rows, columns=_headers)
        df.columns = [re.sub(r'\s+', ' ', c).strip() for c in df.columns]
        df['_ts'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['_scoring_week'] = pd.to_datetime(df['Scoring Week'], errors='coerce')
        for col in ['New Score', 'Score', 'Old Score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

if page == "📋 My QA":
    _cache_bust = datetime.now(eastern).strftime("%Y-%m-%d-%H")

    with st.spinner("Loading QA data..."):
        _qa_raw, _qa_err = load_qa_data(_cache_bust)

    if _qa_raw.empty:
        st.warning(f"QA data unavailable — please try again shortly.")
        if _qa_err:
            st.error(f"Error: {_qa_err}")
        st.stop()

    # Build viewed rep's full name using Teams New as source of truth
    _teams_names = load_teams_new_names(_cache_bust)
    _vr = roster_auth[roster_auth['rep_key'] == viewed_email]
    if not _vr.empty:
        _vfn = _safe(_vr['First_Name'].values[0])
        _vln = _safe(_vr['Last_Name'].values[0])
        _roster_name = f"{_vfn} {_vln}".strip()
    else:
        _parts = viewed_email.split('@')[0].split('.')
        _roster_name = ' '.join(p.title() for p in _parts)
    _viewed_full = _teams_names.get(_roster_name.lower(), _roster_name)

    if 'Agent' not in _qa_raw.columns:
        st.error("Unexpected QA sheet format — 'Agent' column not found.")
        st.stop()

    _agent_df = _qa_raw[
        _qa_raw['Agent'].str.strip().str.lower() == _viewed_full.lower()
    ].copy()
    _agent_df = _agent_df.drop(columns=['Email Address'], errors='ignore')

    st.header(f"📋 QA Feedback — {_viewed_full}")

    if _agent_df.empty:
        st.info(f"No QA observations found for **{_viewed_full}**.")
        st.stop()

    st.caption(f"{len(_agent_df)} total observations on record")

    # Month selector
    _agent_df['_month'] = _agent_df['_scoring_week'].dt.to_period('M')
    _avail = sorted(_agent_df['_month'].dropna().unique(), reverse=True)
    if not _avail:
        st.warning("No dated observations found.")
        st.stop()

    _sel_month_str = st.selectbox(
        "📅 Select month",
        [str(m) for m in _avail],
        key="qa_month_select",
    )
    _sel_period = pd.Period(_sel_month_str, freq='M')
    _month_df = _agent_df[_agent_df['_month'] == _sel_period].copy()

    _score_col = 'New Score' if 'New Score' in _agent_df.columns else 'Score'

    # ── Score summary cards ──────────────────────────────────────────────────
    _m_avg  = _month_df[_score_col].dropna().mean()
    _h1     = _month_df[_month_df['_scoring_week'].dt.day <= 15][_score_col].dropna()
    _h2     = _month_df[_month_df['_scoring_week'].dt.day > 15][_score_col].dropna()

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Monthly QA Avg",  f"{_m_avg:.1f}%" if not pd.isna(_m_avg) else "—")
    _c2.metric("1st–15th Avg",    f"{_h1.mean():.1f}%" if len(_h1) > 0 else "—",
               help="PIP EOR first-half period")
    _c3.metric("16th–End Avg",    f"{_h2.mean():.1f}%" if len(_h2) > 0 else "—",
               help="PIP EOR second-half period")

    # ── 3-month bi-monthly breakdown ────────────────────────────────────────
    st.subheader("📊 QA Score Breakdown — Last 3 Months")
    _recent_3 = sorted(_avail, reverse=True)[:3]
    _breakdown_rows = []
    for _p in _recent_3:
        _md = _agent_df[_agent_df['_month'] == _p]
        _bh1 = _md[_md['_scoring_week'].dt.day <= 15][_score_col].dropna().mean()
        _bh2 = _md[_md['_scoring_week'].dt.day > 15][_score_col].dropna().mean()
        _bma = _md[_score_col].dropna().mean()
        _breakdown_rows.append({
            "Month":         str(_p),
            "1st–15th Avg":  f"{_bh1:.1f}%" if not pd.isna(_bh1) else "—",
            "16th–End Avg":  f"{_bh2:.1f}%" if not pd.isna(_bh2) else "—",
            "Monthly Avg":   f"{_bma:.1f}%" if not pd.isna(_bma) else "—",
        })
    st.dataframe(pd.DataFrame(_breakdown_rows), use_container_width=True, hide_index=True)

    # ── Kudos & improvement (all available data) ─────────────────────────────
    st.subheader("🌟 Strengths & Opportunities")
    st.caption("Based on all observations on record — minimum 5 applicable calls required per area.")

    _kudos, _improve = [], []
    for _label, _raw_col in _QA_RUBRIC_COLS.items():
        _norm_col = re.sub(r'\s+', ' ', _raw_col).strip()
        if _norm_col not in _agent_df.columns:
            continue
        _vals = _agent_df[_norm_col].astype(str).str.strip().str.lower()
        _applicable = _vals[_vals.isin(['yes', 'no'])]
        if len(_applicable) < 5:
            continue
        _rate = (_applicable == 'yes').sum() / len(_applicable)
        if _rate >= 0.90:
            _kudos.append((_label, _rate))
        elif _rate <= 0.70:
            _improve.append((_label, _rate))

    _kc, _ic = st.columns(2)
    with _kc:
        st.markdown("**🏆 Consistently Crushing**")
        if _kudos:
            for _lbl, _r in sorted(_kudos, key=lambda x: -x[1]):
                st.success(f"✅ {_lbl} — {_r*100:.0f}%")
        else:
            st.caption("Not enough data yet to identify top strengths.")
    with _ic:
        st.markdown("**💡 Opportunities to Improve**")
        if _improve:
            for _lbl, _r in sorted(_improve, key=lambda x: x[1]):
                st.warning(f"⚠️ {_lbl} — {_r*100:.0f}%")
                _comment_col = _QA_RUBRIC_TO_COMMENT.get(_lbl)
                _norm_comment_col = re.sub(r'\s+', ' ', _comment_col).strip() if _comment_col else None
                if _norm_comment_col and _norm_comment_col in _agent_df.columns:
                    _raw_col = _QA_RUBRIC_COLS[_lbl]
                    _norm_rubric_col = re.sub(r'\s+', ' ', _raw_col).strip()
                    _failed_rows = _agent_df[
                        _agent_df[_norm_rubric_col].astype(str).str.strip().str.lower() == 'no'
                    ] if _norm_rubric_col in _agent_df.columns else _agent_df
                    _comments = tuple(
                        _failed_rows[_norm_comment_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .loc[lambda s: s.str.lower() != 'nan']
                        .unique()
                        .tolist()
                    )
                    if _comments:
                        _tip_key = f"{_viewed_full}|{_lbl}|{hash(_comments)}"
                        with st.spinner(f"Getting coaching tip for {_lbl}..."):
                            _tip = _get_coaching_tip(_tip_key, _lbl, _comments)
                        if _tip:
                            st.info(f"💬 **Coach:** {_tip}")
                        else:
                            st.caption("⏳ Coaching tip unavailable — check that ANTHROPIC_API_KEY is set in secrets.")
        else:
            st.caption("No consistent gaps identified — keep it up!")

    # ── Observations detail table ────────────────────────────────────────────
    st.subheader(f"📝 Observations — {_sel_month_str}")
    st.caption(f"{len(_month_df)} observation(s) this month")

    _show_cols = [c for c in _QA_DISPLAY_COLS if c in _month_df.columns]
    _show_df = _month_df[_show_cols].sort_values('Timestamp', ascending=False) \
        if 'Timestamp' in _show_cols else _month_df[_show_cols]

    _comment_col_set = {c for c in _show_cols if 'comment' in c.lower() or 'observation' in c.lower()}

    _tbl_rows = ""
    for _, _row in _show_df.iterrows():
        _tbl_rows += "<tr>"
        for _c in _show_cols:
            _v = "" if pd.isna(_row[_c]) or str(_row[_c]).lower() == "nan" else str(_row[_c])
            if _c in _comment_col_set:
                _tbl_rows += f'<td style="padding:8px 10px;border:1px solid #444;white-space:pre-wrap;word-break:break-word;min-width:180px;max-width:320px;vertical-align:top;">{_v}</td>'
            else:
                _tbl_rows += f'<td style="padding:8px 10px;border:1px solid #444;white-space:nowrap;vertical-align:top;">{_v}</td>'
        _tbl_rows += "</tr>"

    _tbl_headers = "".join(
        f'<th style="padding:8px 10px;border:1px solid #555;background:#1e1e2e;text-align:left;white-space:nowrap;">{_c}</th>'
        for _c in _show_cols
    )
    st.markdown(
        f'<div style="overflow-x:auto;font-size:13px;">'
        f'<table style="border-collapse:collapse;width:100%;">'
        f'<thead><tr>{_tbl_headers}</tr></thead>'
        f'<tbody>{_tbl_rows}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


