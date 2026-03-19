#!/usr/bin/env python3
"""
rep_history_sync.py
====================
Pulls daily per-rep metrics from Redshift and writes to data/rep_history.csv.

Metrics pulled:
  - Wins           → bi_lawnstarter_production.orders (closing_user_id)
  - Lawn Treatment → dw_silver.fct_schedules + orders join
  - Mosquito       → dw_silver.fct_schedules + orders join
  - Bush Trimming  → bi_lawnstarter_production.instant_quotes (creating_user_id)
  - Flower Bed Weeding → bi_lawnstarter_production.instant_quotes
  - Leaf Removal   → bi_lawnstarter_production.instant_quotes
  - Pool           → five9.five_9_call_logs_v_3 (T+1 only)
  - Calls          → five9.five_9_call_logs_v_3

Prerequisites:
  - Teleport proxy running on port 55756:
      tsh proxy db --db-user=analytics_read_only --db-name=dev --tunnel --port 55756 bi-test &
  - psycopg2-binary installed: pip install psycopg2-binary

Usage:
  python scripts/rep_history_sync.py            # full historical sync
  python scripts/rep_history_sync.py --today    # today + yesterday only (fast, for cron)
"""

import argparse
import os
import sys
import psycopg2
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REDSHIFT_HOST = "localhost"
REDSHIFT_PORT = 55756
REDSHIFT_DBNAME = "dev"
REDSHIFT_USER = "analytics_read_only"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "rep_history.csv")

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def get_conn():
    return psycopg2.connect(
        host=REDSHIFT_HOST,
        port=REDSHIFT_PORT,
        dbname=REDSHIFT_DBNAME,
        user=REDSHIFT_USER,
        connect_timeout=10,
    )


def run_query(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, conn)


# ---------------------------------------------------------------------------
# SQL Queries
# ---------------------------------------------------------------------------

WINS_SQL = """
SELECT
    CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', o.created_at) AS DATE) AS date,
    u.email                             AS rep,
    u.fname                             AS first_name,
    u.lname                             AS last_name,
    COUNT(DISTINCT o.id)                AS wins
FROM bi_lawnstarter_production.orders o
JOIN bi_lawnstarter_production.users  u ON o.closing_user_id = u.id
WHERE u.is_admin = 1
  AND u.email ILIKE '%@lawnstarter.com%'
  AND o._fivetran_deleted = false
  AND o.closing_user_id IS NOT NULL
  {date_filter}
GROUP BY 1, 2, 3, 4
"""

SCHEDULE_ATTACHES_SQL = """
SELECT
    CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', fs.created_at) AS DATE) AS date,
    u.email                             AS rep,
    SUM(CASE WHEN fs.service_name ILIKE '%Lawn Treatment%'  THEN 1 ELSE 0 END) AS lawn_treatment,
    SUM(CASE WHEN fs.service_name ILIKE '%Mosquito%'        THEN 1 ELSE 0 END) AS mosquito
FROM dw_silver.fct_schedules fs
JOIN bi_lawnstarter_production.orders o ON fs.order_id = o.id
JOIN bi_lawnstarter_production.users  u ON o.closing_user_id = u.id
WHERE u.is_admin = 1
  AND u.email ILIKE '%@lawnstarter.com%'
  AND fs.order_id IS NOT NULL
  {date_filter}
GROUP BY 1, 2
HAVING SUM(CASE WHEN fs.service_name ILIKE '%Lawn Treatment%' THEN 1 ELSE 0 END) > 0
    OR SUM(CASE WHEN fs.service_name ILIKE '%Mosquito%'        THEN 1 ELSE 0 END) > 0
"""

IQ_ATTACHES_SQL = """
SELECT
    CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', iq.created_at) AS DATE) AS date,
    u.email                             AS rep,
    SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteBushTrimming'   THEN 1 ELSE 0 END) AS bush_trimming,
    SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteFlowerBedWeeding' THEN 1 ELSE 0 END) AS flower_bed_weeding,
    SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteLeafRemoval'    THEN 1 ELSE 0 END) AS leaf_removal
FROM bi_lawnstarter_production.instant_quotes iq
JOIN bi_lawnstarter_production.users          u ON iq.creating_user_id = u.id
WHERE iq.instant_quote_status IN ('completed', 'contractor.accepted', 'contractor.pending')
  AND iq._fivetran_deleted = false
  AND u.is_admin = 1
  AND u.email ILIKE '%@lawnstarter.com%'
  {date_filter}
GROUP BY 1, 2
HAVING SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteBushTrimming'     THEN 1 ELSE 0 END) > 0
    OR SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteFlowerBedWeeding' THEN 1 ELSE 0 END) > 0
    OR SUM(CASE WHEN iq.instant_quotable_type = 'App\\\\InstantQuoteLeafRemoval'      THEN 1 ELSE 0 END) > 0
"""

POOL_SQL = """
SELECT
    CAST(REPLACE("date", '/', '-') AS DATE)  AS date,
    agent_email                               AS rep,
    COUNT(DISTINCT call_id)                   AS pool
FROM five9.five_9_call_logs_v_3
WHERE disposition IN ('Pool Closed Won', 'EC Pool')
  AND agent_email ILIKE '%@lawnstarter.com%'
  AND agent != '[None]'
  {date_filter}
GROUP BY 1, 2
"""

CALLS_SQL = """
SELECT
    CAST(REPLACE("date", '/', '-') AS DATE)  AS date,
    agent_email                               AS rep,
    COUNT(DISTINCT call_id)                   AS calls
FROM five9.five_9_call_logs_v_3
WHERE agent_email ILIKE '%@lawnstarter.com%'
  AND agent != '[None]'
  AND (disposition NOT IN ('Support Call', 'Call in Support', 'Test', 'Internal Call', 'Transferred To 3rd Party', 'Provider Inquiry')
       OR disposition IS NULL)
  {date_filter}
GROUP BY 1, 2
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def date_filter_clause(start_date: date | None, end_date: date | None,
                        col: str = "CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', created_at) AS DATE)") -> str:
    """Return an AND clause for date filtering in CST, or empty string."""
    parts = []
    if start_date:
        parts.append(f"AND {col} >= '{start_date}'")
    if end_date:
        parts.append(f"AND {col} <= '{end_date}'")
    return " ".join(parts)


def date_filter_five9(start_date: date | None, end_date: date | None) -> str:
    parts = []
    if start_date:
        parts.append(f"AND CAST(REPLACE(\"date\", '/', '-') AS DATE) >= '{start_date}'")
    if end_date:
        parts.append(f"AND CAST(REPLACE(\"date\", '/', '-') AS DATE) <= '{end_date}'")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# RepData auto-sync from SalesRoster
# ---------------------------------------------------------------------------

def _rowcol_to_a1(row, col):
    """Convert 1-indexed row/col to A1 notation."""
    col_str = ''
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        col_str = chr(65 + rem) + col_str
    return f"{col_str}{row}"


def update_repdata_from_roster(sr, sh):
    """
    Ensures every SalesRoster rep has a row in RepData.
    - Looks up Team Name via TL_TeamMap (Manager_Direct → Team_Name)
    - Backfills Birthday / Start_Date from BirthdayImport tab if present
    Skips silently if TL_TeamMap doesn't exist yet.
    """
    import gspread as _gspread

    # --- TL_TeamMap: Manager_Direct → Team_Name ---
    tl_map = {}
    try:
        tl_ws = sh.worksheet("TL_TeamMap")
        for row in tl_ws.get_all_records():
            mgr = str(row.get('Manager_Direct', '')).strip()
            team = str(row.get('Team_Name', '')).strip()
            if mgr:
                tl_map[mgr] = team
        print(f"  TL_TeamMap: {len(tl_map)} entries loaded")
    except _gspread.exceptions.WorksheetNotFound:
        print("  TL_TeamMap tab not found — Team Name will be blank for new reps")

    # --- BirthdayImport: Email → Birthday, Start_Date ---
    birthday_map = {}
    start_map = {}
    try:
        bday_ws = sh.worksheet("BirthdayImport")
        for row in bday_ws.get_all_records():
            email = str(row.get('Email', '')).lower().strip()
            if email:
                birthday_map[email] = str(row.get('Birthday', '')).strip()
                start_map[email] = str(row.get('Start_Date', '')).strip()
        print(f"  BirthdayImport: {len(birthday_map)} entries loaded")
    except _gspread.exceptions.WorksheetNotFound:
        pass  # Tab not created yet — skip birthday backfill

    # --- Read RepData via gspread ---
    rd_ws = sh.worksheet("RepData")
    all_vals = rd_ws.get_all_values()

    # RepData: row 0 = title row, row 1 = column headers, row 2+ = data
    header_row_idx = None
    for i, row in enumerate(all_vals):
        if 'Rep' in row:
            header_row_idx = i
            break
    if header_row_idx is None:
        print("  Warning: RepData 'Rep' column not found — skipping RepData sync")
        return

    headers = all_vals[header_row_idx]
    col = {h.strip(): i for i, h in enumerate(headers) if h.strip()}
    data_rows = all_vals[header_row_idx + 1:]
    rep_idx = col.get('Rep')
    if rep_idx is None:
        return

    existing_emails = {
        row[rep_idx].lower().strip()
        for row in data_rows
        if len(row) > rep_idx and row[rep_idx].strip()
    }

    # --- Append missing reps ---
    missing = sr[~sr['rep_key'].isin(existing_emails)].copy()
    if not missing.empty:
        print(f"  RepData: adding {len(missing)} new rep(s)")
        new_rows = []
        for _, rep in missing.iterrows():
            email = rep['rep_key']
            mgr = str(rep.get('Manager_Direct', '')).strip()
            new_row = [''] * len(headers)
            for field, val in [
                ('Rep',            email),
                ('First_Name',     rep.get('First_Name', '')),
                ('Last_Name',      rep.get('Last_Name', '')),
                ('Manager_Direct', mgr),
                ('Team Name',      tl_map.get(mgr, '')),
                ('Birthday',       birthday_map.get(email, '')),
                ('Start_Date',     start_map.get(email, '')),
            ]:
                if field in col:
                    new_row[col[field]] = str(val)
            new_rows.append(new_row)
        rd_ws.append_rows(new_rows, value_input_option='USER_ENTERED')
        print(f"  RepData: {len(new_rows)} rep(s) added")
    else:
        print("  RepData: all SalesRoster reps already present")

    # --- Backfill Birthday / Start_Date for existing reps missing them ---
    if birthday_map:
        updates = []
        for i, row in enumerate(data_rows):
            if len(row) <= rep_idx:
                continue
            email = row[rep_idx].lower().strip()
            if not email or email not in birthday_map:
                continue
            sheet_row = header_row_idx + 1 + i + 1  # 1-indexed sheet row
            for field, val_map in [('Birthday', birthday_map), ('Start_Date', start_map)]:
                if field not in col:
                    continue
                cidx = col[field]
                current = row[cidx] if len(row) > cidx else ''
                new_val = val_map.get(email, '')
                if not current.strip() and new_val:
                    updates.append({
                        'range': _rowcol_to_a1(sheet_row, cidx + 1),
                        'values': [[new_val]],
                    })
        if updates:
            rd_ws.batch_update(updates)
            print(f"  RepData: backfilled {len(updates)} birthday/start date(s)")


# ---------------------------------------------------------------------------
# Main sync
# ---------------------------------------------------------------------------

def sync(today_only: bool = False):
    print("Connecting to Redshift...")
    try:
        conn = get_conn()
    except Exception as e:
        print(f"ERROR: Could not connect to Redshift on port {REDSHIFT_PORT}.")
        print("Make sure Teleport proxy is running:")
        print("  tsh proxy db --db-user=analytics_read_only --db-name=dev --tunnel --port 55756 bi-test &")
        print(f"Details: {e}")
        sys.exit(1)

    # Date range
    today = date.today()
    yesterday = today - timedelta(days=1)
    if today_only:
        # Pull today + yesterday (covers both real-time and Five9 T+1)
        start = yesterday
        end = today
        print(f"Pulling data for {start} – {end} (today-only mode)")
    else:
        start = None  # full history
        end = None
        print("Pulling full historical data (this may take 1-2 minutes)...")

    # --- Wins ---
    print("  Fetching Wins...")
    wins_sql = WINS_SQL.format(date_filter=date_filter_clause(start, end, col="CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', o.created_at) AS DATE)"))
    wins_df = run_query(conn, wins_sql)

    # --- Schedule attaches (LT + Mosquito) ---
    print("  Fetching Lawn Treatment + Mosquito...")
    sched_sql = SCHEDULE_ATTACHES_SQL.format(date_filter=date_filter_clause(start, end, col="CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', fs.created_at) AS DATE)"))
    sched_df = run_query(conn, sched_sql)

    # --- IQ attaches (Bush, Flower, Leaf) ---
    print("  Fetching IQ attaches (Bush, Flower, Leaf)...")
    iq_sql = IQ_ATTACHES_SQL.format(date_filter=date_filter_clause(start, end, col="CAST(CONVERT_TIMEZONE('UTC', 'America/Chicago', iq.created_at) AS DATE)"))
    iq_df = run_query(conn, iq_sql)

    # --- Pool (Five9, T+1) ---
    print("  Fetching Pool (Five9)...")
    # Pool only available through yesterday
    pool_end = yesterday
    pool_sql = POOL_SQL.format(date_filter=date_filter_five9(start, pool_end))
    pool_df = run_query(conn, pool_sql)

    # --- Calls (Five9) ---
    print("  Fetching Calls (Five9)...")
    calls_sql = CALLS_SQL.format(date_filter=date_filter_five9(start, pool_end))
    calls_df = run_query(conn, calls_sql)

    conn.close()

    # ---------------------------------------------------------------------------
    # Load roster data from two tabs:
    #   SalesRoster (gid=664880618) → sales-only allowlist + names
    #   RepData     (gid=171451260) → Team Name + Manager_Direct assignments
    # ---------------------------------------------------------------------------
    print("  Fetching team roster from Google Sheet...")
    SALES_ROSTER_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=664880618"
    )
    REPDATA_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=171451260"
    )
    sr = pd.DataFrame(columns=['rep_key', 'First_Name', 'Last_Name', 'Manager_Direct'])
    try:
        # --- SalesRoster: allowlist + names ---
        sr_raw = pd.read_csv(SALES_ROSTER_URL, header=0)
        sr_raw.columns = sr_raw.columns.str.strip()
        sr = sr_raw[sr_raw['Lawnstarter_Email'].notna()].copy()
        sr = sr[sr['Lawnstarter_Email'].astype(str).str.strip() != '']
        sr['rep_key'] = sr['Lawnstarter_Email'].str.lower().str.strip()
        sr['First_Name'] = sr['First_Name'].astype(str).str.strip()
        sr['Last_Name'] = sr['Last_Name'].astype(str).str.strip()
        sr['Manager_Direct'] = sr['Manager_Direct'].astype(str).str.strip() if 'Manager_Direct' in sr.columns else ''
        sales_emails = set(sr['rep_key'])
        name_map = sr[['rep_key', 'First_Name', 'Last_Name']].drop_duplicates('rep_key')
        print(f"  SalesRoster: {len(sales_emails)} sales reps (allowlist)")

        # --- RepData: Team Name + Manager_Direct ---
        rd_raw = pd.read_csv(REPDATA_URL, header=None)
        rd_raw.columns = rd_raw.iloc[1]
        rd_raw = rd_raw.iloc[2:].reset_index(drop=True)
        rd_raw = rd_raw.loc[:, rd_raw.columns.notna() & (rd_raw.columns.astype(str).str.strip() != 'nan')]
        rd_raw.columns = rd_raw.columns.str.strip()
        rd = rd_raw[rd_raw['Rep'].notna() & (rd_raw['Rep'].astype(str).str.strip() != '')].copy()
        rd['rep_key'] = rd['Rep'].str.lower().str.strip()
        rd['Team Name'] = rd['Team Name'].astype(str).str.strip().replace({'nan': None, '': None})
        team_map = rd[['rep_key', 'Team Name', 'Manager_Direct']].drop_duplicates('rep_key')
        print(f"  RepData: {len(team_map)} reps with team assignments")

        # Build mgr → team fallback for reps missing a team in RepData
        has_team = team_map[team_map['Team Name'].notna()]
        mgr_to_team = (
            has_team.groupby('Manager_Direct')['Team Name']
            .agg(lambda x: x.mode()[0])
            .to_dict()
        )

        # Combine: merge names + team into one roster_map
        roster_map = name_map.merge(team_map, on='rep_key', how='left')
        roster_map['Team Name'] = roster_map['Team Name'].fillna(
            roster_map['Manager_Direct'].map(mgr_to_team)
        )

    except Exception as e:
        print(f"  Warning: could not load roster ({e}). Team Name will be 'Unknown', no name fallback.")
        roster_map = pd.DataFrame(columns=['rep_key', 'First_Name', 'Last_Name', 'Team Name', 'Manager_Direct'])
        sales_emails = set()

    # ---------------------------------------------------------------------------
    # Build rep×date spine from all sources (union, not just wins)
    # ---------------------------------------------------------------------------
    print("  Merging datasets...")

    # Normalize email to lowercase
    for df in [wins_df, sched_df, iq_df, pool_df, calls_df]:
        if 'rep' in df.columns:
            df['rep'] = df['rep'].str.lower().str.strip()

    # Rename all dataframes before merging
    wins_df = wins_df.rename(columns={
        'date': 'Date',
        'rep': 'Rep',
        'first_name': 'First_Name',
        'last_name': 'Last_Name',
        'wins': 'Wins',
    })

    sched_df = sched_df.rename(columns={
        'date': 'Date',
        'rep': 'Rep',
        'lawn_treatment': 'Lawn Treatment',
        'mosquito': 'Mosquito',
    })

    iq_df = iq_df.rename(columns={
        'date': 'Date',
        'rep': 'Rep',
        'bush_trimming': 'Bush Trimming',
        'flower_bed_weeding': 'Flower Bed Weeding',
        'leaf_removal': 'Leaf Removal',
    })

    pool_df = pool_df.rename(columns={'date': 'Date', 'rep': 'Rep', 'pool': 'Pool'})
    calls_df = calls_df.rename(columns={'date': 'Date', 'rep': 'Rep', 'calls': 'Calls'})

    # Build spine as union of all rep/date combos across all sources
    spine_parts = []
    for df, name_col in [
        (wins_df, None),
        (sched_df, None),
        (iq_df, None),
        (pool_df, None),
        (calls_df, None),
    ]:
        if not df.empty:
            spine_parts.append(df[['Date', 'Rep']].drop_duplicates())
    spine = pd.concat(spine_parts, ignore_index=True).drop_duplicates(['Date', 'Rep'])

    # First_Name / Last_Name: prefer wins query, fall back to roster
    name_df = wins_df[['Date', 'Rep', 'First_Name', 'Last_Name']].drop_duplicates(['Date', 'Rep'])
    spine = spine.merge(name_df, on=['Date', 'Rep'], how='left')
    # Fill missing names from roster
    if 'First_Name' in roster_map.columns and 'Last_Name' in roster_map.columns:
        name_fill = roster_map[['rep_key', 'First_Name', 'Last_Name']].copy()
        name_fill = name_fill.rename(columns={'rep_key': 'Rep', 'First_Name': '_fn', 'Last_Name': '_ln'})
        spine = spine.merge(name_fill, on='Rep', how='left')
        spine['First_Name'] = spine['First_Name'].where(spine['First_Name'].notna() & (spine['First_Name'].astype(str) != 'nan'), spine['_fn'])
        spine['Last_Name'] = spine['Last_Name'].where(spine['Last_Name'].notna() & (spine['Last_Name'].astype(str) != 'nan'), spine['_ln'])
        spine.drop(columns=['_fn', '_ln'], inplace=True)

    # Merge each metric
    merged = spine.merge(wins_df[['Date', 'Rep', 'Wins']], on=['Date', 'Rep'], how='left')
    merged = merged.merge(sched_df[['Date', 'Rep', 'Lawn Treatment', 'Mosquito']],
                          on=['Date', 'Rep'], how='left')
    merged = merged.merge(iq_df[['Date', 'Rep', 'Bush Trimming', 'Flower Bed Weeding', 'Leaf Removal']],
                          on=['Date', 'Rep'], how='left')
    merged = merged.merge(pool_df[['Date', 'Rep', 'Pool']], on=['Date', 'Rep'], how='left')
    merged = merged.merge(calls_df[['Date', 'Rep', 'Calls']], on=['Date', 'Rep'], how='left')

    # Merge team roster (Team Name, Manager_Direct) — drop name cols already on spine
    roster_team = roster_map.drop(columns=[c for c in ['First_Name', 'Last_Name'] if c in roster_map.columns])
    merged['rep_key'] = merged['Rep'].str.lower().str.strip()
    merged = merged.merge(roster_team, on='rep_key', how='left')
    merged.drop(columns=['rep_key'], inplace=True)
    merged['Team Name'] = merged['Team Name'].fillna('Unknown')
    merged['Manager_Direct'] = merged.get('Manager_Direct', pd.Series('', index=merged.index)).fillna('')

    # Filtering logic:
    # - Current SalesRoster reps → always keep
    # - Reps NOT on roster but with names from Redshift users table → historical sales reps, keep
    # - Reps NOT on roster AND no name from Redshift → OPS/retention showing up via Five9 only, exclude
    before = len(merged)
    if sales_emails:
        is_current_sales = merged['Rep'].isin(sales_emails)
        has_redshift_name = (
            merged['First_Name'].notna() & (merged['First_Name'].astype(str).str.strip() != 'nan') & (merged['First_Name'].astype(str).str.strip() != '')
        )
        merged = merged[is_current_sales | has_redshift_name]
    print(f"  Filtered out {before - len(merged)} rows from non-sales / unlisted reps")

    # Fill NaN numerics with 0
    num_cols = ['Wins', 'Lawn Treatment', 'Mosquito', 'Bush Trimming', 'Flower Bed Weeding',
                'Leaf Removal', 'Pool', 'Calls']
    for c in num_cols:
        merged[c] = pd.to_numeric(merged.get(c, 0), errors='coerce').fillna(0).astype(int)

    # Sort
    merged = merged.sort_values(['Date', 'Rep']).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # If today-only mode: merge with existing CSV, replacing rows for pulled dates
    # ---------------------------------------------------------------------------
    if today_only and os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        existing = pd.read_csv(OUTPUT_CSV)
        existing['Date'] = pd.to_datetime(existing['Date'], errors='coerce')
        merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')

        # Drop existing rows for the dates we just pulled
        pulled_dates = merged['Date'].unique()
        existing = existing[~existing['Date'].isin(pulled_dates)]

        # Append new rows
        combined = pd.concat([existing, merged], ignore_index=True)
        combined = combined.sort_values(['Date', 'Rep']).reset_index(drop=True)
        final = combined
    else:
        final = merged

    # ---------------------------------------------------------------------------
    # Write CSV (local backup)
    # ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(final):,} rows written to {OUTPUT_CSV}")
    min_date = final['Date'].min()
    max_date = final['Date'].max()
    min_str = min_date.date() if hasattr(min_date, 'date') else min_date
    max_str = max_date.date() if hasattr(max_date, 'date') else max_date
    print(f"Date range: {min_str} → {max_str}")
    print(f"Reps: {final['Rep'].nunique()}")
    print(f"Columns: {list(final.columns)}")

    # ---------------------------------------------------------------------------
    # Write to Google Sheet (primary — dashboard reads from here)
    # Requires: pip install gspread gspread-dataframe
    #   and a service account key at ~/.config/gspread/service_account.json
    # ---------------------------------------------------------------------------
    SPREADSHEET_ID = "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    WORKSHEET_NAME = "RepHistory"
    try:
        import gspread
        from gspread_dataframe import set_with_dataframe

        gc = gspread.service_account()          # reads ~/.config/gspread/service_account.json
        sh = gc.open_by_key(SPREADSHEET_ID)
        try:
            ws = sh.worksheet(WORKSHEET_NAME)
            ws.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=len(final) + 200, cols=len(final.columns) + 2)

        final_for_sheet = final.copy()
        final_for_sheet['Date'] = final_for_sheet['Date'].astype(str)
        set_with_dataframe(ws, final_for_sheet)
        print(f"  Written {len(final):,} rows to Google Sheet tab '{WORKSHEET_NAME}'")

        # Sync new SalesRoster reps into RepData
        update_repdata_from_roster(sr, sh)
    except ImportError:
        print("  Skipping Google Sheet write (run: pip install gspread gspread-dataframe)")
    except FileNotFoundError:
        print("  Skipping Google Sheet write (service account key not found at ~/.config/gspread/service_account.json)")
    except Exception as _e:
        print(f"  Warning: Google Sheet write failed: {_e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync rep history from Redshift to CSV")
    parser.add_argument(
        "--today", action="store_true",
        help="Fast mode: pull today + yesterday only, merge into existing CSV"
    )
    args = parser.parse_args()
    sync(today_only=args.today)
