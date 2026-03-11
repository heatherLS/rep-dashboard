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
    # ---------------------------------------------------------------------------
    # Load team roster from Google Sheet (Team Name, Manager_Direct)
    # ---------------------------------------------------------------------------
    print("  Fetching team roster from Google Sheet...")
    ROSTER_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
        "/export?format=csv&gid=171451260"
    )
    SALES_ROLES = {'Sales Tier 1', 'Tier 2 Sales', 'Team Lead', 'Trainee'}
    try:
        # Read raw (row 0 is a date header, row 1 is column names)
        raw = pd.read_csv(ROSTER_URL, header=None)
        raw.columns = raw.iloc[1]
        raw = raw.iloc[2:].reset_index(drop=True)
        raw.columns = raw.columns.str.strip()
        roster = raw[raw['Rep'].notna() & (raw['Rep'].astype(str).str.strip() != '')].copy()
        roster['rep_key'] = roster['Rep'].str.lower().str.strip()
        roster['Team Name'] = roster['Team Name'].astype(str).str.strip().replace({'nan': '', '': None})

        # Infer missing team names from Manager_Direct → team of reps who share that manager
        has_team = roster[roster['Team Name'].notna()]
        mgr_to_team = (
            has_team.groupby('Manager_Direct')['Team Name']
            .agg(lambda x: x.mode()[0])
            .to_dict()
        )
        def resolve_team(row):
            if pd.notna(row['Team Name']) and row['Team Name']:
                return row['Team Name']
            mgr = str(row.get('Manager_Direct', '')).strip()
            return mgr_to_team.get(mgr, None)

        roster['Team Name'] = roster.apply(resolve_team, axis=1)

        # Tag each rep as sales or non-sales
        roster['is_sales'] = roster['Current_Role'].isin(SALES_ROLES)
        roster_map = roster[['rep_key', 'Team Name', 'Manager_Direct', 'is_sales']].drop_duplicates('rep_key')
        sales_emails = set(roster_map.loc[roster_map['is_sales'], 'rep_key'])
        non_sales_emails = set(roster_map.loc[~roster_map['is_sales'], 'rep_key'])
        print(f"  Roster loaded: {len(sales_emails)} sales reps, {len(non_sales_emails)} excluded (non-sales)")
    except Exception as e:
        print(f"  Warning: could not load roster ({e}). Team Name will be 'Unknown'.")
        roster_map = pd.DataFrame(columns=['rep_key', 'Team Name', 'Manager_Direct', 'is_sales'])
        sales_emails = set()
        non_sales_emails = set()

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

    # Carry First_Name / Last_Name from wins onto spine
    name_df = wins_df[['Date', 'Rep', 'First_Name', 'Last_Name']].drop_duplicates(['Date', 'Rep'])
    spine = spine.merge(name_df, on=['Date', 'Rep'], how='left')

    # Merge each metric
    merged = spine.merge(wins_df[['Date', 'Rep', 'Wins']], on=['Date', 'Rep'], how='left')
    merged = merged.merge(sched_df[['Date', 'Rep', 'Lawn Treatment', 'Mosquito']],
                          on=['Date', 'Rep'], how='left')
    merged = merged.merge(iq_df[['Date', 'Rep', 'Bush Trimming', 'Flower Bed Weeding', 'Leaf Removal']],
                          on=['Date', 'Rep'], how='left')
    merged = merged.merge(pool_df[['Date', 'Rep', 'Pool']], on=['Date', 'Rep'], how='left')
    merged = merged.merge(calls_df[['Date', 'Rep', 'Calls']], on=['Date', 'Rep'], how='left')

    # Merge team roster (Team Name, Manager_Direct, is_sales)
    merged['rep_key'] = merged['Rep'].str.lower().str.strip()
    merged = merged.merge(roster_map, on='rep_key', how='left')
    merged.drop(columns=['rep_key'], inplace=True)
    merged['Team Name'] = merged['Team Name'].fillna('Unknown')
    merged['Manager_Direct'] = merged.get('Manager_Direct', pd.Series('', index=merged.index)).fillna('')

    # Exclude reps who are currently in a non-sales role (retention, ops, managers, etc.)
    # Keep historical reps not in the current roster (is_sales is NaN) — they were likely sales reps
    before = len(merged)
    merged = merged[merged['is_sales'].isna() | merged['is_sales']]
    merged.drop(columns=['is_sales'], inplace=True)
    print(f"  Filtered out {before - len(merged)} rows from non-sales reps")

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
    # Write
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
