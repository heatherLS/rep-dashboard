#!/usr/bin/env python3
"""
slack_notify.py
===============
Posts sales performance shoutouts to Slack.

Usage:
  python scripts/slack_notify.py --mode yesterday   # daily 10am CST
  python scripts/slack_notify.py --mode today       # hourly during day

Cron examples (crontab -e):
  # 10:00 AM CST (16:00 UTC) Mon–Fri — yesterday's stats
  0 16 * * 1-5 /path/to/venv/bin/python3 /path/to/scripts/slack_notify.py --mode yesterday >> /path/to/data/sync_log.txt 2>&1

  # Every hour 8am–7pm CST (14:00–01:00 UTC) Mon–Fri — today's live stats
  0 14-23 * * 1-5 /path/to/venv/bin/python3 /path/to/scripts/slack_notify.py --mode today >> /path/to/data/sync_log.txt 2>&1
  0 0,1 * * 2-6 /path/to/venv/bin/python3 /path/to/scripts/slack_notify.py --mode today >> /path/to/data/sync_log.txt 2>&1
"""

import argparse
import os
import json
import sys
import urllib.request
from datetime import date, timedelta, datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Load .env file if SLACK_WEBHOOK_URL not already in environment
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "")

# RepHistory Google Sheet (written by hourly cron sync from Redshift)
HISTORY_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QSX8Me9ZkyNlXJWW_46XrRriHMFY8gIcY_R3FRXcdnU"
    "/export?format=csv&gid=1441799869"
)

# Five9 Google Sheet — Yesterday tab (col 1=Email, col 9=All in Calls YESTERDAY)
FIVE9_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1sO4ZDe-n8-ugc-OsoDisHPXXEAap6b2iINcjsWXz5gU"
    "/export?format=csv&gid=396976453"
)

ATTACH_SERVICES = ['Lawn Treatment', 'Mosquito', 'Bush Trimming', 'Flower Bed Weeding', 'Leaf Removal']

MIN_CALLS_REP  = 5    # min calls for a rep to qualify in conversion rankings
MIN_CALLS_TEAM = 20   # min calls for a team to qualify
MIN_WINS_ATTACH = 1   # min wins for a rep to qualify in attach rankings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_URL)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Rep'].notna() & (df['Rep'].astype(str).str.strip() != '')].copy()
    for col in ['Calls', 'Wins'] + ATTACH_SERVICES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def _enrich_with_five9_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Calls + Wins with Five9 yesterday values (source of truth for conversion %)."""
    try:
        raw = pd.read_csv(FIVE9_URL, header=None)
        # Col 1=Email, Col 9=All in Calls (YESTERDAY), Col 15=Total Wins (YESTERDAY)
        five9_cv = raw.iloc[2:, [1, 9, 15]].copy()
        five9_cv.columns = ['_rep_key', 'five9_calls', 'five9_wins']
        five9_cv['_rep_key']    = five9_cv['_rep_key'].astype(str).str.lower().str.strip()
        five9_cv['five9_calls'] = pd.to_numeric(five9_cv['five9_calls'], errors='coerce').fillna(0)
        five9_cv['five9_wins']  = pd.to_numeric(five9_cv['five9_wins'],  errors='coerce').fillna(0)
        five9_cv = five9_cv[five9_cv['_rep_key'].str.contains('@lawnstarter.com', na=False)]

        df = df.copy()
        df['_rep_key'] = df['Rep'].astype(str).str.lower().str.strip()
        df = df.merge(five9_cv, on='_rep_key', how='left')
        df['Calls'] = df['five9_calls'].combine_first(
            pd.to_numeric(df['Calls'], errors='coerce').fillna(0)
        )
        df['Wins'] = df['five9_wins'].combine_first(
            pd.to_numeric(df['Wins'], errors='coerce').fillna(0)
        )
        df.drop(columns=['_rep_key', 'five9_calls', 'five9_wins'], inplace=True)
    except Exception as e:
        print(f"[slack_notify] Five9 enrich failed: {e}", file=sys.stderr)
    return df


def _full_name(row) -> str:
    fn = str(row.get('First_Name', '')).strip()
    ln = str(row.get('Last_Name', '')).strip()
    name = f"{fn} {ln}".strip()
    return name if name else str(row.get('Rep', 'Unknown')).split('@')[0].title()


def _compute_stats(df: pd.DataFrame):
    """
    Returns:
      top_teams: list of (team_name, conversion_pct, wins, calls) — top 3
      top_conv:  list of (full_name, conversion_pct, wins, calls, team) — top 3 reps by conversion
      top_attach: list of (full_name, attach_pct, attach_count, wins, team) — top 3 reps by attach rate
    """
    df = df.copy()
    df['Full_Name'] = df.apply(_full_name, axis=1)

    # Filter out Unknown/blank teams for team rankings
    valid_teams = (
        df['Team Name'].notna() &
        (df['Team Name'].astype(str).str.strip() != '') &
        (df['Team Name'].astype(str).str.strip().str.lower() != 'unknown')
    )

    # ---- Teams ----
    team_df = df[valid_teams].copy()
    team_totals = team_df.groupby('Team Name').agg(
        Calls=('Calls', 'sum'), Wins=('Wins', 'sum')
    ).reset_index()
    team_totals = team_totals[team_totals['Calls'] >= MIN_CALLS_TEAM]
    if not team_totals.empty:
        team_totals['Conversion'] = (team_totals['Wins'] / team_totals['Calls'].replace(0, pd.NA)) * 100
        team_totals['Conversion'] = team_totals['Conversion'].fillna(0)
        team_totals = team_totals.sort_values(['Conversion', 'Wins'], ascending=[False, False])
    top_teams = [
        (row['Team Name'], float(row['Conversion']), int(row['Wins']), int(row['Calls']))
        for _, row in team_totals.head(3).iterrows()
    ]

    # ---- Reps: Conversion ----
    rep_df = df[df['Calls'] >= MIN_CALLS_REP].copy()
    rep_df['Conversion'] = (rep_df['Wins'] / rep_df['Calls'].replace(0, pd.NA)) * 100
    rep_df['Conversion'] = rep_df['Conversion'].fillna(0)
    rep_df = rep_df.sort_values(['Conversion', 'Wins'], ascending=[False, False])
    top_conv = [
        (row['Full_Name'], float(row['Conversion']), int(row['Wins']), int(row['Calls']),
         str(row.get('Team Name', '')))
        for _, row in rep_df.head(3).iterrows()
    ]

    # ---- Reps: Attach ----
    attach_df = df[df['Wins'] >= MIN_WINS_ATTACH].copy()
    svc_cols = [c for c in ATTACH_SERVICES if c in attach_df.columns]
    attach_df['Attach_Count'] = attach_df[svc_cols].sum(axis=1)
    attach_df['Attach_Rate'] = (attach_df['Attach_Count'] / attach_df['Wins'].replace(0, pd.NA)) * 100
    attach_df['Attach_Rate'] = attach_df['Attach_Rate'].fillna(0)
    attach_df = attach_df.sort_values(['Attach_Count', 'Attach_Rate'], ascending=[False, False])
    top_attach = [
        (row['Full_Name'], float(row['Attach_Rate']), int(row['Attach_Count']), int(row['Wins']),
         str(row.get('Team Name', '')))
        for _, row in attach_df.head(3).iterrows()
    ]

    return top_teams, top_conv, top_attach


def _send_slack(message: str):
    if not SLACK_WEBHOOK:
        print("[slack_notify] ERROR: SLACK_WEBHOOK_URL env var not set.", file=sys.stderr)
        sys.exit(1)
    payload = json.dumps({"text": message}).encode('utf-8')
    req = urllib.request.Request(
        SLACK_WEBHOOK,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        status = resp.status
    if status != 200:
        print(f"[slack_notify] Slack returned HTTP {status}", file=sys.stderr)
        sys.exit(1)


def _format_teams(top_teams):
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (team, conv, wins, calls) in enumerate(top_teams):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"  {medal} *{team}* — {conv:.1f}% conversion ({wins} wins / {calls} calls)")
    return "\n".join(lines) if lines else "  _No data yet_"


def _format_reps_conv(top_conv):
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (name, conv, wins, calls, team) in enumerate(top_conv):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"  {medal} *{name}* ({team}) — {conv:.1f}% ({wins} wins / {calls} calls)")
    return "\n".join(lines) if lines else "  _No data yet_"


def _format_reps_attach(top_attach):
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (name, rate, count, wins, team) in enumerate(top_attach):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"  {medal} *{name}* ({team}) — {rate:.1f}% attach ({count} attaches / {wins} wins)")
    return "\n".join(lines) if lines else "  _No data yet_"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_yesterday():
    print(f"[slack_notify] Posting yesterday's stats...")
    history_df = _load_history()
    yesterday = date.today() - timedelta(days=1)
    df = history_df[history_df['Date'].dt.date == yesterday].copy()

    if df.empty:
        print(f"[slack_notify] No history data for {yesterday}, skipping.")
        return

    df = _enrich_with_five9_calls(df)
    top_teams, top_conv, top_attach = _compute_stats(df)

    date_str = yesterday.strftime("%A, %B %-d")
    msg = (
        f"<!channel> :star: *Yesterday's Sales Highlights — {date_str}* :star:\n\n"
        f":trophy: *Top 3 Teams by Conversion*\n{_format_teams(top_teams)}\n\n"
        f":dart: *Top 3 Converting Reps*\n{_format_reps_conv(top_conv)}\n\n"
        f":seedling: *Top 3 Attach Reps*\n{_format_reps_attach(top_attach)}\n\n"
        f":eyes: *Who's claiming the top spots today? The leaderboard resets in a few hours — let's get after it!*"
    )
    _send_slack(msg)
    print(f"[slack_notify] Yesterday message sent.")


def run_today():
    print(f"[slack_notify] Posting today's live stats...")
    history_df = _load_history()
    today = date.today()
    df = history_df[history_df['Date'].dt.date == today].copy()

    if df.empty:
        print(f"[slack_notify] No history data for {today} yet, skipping.")
        return

    top_teams, top_conv, top_attach = _compute_stats(df)

    now_cst = datetime.utcnow() - timedelta(hours=6)
    time_str = now_cst.strftime("%-I:%M %p CST")
    date_str = today.strftime("%A, %B %-d")
    msg = (
        f"<!channel> :zap: *Live Standings — {date_str} ({time_str})* :zap:\n\n"
        f":trophy: *Top 3 Teams by Conversion*\n{_format_teams(top_teams)}\n\n"
        f":dart: *Top 3 Converting Reps*\n{_format_reps_conv(top_conv)}\n\n"
        f":seedling: *Top 3 Attach Reps*\n{_format_reps_attach(top_attach)}\n\n"
        f":muscle: *Next update in 1 hour — who's making their move?*"
    )
    _send_slack(msg)
    print(f"[slack_notify] Today message sent.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['yesterday', 'today'], required=True,
                        help="'yesterday' for daily 10am shoutout, 'today' for hourly live stats")
    args = parser.parse_args()

    if args.mode == 'yesterday':
        run_yesterday()
    else:
        run_today()
