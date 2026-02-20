import re
import streamlit as st
import pandas as pd
import math
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Canopy — Presentation View", layout="wide")

# ============================================================
# GOOGLE SHEETS DATA LOADING
# ============================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_sheets():
    """Load data directly from Google Sheets with CSV fallback."""
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SHEET_ID = "1bDlQDIyAgggxXSVjLhaKRlZ-YjEY6PlZKaHCYaEhk0s"

    try:
        credentials = Credentials.from_service_account_file(
            "service_account.json", scopes=SCOPES
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(SHEET_ID)

        # Load Agent Stats tab
        agents_sheet = spreadsheet.worksheet("High Level Agent Stats")
        agents_data = agents_sheet.get_all_values()
        df_agents_raw = pd.DataFrame(agents_data)

        # Load TL High Level tab
        tl_sheet = spreadsheet.worksheet("TL High Level")
        tl_data = tl_sheet.get_all_values()
        df_tl_raw = pd.DataFrame(tl_data)

        return df_agents_raw, df_tl_raw, None  # None = no error

    except Exception as e:
        # Fallback to CSV files
        try:
            df_agents_raw = pd.read_csv("agents.csv", header=None)
            df_tl_raw = pd.read_csv("TL_highlevel.csv", header=None)
            return df_agents_raw, df_tl_raw, f"Google Sheets unavailable: {e}. Using CSV fallback."
        except Exception as csv_error:
            return None, None, f"Could not load data: {csv_error}"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_start_date(cycle_str: str) -> pd.Timestamp:
    """Extract first date found in a cycle label (handles newlines and odd dashes)."""
    try:
        if not isinstance(cycle_str, str):
            return pd.to_datetime("1900-01-01")
        s = " ".join(str(cycle_str).split())  # collapse newlines/tabs/multiple spaces
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", s)
        if not m:
            return pd.to_datetime("1900-01-01")
        return pd.to_datetime(m.group(1))
    except Exception:
        return pd.to_datetime("1900-01-01")


def percent_to_float(x):
    if pd.isna(x):
        return math.nan
    try:
        return float(str(x).replace("%", "").replace(",", ""))
    except Exception:
        return math.nan


def momentum_label(values):
    """Classify momentum trend from a list of numeric values (oldest → newest)."""
    if len(values) < 2:
        return "❔ Not enough data"

    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    up = sum(d > 0 for d in diffs)
    down = sum(d < 0 for d in diffs)

    if up == len(diffs):
        return "👍 Improving"
    elif down == len(diffs):
        return "⚠️ Declining"
    else:
        return "↔️ Mixed"


def norm_cycle(x) -> str:
    """Collapse newlines/tabs/extra spaces in a cycle label into a single space."""
    return " ".join(str(x).split())


def ordered_unique(seq):
    """Return unique non-empty normalized values in order, preserving first occurrence."""
    seen = set()
    out = []
    for x in seq:
        if pd.isna(x):
            continue
        x = norm_cycle(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_goal_style(actual_df: pd.DataFrame, goals_df: pd.DataFrame) -> pd.DataFrame:
    """
    For TL tables: return a style DataFrame coloring cells
    green if value >= goal, red otherwise.
    """
    styles = pd.DataFrame("", index=actual_df.index, columns=actual_df.columns)
    for r in actual_df.index:
        for c in actual_df.columns:
            val = actual_df.loc[r, c]
            goal = goals_df.loc[r, c] if (r in goals_df.index and c in goals_df.columns) else math.nan
            if pd.isna(val) or pd.isna(goal):
                continue
            if val >= goal:
                styles.loc[r, c] = "background-color: #6DDA6D;"  # green
            else:
                styles.loc[r, c] = "background-color: #FF7F7F;"  # red
    return styles


def compute_cycle_ranks(df_agent_long: pd.DataFrame, metric: str):
    """
    For a given metric, compute rank per cycle (vs ALL reps),
    and rank change vs previous cycle for that agent.

    Returns:
        ranks_by_cycle: {cycle: {agent: (rank, delta)}}
        where delta = previous_rank - current_rank (positive = improved).
    """
    dfm = df_agent_long[df_agent_long["Metric"] == metric]
    cycles = sorted(dfm["Cycle"].dropna().unique(), key=extract_start_date)

    ranks_by_cycle = {}
    prev_ranks = {}

    for cycle in cycles:
        d = (
            dfm[dfm["Cycle"] == cycle]
            .groupby("Agent", as_index=False)["Value"]
            .mean()
        )
        if d.empty:
            continue

        # SAFE nullable rank
        d["Rank"] = (
            d["Value"]
            .rank(ascending=False, method="dense")
            .astype("Int64")   # allows NA safely
        )

        cycle_ranks = {}
        for _, row in d.iterrows():
            agent = row["Agent"]
            r = row["Rank"]  # may be <Int64> or NA

            if pd.isna(r):
                cycle_ranks[agent] = (math.nan, None)
                continue

            r_int = int(r)

            if agent in prev_ranks and prev_ranks[agent] is not None:
                delta = prev_ranks[agent] - r_int
            else:
                delta = None

            cycle_ranks[agent] = (r_int, delta)

        ranks_by_cycle[cycle] = cycle_ranks

        # Track previous ranks safely
        prev_ranks = {
            row["Agent"]: (int(row["Rank"]) if not pd.isna(row["Rank"]) else None)
            for _, row in d.iterrows()
        }

    return ranks_by_cycle


def build_rep_value_rank_table(
    metric: str,
    df_tl_agents: pd.DataFrame,
    df_agent_long: pd.DataFrame,
    agent_goal_dict: dict,
    cycles_chrono: list,
    cycles_display: list,
):
    """
    Build a wide table for a single metric with:
    Agent | <Cycle1 Value> | <Cycle1 Rank> | <Cycle2 Value> | <Cycle2 Rank> | ...

    - Values are numeric in the underlying data.
    - We'll format them as % for display.
    - Rank cell includes delta vs previous cycle (↑ / ↓ / -).
    - A "Goal" row is added at the top with Value only.
    - Style DataFrame colors Value cells green/red based on goal.
    """
    # Values for this TL only
    temp = df_tl_agents[df_tl_agents["Metric"] == metric]
    pivot_vals = temp.pivot(index="Agent", columns="Cycle", values="Value")

    # Ensure all cycles (even if NaN) appear in newest → oldest order
    pivot_vals = pivot_vals.reindex(columns=cycles_display)

    # Compute global ranks by cycle (all reps)
    ranks_by_cycle = compute_cycle_ranks(df_agent_long, metric)

    # Build columns: [Cycle1 Value, Cycle1 Rank, Cycle2 Value, Cycle2 Rank, ...]
    cols = []
    for cycle in cycles_display:
        cols.append(f"{cycle} Value")
        cols.append(f"{cycle} Rank")

    table = pd.DataFrame(index=pivot_vals.index, columns=cols)

    # Fill agent rows
    for agent in pivot_vals.index:
        for cycle in cycles_display:
            val = pivot_vals.loc[agent, cycle] if cycle in pivot_vals.columns else math.nan
            table.loc[agent, f"{cycle} Value"] = val

            r, delta = ranks_by_cycle.get(cycle, {}).get(agent, (math.nan, None))

            if pd.isna(r):
                rank_text = ""
            else:
                r_int = int(r)
                if delta is None or pd.isna(delta):
                    rank_text = f"{r_int}"
                else:
                    d_int = int(delta)
                    if d_int > 0:
                        rank_text = f"{r_int} (↑{d_int})"
                    elif d_int < 0:
                        rank_text = f"{r_int} (↓{abs(d_int)})"
                    else:
                        rank_text = f"{r_int} (-)"

            table.loc[agent, f"{cycle} Rank"] = rank_text

    # Add GOAL row at top
    goal_row = {}
    for cycle in cycles_display:
        goal_val = agent_goal_dict.get((metric, cycle), math.nan)
        goal_row[f"{cycle} Value"] = goal_val
        goal_row[f"{cycle} Rank"] = ""
    table_with_goal = pd.concat([pd.DataFrame([goal_row], index=["Goal"]), table])

    # Style DF: color only the Value columns (not ranks)
    style_df = pd.DataFrame("", index=table_with_goal.index, columns=table_with_goal.columns)
    for cycle in cycles_display:
        col_val = f"{cycle} Value"
        goal_val = agent_goal_dict.get((metric, cycle), math.nan)
        if pd.isna(goal_val):
            continue
        for idx in table_with_goal.index:
            if idx == "Goal":
                continue
            v = table_with_goal.loc[idx, col_val]
            if pd.isna(v):
                continue
            if v >= goal_val:
                style_df.loc[idx, col_val] = "background-color: #6DDA6D;"  # green
            else:
                style_df.loc[idx, col_val] = "background-color: #FF7F7F;"  # red

    # Make a pretty copy for display: format Value as % strings
    display = table_with_goal.copy()
    for cycle in cycles_display:
        col_val = f"{cycle} Value"
        display[col_val] = display[col_val].apply(
            lambda x: "" if pd.isna(x) else f"{x:.2f}%"
        )

    return display, style_df


# ============================================================
# LOAD DATA FROM GOOGLE SHEETS (with CSV fallback)
# ============================================================

df_agents_raw, df_tl_raw, load_error = load_data_from_sheets()

if df_agents_raw is None or df_tl_raw is None:
    st.error("ERROR LOADING DATA")
    st.code(load_error)
    st.stop()

if load_error:
    st.warning(load_error)
else:
    st.success("Data Loaded from Google Sheets")

# Row 0: metric name (Conversion / Attach / LT Attach / QA)
# Row 1: cycle labels (newest → oldest)
# Row 2: goals
metric_row_agents = df_agents_raw.iloc[0].replace("", pd.NA).ffill()
cycle_row_agents = df_agents_raw.iloc[1].ffill()
goal_row_agents = df_agents_raw.iloc[2]


# Build agent goal lookup: {(metric, cycle): goal}
agent_goal_dict = {}
for col in range(3, df_agents_raw.shape[1]):
    metric = metric_row_agents[col]
    cycle = norm_cycle(cycle_row_agents[col])
    if metric in ["Conversion", "Attach", "LT Attach", "QA"]:
        goal_val = percent_to_float(goal_row_agents[col])
        if not pd.isna(goal_val):
            agent_goal_dict[(metric, cycle)] = goal_val

# Actual agent data starts at row 3
df_agents = df_agents_raw.iloc[3:].copy()
df_agents.columns = range(df_agents.shape[1])
df_agents = df_agents.rename(columns={0: "TL", 1: "Agent"})

agent_records = []
for col in range(3, df_agents.shape[1]):
    metric = metric_row_agents[col]
    cycle = norm_cycle(cycle_row_agents[col])
    if metric in ["Conversion", "Attach", "LT Attach", "QA"]:
        for _, row in df_agents.iterrows():
            raw_val = row[col]
            if pd.notna(raw_val):
                val = percent_to_float(raw_val)
                agent_records.append(
                    {
                        "Agent": row["Agent"],
                        "TL": row["TL"],
                        "Metric": metric,
                        "Cycle": cycle,
                        "Value": val,
                    }
                )

df_agent_long = pd.DataFrame(agent_records)

# ============================================================
# PROCESS TL DATA
# ============================================================

# --- TL headers ---
tl_metrics_row    = df_tl_raw.iloc[0, 4:].replace("", pd.NA).ffill()
tl_cycles_row_raw = df_tl_raw.iloc[1, 4:]          # NO ffill — used as a mask
tl_cycles_row     = df_tl_raw.iloc[1, 4:].ffill()
tl_goals_row      = df_tl_raw.iloc[2, 4:]

# Only process offsets where the raw cycle cell is non-blank (skips separator columns)
valid_offsets = [
    offset for offset, raw in enumerate(tl_cycles_row_raw.tolist())
    if str(raw).strip() not in ("", "nan")
]

# Build TL goal dict (valid offsets only)
tl_goal_dict = {}
for offset in valid_offsets:
    metric = tl_metrics_row.iloc[offset]
    cycle  = norm_cycle(tl_cycles_row.iloc[offset])
    if metric in ["Conversion", "Attach", "LT Attach", "QA"]:
        goal_val = percent_to_float(tl_goals_row.iloc[offset])
        if not pd.isna(goal_val):
            tl_goal_dict[(metric, cycle)] = goal_val

# TL data rows
df_tl_data = df_tl_raw.iloc[3:].reset_index(drop=True)
df_tl_data["TL"] = df_tl_data[0]

tl_records = []
for _, row in df_tl_data.iterrows():
    tl_name = row["TL"]
    if pd.isna(tl_name):
        continue
    for offset in valid_offsets:
        col    = 4 + offset
        metric = tl_metrics_row.iloc[offset]
        cycle  = norm_cycle(tl_cycles_row.iloc[offset])
        if metric not in ["Conversion", "Attach", "LT Attach", "QA"]:
            continue
        raw_val = row[col]
        val     = percent_to_float(raw_val)
        if pd.isna(val):
            continue  # skip separator / truly empty cells
        goal_val = tl_goal_dict.get((metric, cycle), math.nan)
        tl_records.append({"TL": tl_name, "Metric": metric, "Cycle": cycle, "Value": val, "Goal": goal_val})

df_tl_long = pd.DataFrame(tl_records)

# Global cycles (oldest → newest)
cycles_chrono = sorted(
    df_agent_long["Cycle"].dropna().unique(), key=extract_start_date
)
# Display newest → oldest to match sheet
cycles_display = cycles_chrono[::-1]
latest_cycle = cycles_chrono[-1] if cycles_chrono else None
prev_cycle = cycles_chrono[-2] if len(cycles_chrono) > 1 else None

# ============================================================
# UI — TL SELECTION
# ============================================================

st.title("🌳 Canopy — Supervisor Presentation View")
st.markdown(
    "_Use this view live in the Wednesday meeting. Everything is ordered, "
    "ranked, and annotated for you._"
)
st.markdown("---")

all_tls = sorted(df_agents["TL"].dropna().unique())
selected_tl = st.sidebar.selectbox("Choose a Team Lead", all_tls)

st.sidebar.markdown("---")
st.sidebar.write("This entire page adjusts to the selected TL.")

# Filter agent data for selected TL
df_tl_agents = df_agent_long[df_agent_long["TL"] == selected_tl]
metrics = ["Conversion", "Attach", "LT Attach", "QA"]

# TL subset
tl_subset = df_tl_long[df_tl_long["TL"] == selected_tl]

# ============================================================
# TEAM SUMMARY CARD
# ============================================================

st.header(f"👤 Team Lead: {selected_tl}")
st.subheader("🌿 Team Summary (Latest Cycle)")

summary_rows = []
metric_momentum = []

for metric in metrics:
    # TL latest value & rank
    tl_metric_all = df_tl_long[(df_tl_long["Metric"] == metric)]
    tl_pivot = tl_metric_all.pivot(index="TL", columns="Cycle", values="Value")
    tl_pivot = tl_pivot.reindex(columns=cycles_chrono)

    latest_val = None
    latest_rank = None
    out_of_tls = None

    if latest_cycle in tl_pivot.columns and selected_tl in tl_pivot.index:
        latest_val = tl_pivot.loc[selected_tl, latest_cycle]
        latest_col_series = tl_pivot[latest_cycle].dropna()
        if not pd.isna(latest_val):
            rank_df = (
                latest_col_series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={latest_cycle: "Score"})
            )
            rank_df["Rank"] = rank_df["Score"].rank(
                ascending=False, method="dense"
            ).astype(int)
            out_of_tls = len(rank_df)
            row_match = rank_df[rank_df["TL"] == selected_tl]
            if not row_match.empty:
                latest_rank = int(row_match["Rank"].iloc[0])

    # team % meeting goal (Option A: only reps with latest-cycle values)
    goal_val = agent_goal_dict.get((metric, latest_cycle), math.nan)
    count_met = 0
    count_total = 0
    if not pd.isna(goal_val):
        tmp = df_tl_agents[
            (df_tl_agents["Metric"] == metric)
            & (df_tl_agents["Cycle"] == latest_cycle)
        ]
        if not tmp.empty:
            latest_rep_vals = (
                tmp.groupby("Agent", as_index=False)["Value"].mean()
            )
            for _, r in latest_rep_vals.iterrows():
                v = r["Value"]
                if pd.isna(v):
                    continue
                count_total += 1
                if v >= goal_val:
                    count_met += 1

    pct_on_goal = (count_met / count_total * 100.0) if count_total > 0 else None

    # momentum for this metric from TL perspective
    tl_this = tl_subset[tl_subset["Metric"] == metric]
    tl_p = tl_this.pivot(index="Metric", columns="Cycle", values="Value")
    tl_p = tl_p.reindex(columns=cycles_chrono)
    trend_label = "❔ Not enough data"
    if not tl_p.empty:
        vals = tl_p.loc[metric].dropna().tolist()
        if len(vals) >= 2:
            trend_label = momentum_label(vals)

    metric_momentum.append((metric, trend_label))

    summary_rows.append(
        {
            "Metric": metric,
            "Latest Value": latest_val,
            "Rank": latest_rank,
            "Out of TLs": out_of_tls,
            "% Reps on Goal": pct_on_goal,
        }
    )

summary_df = pd.DataFrame(summary_rows)

def fmt_pct(x):
    return "" if pd.isna(x) else f"{x:.2f}%"

def fmt_int(x):
    return "" if pd.isna(x) else str(int(x))

def fmt_goal_pct(x):
    return "" if x is None or pd.isna(x) else f"{x:.0f}%"

display_summary = summary_df.copy()
display_summary["Latest Value"] = display_summary["Latest Value"].apply(fmt_pct)
display_summary["Rank"] = display_summary["Rank"].apply(fmt_int)
display_summary["Out of TLs"] = display_summary["Out of TLs"].apply(fmt_int)
display_summary["% Reps on Goal"] = display_summary["% Reps on Goal"].apply(fmt_goal_pct)

st.dataframe(display_summary, use_container_width=True)

# Momentum blurb
improving_count = sum(1 for m, lab in metric_momentum if "Improving" in lab)
declining_count = sum(1 for m, lab in metric_momentum if "Declining" in lab)

st.markdown("**Team Momentum:**")
st.markdown(
    f"- Improving in **{improving_count}** of 4 metrics\n"
    f"- Declining in **{declining_count}** of 4 metrics\n"
)

st.markdown("---")

# ============================================================
# 3-CYCLE SIDE-BY-SIDE TL SUMMARY TABLE
# ============================================================

st.subheader("📋 3-Cycle Side-by-Side Summary")

# Build cycle list directly from header rows (handles newlines/blanks/merged cells)
cycles_from_agent_header = ordered_unique(cycle_row_agents.iloc[3:].tolist())
cycles_from_tl_header    = ordered_unique(tl_cycles_row.tolist())

_all_cycles = sorted(
    set(cycles_from_agent_header) | set(cycles_from_tl_header),
    key=extract_start_date,
    reverse=True,  # newest first
)
recent_3 = _all_cycles[:3]

if recent_3:
    table_rows = []
    style_rows = []
    row_labels = []

    # One extra older cycle for rank-delta on the oldest of the 3 shown
    extra_cycle = _all_cycles[3] if len(_all_cycles) > 3 else None

    for metric in metrics:
        # Goal row
        goal_data = []
        for cycle in recent_3:
            g = tl_goal_dict.get((metric, cycle), agent_goal_dict.get((metric, cycle), math.nan))
            goal_data.append(f"{g:.0f}%" if not pd.isna(g) else "—")
        table_rows.append(goal_data)
        style_rows.append(["font-weight: bold;"] * len(recent_3))
        row_labels.append(f"{metric} Goal")

        # Value row
        val_data = []
        val_styles = []
        for cycle in recent_3:
            match = df_tl_long[
                (df_tl_long["TL"] == selected_tl)
                & (df_tl_long["Metric"] == metric)
                & (df_tl_long["Cycle"] == cycle)
            ]["Value"]
            val = match.iloc[0] if not match.empty else math.nan
            g = tl_goal_dict.get((metric, cycle), agent_goal_dict.get((metric, cycle), math.nan))
            val_data.append(f"{val:.2f}%" if not pd.isna(val) else "—")
            if not pd.isna(val) and not pd.isna(g):
                val_styles.append("background-color: #6DDA6D;" if val >= g else "background-color: #FF7F7F;")
            else:
                val_styles.append("")
        table_rows.append(val_data)
        style_rows.append(val_styles)
        row_labels.append(metric)

        # % Team on Goal row
        pct_data = []
        for cycle in recent_3:
            g = tl_goal_dict.get((metric, cycle), agent_goal_dict.get((metric, cycle), math.nan))
            if pd.isna(g):
                pct_data.append("—")
                continue
            tmp = df_tl_agents[
                (df_tl_agents["Metric"] == metric)
                & (df_tl_agents["Cycle"] == cycle)
            ]
            if tmp.empty:
                pct_data.append("—")
                continue
            rep_vals = tmp.groupby("Agent", as_index=False)["Value"].mean()
            valid = rep_vals["Value"].dropna()
            met = (valid >= g).sum()
            total = len(valid)
            pct_data.append(f"{met}/{total} ({met/total*100:.0f}%)" if total > 0 else "—")
        table_rows.append(pct_data)
        style_rows.append(["text-align: center;"] * len(recent_3))
        row_labels.append(f"{metric} % on Goal")  # unique per metric

        # Rank row with delta arrows and confetti
        tl_metric_pivot = df_tl_long[df_tl_long["Metric"] == metric].pivot(
            index="TL", columns="Cycle", values="Value"
        )

        def tl_rank_for_cycle(cycle, pivot=tl_metric_pivot):
            if cycle not in pivot.columns or selected_tl not in pivot.index:
                return None
            if pd.isna(pivot.loc[selected_tl, cycle]):
                return None
            series = pivot[cycle].dropna()
            return int(series.rank(ascending=False, method="dense")[selected_tl]), len(series)

        rank_data = []
        for i, cycle in enumerate(recent_3):
            result = tl_rank_for_cycle(cycle)
            if result is None:
                rank_data.append("—")
                continue
            r, out_of = result

            # Determine previous cycle for delta
            if i + 1 < len(recent_3):
                prev_c = recent_3[i + 1]
            elif extra_cycle:
                prev_c = extra_cycle
            else:
                prev_c = None

            delta_str = ""
            if prev_c:
                prev_result = tl_rank_for_cycle(prev_c)
                if prev_result:
                    prev_r, _ = prev_result
                    delta = prev_r - r  # positive = rank number went down = improvement
                    if delta > 0:
                        delta_str = f" ↑{delta}"
                    elif delta < 0:
                        delta_str = f" ↓{abs(delta)}"
                    else:
                        delta_str = " →"

            confetti = " 🎉" if r <= 3 else ""
            rank_data.append(f"{r} of {out_of}{delta_str}{confetti}")

        table_rows.append(rank_data)
        style_rows.append(["text-align: center;"] * len(recent_3))
        row_labels.append(f"{metric} Ranking")  # unique per metric

    df_3cycle = pd.DataFrame(table_rows, index=row_labels, columns=recent_3)
    style_3cycle = pd.DataFrame(style_rows, index=row_labels, columns=recent_3)

    # Add centering to all existing style cells
    for r in style_3cycle.index:
        for c in style_3cycle.columns:
            if "text-align" not in style_3cycle.loc[r, c]:
                style_3cycle.loc[r, c] += " text-align: center;"

    st.dataframe(
        df_3cycle.style.apply(lambda _: style_3cycle, axis=None),
        use_container_width=True,
    )
else:
    st.write("Not enough cycle data available.")

st.markdown("---")

# ============================================================
# METRIC-BY-METRIC PRESENTATION BLOCKS
# ============================================================

st.header("📊 Metric Deep-Dive — Rep Performance")

# Precompute global ranks per metric (for fastest reuse)
global_ranks_by_metric = {
    metric: compute_cycle_ranks(df_agent_long, metric) for metric in metrics
}

for metric in metrics:
    st.subheader(f"🎯 {metric}")

    # If no data for this metric for this TL, skip
    if df_tl_agents[df_tl_agents["Metric"] == metric].empty:
        st.write("No data for this metric for this team.")
        st.markdown("---")
        continue

    # ---------- Goal attainment (latest cycle) ----------
    goal_val = agent_goal_dict.get((metric, latest_cycle), math.nan)
    count_met = 0
    count_total = 0
    latest_rep_vals = None
    latest_metric_team = df_tl_agents[
        (df_tl_agents["Metric"] == metric)
        & (df_tl_agents["Cycle"] == latest_cycle)
    ]
    if not latest_metric_team.empty:
        latest_rep_vals = (
            latest_metric_team.groupby("Agent", as_index=False)["Value"].mean()
        )
        for _, r in latest_rep_vals.iterrows():
            v = r["Value"]
            if pd.isna(v):
                continue
            count_total += 1
            if not pd.isna(goal_val) and v >= goal_val:
                count_met += 1

    if count_total > 0 and not pd.isna(goal_val):
        pct_goal = count_met / count_total * 100.0
        st.markdown(
            f"🌱 **Goal Attainment:** {count_met} of {count_total} reps "
            f"meeting goal ({pct_goal:.0f}%)"
        )
    else:
        st.markdown("🌱 **Goal Attainment:** Not enough data for latest cycle.")

    # ---------- Top performer / most improved / biggest drop ----------
    ranks_by_cycle = global_ranks_by_metric[metric]
    latest_cycle_ranks = ranks_by_cycle.get(latest_cycle, {})

    # Filter to this TL's reps only
    team_agents = sorted(df_tl_agents["Agent"].unique())
    team_rank_records = []
    for agent in team_agents:
        r, delta = latest_cycle_ranks.get(agent, (math.nan, None))
        # Need latest value for display
        val = None
        if latest_rep_vals is not None:
            row = latest_rep_vals[latest_rep_vals["Agent"] == agent]
            if not row.empty:
                val = row["Value"].iloc[0]

        team_rank_records.append(
            {
                "Agent": agent,
                "Latest Value": val,
                "Latest Rank": r,
                "Rank Delta": delta,
            }
        )

    team_rank_df = pd.DataFrame(team_rank_records)

    # Top performer = smallest rank (1 is best) with non-NaN
    top_perf = None
    most_improved = None
    biggest_drop = None

    valid_rank_df = team_rank_df.dropna(subset=["Latest Rank"])

    if not valid_rank_df.empty:
        # Top performer
        top_perf = valid_rank_df.sort_values("Latest Rank", ascending=True).iloc[0]

        # Most improved = largest positive Rank Delta
        improved_df = valid_rank_df.dropna(subset=["Rank Delta"])
        improved_df_pos = improved_df[improved_df["Rank Delta"] > 0]
        if not improved_df_pos.empty:
            most_improved = improved_df_pos.sort_values(
                "Rank Delta", ascending=False
            ).iloc[0]

        # Biggest drop = most negative Rank Delta (only latest cycle, per your choice B)
        dropped_df = improved_df[improved_df["Rank Delta"] < 0]
        if not dropped_df.empty:
            biggest_drop = dropped_df.sort_values(
                "Rank Delta", ascending=True
            ).iloc[0]

    # Display summary callouts
    if top_perf is not None:
        st.markdown(
            f"🏆 **Top Performer:** {top_perf['Agent']} — "
            f"{'' if pd.isna(top_perf['Latest Value']) else f'{top_perf['Latest Value']:.2f}%'} "
            f"(Rank {int(top_perf['Latest Rank']) if not pd.isna(top_perf['Latest Rank']) else '?'} overall)"
        )
    if most_improved is not None:
        st.markdown(
            f"🚀 **Most Improved:** {most_improved['Agent']} — "
            f"moved up {int(most_improved['Rank Delta'])} ranks this cycle"
        )
    if biggest_drop is not None:
        st.markdown(
            f"⚠ **Biggest Drop:** {biggest_drop['Agent']} — "
            f"dropped {abs(int(biggest_drop['Rank Delta']))} ranks this cycle"
        )

    # ---------- Full rep table with goal row & ranks ----------
    rep_table, rep_style = build_rep_value_rank_table(
        metric,
        df_tl_agents,
        df_agent_long,
        agent_goal_dict,
        cycles_chrono,
        cycles_display,
    )

    st.markdown("**Scores & Ranks (Goal Row on Top, Ranks vs All Sales):**")
    st.dataframe(
        rep_table.style.apply(lambda _: rep_style, axis=None),
        use_container_width=True,
    )

    # ---------- Insights block ----------
    st.markdown("**📌 Insights:**")
    insight_lines = []

    if count_total > 0 and not pd.isna(goal_val):
        pct_goal = count_met / count_total * 100.0
        insight_lines.append(
            f"- {pct_goal:.0f}% of the team is meeting **{metric}** goal this cycle."
        )

    if top_perf is not None:
        insight_lines.append(
            f"- Top performer: **{top_perf['Agent']}** "
            f"(Rank {int(top_perf['Latest Rank']) if not pd.isna(top_perf['Latest Rank']) else '?'} overall)."
        )

    if most_improved is not None:
        insight_lines.append(
            f"- Most improved: **{most_improved['Agent']}** "
            f"(+{int(most_improved['Rank Delta'])} rank movement)."
        )

    if biggest_drop is not None:
        insight_lines.append(
            f"- Biggest drop: **{biggest_drop['Agent']}** "
            f"(-{abs(int(biggest_drop['Rank Delta']))} ranks); be ready to speak to what changed."
        )

    # TL vs last cycle for this metric
    tl_metric_this = tl_subset[tl_subset["Metric"] == metric]
    if not tl_metric_this.empty and latest_cycle in tl_metric_this["Cycle"].values:
        latest_val_tl = tl_metric_this[tl_metric_this["Cycle"] == latest_cycle]["Value"].iloc[0]
        if prev_cycle and prev_cycle in tl_metric_this["Cycle"].values:
            prev_val_tl = tl_metric_this[tl_metric_this["Cycle"] == prev_cycle]["Value"].iloc[0]
            if not pd.isna(latest_val_tl) and not pd.isna(prev_val_tl):
                delta_tl = latest_val_tl - prev_val_tl
                direction = "up" if delta_tl > 0 else "down" if delta_tl < 0 else "flat"
                if direction == "up":
                    insight_lines.append(
                        f"- Team {metric} is **up {delta_tl:.2f} pts** vs last cycle."
                    )
                elif direction == "down":
                    insight_lines.append(
                        f"- Team {metric} is **down {abs(delta_tl):.2f} pts** vs last cycle."
                    )
                else:
                    insight_lines.append(
                        f"- Team {metric} is roughly **flat vs last cycle**."
                    )

    if insight_lines:
        st.markdown("\n".join(insight_lines))
    else:
        st.markdown("- No strong signals yet; use this section to add your own notes.")

    st.markdown("---")

# ============================================================
# COACHING FOCUS REP (ONE REP TL IS WORKING CLOSEST WITH)
# ============================================================

st.header("🛠 Coaching Focus Rep")

team_agents = sorted(df_tl_agents["Agent"].unique())
focus_rep = st.selectbox(
    "Which rep are you working closest with this cycle?",
    ["(None)"] + team_agents,
)

if focus_rep != "(None)":
    st.subheader(f"Focus: {focus_rep}")

    focus_rows = []
    for metric in metrics:
        metric_data = df_tl_agents[df_tl_agents["Metric"] == metric]
        if metric_data.empty:
            continue

        # latest value
        latest_focus = metric_data[metric_data["Cycle"] == latest_cycle]
        latest_val = None
        if not latest_focus.empty:
            latest_val = (
                latest_focus[latest_focus["Agent"] == focus_rep]["Value"]
                .mean()
            )

        # rank and delta
        ranks_by_cycle = global_ranks_by_metric[metric]
        r, delta = ranks_by_cycle.get(latest_cycle, {}).get(focus_rep, (math.nan, None))

        focus_rows.append(
            {
                "Metric": metric,
                "Latest Value": latest_val,
                "Latest Rank": r,
                "Rank Δ": delta,
            }
        )

    if focus_rows:
        focus_df = pd.DataFrame(focus_rows)

        def fmt_val(x):
            return "" if pd.isna(x) else f"{x:.2f}%"

        def fmt_rank(x):
            return "" if pd.isna(x) else str(int(x))

        def fmt_delta(x):
            if x is None or pd.isna(x):
                return ""
            d = int(x)
            if d > 0:
                return f"↑{d}"
            if d < 0:
                return f"↓{abs(d)}"
            return "-"

        display_focus = focus_df.copy()
        display_focus["Latest Value"] = display_focus["Latest Value"].apply(fmt_val)
        display_focus["Latest Rank"] = display_focus["Latest Rank"].apply(fmt_rank)
        display_focus["Rank Δ"] = display_focus["Rank Δ"].apply(fmt_delta)

        st.markdown("**Current Snapshot (Latest Cycle):**")
        st.dataframe(display_focus, use_container_width=True)

    coaching_notes = st.text_area(
        "Coaching Notes / Plan",
        placeholder="Example: Focusing on opening, slowing down the quote, and clearer LT value framing. Reviewing 2 calls per day..."
    )
else:
    st.write("Select a rep above to show their current stats across metrics.")

st.markdown("---")

# ============================================================
# TL RANK VS OTHER TEAMS (SUMMARY)
# ============================================================

st.header("📊 TL Rank vs Other Teams (Latest Cycle)")

tl_rank_rows = []
for metric in metrics:
    temp = df_tl_long[df_tl_long["Metric"] == metric]
    pivot_tl = temp.pivot(index="TL", columns="Cycle", values="Value")
    if latest_cycle and latest_cycle in pivot_tl.columns and selected_tl in pivot_tl.index:
        series = pivot_tl[latest_cycle].dropna()
        if not series.empty:
            rank_df = (
                series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={latest_cycle: "Score"})
            )
            rank_df["Rank"] = rank_df["Score"].rank(
                ascending=False, method="dense"
            ).astype(int)
            rank_df["Out of TLs"] = len(rank_df)
            row = rank_df[rank_df["TL"] == selected_tl].iloc[0]
            tl_rank_rows.append(
                {
                    "Metric": metric,
                    "Cycle": latest_cycle,
                    "Score": row["Score"],
                    "Rank": int(row["Rank"]),
                    "Out of TLs": int(row["Out of TLs"]),
                }
            )

if tl_rank_rows:
    tl_rank_table = pd.DataFrame(tl_rank_rows)
    st.dataframe(tl_rank_table, use_container_width=True)
else:
    st.write("No TL rank data available for the latest cycle.")

st.markdown("---")

# ============================================================
# SUPERVISOR READOUT HELPER
# ============================================================

st.header("📣 Supervisor Readout Helper")

st.write(
    "Use this section to prep your talking points. You can copy/paste this into "
    "your slide or Slack message."
)

improvement_area = st.text_area(
    "1️⃣ What was your main improvement area this cycle?",
    placeholder="Example: LT attach was below goal; team was struggling on 3-cut minimum objections; QA dipped below 90%..."
)

actions_taken = st.text_area(
    "2️⃣ What specific actions did you take?",
    placeholder="Example: Daily 10-minute huddle on LT pitch; shared 3 best call clips; side-by-side coaching with Ashley and Mary; reinforced QA checklist..."
)

follow_through = st.text_area(
    "3️⃣ How did you follow through, and what results did you see?",
    placeholder="Example: LT attach up +3.5 pts; 60% of team now on goal; Mary jumped 39 ranks in conversion; QA back above 90%..."
)

if improvement_area or actions_taken or follow_through:
    st.subheader("📝 Copy/Paste Summary")
    st.markdown(
        f"- **Improvement Area:** {improvement_area or '—'}\n"
        f"- **Actions Taken:** {actions_taken or '—'}\n"
        f"- **Follow Through / Results:** {follow_through or '—'}"
    )
else:
    st.write("Fill in the fields above to generate a summary here.")
