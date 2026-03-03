import streamlit as st
import pandas as pd
import requests, zipfile, io, time, calendar
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="LTA Train Station Passenger Volume", layout="wide")

# --- API Key input in sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("LTA API Key", type="password", placeholder="Paste your API key here")
    if not api_key:
        st.warning("Please enter your API key to continue.")
        st.stop()

BASE_URL = "https://datamall2.mytransport.sg/ltaodataservice/PV/Train"

def month_starts_between(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    s = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    e = end_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    cur = s
    while cur <= e:
        out.append(cur)
        y = cur.year + (cur.month // 12)
        m = (cur.month % 12) + 1
        cur = cur.replace(year=y, month=m)
    return out

def last_n_completed_months(n: int = 3) -> set[str]:
    today = datetime.today().replace(day=1)
    months = []
    cur = today
    for _ in range(n):
        y = cur.year if cur.month > 1 else cur.year - 1
        m = cur.month - 1 if cur.month > 1 else 12
        cur = cur.replace(year=y, month=m)
        months.append(cur.strftime("%Y%m"))
    return set(months)

def get_with_retry(session: requests.Session, url: str, *, headers=None, params=None, timeout=30, max_tries=5):
    backoff = 1.0
    for attempt in range(1, max_tries + 1):
        resp = session.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = backoff + (0.1 * attempt)
            time.sleep(sleep_s)
            backoff *= 2
            continue
        return resp
    return resp

def download_zip_csv_to_df(session: requests.Session, link: str) -> pd.DataFrame:
    r = get_with_retry(session, link, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip().upper() for c in df.columns]
    return df

@st.cache_data(ttl=6 * 3600)
def fetch_pv_train(api_key: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:

    months = [d.strftime("%Y%m") for d in month_starts_between(start_dt, end_dt)]

    allowed = last_n_completed_months(3)
    months = [m for m in months if m in allowed]
    months = sorted(set(months))

    headers = {"AccountKey": api_key, "accept": "application/json"}
    session = requests.Session()

    frames = []
    for ym in months:
        params = {"Date": ym}
        resp = get_with_retry(session, BASE_URL, headers=headers, params=params, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(f"PV/Train failed for {ym}: {resp.status_code} {resp.text[:200]}")

        data = resp.json()
        values = data.get("value", [])
        if not values or "Link" not in values[0]:
            raise RuntimeError(f"No download link returned for {ym}. Response: {str(data)[:300]}")

        link = values[0]["Link"]
        df_month = download_zip_csv_to_df(session, link)
        frames.append(df_month)

        time.sleep(0.2)

    return pd.concat(frames, ignore_index=True)

st.title("LTA Train Station Passenger Volume Analysis")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=120))
with col2:
    end_date = st.date_input("End Date", value=datetime.now() - timedelta(days=60))

try:
    raw_df = fetch_pv_train(api_key, datetime.combine(start_date, datetime.min.time()),
                            datetime.combine(end_date, datetime.min.time()))
    st.success(f"Fetched {len(raw_df):,} rows.")

    import altair as alt
    import numpy as np

    STATION_CODES = {
        "Tanjong Pagar": ["EW15"],
        "Raffles Place": ["EW14/NS26", "NS26/EW14", "EW14", "NS26"],
        "City Hall": ["EW13/NS25", "NS25/EW13", "EW13", "NS25"],
        "Woodlands": ["NS9/TE2", "TE2/NS9", "NS9", "TE2"],
    }

    def month_daytype_counts(year: int, month: int) -> dict[str, int]:
        days_in_month = calendar.monthrange(year, month)[1]
        wd = 0
        we = 0
        for d in range(1, days_in_month + 1):
            dow = datetime(year, month, d).weekday()
            if dow < 5:
                wd += 1
            else:
                we += 1
        return {"WEEKDAY": wd, "WEEKENDS/HOLIDAY": we}

    def compute_monthly_avg_inflow(raw_df: pd.DataFrame, *, stations: list[str], start_hour=7, end_hour=9):
        df = raw_df.copy()

        df["TIME_PER_HOUR"] = pd.to_numeric(df["TIME_PER_HOUR"], errors="coerce")
        df = df.dropna(subset=["TIME_PER_HOUR"])

        hours = list(range(start_hour, end_hour))

        station_codes = set()
        for s in stations:
            station_codes.update(STATION_CODES.get(s, []))
        
        print("Before", df["PT_CODE"].unique())
        df = df[(df["PT_TYPE"] == "TRAIN") & (df["PT_CODE"].isin(station_codes)) & (df["TIME_PER_HOUR"].isin(hours))]
        print("After", df["PT_CODE"].unique())

        ym = df["YEAR_MONTH"].astype(str).str.strip()
        df["MONTH"] = pd.to_datetime(
            np.where(ym.str.contains("-"), ym, ym.str.slice(0, 4) + "-" + ym.str.slice(4, 6)),
            format="%Y-%m",
            errors="coerce",
        )
        df = df.dropna(subset=["MONTH"])

        g = df.groupby(["MONTH", "DAY_TYPE", "PT_CODE"], as_index=False)["TOTAL_TAP_IN_VOLUME"].sum()

        def _daycount(row):
            y = int(row["MONTH"].year)
            m = int(row["MONTH"].month)
            counts = month_daytype_counts(y, m)
            return counts.get(row["DAY_TYPE"], np.nan)

        g["DAY_COUNT"] = g.apply(_daycount, axis=1)
        g["AVG_TAP_IN_PER_DAY_7_9"] = g["TOTAL_TAP_IN_VOLUME"] / g["DAY_COUNT"]

        inv = {}
        for name, codes in STATION_CODES.items():
            for c in codes:
                inv[c] = name
        g["STATION"] = g["PT_CODE"].map(inv).fillna(g["PT_CODE"])

        return g.sort_values(["MONTH", "STATION", "DAY_TYPE"])

    st.subheader("Monthly Average Inflow — 7–9am")

    stations_sel = st.multiselect(
        "Stations",
        options=list(STATION_CODES.keys()),
        default=["Tanjong Pagar", "Raffles Place", "City Hall", "Woodlands"],
    )

    daytype_sel = st.multiselect(
        "Day type",
        options=["WEEKDAY", "WEEKENDS/HOLIDAY"],
        default=["WEEKDAY"],
    )

    hour_window = st.slider("Hour window (start inclusive, end exclusive)", 0, 24, (7, 9))
    start_h, end_h = hour_window

    monthly_avg = compute_monthly_avg_inflow(raw_df, stations=stations_sel, start_hour=start_h, end_hour=end_h)
    monthly_avg = monthly_avg[monthly_avg["DAY_TYPE"].isin(daytype_sel)]

    st.dataframe(monthly_avg, use_container_width=True)

    chart = (
        alt.Chart(monthly_avg)
        .mark_line(point=True)
        .encode(
            x=alt.X("MONTH:T", title="Month"),
            y=alt.Y("AVG_TAP_IN_PER_DAY_7_9:Q", title="Avg tap-in per day (selected hours)"),
            color=alt.Color("STATION:N", title="Station"),
            tooltip=["MONTH:T", "STATION:N", "DAY_TYPE:N", "TOTAL_TAP_IN_VOLUME:Q", "DAY_COUNT:Q", "AVG_TAP_IN_PER_DAY_7_9:Q"],
        )
        .properties(height=380)
    )

    if len(daytype_sel) > 1:
        chart = chart.facet(row=alt.Row("DAY_TYPE:N", title=None))

    st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(str(e))
