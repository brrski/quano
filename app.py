"""
NASDAQ-100 Momentum & Volume-Confirmation Monitor
-------------------------------------------------
Run:  pip install streamlit yfinance pandas numpy altair requests lxml
      streamlit run ndx_momentum_monitor.py

Methodology
-----------
PRICE (relative strength via simple moving averages, hourly bars):
  p_fast  : Price / SMA(fast) - 1            short-term extension
  p_slow  : Price / SMA(slow) - 1            medium-term extension
  spread  : SMA(fast) / SMA(slow) - 1        trend alignment
  slope   : 6-bar % change of SMA(fast)      trend acceleration
  d_dist  : Price / daily SMA(n) - 1         higher-timeframe trend filter
  rs      : N-bar return minus QQQ return    cross-sectional relative strength
  Each factor is cross-sectionally z-scored (winsorised at +/-3), averaged, and
  re-standardised -> "Mom Z". Standouts are |Mom Z| >= threshold.

VOLUME (validation of price action):
  RVOL (hr)  : current-bar volume / (time-prorated) baseline hourly average volume.
               Baseline = trailing N sessions, same intraday slot (or all slots).
  RVOL (day) : cumulative volume today / expected cumulative volume to this time.
  CMF        : Chaikin Money Flow over K bars (close location within bar x volume).
  Up-Vol %   : share of volume traded in up bars (close > open) over K bars.
  A move is "Confirmed" when trend + Mom Z agree AND RVOL(day) >= threshold AND
  CMF / Up-Vol% agree with the direction. Otherwise it is flagged "Unconfirmed".
"""

from datetime import datetime
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NDX-100 Momentum Monitor", layout="wide", page_icon="📈")

NY = "America/New_York"
BENCH = "QQQ"
HIST_PERIOD = "6mo"  # ~125 sessions of 1h bars -> enough for a daily SMA(50)

# Fallback constituent list (verify against current index; use the Wikipedia refresh in the sidebar).
FALLBACK_NDX = list(dict.fromkeys("""
AAPL MSFT NVDA AMZN META AVGO GOOGL TSLA QQQ COST WMT NFLX PLTR ASML AMD COHR LITE SNDK STX SKHY
CLS ARM AMAT GILD LLY V SHOP PANW MU PURR CRWD MELI PBRS SOFI COIN HOOD IBKR SPY IWM
LRCX CDNS MRVL TTWO SLS GSIT ONDS RKLB SPCX ALAB AMBA RBRK FLEX RXT NOK HPE LUNR
""".split()))

SIGNAL_COLORS = {
    "✅ Confirmed Long": "#16a34a",
    "⚠️ Unconfirmed Long": "#eab308",
    "Neutral": "#9ca3af",
    "⚠️ Unconfirmed Weakness": "#f97316",
    "🔻 Confirmed Weakness": "#dc2626",
}


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_constituents():
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        for tb in pd.read_html(StringIO(r.text)):
            if "Ticker" in tb.columns:
                tk = tb["Ticker"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
                if len(tk) >= 90:
                    return tk
    except Exception:
        pass
    return None


@st.cache_data(ttl=45, show_spinner=False)
def load_data(tickers: tuple):
    """One batched hourly download. TTL keeps repeated reruns from hammering Yahoo."""
    raw = yf.download(
        list(tickers),
        period=HIST_PERIOD,
        interval="1h",
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
        threads=True,
        progress=False,
    )
    cols = ["Open", "High", "Low", "Close", "Volume"]
    out = {}
    for t in tickers:
        try:
            d = raw[t][cols].dropna(subset=["Close"]).copy()
        except (KeyError, TypeError):
            continue
        if d.empty:
            continue
        idx = d.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        d.index = idx.tz_convert(NY)
        d["Volume"] = d["Volume"].fillna(0)
        out[t] = d
    return out


# ----------------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------------
def analyze(ticker, df, bench_ret, p):
    if len(df) < max(p["slow"], p["rs_bars"]) + 10:
        return None

    df = df.copy()
    df["date"] = df.index.date
    df["slot"] = df.groupby("date").cumcount()
    close = df["Close"]

    sma_f = close.rolling(p["fast"]).mean()
    sma_s = close.rolling(p["slow"]).mean()
    px = float(close.iloc[-1])

    daily_close = df.groupby("date")["Close"].last()
    sma_d = daily_close.rolling(p["daily"]).mean().iloc[-1]
    last_date = df["date"].iloc[-1]
    prev = daily_close[daily_close.index < last_date]
    chg = px / prev.iloc[-1] - 1 if len(prev) else np.nan

    n = p["rs_bars"]
    rs = (px / close.iloc[-1 - n] - 1) - bench_ret
    slope = sma_f.iloc[-1] / sma_f.iloc[-7] - 1 if len(sma_f.dropna()) > 7 else np.nan

    # ---- current bar state (live / prorated) ----
    last_pos = len(df) - 1
    ts = df.index[last_pos]
    end = min(ts + pd.Timedelta(hours=1), ts.normalize() + pd.Timedelta(hours=16))
    now = pd.Timestamp.now(tz=NY)
    live = bool(ts <= now < end)
    pos, frac = last_pos, 1.0
    if live:
        if p["basis"] == "Last completed bar":
            pos = last_pos - 1
        else:
            frac = float(np.clip((now - ts) / (end - ts), 0.10, 1.0))

    ref = df.iloc[pos]
    ref_date, slot = ref["date"], int(ref["slot"])

    dates = list(dict.fromkeys(df["date"]))
    prior = [d for d in dates if d < ref_date][-p["vol_days"]:]
    pivot = df[df["date"].isin(prior)].pivot(index="date", columns="slot", values="Volume")
    if pivot.empty:
        return None

    base_all = float(np.nanmean(pivot.values))
    base_slot = float(pivot[slot].mean()) if slot in pivot.columns else np.nan
    if np.isnan(base_slot):
        base_slot = base_all
    baseline = base_slot if p["same_slot"] else base_all

    vol_cur = float(ref["Volume"])
    rvol_hr = vol_cur / (baseline * frac) if baseline > 0 else np.nan

    today_cum = float(df[(df["date"] == ref_date) & (df["slot"] <= slot)]["Volume"].sum())
    exp_prev = pivot.loc[:, pivot.columns < slot].sum(axis=1).mean() if slot > 0 else 0.0
    exp_prev = 0.0 if pd.isna(exp_prev) else float(exp_prev)
    exp_cum = exp_prev + frac * base_slot
    rvol_day = today_cum / exp_cum if exp_cum > 0 else np.nan

    # ---- buying-pressure measures over the last K bars ----
    w = df.iloc[max(0, pos - p["cmf"] + 1): pos + 1]
    hl = (w["High"] - w["Low"]).replace(0, np.nan)
    mfm = (((w["Close"] - w["Low"]) - (w["High"] - w["Close"])) / hl).fillna(0)
    tot_v = w["Volume"].sum()
    cmf = float((mfm * w["Volume"]).sum() / tot_v) if tot_v > 0 else np.nan
    upvol = float(w.loc[w["Close"] > w["Open"], "Volume"].sum() / tot_v) if tot_v > 0 else np.nan

    return {
        "Ticker": ticker,
        "Price": px,
        "Chg %": chg * 100,
        "p_fast": (px / sma_f.iloc[-1] - 1) * 100,
        "p_slow": (px / sma_s.iloc[-1] - 1) * 100,
        "spread": (sma_f.iloc[-1] / sma_s.iloc[-1] - 1) * 100,
        "slope": slope * 100,
        "d_dist": (px / sma_d - 1) * 100 if pd.notna(sma_d) else np.nan,
        "rs": rs * 100,
        "Bar Vol": vol_cur,
        "Hourly Avg Vol": baseline,
        "RVOL (hr)": rvol_hr,
        "RVOL (day)": rvol_day,
        "CMF": cmf,
        "Up-Vol %": upvol * 100 if pd.notna(upvol) else np.nan,
        "trend_up": bool(px > sma_f.iloc[-1] > sma_s.iloc[-1]),
        "trend_dn": bool(px < sma_f.iloc[-1] < sma_s.iloc[-1]),
        "live": live,
        "last_bar": ts,
    }


def build_table(data, p):
    bench = data.get(BENCH)
    if bench is None or len(bench) <= p["rs_bars"] + 1:
        return None
    bc = bench["Close"]
    bench_ret = bc.iloc[-1] / bc.iloc[-1 - p["rs_bars"]] - 1

    rows = [r for t, d in data.items() if t != BENCH
            for r in [analyze(t, d, bench_ret, p)] if r is not None]
    if not rows:
        return None
    df = pd.DataFrame(rows)

    feats = ["p_fast", "p_slow", "spread", "slope", "d_dist", "rs"]
    Z = (df[feats] - df[feats].mean()) / df[feats].std(ddof=0).replace(0, np.nan)
    raw = Z.clip(-3, 3).mean(axis=1, skipna=True)
    df["Mom Z"] = (raw - raw.mean()) / raw.std(ddof=0)

    vol_ok = df["RVOL (day)"] >= p["rvol_thr"]
    buy = (df["CMF"] > 0) & (df["Up-Vol %"] > 50)
    sell = (df["CMF"] < 0) & (df["Up-Vol %"] < 50)
    long_px = (df["Mom Z"] > 0) & df["trend_up"]
    short_px = (df["Mom Z"] < 0) & df["trend_dn"]

    df["Signal"] = np.select(
        [long_px & vol_ok & buy, long_px, short_px & vol_ok & sell, short_px],
        ["✅ Confirmed Long", "⚠️ Unconfirmed Long", "🔻 Confirmed Weakness", "⚠️ Unconfirmed Weakness"],
        default="Neutral",
    )
    return df.sort_values("Mom Z", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_chart(ch):
    try:
        st.altair_chart(ch, width="stretch")
    except TypeError:
        st.altair_chart(ch, use_container_width=True)


COLCFG = {
    "Price": st.column_config.NumberColumn(format="$%.2f"),
    "Chg %": st.column_config.NumberColumn(format="%+.2f%%"),
    "Mom Z": st.column_config.NumberColumn(format="%+.2f"),
    "vs SMA fast %": st.column_config.NumberColumn(format="%+.2f%%"),
    "vs SMA slow %": st.column_config.NumberColumn(format="%+.2f%%"),
    "SMA spread %": st.column_config.NumberColumn(format="%+.2f%%"),
    "SMA slope %": st.column_config.NumberColumn(format="%+.2f%%"),
    "vs Daily SMA %": st.column_config.NumberColumn(format="%+.2f%%"),
    "RS vs QQQ %": st.column_config.NumberColumn(format="%+.2f%%"),
    "Bar Vol": st.column_config.NumberColumn(format="%d"),
    "Hourly Avg Vol": st.column_config.NumberColumn(format="%d"),
    "RVOL (hr)": st.column_config.NumberColumn(format="%.2fx"),
    "RVOL (day)": st.column_config.NumberColumn(format="%.2fx"),
    "CMF": st.column_config.NumberColumn(format="%+.2f"),
    "Up-Vol %": st.column_config.NumberColumn(format="%.0f%%"),
}
RENAME = {
    "p_fast": "vs SMA fast %", "p_slow": "vs SMA slow %", "spread": "SMA spread %",
    "slope": "SMA slope %", "d_dist": "vs Daily SMA %", "rs": "RS vs QQQ %",
}


def drilldown(data, ticker, p):
    d = data[ticker].tail(160).copy()
    d["SMA fast"] = data[ticker]["Close"].rolling(p["fast"]).mean().tail(160)
    d["SMA slow"] = data[ticker]["Close"].rolling(p["slow"]).mean().tail(160)
    d["bar"] = range(len(d))
    d["time"] = d.index.strftime("%m-%d %H:%M")
    long = d.melt(id_vars=["bar", "time"], value_vars=["Close", "SMA fast", "SMA slow"],
                  var_name="series", value_name="value")
    price = (
        alt.Chart(long).mark_line().encode(
            x=alt.X("bar:Q", axis=None),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="Price"),
            color=alt.Color("series:N", scale=alt.Scale(
                domain=["Close", "SMA fast", "SMA slow"], range=["#e5e7eb", "#38bdf8", "#f59e0b"])),
            tooltip=["time", "series", alt.Tooltip("value:Q", format=".2f")],
        ).properties(height=260)
    )
    vol = (
        alt.Chart(d).mark_bar().encode(
            x=alt.X("bar:Q", title="Hourly bars (most recent at right)"),
            y=alt.Y("Volume:Q", title="Volume"),
            color=alt.condition(alt.datum.Close >= alt.datum.Open, alt.value("#16a34a"), alt.value("#dc2626")),
            tooltip=["time", "Volume"],
        ).properties(height=110)
    )
    show_chart(alt.vconcat(price, vol).resolve_scale(x="shared"))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("📈 NASDAQ Momentum & Volume Scan")

with st.sidebar:
    st.header("Universe")
    use_wiki = st.checkbox("Refresh constituents from Wikipedia", value=False)
    st.header("Price strength (SMA)")
    fast = st.number_input("Fast SMA (hourly bars)", 5, 100, 20)
    slow = st.number_input("Slow SMA (hourly bars)", 10, 300, 50)
    daily = st.number_input("Daily SMA (sessions)", 10, 100, 50)
    rs_bars = st.number_input("Rel. strength lookback (bars, ~7/day)", 7, 140, 35)
    z_thr = st.slider("Standout threshold (|Mom Z|)", 0.5, 3.0, 1.0, 0.1)
    st.header("Volume validation")
    basis = st.radio("Current-bar basis", ["Current bar (time-prorated)", "Last completed bar"])
    same_slot = st.checkbox("Baseline = same intraday slot", value=True,
                            help="Removes U-shaped intraday volume bias (open/close are naturally heavier).")
    vol_days = st.slider("Baseline lookback (sessions)", 5, 60, 20)
    rvol_min = st.slider("Min RVOL (hr) for volume list", 1.0, 5.0, 1.0, 0.1)
    rvol_thr = st.slider("RVOL (day) needed to confirm", 0.5, 3.0, 1.0, 0.1)
    cmf_bars = st.slider("CMF / Up-Vol window (bars)", 7, 42, 14)
    st.header("Refresh")
    auto = st.checkbox("Auto-refresh", value=True)
    every = st.slider("Interval (seconds)", 30, 300, 60, 10)
    if st.button("Refresh now"):
        st.cache_data.clear()

params = dict(
    fast=int(fast), slow=int(slow), daily=int(daily), rs_bars=int(rs_bars),
    basis=basis, same_slot=same_slot, vol_days=vol_days, rvol_thr=rvol_thr, cmf=cmf_bars,
)
if params["fast"] >= params["slow"]:
    st.sidebar.error("Fast SMA must be shorter than slow SMA.")
    st.stop()

universe = (fetch_constituents() if use_wiki else None) or FALLBACK_NDX


# ----------------------------------------------------------------------------
# Live dashboard (re-runs on a timer without re-running the sidebar)
# ----------------------------------------------------------------------------
def dashboard():
    with st.spinner("Pulling hourly bars from Yahoo Finance..."):
        data = load_data(tuple(universe + [BENCH]))
    tbl = build_table(data, params) if data else None
    if tbl is None or tbl.empty:
        st.error("No data returned (rate-limited or offline). Try again shortly.")
        return

    live = bool(tbl["live"].mean() > 0.5)
    last_bar = tbl["last_bar"].max()
    status = "🟢 Live session" if live else "⚪ Market closed - showing last session"
    st.caption(f"{status}  |  last bar: {last_bar:%Y-%m-%d %H:%M} ET  |  "
               f"refreshed {datetime.now():%H:%M:%S}  |  {len(tbl)} tickers  |  "
               "Yahoo data may be delayed ~15 min")

    above_vol = tbl[tbl["RVOL (hr)"] > rvol_min]
    c = st.columns(5)
    c[0].metric("Above fast SMA", f"{(tbl['p_fast'] > 0).mean():.0%}")
    c[1].metric("Above daily SMA", f"{(tbl['d_dist'] > 0).mean():.0%}")
    c[2].metric("Confirmed longs", int((tbl["Signal"] == "✅ Confirmed Long").sum()))
    c[3].metric("Confirmed weakness", int((tbl["Signal"] == "🔻 Confirmed Weakness").sum()))
    c[4].metric("Above hourly avg vol", len(above_vol))

    disp = tbl.rename(columns=RENAME)
    price_cols = ["Ticker", "Price", "Chg %", "Mom Z", "vs SMA fast %", "vs SMA slow %", "SMA spread %",
                  "SMA slope %", "vs Daily SMA %", "RS vs QQQ %", "RVOL (day)", "CMF", "Up-Vol %", "Signal"]
    vol_cols = ["Ticker", "Price", "Chg %", "Bar Vol", "Hourly Avg Vol", "RVOL (hr)", "RVOL (day)",
                "CMF", "Up-Vol %", "Mom Z", "Signal"]

    t1, t2, t3, t4 = st.tabs(["🏆 Momentum Standouts", "📊 Above Hourly Avg Volume",
                              "🔍 Ticker Drilldown", "🗂 Full Universe"])

    with t1:
        lead = disp[disp["Mom Z"] >= z_thr]
        lag = disp[disp["Mom Z"] <= -z_thr].sort_values("Mom Z")
        a, b = st.columns(2)
        with a:
            st.subheader(f"Leaders (Z >= {z_thr:.1f}) - {len(lead)}")
            show_df(lead[price_cols], column_config=COLCFG)
        with b:
            st.subheader(f"Laggards (Z <= -{z_thr:.1f}) - {len(lag)}")
            show_df(lag[price_cols], column_config=COLCFG)

        st.subheader("Price strength vs. volume support")
        sc = tbl.dropna(subset=["Mom Z", "RVOL (day)"])
        pts = alt.Chart(sc).mark_circle(size=90, opacity=0.85).encode(
            x=alt.X("Mom Z:Q", title="Momentum Z (SMA-based relative strength)"),
            y=alt.Y("RVOL (day):Q", title="Relative volume (day, time-adjusted)"),
            color=alt.Color("Signal:N", scale=alt.Scale(
                domain=list(SIGNAL_COLORS), range=list(SIGNAL_COLORS.values()))),
            tooltip=["Ticker", alt.Tooltip("Mom Z:Q", format="+.2f"),
                     alt.Tooltip("RVOL (day):Q", format=".2f"), "Signal"],
        )
        txt = alt.Chart(sc[sc["Mom Z"].abs() >= z_thr]).mark_text(dy=-9, fontSize=10, color="#d1d5db").encode(
            x="Mom Z:Q", y="RVOL (day):Q", text="Ticker")
        rules = alt.Chart(pd.DataFrame({"y": [rvol_thr]})).mark_rule(strokeDash=[4, 4], color="#6b7280").encode(y="y:Q")
        vr = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#6b7280").encode(x="x:Q")
        show_chart((pts + txt + rules + vr).properties(height=420).interactive())
        st.caption("Top-right = strong trend with volume support. Top-left = selling with volume. "
                   "Bottom-right = strength on thin volume (fade risk / unconfirmed).")

    with t2:
        st.subheader(f"Stocks with current-hour volume above hourly average (RVOL > {rvol_min:.1f}x)")
        opts = list(SIGNAL_COLORS)
        pick = st.multiselect("Filter by signal", opts, default=opts, key="sigfilter")
        v = disp[(disp["RVOL (hr)"] > rvol_min) & disp["Signal"].isin(pick)] \
            .sort_values("RVOL (hr)", ascending=False)
        show_df(v[vol_cols], column_config=COLCFG)
        st.caption(
            "Current-bar volume is compared with a time-prorated baseline so a half-finished hour is not "
            "penalised. Baseline = trailing sessions' average volume for the same intraday slot (sidebar)."
        )

    with t3:
        sel = st.selectbox("Ticker", tbl["Ticker"].tolist(), key="dd_ticker")
        r = tbl[tbl["Ticker"] == sel].iloc[0]
        m = st.columns(5)
        m[0].metric("Price", f"${r['Price']:.2f}", f"{r['Chg %']:+.2f}%")
        m[1].metric("Mom Z", f"{r['Mom Z']:+.2f}")
        m[2].metric("RVOL (hr)", f"{r['RVOL (hr)']:.2f}x")
        m[3].metric("CMF", f"{r['CMF']:+.2f}")
        m[4].metric("Signal", r["Signal"])
        drilldown(data, sel, params)

    with t4:
        show_df(disp[price_cols[:-1] + ["Bar Vol", "Hourly Avg Vol", "RVOL (hr)", "Signal"]],
                column_config=COLCFG)
        st.download_button("Download CSV", disp.drop(columns=["last_bar"]).to_csv(index=False),
                           file_name=f"ndx_momentum_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")


if hasattr(st, "fragment"):
    st.fragment(run_every=(every if auto else None))(dashboard)()
else:
    dashboard()
    if auto:
        st.info("Upgrade Streamlit (>=1.37) for timer-based auto-refresh; use the browser refresh meanwhile.")
