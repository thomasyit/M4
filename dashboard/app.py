import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st


STATE_DIR = Path(os.getenv("BOT_STATE_DIR", "state"))
RECEIPTS_DIR = Path(os.getenv("RECEIPTS_DIR", STATE_DIR / "receipts"))
LOG_FILE = Path(os.getenv("BOT_LOG_FILE", STATE_DIR / "bot.log"))
RUN_SUMMARY_DIR = Path(os.getenv("RUN_SUMMARY_DIR", STATE_DIR / "run_summaries"))


st.set_page_config(
    page_title="M4 Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("M4 Bot Dashboard")

# Style similar to ai_bot dashboard
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 16px;
        border-radius: 10px;
        margin: 8px 0;
        color: #000000;
    }
    .profit { color: #00cc00; font-weight: bold; }
    .loss { color: #ff0000; font-weight: bold; }
    .neutral { color: #888888; font-weight: bold; }
    .sentiment-bullish { color: #00cc00; font-weight: bold; font-size: 1.1em; }
    .sentiment-bearish { color: #ff0000; font-weight: bold; font-size: 1.1em; }
    .sentiment-neutral { color: #888888; font-weight: bold; font-size: 1.1em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _utc_to_sgt(ts: int) -> str:
    try:
        utc = datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.UTC)
        sgt = utc.astimezone(pytz.timezone("Asia/Singapore"))
        return sgt.strftime("%Y-%m-%d %H:%M:%S SGT")
    except Exception:
        return "N/A"
st.sidebar.header("Chart Settings")
default_symbol = os.getenv("DASHBOARD_SYMBOL", "BTCUSDT")
symbol = st.sidebar.text_input("Symbol", value=default_symbol)
interval = st.sidebar.selectbox("Candle Interval", ["15m", "1h", "4h", "1d"], index=1)
candle_limit = st.sidebar.slider("Candle Count", min_value=50, max_value=300, value=150, step=10)


def _load_receipts():
    if not RECEIPTS_DIR.exists():
        return []
    rows = []
    for path in sorted(RECEIPTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        intent = payload.get("intent", {})
        details = payload.get("details", {})
        rows.append(
            {
                "run_id": payload.get("run_id") or path.stem,
                "status": payload.get("status"),
                "reason": details.get("reason"),
                "symbol": intent.get("symbol"),
                "side": intent.get("side"),
                "intent": intent.get("intent"),
                "confidence": intent.get("confidence"),
                "connector": intent.get("execution", {}).get("connector"),
                "timestamp_epoch": intent.get("ts"),
                "timestamp (SGT)": _utc_to_sgt(intent.get("ts") or 0),
                "sentiment": intent.get("metadata", {}).get("sentiment"),
                "analysis_model": intent.get("metadata", {}).get("analysis_model"),
                "path": str(path),
            }
        )
    return rows


def _load_run_summaries():
    if not RUN_SUMMARY_DIR.exists():
        return []
    rows = []
    for path in sorted(RUN_SUMMARY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "run_id": payload.get("run_id") or path.stem,
                "mode": payload.get("mode"),
                "decision": payload.get("decision"),
                "reason": payload.get("reason"),
                "chatgpt": payload.get('chatgpt_sentiment'),
                "chatgpt_conf": payload.get("chatgpt_confidence"),
                "synternet": payload.get('synternet_sentiment'),
                "synternet_conf": payload.get("synternet_confidence"),
                "ta_sentiment": payload.get("ta_sentiment"),
                "ta_conf": payload.get("ta_confidence"),
                "freshness_h": payload.get("data_fresh_age_hours"),
                "timestamp_epoch": payload.get("ts"),
                "timestamp (SGT)": _utc_to_sgt(payload.get("ts") or 0),
                "path": str(path),
            }
        )
    return rows


def _tail_log(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return "Log file not found."
    try:
        data = path.read_text(errors="ignore").splitlines()
    except Exception as exc:
        return f"Failed to read log: {exc}"
    return "\n".join(data[-lines:])


st.subheader("Recent Runs (decision pipeline)")
run_summaries = _load_run_summaries()
if run_summaries:
    display_runs = [{k: v for k, v in r.items() if k != "timestamp_epoch"} for r in run_summaries]
    st.dataframe(display_runs, use_container_width=True)
else:
    st.info("No run summaries found yet.")

st.subheader("Recent Receipts (execution outcome)")
receipts = _load_receipts()
if receipts:
    status_options = sorted({r["status"] for r in receipts if r.get("status")})
    status_filter = st.multiselect("Status", status_options, default=status_options)
    date_range = st.date_input(
        "Date range",
        value=(datetime.utcnow().date() - timedelta(days=7), datetime.utcnow().date()),
    )
    start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)

    filtered = []
    for r in receipts:
        ts = r.get("timestamp_epoch")
        if ts:
            dt = datetime.utcfromtimestamp(ts).date()
        else:
            dt = datetime.utcfromtimestamp(Path(r["path"]).stat().st_mtime).date()

        if status_filter and r.get("status") not in status_filter:
            continue
        if dt < start_date or dt > end_date:
            continue
        filtered.append(r)

    display_filtered = [{k: v for k, v in r.items() if k != "timestamp_epoch"} for r in filtered]
    st.dataframe(display_filtered, use_container_width=True)
else:
    st.info("No receipts found yet.")

st.subheader("Last 24h Stats")
if receipts:
    now = datetime.utcnow()
    last_day = []
    for r in receipts:
        ts = r.get("timestamp_epoch")
        dt = datetime.utcfromtimestamp(ts) if ts else datetime.utcfromtimestamp(Path(r["path"]).stat().st_mtime)
        if dt >= now - timedelta(hours=24):
            last_day.append(r)
    total = len(last_day)
    by_status = {}
    for r in last_day:
        by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    st.metric("Runs (24h)", total)
    if by_status:
        st.write(by_status)
else:
    st.info("No data for last 24h yet.")

st.subheader("Snapshot")
if receipts:
    last = receipts[0]
    status = last.get("status", "unknown")
    sentiment = (last.get("sentiment") or "neutral").lower()
    sentiment_class = f"sentiment-{sentiment}"
    st.markdown(
        f"""
        <div class="metric-card">
          <div><strong>Last Run</strong>: {_utc_to_sgt(last.get("timestamp_epoch") or 0)}</div>
          <div><strong>Status</strong>: {status}</div>
          <div><strong>Sentiment</strong>: <span class="{sentiment_class}">{sentiment.upper()}</span></div>
          <div><strong>Model</strong>: {last.get('analysis_model') or 'N/A'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No runs yet.")

st.subheader("Reasoning")
if run_summaries:
    latest_id = run_summaries[0]["run_id"]
    latest_path = RUN_SUMMARY_DIR / f"{latest_id}.json"
    try:
        latest_payload = json.loads(latest_path.read_text())
    except json.JSONDecodeError:
        latest_payload = {}
    col_chatgpt, col_synternet = st.columns(2)
    with col_chatgpt:
        st.markdown("**ChatGPT Reasoning**")
        st.text_area(
            "ChatGPT reasoning",
            value=latest_payload.get("chatgpt_reasoning") or "N/A",
            height=220,
            label_visibility="collapsed",
        )
    with col_synternet:
        st.markdown("**Synternet Reasoning**")
        st.text_area(
            "Synternet reasoning",
            value=latest_payload.get("synternet_reasoning") or "N/A",
            height=220,
            label_visibility="collapsed",
        )
else:
    st.info("No reasoning output available yet.")

st.subheader("Last Intent")
if receipts:
    last_id = receipts[0]["run_id"]
    last_path = RECEIPTS_DIR / f"{last_id}.json"
    st.code(last_path.read_text(), language="json")
else:
    st.info("No intent available.")

st.subheader("Trade Chart")
if receipts:
    points = []
    overlays = []
    for r in receipts:
        path = Path(r["path"])
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        status = payload.get("status")
        intent = payload.get("intent", {})
        details = payload.get("details", {})
        ts = intent.get("ts")
        if not ts:
            continue
        order_payload = details.get("order_payload", {})
        price = details.get("price") or order_payload.get("price")
        if not price:
            price = intent.get("snapshot", {}).get("price")
        if not price:
            continue

        time_val = datetime.utcfromtimestamp(ts)
        entry_price = float(price)
        points.append(
            {
                "time": time_val,
                "price": entry_price,
                "side": intent.get("side"),
                "status": status,
            }
        )

        risk = intent.get("risk", {}) or {}
        stop_bps = risk.get("stop_bps")
        tp_bps = risk.get("takeprofit_bps")
        if stop_bps:
            stop_price = entry_price * (1 - float(stop_bps) / 10_000.0)
            overlays.append({"time": time_val, "price": stop_price, "label": "stop"})
        if tp_bps:
            tp_price = entry_price * (1 + float(tp_bps) / 10_000.0)
            overlays.append({"time": time_val, "price": tp_price, "label": "take_profit"})

    if points:
        df = pd.DataFrame(points).sort_values("time")
        fig = go.Figure()

        def _fetch_candles(symbol_value: str, interval_value: str, limit: int) -> pd.DataFrame:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol_value, "interval": interval_value, "limit": limit}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for row in data:
                records.append(
                    {
                        "time": datetime.utcfromtimestamp(row[0] / 1000),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )
            return pd.DataFrame(records)

        try:
            candles = _fetch_candles(symbol, interval, candle_limit)
            fig.add_trace(
                go.Candlestick(
                    x=candles["time"],
                    open=candles["open"],
                    high=candles["high"],
                    low=candles["low"],
                    close=candles["close"],
                    name=f"{symbol} {interval}",
                )
            )
        except Exception as exc:
            st.warning(f"Failed to load candles: {exc}")

        for side, color in [("BUY", "#2ca02c"), ("SELL", "#d62728")]:
            sub = df[df["side"] == side]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub["time"],
                        y=sub["price"],
                        mode="markers",
                        name=side,
                        marker={"color": color, "size": 9},
                    )
                )

        if overlays:
            overlay_df = pd.DataFrame(overlays)
            for label, color in [("stop", "#ff7f0e"), ("take_profit", "#1f77b4")]:
                sub = overlay_df[overlay_df["label"] == label]
                if not sub.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["time"],
                            y=sub["price"],
                            mode="markers",
                            name=label,
                            marker={"color": color, "symbol": "x", "size": 7},
                        )
                    )

        fig.update_layout(
            xaxis_title="Time (UTC)",
            yaxis_title="Price",
            legend_title="Markers",
            height=420,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trade prices found in receipts yet.")
else:
    st.info("No receipts available for charting.")

st.subheader("Error Logs (tail)")
log_text = _tail_log(LOG_FILE)
st.code(log_text, language="text")
