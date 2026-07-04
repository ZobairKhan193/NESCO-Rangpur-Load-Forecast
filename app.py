"""
NESCO Rangpur — Hourly Load Forecast (Streamlit web app)
========================================================

Serves a day-ahead hourly demand forecast for the Rangpur distribution zone and
produces the OIS-ready .xlsx. Login-gated; credentials live in Streamlit Secrets.

Assumptions
-----------
* The 5 training artifacts (config.json, best_model.pkl/.keras, feat_scaler.pkl,
  target_scaler.pkl, history_tail.csv) sit either in ./artifacts or next to this file.
* The feature builder below is BYTE-IDENTICAL to train.ipynb and forecast.ipynb.
* Target date is restricted to [today, today+2] in Asia/Dhaka.
* OIS: MVAR = MW * tan(arccos(power_factor=0.9)); half-hourly 18:30/19:30 = mean of
  adjacent hours.
"""
import io
import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="NESCO Rangpur Load Forecast", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --accent:#0d9488; --accent-2:#10b981; --danger:#c15650;
  --ink:#0f172a; --muted:#64748b; --muted-2:#94a3b8;
  --line:#e2e8f0; --page:#f6f8fb; --card:#ffffff;
  --navy:linear-gradient(120deg,#101d34 0%,#0f1b30 60%,#0d192c 100%);
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
html, body, .stApp, [class*="css"], .stMarkdown {
  font-family:'IBM Plex Sans', system-ui, sans-serif; color: var(--ink);
}
.stApp {background: var(--page);}
.block-container {padding-top: 1.4rem;}
h1,h2,h3,h4 {color: var(--ink); letter-spacing:-0.01em;}
.hero {
    background: var(--navy); color: #fff; padding: 1.4rem 1.8rem;
    border-radius: 16px; margin-bottom: 1.2rem;
    border: 1px solid rgba(13,148,136,0.28);
    box-shadow: 0 2px 8px rgba(15,23,42,0.20);
}
.hero h1 {margin: 0; font-size: 1.7rem; font-weight: 600; color: #fff;}
.hero p {margin: 0.35rem 0 0 0; color: #cbd5e1; font-size: .95rem;}
.card {
    background: var(--card); border: 1px solid var(--line); border-radius: 16px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.card .label {
    font-family:'IBM Plex Mono', monospace; font-size: .72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .12em;
}
.card .value {
    font-family:'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 600;
    color: var(--ink); margin-top: .2rem;
}
.badge {
    display: inline-block; background: var(--accent); color: #fff; padding: .28rem .7rem;
    border-radius: 999px; font-weight: 600; font-size: .85rem; letter-spacing: .02em;
    box-shadow: 0 4px 14px rgba(13,148,136,0.35);
}
.stButton > button[kind="primary"] {
    background: var(--accent); border: 0; box-shadow: 0 4px 14px rgba(13,148,136,0.35);
}
.app-footer {
    text-align: center; color: var(--muted-2); font-size: .8rem; margin-top: 2.5rem;
    font-family:'IBM Plex Mono', monospace; letter-spacing: .04em;
}
</style>
""", unsafe_allow_html=True)

TZ = "Asia/Dhaka"

# --------------------------------------------------------------------------- #
# Artifact resolver (tolerant of artifacts/ subfolder OR repo root)
# --------------------------------------------------------------------------- #
def resolve_artifact_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "artifacts"), here,
                  os.path.join(here, "..", "artifacts"), "."]
    for d in candidates:
        if os.path.exists(os.path.join(d, "config.json")):
            return os.path.abspath(d)
    return os.path.abspath(candidates[0])

ARTIFACT_DIR = resolve_artifact_dir()

# --------------------------------------------------------------------------- #
# Login gate
# --------------------------------------------------------------------------- #
def check_login():
    if st.session_state.get("authed"):
        return True
    st.markdown('<div class="hero"><h1>⚡ NESCO Rangpur Load Forecast</h1>'
                '<p>Please sign in to continue.</p></div>', unsafe_allow_html=True)
    users = {}
    try:
        users = dict(st.secrets["users"])
    except Exception:
        st.error("No `[users]` configured in Streamlit Secrets. "
                 "See secrets.toml.example.")
        st.stop()
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        if u in users and str(users[u]) == p:
            st.session_state["authed"] = True
            st.session_state["user"] = u
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# --------------------------------------------------------------------------- #
# Load artifacts (cached)
# --------------------------------------------------------------------------- #
def artifacts_version(art_dir):
    """Newest mtime across the artifact files — used as a cache key so that when new
    artifacts land (e.g. after a weekly update), the app reloads them WITHOUT needing a
    manual reboot. (`version` is a real arg, not underscore-prefixed, so it is hashed.)"""
    files = ["config.json", "best_model.pkl", "best_model.keras",
             "feat_scaler.pkl", "target_scaler.pkl", "history_tail.csv"]
    mtimes = [os.path.getmtime(os.path.join(art_dir, f))
              for f in files if os.path.exists(os.path.join(art_dir, f))]
    return max(mtimes, default=0.0)

@st.cache_resource(show_spinner="Loading model & artifacts…")
def load_artifacts(art_dir, version):
    with open(os.path.join(art_dir, "config.json")) as f:
        config = json.load(f)
    feat_scaler = joblib.load(os.path.join(art_dir, "feat_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(art_dir, "target_scaler.pkl"))
    kind = config["model_kind"]
    if kind in ("xgboost", "lightgbm"):
        model = joblib.load(os.path.join(art_dir, "best_model.pkl"))
    else:
        import tensorflow as tf
        from tensorflow.keras.layers import Layer

        class AttentionPooling(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1),
                                         initializer="glorot_uniform", trainable=True)
                self.b = self.add_weight(name="att_b", shape=(1,),
                                         initializer="zeros", trainable=True)
                super().build(input_shape)

            def call(self, x):
                score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
                weights = tf.nn.softmax(score, axis=1)
                return tf.reduce_sum(x * weights, axis=1)

            def get_config(self):
                return super().get_config()

        model = tf.keras.models.load_model(
            os.path.join(art_dir, "best_model.keras"),
            custom_objects={"AttentionPooling": AttentionPooling})
    history = pd.read_csv(os.path.join(art_dir, "history_tail.csv"),
                          parse_dates=["Time"]).set_index("Time").sort_index()
    return config, model, feat_scaler, target_scaler, history

# --------------------------------------------------------------------------- #
# Feature builder (CANONICAL — identical to train.ipynb & forecast.ipynb)
# --------------------------------------------------------------------------- #
def build_features(df, holiday_dates):
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["dayofyear"] = idx.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["is_weekend"] = idx.dayofweek.isin([4, 5]).astype(int)  # Fri=4, Sat=5
    dstr = pd.Series(idx.strftime("%Y-%m-%d"), index=idx)
    df["is_holiday"] = dstr.isin(holiday_dates).astype(int)
    for L in [1, 2, 3, 24, 48, 72, 168]:
        df[f"lag_{L}"] = df["Demand"].shift(L)
    s = df["Demand"].shift(1)
    df["roll_mean_6"] = s.rolling(6).mean()
    df["roll_mean_12"] = s.rolling(12).mean()
    df["roll_mean_24"] = s.rolling(24).mean()
    df["roll_mean_168"] = s.rolling(168).mean()
    df["roll_std_24"] = s.rolling(24).std()
    df["roll_std_168"] = s.rolling(168).std()
    df["temp_lag24"] = df["temperature_2m"].shift(24)
    df["hum_lag24"] = df["relative_humidity_2m"].shift(24)
    df["precip_lag24"] = df["precipitation"].shift(24)
    df["temp_squared"] = df["temperature_2m"] ** 2
    df["temp_hour_sin"] = df["temperature_2m"] * df["hour_sin"]
    df["temp_hour_cos"] = df["temperature_2m"] * df["hour_cos"]
    df["humidex"] = df["temperature_2m"] + 0.1 * df["relative_humidity_2m"]
    return df

# --------------------------------------------------------------------------- #
# Weather + iterative predictor
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_forecast(start_date, end_date, lat, lon, weather_vars):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "start_date": start_date,
              "end_date": end_date, "hourly": ",".join(weather_vars), "timezone": TZ}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame(h)
    w["Time"] = pd.to_datetime(w["time"])
    w = w.drop(columns=["time"]).set_index("Time").sort_index()
    w = w[~w.index.duplicated(keep="first")]
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
    w = w.interpolate(method="time", limit_direction="both")
    return w[weather_vars]

def predict_window(ts, feat_full, cfg, model, feat_scaler, target_scaler):
    cols, kind, lookback = cfg["feature_columns"], cfg["model_kind"], cfg["lookback"]
    if kind in ("xgboost", "lightgbm"):
        return float(model.predict(feat_full.loc[[ts], cols].values)[0])
    win = feat_full.loc[:ts, cols].tail(lookback)
    Xs = feat_scaler.transform(win.values)
    yps = model.predict(Xs[None, ...], verbose=0)
    return float(target_scaler.inverse_transform(yps.reshape(-1, 1))[0, 0])

def run_forecast(cfg, model, feat_scaler, target_scaler, history, target_date, progress=None):
    weather_vars = cfg["weather_vars"]
    holiday_dates = set(cfg["holidays"].keys())
    lookback = cfg["lookback"]
    bias = float(cfg["bias_correction"])
    if not np.isfinite(bias):   # guard against a NaN bias poisoning every forecast
        bias = 0.0
    clip_min, clip_max = cfg.get("clip_min", 30.0), cfg.get("clip_max", 700.0)

    target_idx = pd.date_range(target_date, periods=24, freq="h")
    margin_start = target_date - pd.Timedelta(hours=lookback + 12)
    wx_start = min(history.index.max() + pd.Timedelta(hours=1), margin_start)
    wx = fetch_weather_forecast(wx_start.strftime("%Y-%m-%d"),
                                target_date.strftime("%Y-%m-%d"),
                                cfg["lat"], cfg["lon"], weather_vars)

    full_idx = pd.date_range(history.index.min(), target_idx[-1], freq="h")
    work = pd.DataFrame(index=full_idx)
    work["Demand"] = history["Demand"].reindex(full_idx)
    for v in weather_vars:
        work[v] = history[v].reindex(full_idx).fillna(wx[v].reindex(full_idx))
    work[weather_vars] = work[weather_vars].interpolate(method="time", limit_direction="both")

    # Gap between the last actual and the target day.
    gap_idx = pd.date_range(history.index.max() + pd.Timedelta(hours=1),
                            target_date - pd.Timedelta(hours=1), freq="h")
    gap_idx = gap_idx[gap_idx.isin(full_idx)]

    # FORECAST the gap (predict it iteratively) instead of seasonal-copying it: copying a
    # single reference day injects that day's level error into the lags and biases the whole
    # forecast high/low (validated: gap MAPE ~5% copied vs ~1.6% forecast). Only an unusually
    # long gap (> MAX_FC_GAP_H) is seasonal-bridged at its oldest part to bound compounding.
    MAX_FC_GAP_H = 168
    if len(gap_idx) > MAX_FC_GAP_H:
        early = gap_idx[:-MAX_FC_GAP_H]
        for ts in early:
            for back in (7, 14, 21):
                src = ts - pd.Timedelta(days=back)
                if src in work.index and pd.notna(work.at[src, "Demand"]):
                    work.at[ts, "Demand"] = work.at[src, "Demand"]
                    break
        work["Demand"] = work["Demand"].ffill(limit=24)
        fc_idx = gap_idx[-MAX_FC_GAP_H:].append(target_idx)
    else:
        fc_idx = gap_idx.append(target_idx)
    work.loc[fc_idx, "Demand"] = np.nan

    # iterative prediction over the gap (if any) + target day; commit each hour before next
    out = {}
    n = len(fc_idx)
    for i, ts in enumerate(fc_idx):
        feat_full = build_features(work, holiday_dates)
        yhat = predict_window(ts, feat_full, cfg, model, feat_scaler, target_scaler) + bias
        # Fail loud rather than serve a blank/NaN forecast. NaN here usually means the
        # bundled history_tail is too short for this model's lookback window -> retrain.
        if not np.isfinite(yhat):
            raise ValueError(
                f"NaN forecast at {ts}: history_tail ({len(history)} rows) is too short for "
                f"'{cfg['model_name']}'. Retrain with the updated train.ipynb and re-upload "
                f"the artifacts.")
        yhat = float(np.clip(yhat, clip_min, clip_max))
        work.loc[ts, "Demand"] = yhat
        out[ts] = yhat
        if progress is not None:
            progress.progress((i + 1) / n, text=f"Predicting {ts.strftime('%m-%d %H:%M')}…")
    return pd.DataFrame({"Time": target_idx, "Forecast_MW": [out[t] for t in target_idx]})

def build_ois(fc, target_date, power_factor):
    factor = np.tan(np.arccos(power_factor))
    mw = {ts.strftime("%H:%M"): v for ts, v in zip(fc["Time"], fc["Forecast_MW"])}
    mw["18:30"] = (mw["18:00"] + mw["19:00"]) / 2
    mw["19:30"] = (mw["19:00"] + mw["20:00"]) / 2
    order = sorted(mw.keys())
    col0 = f"date_time({target_date.strftime('%Y-%m-%d')})"
    return pd.DataFrame([{col0: t, "Forecast (MW)": round(mw[t], 3),
                          "Forecast (MVAR)": round(mw[t] * factor, 3)} for t in order])

# --------------------------------------------------------------------------- #
# Accuracy helpers (robust to missing keys)
# --------------------------------------------------------------------------- #
def accuracy_table(cfg):
    holdout = {r["model"]: r.get("mape") for r in cfg.get("holdout_leaderboard", [])}
    backtest = cfg.get("backtest_leaderboard", {})
    names = list(dict.fromkeys(list(backtest.keys()) + list(holdout.keys())))
    rows = []
    for n in names:
        rows.append({"Model": n,
                     "Day-ahead MAPE (%)": round(backtest[n], 2) if n in backtest else None,
                     "Holdout MAPE (%)": round(holdout[n], 2) if holdout.get(n) is not None else None})
    df = pd.DataFrame(rows)
    if "Day-ahead MAPE (%)" in df:
        df = df.sort_values("Day-ahead MAPE (%)", na_position="last").reset_index(drop=True)
    return df

# --------------------------------------------------------------------------- #
# App body
# --------------------------------------------------------------------------- #
check_login()

if not os.path.exists(os.path.join(ARTIFACT_DIR, "config.json")):
    st.error(f"No artifacts found. Looked in: {ARTIFACT_DIR}. "
             "Upload the 5 files into an `artifacts/` folder.")
    st.stop()

cfg, model, feat_scaler, target_scaler, history = load_artifacts(
    ARTIFACT_DIR, artifacts_version(ARTIFACT_DIR))

best_mape = cfg.get("selected_backtest_mape")
last_actual = history.index.max()
trained_at = cfg.get("trained_at", "?")
data_through = cfg.get("data_range", ["?", "?"])[1]

# ---- Sidebar ----
with st.sidebar:
    st.markdown(f"**👤 {st.session_state.get('user','user')}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.markdown("### 🏆 Best model")
    st.markdown(f'<span class="badge">{cfg["model_name"]}</span>', unsafe_allow_html=True)
    if best_mape is not None:
        st.metric("Day-ahead MAPE (backtest)", f"{best_mape:.2f}%")
    st.caption(f"Trained: {trained_at}")
    st.caption(f"Data through: {data_through}")
    st.caption(f"Last actual: {last_actual}")
    # staleness warning
    age_days = (pd.Timestamp.now() - last_actual).days
    if age_days > 10:
        st.warning(f"⚠️ History is {age_days} days old — retrain soon.")
    with st.expander("📊 Models tested (accuracy)"):
        st.dataframe(accuracy_table(cfg), hide_index=True, use_container_width=True)

# ---- Hero ----
st.markdown('<div class="hero"><h1>⚡ NESCO Rangpur — Hourly Load Forecast</h1>'
            '<p>Day-ahead demand forecasting · System Planning, NESCO</p></div>',
            unsafe_allow_html=True)

# ---- Landing: selected model + evaluation table ----
c1, c2 = st.columns([1, 2])
with c1:
    st.markdown("#### Selected model")
    st.markdown(f'<span class="badge">{cfg["model_name"]}</span>', unsafe_allow_html=True)
    if best_mape is not None:
        st.caption(f"Day-ahead MAPE: **{best_mape:.2f}%**")
with c2:
    st.markdown("#### Models evaluated")
    tbl = accuracy_table(cfg)
    if not tbl.empty:
        best_name = cfg["model_name"]
        st.dataframe(tbl.style.apply(
            lambda r: ["background-color: rgba(13,148,136,0.16)" if r["Model"] == best_name
                       else "" for _ in r], axis=1),
            hide_index=True, use_container_width=True)

st.divider()

# ---- Date picker + generate ----
today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
sel = st.date_input("Target date", value=(today + pd.Timedelta(days=1)).date(),
                    min_value=today.date(), max_value=(today + pd.Timedelta(days=2)).date())
target_date = pd.Timestamp(sel)

if st.button("🚀 Generate Forecast", type="primary"):
    bar = st.progress(0.0, text="Starting…")
    try:
        fc = run_forecast(cfg, model, feat_scaler, target_scaler, history, target_date, bar)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()
    bar.empty()
    st.session_state["fc"] = fc
    st.session_state["fc_date"] = target_date

if "fc" in st.session_state:
    fc = st.session_state["fc"]
    target_date = st.session_state["fc_date"]
    date_str = target_date.strftime("%Y-%m-%d")

    # summary cards
    peak, mn, avg = fc.Forecast_MW.max(), fc.Forecast_MW.min(), fc.Forecast_MW.mean()
    energy = fc.Forecast_MW.sum()
    cols = st.columns(4)
    for col, label, val, unit in [
        (cols[0], "Peak", peak, "MW"), (cols[1], "Minimum", mn, "MW"),
        (cols[2], "Average", avg, "MW"), (cols[3], "Energy", energy, "MWh")]:
        col.markdown(f'<div class="card"><div class="label">{label}</div>'
                     f'<div class="value">{val:,.1f} <span style="font-size:.8rem">{unit}</span>'
                     f'</div></div>', unsafe_allow_html=True)

    st.markdown(f"#### Hourly forecast — {date_str}")
    chart = fc.set_index("Time")["Forecast_MW"]
    st.line_chart(chart, height=320)

    ois = build_ois(fc, target_date, cfg.get("power_factor", 0.9))
    st.markdown("#### OIS table")
    st.dataframe(ois, hide_index=True, use_container_width=True)
    st.caption("MVAR computed at power factor 0.9; half-hourly rows (18:30, 19:30) = "
               "average of the adjacent hourly values.")

    # downloads
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        ois.to_excel(xw, sheet_name="forecast", index=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download OIS file (.xlsx)", xbuf.getvalue(),
                       file_name=f"NESCO-Rangpur_forecast_{date_str}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇️ Download CSV", fc.to_csv(index=False).encode(),
                       file_name=f"NESCO-Rangpur_forecast_{date_str}.csv", mime="text/csv")

# ---- Footer ----
year = datetime.now().year
st.markdown(f'<div class="app-footer">© {year} Zobair Hossain Khan · '
            'Load Forecast by Planning — NESCO System Planning</div>',
            unsafe_allow_html=True)
