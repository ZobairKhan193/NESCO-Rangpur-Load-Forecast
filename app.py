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
  --accent:#e0723a; --accent-soft:rgba(224,114,58,0.14);
  --purple:#7c3aed; --teal:#0d9488; --emerald:#10b981; --emerald-txt:#6ee7b7;
  --red:#dc2626; --blue:#2f6fed; --amber:#c2892f;
  --page:#0b1120; --panel:#0f1b30; --line:#233049;
  --ink:#e2e8f0; --muted:#94a3b8; --dim:#64748b;
  --navy:linear-gradient(120deg,#101d34 0%,#0f1b30 60%,#0d192c 100%);
}
#MainMenu, footer, header {visibility: hidden;}
html, body, .stApp, [class*="css"], .stMarkdown {
  font-family:'IBM Plex Sans', system-ui, sans-serif; color: var(--ink);
}
.stApp {background: var(--page);}
.block-container {padding-top: 1.2rem;}
h1,h2,h3,h4 {color: var(--ink); letter-spacing:-0.01em;}
.mono {font-family:'IBM Plex Mono', monospace;}
.sec {font-family:'IBM Plex Mono', monospace; font-size:.68rem; color:var(--muted);
      text-transform:uppercase; letter-spacing:.14em; margin:.1rem 0 .5rem 0;}

/* hero */
.hero {position:relative; background:var(--navy); border:1px solid var(--line);
    border-radius:16px; padding:1.2rem 1.5rem 1.2rem 1.6rem; margin-bottom:1.1rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.35); overflow:hidden;}
.hero::before {content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent);}
.hero-row {display:flex; align-items:center; gap:.9rem;}
.hero-ico {width:44px; height:44px; border-radius:12px; background:rgba(13,148,136,0.16);
    border:1px solid rgba(13,148,136,0.35); display:flex; align-items:center;
    justify-content:center; font-size:1.25rem; flex:0 0 auto;}
.hero h1 {margin:0; font-size:1.5rem; font-weight:600; color:#fff;}
.hero p {margin:.15rem 0 0 0; color:var(--muted); font-size:.88rem;}
.grow {flex:1 1 auto;}

/* pills */
.pill {display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .75rem;
    border-radius:999px; font-size:.72rem; font-weight:600; letter-spacing:.09em;
    text-transform:uppercase; font-family:'IBM Plex Mono', monospace;}
.pill-live {background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.4); color:var(--emerald-txt);}
.pill .dot {width:7px; height:7px; border-radius:50%; background:var(--emerald);}

/* cards — off-white boxes on the dark page (dark text inside) */
.card {background:#f6f8fb; border:1px solid #e4e9f2; border-radius:16px; color:#0f172a;
    padding:1.05rem 1.15rem; box-shadow:0 4px 14px rgba(0,0,0,0.28);}
.card .label {font-family:'IBM Plex Mono', monospace; font-size:.68rem; color:#64748b;
    text-transform:uppercase; letter-spacing:.12em; display:flex; align-items:center; gap:.45rem;}
.card .cdot {width:8px; height:8px; border-radius:50%; display:inline-block;}
.card .value {font-family:'IBM Plex Mono', monospace; font-size:1.7rem; font-weight:600;
    color:#0f172a; margin-top:.35rem; letter-spacing:-0.01em;}
.card .value .u {font-size:.78rem; color:#64748b; font-weight:500; margin-left:.2rem;}
.card .sub {font-family:'IBM Plex Mono', monospace; font-size:.7rem; color:#94a3b8; margin-top:.4rem;}
/* dark-context helpers re-colored for the off-white cards */
.card .sec {color:#64748b;}
.card .bignum {color:#0f172a;}
.card .badge-teal {background:rgba(13,148,136,0.12); border:1px solid rgba(13,148,136,0.4); color:#0f766e;}
.card .badge-teal .dot {background:#0d9488;}
.card .evt th {color:#64748b;}
.card .evt td {color:#334155; border-top:1px solid #e4e9f2;}
.card .evt td.num {color:#64748b;}
.card .evt tr.best td {color:#0f172a;}
.card .evt tr.best td.num {color:#e0723a;}

/* badges */
.badge {display:inline-flex; align-items:center; gap:.45rem; background:var(--accent);
    color:#0b1120; padding:.3rem .75rem; border-radius:999px; font-weight:600; font-size:.82rem;
    font-family:'IBM Plex Mono', monospace;}
.badge .dot {width:7px; height:7px; border-radius:50%; background:#0b1120;}
.badge-teal {background:rgba(13,148,136,0.14); border:1px solid rgba(13,148,136,0.45);
    color:#5eead4;}
.badge-teal .dot {background:#0d9488;}
.bignum {font-family:'IBM Plex Mono', monospace; font-size:2rem; font-weight:600; color:var(--ink); line-height:1;}
.bignum .p {font-size:1rem; color:var(--accent); margin-left:.1rem;}

/* sidebar user */
.avatar {width:38px; height:38px; border-radius:10px; background:var(--accent); color:#0b1120;
    display:flex; align-items:center; justify-content:center; font-weight:700;
    font-family:'IBM Plex Mono', monospace; flex:0 0 auto;}
.side-user {display:flex; align-items:center; gap:.6rem;}
.side-user .nm {font-weight:600; color:var(--ink); font-size:.95rem; line-height:1.1;}
.side-user .rl {font-family:'IBM Plex Mono', monospace; font-size:.6rem; color:var(--muted);
    text-transform:uppercase; letter-spacing:.14em;}
.kv {display:flex; justify-content:space-between; font-family:'IBM Plex Mono', monospace;
    font-size:.72rem; margin:.28rem 0;}
.kv .k {color:var(--dim);} .kv .v {color:var(--ink);}

/* models-evaluated table */
.evt {width:100%; border-collapse:collapse; font-size:.86rem;}
.evt th {font-family:'IBM Plex Mono', monospace; font-size:.62rem; color:var(--muted);
    text-transform:uppercase; letter-spacing:.1em; text-align:right; padding:.2rem .3rem .55rem; font-weight:600;}
.evt th:first-child {text-align:left;}
.evt td {padding:.5rem .3rem; border-top:1px solid var(--line); font-family:'IBM Plex Mono', monospace;}
.evt td.num {text-align:right; color:var(--muted);}
.evt tr.best td {color:var(--ink);}
.evt tr.best td.num {color:var(--accent);}
.evt .mdot {width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:.55rem; vertical-align:middle;}

/* primary (purple) button + card wrappers */
.stButton > button[kind="primary"] {background:var(--purple); border:0; border-radius:10px;
    font-weight:600; box-shadow:0 4px 14px rgba(124,58,237,0.35);}
.app-footer {text-align:center; color:var(--dim); font-size:.75rem; margin-top:2.5rem;
    font-family:'IBM Plex Mono', monospace; letter-spacing:.04em;}
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

def models_eval_html(cfg):
    """Custom dark table with a coloured dot per model (best = orange)."""
    tbl, best = accuracy_table(cfg), cfg["model_name"]
    body = ""
    for _, r in tbl.iterrows():
        is_best = r["Model"] == best
        dot = "var(--accent)" if is_best else "var(--dim)"
        da = "—" if pd.isna(r["Day-ahead MAPE (%)"]) else f'{r["Day-ahead MAPE (%)"]:.2f}'
        ho = "—" if pd.isna(r["Holdout MAPE (%)"]) else f'{r["Holdout MAPE (%)"]:.2f}'
        body += (f'<tr class="{"best" if is_best else ""}">'
                 f'<td><span class="mdot" style="background:{dot}"></span>{r["Model"]}</td>'
                 f'<td class="num">{da}</td><td class="num">{ho}</td></tr>')
    return ('<table class="evt"><tr><th>Model</th><th>Day-ahead</th>'
            f'<th>Holdout</th></tr>{body}</table>')

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
    user = st.session_state.get("user", "user")
    initials = ("".join(p[0] for p in user.split()[:2]) or user[:2]).upper()
    st.markdown(f'<div class="side-user"><div class="avatar">{initials}</div>'
                f'<div><div class="nm">{user}</div>'
                f'<div class="rl">System Planning</div></div></div>', unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('<div class="sec" style="margin-top:1rem">Best model</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge"><span class="dot"></span>{cfg["model_name"]}</span>',
                unsafe_allow_html=True)
    if best_mape is not None:
        st.markdown('<div class="sec" style="margin-top:.9rem">Day-ahead MAPE · backtest</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="bignum">{best_mape:.2f}<span class="p">%</span></div>',
                    unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-top:1rem">'
        f'<div class="kv"><span class="k">Trained</span><span class="v">{str(trained_at)[:16].replace("T"," ")}</span></div>'
        f'<div class="kv"><span class="k">Data through</span><span class="v">{str(data_through)[:16]}</span></div>'
        f'<div class="kv"><span class="k">Last actual</span><span class="v">{str(last_actual)[:16]}</span></div>'
        '</div>', unsafe_allow_html=True)
    age_days = (pd.Timestamp.now() - last_actual).days
    if age_days > 10:
        st.warning(f"⚠️ History is {age_days} days old — retrain soon.")
    with st.expander("Models tested (accuracy)"):
        st.dataframe(accuracy_table(cfg), hide_index=True, use_container_width=True)

# ---- Hero ----
st.markdown(
    '<div class="hero"><div class="hero-row">'
    '<div class="hero-ico">⚡</div>'
    '<div class="grow"><h1>NESCO Rangpur — Hourly Load Forecast</h1>'
    '<p>Day-ahead demand forecasting · System Planning, NESCO</p></div>'
    '<span class="pill pill-live"><span class="dot"></span>Model Live</span>'
    '</div></div>', unsafe_allow_html=True)

# ---- Landing: selected model card + models-evaluated table ----
c1, c2 = st.columns([1, 2], gap="medium")
with c1:
    st.markdown(
        '<div class="card"><div class="sec">Selected model</div>'
        f'<span class="badge badge-teal"><span class="dot"></span>{cfg["model_name"]}</span>'
        '<div class="sec" style="margin-top:1rem">Day-ahead MAPE</div>'
        + (f'<div class="bignum">{best_mape:.2f}<span class="p">%</span></div>'
           if best_mape is not None else '<div class="bignum">—</div>')
        + '<div style="color:var(--dim);font-size:.8rem;margin-top:.8rem;line-height:1.4">'
        'Lowest error across the evaluated set. Click a row in the table to compare.</div>'
        '</div>', unsafe_allow_html=True)
with c2:
    n_models = len(accuracy_table(cfg))
    st.markdown(
        '<div class="card">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline">'
        '<div class="sec">Models evaluated</div>'
        f'<div class="sec">{n_models} models · MAPE %</div></div>'
        + models_eval_html(cfg) + '</div>', unsafe_allow_html=True)

# ---- Target date + generate (button raised next to the date for easy click) ----
st.markdown('<div class="sec" style="margin-top:1.2rem">Target date</div>', unsafe_allow_html=True)
today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
dcol, bcol, scol = st.columns([2, 1.5, 2], gap="medium")
with dcol:
    sel = st.date_input("Target date", value=(today + pd.Timedelta(days=1)).date(),
                        min_value=today.date(), max_value=(today + pd.Timedelta(days=2)).date(),
                        label_visibility="collapsed")
target_date = pd.Timestamp(sel)
with bcol:
    go = st.button("⚡ Generate Forecast", type="primary", use_container_width=True)
with scol:
    st.markdown('<div style="text-align:right;padding-top:.35rem">'
                '<span class="pill pill-live"><span class="dot"></span>Forecast ready · run</span>'
                '</div>', unsafe_allow_html=True)

if go:
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

    # ---- downloads: placed right below the Generate Forecast button ----
    ois = build_ois(fc, target_date, cfg.get("power_factor", 0.9))
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        ois.to_excel(xw, sheet_name="forecast", index=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download OIS file (.xlsx)", xbuf.getvalue(),
                       file_name=f"NESCO-Rangpur_forecast_{date_str}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    d2.download_button("⬇️ Download CSV", fc.to_csv(index=False).encode(),
                       file_name=f"NESCO-Rangpur_forecast_{date_str}.csv", mime="text/csv",
                       use_container_width=True)

    # summary cards (coloured dot + sub-caption, matching the design)
    peak, mn, avg = fc.Forecast_MW.max(), fc.Forecast_MW.min(), fc.Forecast_MW.mean()
    energy = fc.Forecast_MW.sum()
    peak_h = fc.loc[fc.Forecast_MW.idxmax(), "Time"].strftime("%H:00")
    min_h = fc.loc[fc.Forecast_MW.idxmin(), "Time"].strftime("%H:00")
    metrics = [
        ("Peak", f"{peak:,.1f}", "MW", "var(--red)", f"at {peak_h}"),
        ("Minimum", f"{mn:,.1f}", "MW", "var(--blue)", f"at {min_h}"),
        ("Average", f"{avg:,.1f}", "MW", "var(--accent)", "24h mean"),
        ("Energy", f"{energy:,.0f}", "MWh", "var(--amber)", "day-ahead total"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, val, unit, dot, sub) in zip(cols, metrics):
        col.markdown(
            f'<div class="card"><div class="label">'
            f'<span class="cdot" style="background:{dot}"></span>{label}</div>'
            f'<div class="value">{val}<span class="u">{unit}</span></div>'
            f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sec" style="margin-top:1.3rem">Hourly forecast — {date_str}</div>',
                unsafe_allow_html=True)
    chart = fc.set_index("Time")["Forecast_MW"]
    st.line_chart(chart, height=300, color="#0d9488")

    st.markdown('<div class="sec" style="margin-top:1rem">OIS table</div>', unsafe_allow_html=True)
    st.dataframe(ois, hide_index=True, use_container_width=True)
    st.caption("MVAR computed at power factor 0.9; half-hourly rows (18:30, 19:30) = "
               "average of the adjacent hourly values.")

# ---- Footer ----
year = datetime.now().year
st.markdown(f'<div class="app-footer">© {year} Zobair Hossain Khan · '
            'Load Forecast by Planning — NESCO System Planning</div>',
            unsafe_allow_html=True)
