"""NBA ML Project — Professional Streamlit Dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLOTS_DIR    = _PROJECT_ROOT / "nba_data" / "plots"
_RESULTS_FILE = _PROJECT_ROOT / "results" / "model_metrics.csv"
_MODELS_E1    = _PROJECT_ROOT / "models" / "engine1"
_MODELS_E2    = _PROJECT_ROOT / "models" / "engine2"
_AWARDS_CSV   = _PROJECT_ROOT / "nba_data" / "processed" / "awards_features_labeled.csv"

_E1_FEATURES = [
    "HOME",
    "ROLL5_PTS", "ROLL10_PTS",
    "ROLL5_FG_PCT", "ROLL10_FG_PCT",
    "ROLL5_FG3_PCT", "ROLL10_FG3_PCT",
    "ROLL5_FT_PCT", "ROLL10_FT_PCT",
    "ROLL5_OREB", "ROLL10_OREB",
    "ROLL5_DREB", "ROLL10_DREB",
    "ROLL5_AST", "ROLL10_AST",
    "ROLL5_STL", "ROLL10_STL",
    "ROLL5_BLK", "ROLL10_BLK",
    "ROLL5_TOV", "ROLL10_TOV",
    "ROLL5_PLUS_MINUS", "ROLL10_PLUS_MINUS",
    "DAYS_REST", "BACK_TO_BACK", "GAMES_LAST_7D", "WIN_STREAK",
    "ROLL5_PTS_IS_VALID", "ROLL10_PTS_IS_VALID",
    "ROLL5_FG_PCT_IS_VALID", "ROLL10_FG_PCT_IS_VALID",
    "ROLL5_FG3_PCT_IS_VALID", "ROLL10_FG3_PCT_IS_VALID",
    "ROLL5_FT_PCT_IS_VALID", "ROLL10_FT_PCT_IS_VALID",
    "ROLL5_OREB_IS_VALID", "ROLL10_OREB_IS_VALID",
    "ROLL5_DREB_IS_VALID", "ROLL10_DREB_IS_VALID",
    "ROLL5_AST_IS_VALID", "ROLL10_AST_IS_VALID",
    "ROLL5_STL_IS_VALID", "ROLL10_STL_IS_VALID",
    "ROLL5_BLK_IS_VALID", "ROLL10_BLK_IS_VALID",
    "ROLL5_TOV_IS_VALID", "ROLL10_TOV_IS_VALID",
    "ROLL5_PLUS_MINUS_IS_VALID", "ROLL10_PLUS_MINUS_IS_VALID",
]

_E2_FEATURES = [
    "PTS_AVG", "REB_AVG", "AST_AVG", "STL_AVG", "BLK_AVG",
    "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS_AVG",
    "FANTASY_AVG", "MIN_AVG",
    "DD2_RATE", "TD3_RATE", "CONSISTENCY",
    "RANK_PTS_AVG", "RANK_AST_AVG", "RANK_REB_AVG",
    "RANK_BLK_AVG", "RANK_STL_AVG", "RANK_PLUS_MINUS_AVG",
    "TOP5_PCT_FLAG", "GP",
    "DD2_TOTAL", "TD3_TOTAL", "TOV_AVG",
]

_AWARDS = ["MVP", "DPOY", "ROY", "6MOY"]

_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Base & Typography ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── App header banner ── */
.nba-header {
    background: linear-gradient(135deg, #0D1B3E 0%, #1D428A 60%, #C9A84C 100%);
    padding: 2.2rem 2rem 1.8rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border: 1px solid #1D428A;
}
.nba-header h1 { color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 700; }
.nba-header p  { color: #C8D6F0; margin: 0.4rem 0 0; font-size: 1rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #161B22;
    border-left: 4px solid #1D428A;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: transform 0.15s ease;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px); }
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
}

/* ── Section cards ── */
.info-card {
    background: #161B22;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #30363D;
    margin-bottom: 1rem;
}
.info-card h4 { color: #C9A84C; margin-top: 0; }

/* ── Engine badge labels ── */
.engine-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
}
.e1-badge { background: #1D428A33; color: #5B9BD5; border: 1px solid #1D428A; }
.e2-badge { background: #C9A84C22; color: #C9A84C; border: 1px solid #C9A84C66; }

/* ── Ranking table ── */
.rank-table { width: 100%; border-collapse: collapse; }
.rank-table th {
    background: #1D428A;
    color: white;
    padding: 0.5rem 1rem;
    text-align: left;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.rank-table td { padding: 0.5rem 1rem; border-bottom: 1px solid #21262D; font-size: 0.9rem; }
.rank-table tr:first-child td { color: #C9A84C; font-weight: 700; }
.rank-table tr:hover td { background: #1C2128; }

/* ── Win probability gauge ── */
.prob-display {
    text-align: center;
    padding: 1.5rem;
    background: #161B22;
    border-radius: 14px;
    border: 1px solid #30363D;
}
.prob-value { font-size: 3.5rem; font-weight: 800; }
.prob-label { font-size: 0.9rem; color: #8B949E; margin-top: 0.3rem; }

/* ── Divider ── */
hr { border-color: #21262D; }
</style>
"""

_PLOT_GROUPS = {
    "Data Quality & Distributions": [
        ("01_missing_values.png", "Missing values by feature"),
        ("04_correlations.png", "Feature correlation heatmap"),
        ("16_pca_analysis.png", "PCA — 68% variance in 3 components"),
    ],
    "Home-Court Advantage & COVID Impact": [
        ("02_wl_home_away.png", "Win rate: home vs away (2014–2024)"),
        ("05_covid_home.png", "COVID (2019–21) nearly eliminated home advantage"),
    ],
    "Game Evolution & Head-to-Head": [
        ("03_evolution_jeu.png", "3-point revolution: 22 → 35 attempts/game"),
        ("15_h2h_heatmap.png", "Head-to-head franchise matchup patterns"),
    ],
    "Fatigue & Rolling Features (Engine 1)": [
        ("06_fatigue_features.png", "Rest days & back-to-back effect on win rate"),
        ("07_rolling_correlations.png", "Rolling average correlations with outcome"),
    ],
    "Player Analysis (Engine 2)": [
        ("08_absences.png", "Player absence patterns and team impact"),
        ("09_player_impact.png", "Key-player impact on win rate"),
        ("10_rolling_players.png", "Rolling player statistics"),
        ("11_consistency.png", "Performance consistency tiers"),
        ("12_usage.png", "Player usage rate distribution"),
        ("13_mvp_candidates.png", "MVP candidate profiles by season"),
        ("14_leaders_by_season.png", "Statistical leaders per season"),
    ],
}


# ── Model cache ────────────────────────────────────────────────────────────

@st.cache_resource
def _load_e1_model(name: str):
    import joblib
    path = _MODELS_E1 / f"{name}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def _load_e2_model(award: str, name: str):
    import joblib
    path = _MODELS_E2 / f"{award}_{name}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def _load_e2_test_data():
    if not _AWARDS_CSV.exists():
        return None
    df = pd.read_csv(_AWARDS_CSV)
    seasons = sorted(df["SEASON_YEAR"].unique())
    test_seasons = set(seasons[-2:])
    return df[df["SEASON_YEAR"].isin(test_seasons)].reset_index(drop=True)


@st.cache_data
def _evaluate_e1_models():
    """Run Engine 1 evaluation inline and return metrics DataFrame."""
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from data import load_dataset_split_engine1
    from metrics import compute_metrics

    try:
        _, X_test, _, y_test = load_dataset_split_engine1()
    except Exception:
        return None

    rows = []
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = _load_e1_model(name)
        if model is None:
            continue
        try:
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
            m = compute_metrics(y_test, y_pred)
            rows.append({"Model": name.replace("_", " ").title(), **m})
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else None


# ── Entry point ────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="NBA ML Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="nba-header">'
        '<h1>🏀 NBA Machine Learning Dashboard</h1>'
        '<p>Two prediction engines trained on 10 seasons (2014–2024) &nbsp;·&nbsp; '
        'Engine 1: Match Outcomes &nbsp;·&nbsp; Engine 2: Award Winners</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    tab_proj, tab_eda, tab_models, tab_e1, tab_e2 = st.tabs([
        "📋 Project",
        "📊 EDA",
        "🤖 Model Comparison",
        "⚡ Match Predictor",
        "🏆 Awards Predictor",
    ])

    with tab_proj:
        _render_project()
    with tab_eda:
        _render_eda()
    with tab_models:
        _render_model_comparison()
    with tab_e1:
        _render_engine1_demo()
    with tab_e2:
        _render_engine2_demo()


# ── Tab 1: Project ─────────────────────────────────────────────────────────

def _render_project() -> None:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            '<span class="engine-badge e1-badge">ENGINE 1</span>', unsafe_allow_html=True
        )
        st.subheader("Match Outcome Prediction")
        st.markdown("""
**Objective:** Predict whether a team wins or loses a given game.

| | |
|---|---|
| **Task** | Binary classification (W / L) |
| **Dataset** | 27 313 team-game rows |
| **Features** | 49 (rolling avgs, fatigue, momentum) |
| **Train** | 8 seasons — 2014–2022 |
| **Test** | 2 seasons — 2022–2024 |

**Models:** Logistic Regression · Random Forest · XGBoost

**Primary metric:** ROC-AUC
**Targets:** Log Loss < 0.60 · ROC-AUC > 0.70 · Accuracy > 60%
        """)

    with col2:
        st.markdown(
            '<span class="engine-badge e2-badge">ENGINE 2</span>', unsafe_allow_html=True
        )
        st.subheader("Award Winner Prediction")
        st.markdown("""
**Objective:** Rank players and identify the award winner each season.

| | |
|---|---|
| **Task** | Binary ranking (winner = 1) |
| **Awards** | MVP · DPOY · ROY · 6MOY |
| **Dataset** | 4 654 player-season rows |
| **Features** | 25 (averages, ranks, efficiency) |
| **Train** | 8 seasons — 2014–2022 |
| **Test** | 2 seasons — 2022–2024 |

**Models:** Logistic Regression · Decision Tree · Random Forest

**Primary metrics:** Top-1 Accuracy · Precision@3
        """)

    st.divider()

    st.subheader("Feature Engineering Highlights")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.metric("Rolling windows", "2 (5 & 10 games)", help="ROLL5 / ROLL10 on 11 stats")
    with feat_col2:
        st.metric("Validity masks", "22 IS_VALID flags", help="1 if enough games played, else 0")
    with feat_col3:
        st.metric("Fatigue features", "4 indicators", help="Days rest, B2B, games/7d, win streak")

    st.divider()
    st.subheader("Evaluation Baselines")
    baseline = pd.DataFrame({
        "Engine": ["Engine 1"] * 3 + ["Engine 2"] * 2,
        "Metric": ["ROC-AUC", "Log Loss", "Accuracy", "Top-1 Accuracy", "Precision@3"],
        "Random baseline": ["0.50", "0.693", "50%", "~0.2%", "~0.6%"],
        "Target": ["> 0.70", "< 0.60", "> 60%", "≥ 50%", "≥ 33%"],
    })
    st.dataframe(baseline, use_container_width=True, hide_index=True)


# ── Tab 2: EDA ─────────────────────────────────────────────────────────────

def _render_eda() -> None:
    st.subheader("Exploratory Data Analysis")
    st.caption("16 visualisations across 5 themes — generated during project phase 1 & 2.")

    if not _PLOTS_DIR.exists():
        st.warning(f"Plots directory not found: `{_PLOTS_DIR}`")
        return

    for group, plots in _PLOT_GROUPS.items():
        st.markdown(f"#### {group}")
        ncols = min(len(plots), 3)
        cols = st.columns(ncols)
        for i, (fname, cap) in enumerate(plots):
            p = _PLOTS_DIR / fname
            with cols[i % ncols]:
                if p.exists():
                    st.image(str(p), caption=cap, use_container_width=True)
                else:
                    st.info(f"`{fname}` not generated yet")
        st.divider()


# ── Tab 3: Model Comparison ────────────────────────────────────────────────

def _render_model_comparison() -> None:
    st.subheader("Engine 1 — Model Performance Comparison")

    # Try results CSV first, fall back to inline evaluation
    df = None
    if _RESULTS_FILE.exists():
        df = pd.read_csv(_RESULTS_FILE)
        if "model_name" in df.columns:
            df = df.rename(columns={"model_name": "Model"})
        source = "results/model_metrics.csv"
    else:
        with st.spinner("Running evaluation on trained models..."):
            df = _evaluate_e1_models()
        source = "inline evaluation (models loaded directly)"

    if df is None or df.empty:
        st.warning(
            "No trained models found in `models/engine1/`. "
            "Run `python src/train.py --engine 1` first."
        )
        return

    st.caption(f"Source: {source}")

    # KPI row
    if {"log_loss", "roc_auc", "accuracy"}.issubset(df.columns):
        best = df.loc[df["roc_auc"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", str(best.get("Model", "—")))
        c2.metric("Best ROC-AUC", f"{best['roc_auc']:.4f}", help="Higher is better")
        c3.metric("Best Log Loss", f"{best['log_loss']:.4f}", help="Lower is better")
        c4.metric("Best Accuracy", f"{best['accuracy']:.1%}")

        st.divider()

        col_chart, col_table = st.columns([3, 2], gap="large")
        with col_chart:
            _plot_e1_comparison(df)
        with col_table:
            st.markdown("**Full metrics table**")
            display = df[["Model", "roc_auc", "log_loss", "accuracy"]].copy()
            display = display.sort_values("roc_auc", ascending=False)
            st.dataframe(
                display.style
                    .format({"roc_auc": "{:.4f}", "log_loss": "{:.4f}", "accuracy": "{:.4f}"})
                    .highlight_max(subset=["roc_auc", "accuracy"], color="#1D428A55")
                    .highlight_min(subset=["log_loss"], color="#1D428A55"),
                use_container_width=True,
                hide_index=True,
            )


def _plot_e1_comparison(df: pd.DataFrame) -> None:
    metrics = ["roc_auc", "log_loss", "accuracy"]
    labels  = ["ROC-AUC", "Log Loss", "Accuracy"]
    colors  = ["#1D428A", "#C9A84C", "#28A745"]

    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors):
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=label,
            x=df["Model"] if "Model" in df.columns else df.index,
            y=df[metric],
            marker_color=color,
            text=[f"{v:.4f}" for v in df[metric]],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison — Engine 1",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Engine 1 Demo ───────────────────────────────────────────────────

def _render_engine1_demo() -> None:
    st.subheader("Match Win Probability Predictor")
    st.caption(
        "Enter rolling team stats to predict the win probability using the trained XGBoost model."
    )

    model = _load_e1_model("xgboost") or _load_e1_model("random_forest") or _load_e1_model("logistic_regression")
    model_name = (
        "XGBoost" if _load_e1_model("xgboost") is not None
        else "Random Forest" if _load_e1_model("random_forest") is not None
        else "Logistic Regression" if _load_e1_model("logistic_regression") is not None
        else None
    )

    if model is None:
        st.warning("No Engine 1 models found. Run `python src/train.py --engine 1` first.")
        return

    st.success(f"Using trained **{model_name}** model")

    st.markdown("#### Team Rolling Stats (last 5 games)")
    col1, col2, col3 = st.columns(3)

    with col1:
        home       = st.checkbox("Playing at HOME", value=True)
        pts        = st.slider("Points (PTS)", 85.0, 135.0, 112.0, 0.5)
        fg_pct     = st.slider("Field Goal % (FG%)", 0.35, 0.58, 0.47, 0.01, format="%.2f")
        fg3_pct    = st.slider("3-Point % (FG3%)", 0.28, 0.52, 0.36, 0.01, format="%.2f")
        ft_pct     = st.slider("Free Throw % (FT%)", 0.60, 0.90, 0.77, 0.01, format="%.2f")

    with col2:
        oreb       = st.slider("Off. Rebounds (OREB)", 4.0, 18.0, 10.0, 0.5)
        dreb       = st.slider("Def. Rebounds (DREB)", 25.0, 52.0, 35.0, 0.5)
        ast        = st.slider("Assists (AST)", 15.0, 38.0, 25.0, 0.5)
        stl        = st.slider("Steals (STL)", 4.0, 14.0, 7.5, 0.5)

    with col3:
        blk        = st.slider("Blocks (BLK)", 2.0, 12.0, 5.0, 0.5)
        tov        = st.slider("Turnovers (TOV)", 8.0, 22.0, 14.0, 0.5)
        plus_minus = st.slider("Plus/Minus (±)", -15.0, 15.0, 2.0, 0.5)
        days_rest  = st.selectbox("Days of rest", [0, 1, 2, 3, 4, 5], index=2)
        back2back  = st.checkbox("Back-to-back game", value=False)
        win_streak = st.slider("Win streak", -8, 12, 1)

    if st.button("Predict Win Probability", type="primary", use_container_width=True):
        row = {
            "HOME": int(home),
            "ROLL5_PTS": pts, "ROLL10_PTS": pts,
            "ROLL5_FG_PCT": fg_pct, "ROLL10_FG_PCT": fg_pct,
            "ROLL5_FG3_PCT": fg3_pct, "ROLL10_FG3_PCT": fg3_pct,
            "ROLL5_FT_PCT": ft_pct, "ROLL10_FT_PCT": ft_pct,
            "ROLL5_OREB": oreb, "ROLL10_OREB": oreb,
            "ROLL5_DREB": dreb, "ROLL10_DREB": dreb,
            "ROLL5_AST": ast, "ROLL10_AST": ast,
            "ROLL5_STL": stl, "ROLL10_STL": stl,
            "ROLL5_BLK": blk, "ROLL10_BLK": blk,
            "ROLL5_TOV": tov, "ROLL10_TOV": tov,
            "ROLL5_PLUS_MINUS": plus_minus, "ROLL10_PLUS_MINUS": plus_minus,
            "DAYS_REST": days_rest,
            "BACK_TO_BACK": int(back2back),
            "GAMES_LAST_7D": 3,
            "WIN_STREAK": win_streak,
        }
        # All IS_VALID = 1 (simulating a mid-season team)
        for feat in _E1_FEATURES:
            if feat.endswith("_IS_VALID"):
                row[feat] = 1

        X = pd.DataFrame([row])[_E1_FEATURES]

        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0, 1])
            else:
                prob = float(model.predict(X)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        pct = f"{prob:.1%}"
        color = "#28A745" if prob >= 0.55 else "#DC3545" if prob < 0.45 else "#C9A84C"
        verdict = "WIN predicted" if prob >= 0.5 else "LOSS predicted"

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(
                f'<div class="prob-display">'
                f'<div class="prob-value" style="color:{color}">{pct}</div>'
                f'<div class="prob-label">Win probability</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with res_col2:
            _plot_probability_gauge(prob)

        if prob >= 0.55:
            st.success(f"Model predicts: **{verdict}** ({model_name})")
        elif prob < 0.45:
            st.error(f"Model predicts: **{verdict}** ({model_name})")
        else:
            st.warning(f"Model predicts: **{verdict}** — but this is a close match ({model_name})")


def _plot_probability_gauge(prob: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
            "bar": {"color": "#1D428A"},
            "bgcolor": "#161B22",
            "bordercolor": "#30363D",
            "steps": [
                {"range": [0, 40],  "color": "#DC354522"},
                {"range": [40, 60], "color": "#C9A84C22"},
                {"range": [60, 100],"color": "#28A74522"},
            ],
            "threshold": {
                "line": {"color": "#C9A84C", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Win Probability", "font": {"color": "#8B949E", "size": 14}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        height=260,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Engine 2 Demo ───────────────────────────────────────────────────

def _render_engine2_demo() -> None:
    st.subheader("Award Winner Predictor — Engine 2")
    st.caption(
        "Select an award and model to see how the trained model ranks candidates "
        "in the 2022–23 and 2023–24 test seasons."
    )

    test_df = _load_e2_test_data()
    if test_df is None:
        st.warning("Awards data not found. Check `nba_data/processed/awards_features_labeled.csv`.")
        return

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        award = st.selectbox("Award", _AWARDS)
    with ctrl_col2:
        model_name = st.selectbox("Model", ["random_forest", "logistic_regression", "decision_tree"])
    with ctrl_col3:
        season_filter = st.selectbox("Season", ["All test seasons"] + sorted(test_df["SEASON_YEAR"].unique().tolist(), reverse=True))

    model = _load_e2_model(award, model_name)
    if model is None:
        st.warning(f"Model `{award}_{model_name}.pkl` not found in `models/engine2/`. Train first.")
        return

    df = test_df.copy()
    if season_filter != "All test seasons":
        df = df[df["SEASON_YEAR"] == season_filter]

    avail_feats = [f for f in _E2_FEATURES if f in df.columns]
    X = df[avail_feats].fillna(0)

    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    df = df.copy()
    df["score"] = scores
    df["true_winner"] = df[award].astype(int) if award in df.columns else 0
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Summary metrics
    true_winner_rank = df.loc[df["true_winner"] == 1, "rank"].min() if df["true_winner"].sum() > 0 else None
    top1_correct = int(df.iloc[0]["true_winner"] == 1) if not df.empty else 0
    top3_winners = int(df.head(3)["true_winner"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top-1 Accuracy", "Correct" if top1_correct else "Incorrect",
              delta="Winner ranked #1" if top1_correct else f"Winner at #{true_winner_rank}")
    m2.metric("Precision@3", f"{top3_winners}/3 correct")
    m3.metric("Award", award)
    m4.metric("Candidates ranked", len(df))

    st.divider()

    col_rank, col_chart = st.columns([2, 3], gap="large")

    with col_rank:
        st.markdown(f"**Top 15 candidates — {award}**")
        top15 = df.head(15)[["rank", "PLAYER_NAME", "SEASON_YEAR", "score", "true_winner"]].copy()
        top15 = top15.rename(columns={
            "PLAYER_NAME": "Player", "SEASON_YEAR": "Season",
            "score": "Score", "true_winner": "Winner"
        })
        top15["Score"] = top15["Score"].round(4)
        top15["Winner"] = top15["Winner"].map({1: "YES", 0: ""})

        def _row_style(row):
            if row["Winner"] == "YES":
                return ["background-color: #C9A84C22; color: #C9A84C"] * len(row)
            return [""] * len(row)

        st.dataframe(
            top15.style.apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
            height=450,
        )

    with col_chart:
        _plot_e2_scores(df.head(20), award)


def _plot_e2_scores(df: pd.DataFrame, award: str) -> None:
    players = df["PLAYER_NAME"] + " (" + df["SEASON_YEAR"] + ")"
    colors = ["#C9A84C" if w == 1 else "#1D428A" for w in df["true_winner"]]

    fig = go.Figure(go.Bar(
        x=df["score"],
        y=players,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in df["score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top-20 Predicted {award} Candidates",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        yaxis=dict(autorange="reversed"),
        height=520,
        margin=dict(t=50, b=30, l=10, r=60),
    )
    fig.add_annotation(
        text="Gold = True winner",
        xref="paper", yref="paper",
        x=1.0, y=-0.05,
        showarrow=False,
        font=dict(color="#C9A84C", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    build_app()
