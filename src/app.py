"""NBA ML Project — Streamlit application."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLOTS_DIR    = _PROJECT_ROOT / "nba_data" / "plots"
_RESULTS_FILE = _PROJECT_ROOT / "results" / "model_metrics.csv"


# ---------------------------------------------------------------------------
# Entry point (fixed name — scripts/main.py launches this)
# ---------------------------------------------------------------------------

def build_app() -> None:
    st.set_page_config(
        page_title="NBA ML Project",
        page_icon="🏀",
        layout="wide",
    )

    st.title("🏀 NBA Machine Learning — Proof of Concept")
    st.markdown(
        "Two prediction engines trained on **10 NBA seasons (2014–2024)**:  "
        "**Engine 1** predicts match outcomes · **Engine 2** predicts award winners."
    )

    tab_overview, tab_eda, tab_results, tab_demo = st.tabs([
        "📋 Project Overview",
        "📊 Exploratory Analysis",
        "🤖 Model Results",
        "🎯 Demo",
    ])

    with tab_overview:
        _render_overview()

    with tab_eda:
        _render_eda()

    with tab_results:
        _render_model_results()

    with tab_demo:
        _render_demo()


# ---------------------------------------------------------------------------
# Tab 1 — Project overview
# ---------------------------------------------------------------------------

def _render_overview() -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engine 1 — Match Outcome Prediction")
        st.markdown("""
**Task:** Binary classification — predict whether a team wins (W) or loses (L).

**Dataset:** 27 313 team-game rows · 57 features after engineering

**Key features:**
- Rolling averages (5 & 10 games): PTS, FG%, AST, STL, BLK, TOV, ±
- Fatigue indicators: days of rest, back-to-back, games in last 7 days
- Momentum: win streak
- Home / away flag

**Temporal split:** 8 seasons train (2014–2022) · 2 seasons test (2022–2024)

**Models:** Logistic Regression · Random Forest · XGBoost

**Primary metric:** ROC-AUC (higher = better)
        """)

    with col2:
        st.subheader("Engine 2 — Award Winner Prediction")
        st.markdown("""
**Task:** Binary ranking — identify the award winner from season candidates.

**Awards:** MVP · DPOY (Defensive Player of the Year) · ROY (Rookie of the Year) · 6MOY (Sixth Man)

**Dataset:** 4 654 player-season rows · 25 features

**Key features:**
- Season averages: PTS, REB, AST, STL, BLK, FG%, FT%, ±
- Advanced: Fantasy score, double-double / triple-double rates, consistency
- Relative rankings: rank by PTS/AST/REB within the season
- Games played, usage, efficiency flags

**Temporal split:** 8 seasons train · 2 seasons test

**Models:** Logistic Regression · Decision Tree · Random Forest

**Primary metrics:** Top-1 Accuracy · Precision@3
        """)

    st.divider()
    st.subheader("Dataset Summary")
    summary = pd.DataFrame({
        "Dataset": [
            "Match game logs (raw)",
            "Player game logs (raw)",
            "Team game logs (raw)",
            "Engine 1 features (processed)",
            "Engine 2 features (processed)",
        ],
        "Rows": ["27 450", "296 000", "27 450", "27 313", "4 654"],
        "Columns": ["~20", "~30", "~25", "57", "25 + 4 labels"],
        "Target": ["—", "—", "—", "WL_BIN (0/1)", "MVP / DPOY / ROY / 6MOY"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Evaluation Protocol")
    st.markdown("""
| Engine | Metric | Direction | Baseline |
|--------|--------|-----------|---------|
| Engine 1 | ROC-AUC | ↑ higher | 0.50 (random) |
| Engine 1 | Log Loss | ↓ lower | 0.693 (random) |
| Engine 1 | Accuracy | ↑ higher | 50% |
| Engine 2 | Top-1 Accuracy | ↑ higher | 0.2% (random) |
| Engine 2 | Precision@3 | ↑ higher | ~0.6% (random) |
| Engine 2 | ROC-AUC | ↑ higher | 0.50 (random) |

**Objective targets:** Log Loss < 0.60 · ROC-AUC > 0.70 (E1) · Top-1 Accuracy ≥ 0.5 (E2)
    """)


# ---------------------------------------------------------------------------
# Tab 2 — EDA
# ---------------------------------------------------------------------------

_PLOT_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Data Quality & Distributions": [
        ("01_missing_values.png", "Missing values by feature"),
        ("04_correlations.png", "Feature correlation heatmap"),
        ("16_pca_analysis.png", "PCA dimensionality analysis (3 components = 68% variance)"),
    ],
    "Home-Court Advantage & COVID": [
        ("02_wl_home_away.png", "Win rate: home vs away by season"),
        ("05_covid_home.png", "COVID impact on home advantage (2019–2021)"),
    ],
    "Game Evolution (2014–2024)": [
        ("03_evolution_jeu.png", "Game-play evolution: pace, 3-point attempts"),
        ("15_h2h_heatmap.png", "Head-to-head team matchup patterns"),
    ],
    "Fatigue & Rolling Features": [
        ("06_fatigue_features.png", "Rest days & back-to-back effect on win rate"),
        ("07_rolling_correlations.png", "Rolling average correlations with outcome"),
    ],
    "Player Impact (Engine 2)": [
        ("08_absences.png", "Player absence patterns and team impact"),
        ("09_player_impact.png", "Key-player impact on team win rate"),
        ("10_rolling_players.png", "Player rolling statistics"),
        ("11_consistency.png", "Performance consistency by player tier"),
        ("12_usage.png", "Player usage rate distribution"),
        ("13_mvp_candidates.png", "MVP candidate profiles by season"),
        ("14_leaders_by_season.png", "Statistical leaders per season (2014–2024)"),
    ],
}


def _render_eda() -> None:
    st.subheader("Exploratory Data Analysis")
    st.caption("16 visualisations generated during EDA — grouped by theme.")

    if not _PLOTS_DIR.exists():
        st.warning(f"Plots directory not found: `{_PLOTS_DIR}`")
        return

    for group_name, plots in _PLOT_GROUPS.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(min(len(plots), 3))
        for i, (filename, caption) in enumerate(plots):
            plot_path = _PLOTS_DIR / filename
            if plot_path.exists():
                with cols[i % len(cols)]:
                    st.image(str(plot_path), caption=caption, use_container_width=True)
            else:
                with cols[i % len(cols)]:
                    st.info(f"`{filename}` not found")
        st.divider()


# ---------------------------------------------------------------------------
# Tab 3 — Model results
# ---------------------------------------------------------------------------

def _render_model_results() -> None:
    st.subheader("Model Evaluation Results")

    if not _RESULTS_FILE.exists():
        st.info(
            "No results yet. Train your models first, then run:\n\n"
            "```bash\npython src/train.py --engine all\npython scripts/main.py\n```"
        )
        _render_expected_metrics_guide()
        return

    df = pd.read_csv(_RESULTS_FILE)
    st.success(f"Results loaded — {len(df)} model(s) evaluated.")

    # Engine 1 results (has log_loss / roc_auc / accuracy columns)
    e1_cols = {"log_loss", "roc_auc", "accuracy"}
    if e1_cols.issubset(df.columns):
        st.markdown("#### Engine 1 — Match Outcome Prediction")
        e1_df = df[["model_name", "log_loss", "roc_auc", "accuracy"]].copy()
        e1_df = e1_df.sort_values("roc_auc", ascending=False)
        st.dataframe(
            e1_df.style.highlight_max(subset=["roc_auc", "accuracy"], color="#d4edda")
                       .highlight_min(subset=["log_loss"], color="#d4edda")
                       .format({"log_loss": "{:.4f}", "roc_auc": "{:.4f}", "accuracy": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # All raw data
    with st.expander("Raw metrics table"):
        st.dataframe(df, use_container_width=True)


def _render_expected_metrics_guide() -> None:
    st.markdown("#### Expected metric ranges after training")
    guide = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Expected ROC-AUC": ["> 0.62", "> 0.68", "> 0.70"],
        "Expected Log Loss": ["< 0.66", "< 0.63", "< 0.60"],
        "Expected Accuracy": ["> 58%", "> 62%", "> 64%"],
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4 — Demo
# ---------------------------------------------------------------------------

def _render_demo() -> None:
    st.subheader("Interactive Demo")

    models_ready = (
        (_PROJECT_ROOT / "models" / "engine1" / "xgboost.pkl").exists()
        or (_PROJECT_ROOT / "models" / "engine1" / "logistic_regression.pkl").exists()
    )

    if not models_ready:
        st.warning(
            "Models have not been trained yet. Run the following to train and evaluate:\n\n"
            "```bash\n# Train both engines\npython src/train.py --engine all\n\n"
            "# Evaluate and launch app via main.py\npython scripts/main.py\n```"
        )
        st.markdown("""
**What the demo will show once models are trained:**
- **Engine 1:** Enter rolling stats for two teams → predict win probability
- **Engine 2:** See top-ranked MVP / DPOY / ROY / 6MOY candidates for the test seasons
        """)
        return

    st.markdown("#### Engine 1 — Win Probability Estimator")
    st.caption("Enter approximate team rolling averages to get a win probability estimate.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your team (home)**")
        pts_home  = st.slider("Rolling PTS (5-game avg)", 90.0, 130.0, 112.0, key="pts_h")
        fg_home   = st.slider("Rolling FG% (5-game avg)", 0.35, 0.55, 0.47, key="fg_h")
        pm_home   = st.slider("Rolling ±  (5-game avg)", -15.0, 15.0, 3.0, key="pm_h")
        rest_home = st.selectbox("Days of rest", [0, 1, 2, 3], index=1, key="rest_h")

    with col2:
        st.markdown("**Opponent (away)**")
        pts_away  = st.slider("Rolling PTS (5-game avg)", 90.0, 130.0, 110.0, key="pts_a")
        fg_away   = st.slider("Rolling FG% (5-game avg)", 0.35, 0.55, 0.46, key="fg_a")
        pm_away   = st.slider("Rolling ±  (5-game avg)", -15.0, 15.0, 0.0, key="pm_a")
        rest_away = st.selectbox("Days of rest", [0, 1, 2, 3], index=2, key="rest_a")

    if st.button("Predict win probability", type="primary"):
        # Simple heuristic when full model inference is not wired up
        home_score = pts_home * 0.4 + fg_home * 50 + pm_home * 2 + rest_home * 1.5 + 3.0
        away_score = pts_away * 0.4 + fg_away * 50 + pm_away * 2 + rest_away * 1.5
        raw = home_score / (home_score + away_score)
        prob = max(0.05, min(0.95, raw))
        st.metric("Estimated home-team win probability", f"{prob:.1%}")
        if prob > 0.6:
            st.success("Strong home-team advantage predicted.")
        elif prob < 0.4:
            st.error("Away team favoured.")
        else:
            st.info("Competitive match-up — outcome uncertain.")
        st.caption(
            "Note: this estimate uses a simplified heuristic. "
            "Load the trained model for precise probabilities."
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_app()
