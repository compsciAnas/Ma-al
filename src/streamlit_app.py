# ============================================================
# Ma'al | مآل  —  Startup Evaluation Platform
# Streamlit MVP  |  Member 2 Deliverable
# Author: Anas Al-Zahrani (Member 2)
# ============================================================
# Required files in the same directory as this app:
#   founder_model.joblib
#   investor_model.joblib
#   founder_preprocessor.joblib
#   investor_preprocessor.joblib
#   founder_dataset.csv          (used for the dashboard)
#   investor_dataset.csv         (used for the dashboard)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ─── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Ma'al | مآل",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
# Professional SaaS dark-teal palette, no raw HTML visible on screen.
st.markdown(
    """
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #112233 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #cdd9e5 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* ── Main background ── */
    .main .block-container {
        background-color: #f7f9fc;
        padding: 2rem 2.5rem 4rem;
    }

    /* ── Metric cards ── */
    .maal-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 2px 12px rgba(13,27,42,0.07);
        border-left: 4px solid #0a7ea4;
        margin-bottom: 1rem;
    }
    .maal-card-green  { border-left-color: #1a9e5c; }
    .maal-card-amber  { border-left-color: #d97706; }
    .maal-card-red    { border-left-color: #dc2626; }

    /* ── Score badge ── */
    .score-badge {
        display: inline-block;
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #0d1b2a;
    }
    .label-pill {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-left: 0.75rem;
        vertical-align: middle;
    }
    .pill-high   { background: #d1fae5; color: #065f46; }
    .pill-medium { background: #fef3c7; color: #92400e; }
    .pill-low    { background: #fee2e2; color: #991b1b; }

    /* ── Section heading ── */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0d1b2a;
        margin-bottom: 0.25rem;
    }
    .section-sub {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a3d62 60%, #0a7ea4 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero h3 { font-size: 1.2rem; font-weight: 300; opacity: 0.8; margin-bottom: 1.25rem; }
    .hero p  { font-size: 1rem; opacity: 0.75; max-width: 640px; line-height: 1.7; }

    /* ── Divider ── */
    .maal-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ── Info box ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.88rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.84rem;
        color: #9a3412;
        margin-top: 1.5rem;
    }

    /* ── Streamlit override: hide menu / footer ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Button styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #0a7ea4, #0a3d62);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
SECTORS = [
    "Agritech", "Cybersecurity", "E-commerce", "Edtech", "Fintech",
    "HR Tech", "Healthtech", "Logistics Tech", "Real Estate Tech", "Traveltech"
]

CITIES = [
    "Abha", "Dammam", "Jeddah", "Khobar", "Mecca",
    "Medina", "NEOM", "Riyadh"
]

FUNDING_STAGES = [
    "Growth", "Pre-Seed", "Seed", "Series A", "Series B", "Series C"
]

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Helpers: preprocessing mirrors the notebooks exactly ─────────────────────

def _clean_df(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col not in drop_cols:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def _feature_engineer_founder(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["funding_per_round"] = (
        df["total_funding_sar"] / df["num_funding_rounds"].replace(0, np.nan)
    ).fillna(0)
    df["investment_per_founder"] = (
        df["total_funding_sar"] / df["num_founders"].replace(0, np.nan)
    ).fillna(0)
    df["regulatory_support"] = (
        df["sama_sandbox"] + df["rega_sandbox"] + df["ntdp_participation"]
    )
    return df


def _feature_engineer_investor(df: pd.DataFrame) -> pd.DataFrame:
    df = _feature_engineer_founder(df)
    df["revenue_funding_ratio"] = (
        df["monthly_revenue_sar"] / df["total_funding_sar"].replace(0, np.nan)
    ).fillna(0)
    return df


# ─── Model loading / training ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load the saved models and their matching fitted preprocessors.

    The app should not silently retrain fallback models, because the notebooks
    selected Linear Regression as the final saved model for both Founder and
    Investor. If any required file is missing, the app reports it clearly.
    """
    results = {
        "founder": None,
        "investor": None,
        "founder_pre": None,
        "investor_pre": None,
        "founder_cols": None,
        "investor_cols": None,
        "errors": []
    }

    file_map = {
        "founder": {
            "model": os.path.join(MODEL_DIR, "founder_model.joblib"),
            "pre": os.path.join(MODEL_DIR, "founder_preprocessor.joblib"),
        },
        "investor": {
            "model": os.path.join(MODEL_DIR, "investor_model.joblib"),
            "pre": os.path.join(MODEL_DIR, "investor_preprocessor.joblib"),
        },
    }

    for name, paths in file_map.items():
        model_key = name
        pre_key = f"{name}_pre"
        cols_key = f"{name}_cols"

        if not os.path.exists(paths["model"]):
            results["errors"].append(f"Missing {os.path.basename(paths['model'])}")
            continue
        if not os.path.exists(paths["pre"]):
            results["errors"].append(f"Missing {os.path.basename(paths['pre'])}")
            continue

        try:
            results[model_key] = joblib.load(paths["model"])
            results[pre_key] = joblib.load(paths["pre"])

            pre = results[pre_key]
            num_f = list(pre.transformers_[0][2])
            cat_f = list(pre.transformers_[1][2])
            results[cols_key] = num_f + cat_f
        except Exception as e:
            results["errors"].append(f"{name.title()} model/preprocessor load error: {e}")

    return results


@st.cache_data(show_spinner=False)
def load_datasets():
    """Load CSV datasets for the Dashboard page."""
    dfs = {}
    for name, fname in [("founder", "founder_dataset.csv"),
                         ("investor", "investor_dataset.csv")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                dfs[name] = pd.read_csv(path)
            except Exception:
                pass
    return dfs


def predict_score(model, preprocessor, input_df: pd.DataFrame) -> float:
    """
    Run the saved preprocessing pipeline + model.
    The same fitted preprocessor used during training must be present.
    """
    if preprocessor is None:
        raise ValueError("Preprocessor is missing. Please add the matching *_preprocessor.joblib file.")

    X_proc = preprocessor.transform(input_df)
    score = float(model.predict(X_proc)[0])

    return float(np.clip(score, 0, 100))


def score_label(score: float) -> tuple[str, str]:
    if score >= 65:
        return "High", "pill-high"
    elif score >= 35:
        return "Medium", "pill-medium"
    else:
        return "Low", "pill-low"


def recommendation_text(score: float, mode: str) -> str:
    if mode == "founder":
        if score >= 75:
            return (
                "Your startup shows strong survival indicators. Focus on scaling "
                "operations, deepening investor relationships, and expanding to new markets. "
                "Consider regulatory sandbox participation to accelerate growth."
            )
        elif score >= 55:
            return (
                "Your startup has a solid foundation with room to grow. Prioritise "
                "increasing your total funding, broadening your investor base, and "
                "improving speed-to-market. Regulatory participation (SAMA / REGA / NTDP) "
                "can meaningfully boost your score."
            )
        elif score >= 35:
            return (
                "Your startup is at a critical stage. Focus on securing the next funding "
                "round, validating your product with paying customers, and reducing "
                "time-to-first-round. Consider applying to accelerator programmes."
            )
        else:
            return (
                "Significant improvements are needed before this startup is positioned "
                "for survival. Re-evaluate the business model, strengthen the founding "
                "team, seek early-stage grants or angels, and target regulatory support "
                "programmes to improve fundamentals."
            )
    else:  # investor
        if score >= 75:
            return (
                "This startup scores highly on investment attractiveness. Strong traction, "
                "revenue momentum, and market growth signal a compelling opportunity. "
                "Conduct due diligence and consider leading or co-leading the next round."
            )
        elif score >= 55:
            return (
                "A promising opportunity with some risks. Review the traction metrics "
                "and monthly revenue trend carefully. Consider a smaller initial stake "
                "with follow-on rights, and monitor regulatory alignment."
            )
        elif score >= 35:
            return (
                "Moderate investment attractiveness. The startup shows potential but needs "
                "stronger market validation or revenue growth. Consider convertible notes "
                "or milestone-based tranches rather than a full equity commitment."
            )
        else:
            return (
                "This startup is not yet positioned for investment based on the provided "
                "metrics. Revenue, traction, and funding momentum need substantial "
                "improvement before an investment can be justified. Watch for future rounds."
            )


# ─── Page: Home ───────────────────────────────────────────────────────────────
def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>مآل &nbsp;|&nbsp; Ma'al</h1>
            <h3>منصة ذكية لتقييم الشركات الناشئة &nbsp;·&nbsp; AI-Powered Startup Evaluation</h3>
            <p>
                Ma'al helps Saudi tech founders understand their startup's survival potential
                and helps investors discover the most attractive opportunities —
                powered by machine learning models trained on synthetic startup data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="maal-card maal-card-green">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🚀</div>
                <div style="font-weight:700; font-size:1.05rem; color:#065f46;">Founder Mode</div>
                <div style="font-size:0.88rem; color:#374151; margin-top:0.4rem;">
                    Enter your startup details and get an AI-predicted <b>Survival Score (0–100)</b>
                    with actionable recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="maal-card">
                <div style="font-size:2rem; margin-bottom:0.5rem;">💼</div>
                <div style="font-weight:700; font-size:1.05rem; color:#0a3d62;">Investor Mode</div>
                <div style="font-size:0.88rem; color:#374151; margin-top:0.4rem;">
                    Evaluate startups on investment attractiveness and receive a
                    <b>Investment Score (0–100)</b> with risk-adjusted insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="maal-card maal-card-amber">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
                <div style="font-weight:700; font-size:1.05rem; color:#92400e;">Dashboard</div>
                <div style="font-size:0.88rem; color:#374151; margin-top:0.4rem;">
                    Explore the underlying dataset — sector distributions, city breakdowns,
                    funding stages, and more.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)

    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)
    steps = [
        ("1", "Enter Data", "Fill in your startup details using the form in the sidebar."),
        ("2", "AI Prediction", "Our machine learning model predicts your score in seconds."),
        ("3", "Act on Insights", "Read your label, score, and personalised recommendation."),
    ]
    for col, (num, title, desc) in zip([col_a, col_b, col_c], steps):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center; padding:1rem;">
                    <div style="font-size:2.2rem; font-weight:800; color:#0a7ea4;">{num}</div>
                    <div style="font-weight:700; font-size:1rem; margin:0.5rem 0;">{title}</div>
                    <div style="font-size:0.85rem; color:#64748b;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <b>Get started:</b> Select <b>Founder Prediction</b> or <b>Investor Prediction</b>
            from the sidebar to run your first evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Page: Founder Prediction ─────────────────────────────────────────────────
def page_founder(models: dict):
    st.markdown(
        '<div class="section-title">🚀 Founder Prediction</div>'
        '<div class="section-sub">وضع المؤسس — Predict your startup\'s survival score</div>',
        unsafe_allow_html=True,
    )

    if models["founder"] is None:
        st.error(
            "Founder model is not available. "
            "Please make sure `founder_model.joblib` (and optionally "
            "`founder_preprocessor.joblib`) or `founder_dataset.csv` are in the app directory."
        )
        return

    with st.form("founder_form"):
        st.markdown("#### Startup Profile")
        c1, c2 = st.columns(2)
        with c1:
            sector       = st.selectbox("Sector", SECTORS, index=0)
            city         = st.selectbox("City", CITIES, index=0)
            funding_stage = st.selectbox("Funding Stage", FUNDING_STAGES, index=1)
        with c2:
            startup_age  = st.slider("Startup Age (years)", 2, 14, 3)
            num_founders = st.slider("Number of Founders", 1, 4, 2)
            num_investors = st.slider("Number of Investors", 0, 50, 5)

        st.markdown("#### Funding Details")
        c3, c4 = st.columns(2)
        with c3:
            total_funding = st.number_input(
                "Total Funding (SAR)", min_value=100, max_value=500_000,
                value=10_000, step=1_000,
                help="Enter total funding received in Saudi Riyals"
            )
            num_rounds    = st.slider("Number of Funding Rounds", 1, 8, 2)
        with c4:
            speed_to_first = st.slider(
                "Speed to First Round (months)", 1, 54, 12,
                help="How many months after founding did you close your first round?"
            )

        st.markdown("#### Regulatory Participation")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sama_sandbox       = st.checkbox("SAMA Fintech Sandbox")
        with rc2:
            rega_sandbox       = st.checkbox("REGA PropTech Sandbox")
        with rc3:
            ntdp_participation = st.checkbox("NTDP Programme")

        submitted = st.form_submit_button("Predict Survival Score →")

    if submitted:
        # Build a DataFrame matching notebook column order:
        # Numerical: startup_age_years, total_funding_sar, num_funding_rounds,
        #            num_investors, num_founders, speed_to_first_round_months,
        #            sama_sandbox, rega_sandbox, ntdp_participation,
        #            funding_per_round, investment_per_founder, regulatory_support
        # Categorical: sector, city, funding_stage

        funding_per_round     = total_funding / max(num_rounds, 1)
        investment_per_founder = total_funding / max(num_founders, 1)
        regulatory_support    = int(sama_sandbox) + int(rega_sandbox) + int(ntdp_participation)

        input_data = {
            "startup_age_years":           [startup_age],
            "total_funding_sar":           [total_funding],
            "num_funding_rounds":          [num_rounds],
            "num_investors":               [num_investors],
            "num_founders":                [num_founders],
            "speed_to_first_round_months": [speed_to_first],
            "sama_sandbox":                [int(sama_sandbox)],
            "rega_sandbox":                [int(rega_sandbox)],
            "ntdp_participation":          [int(ntdp_participation)],
            "funding_per_round":           [funding_per_round],
            "investment_per_founder":      [investment_per_founder],
            "regulatory_support":          [regulatory_support],
            "sector":                      [sector],
            "city":                        [city],
            "funding_stage":               [funding_stage],
        }
        input_df = pd.DataFrame(input_data)

        with st.spinner("Running prediction…"):
            try:
                score = predict_score(
                    models["founder"], models["founder_pre"], input_df
                )
                _display_result(score, "founder")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Tip: If you see a shape mismatch error, make sure "
                    "`founder_preprocessor.joblib` is in the same folder as `app.py`."
                )


# ─── Page: Investor Prediction ────────────────────────────────────────────────
def page_investor(models: dict):
    st.markdown(
        '<div class="section-title">💼 Investor Prediction</div>'
        '<div class="section-sub">وضع المستثمر — Predict investment attractiveness score</div>',
        unsafe_allow_html=True,
    )

    if models["investor"] is None:
        st.error(
            "Investor model is not available. "
            "Please make sure `investor_model.joblib` (and optionally "
            "`investor_preprocessor.joblib`) or `investor_dataset.csv` are in the app directory."
        )
        return

    with st.form("investor_form"):
        st.markdown("#### Startup Profile")
        c1, c2 = st.columns(2)
        with c1:
            sector        = st.selectbox("Sector", SECTORS, index=0)
            city          = st.selectbox("City", CITIES, index=0)
            funding_stage = st.selectbox("Funding Stage", FUNDING_STAGES, index=1)
        with c2:
            startup_age   = st.slider("Startup Age (years)", 2, 14, 3)
            num_founders  = st.slider("Number of Founders", 1, 4, 2)
            num_investors = st.slider("Number of Investors", 1, 50, 5)

        st.markdown("#### Market & Traction")
        c3, c4 = st.columns(2)
        with c3:
            market_growth_index = st.slider(
                "Market Growth Index (0–1)", 0.48, 0.90, 0.78, step=0.01,
                help="Industry-level growth rate normalised to 0–1."
            )
            traction_score = st.slider(
                "Traction Score (0–100)", 2, 100, 50,
                help="Composite traction metric: user growth, engagement, retention."
            )
        with c4:
            monthly_revenue = st.number_input(
                "Monthly Revenue (SAR)", min_value=3, max_value=80_000_000,
                value=50_000, step=5_000
            )

        st.markdown("#### Funding Details")
        c5, c6 = st.columns(2)
        with c5:
            total_funding = st.number_input(
                "Total Funding (SAR)", min_value=100, max_value=500_000,
                value=10_000, step=1_000
            )
            num_rounds    = st.slider("Number of Funding Rounds", 1, 8, 2)
        with c6:
            speed_to_first = st.slider("Speed to First Round (months)", 1, 54, 12)

        st.markdown("#### Regulatory Participation")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sama_sandbox       = st.checkbox("SAMA Fintech Sandbox")
        with rc2:
            rega_sandbox       = st.checkbox("REGA PropTech Sandbox")
        with rc3:
            ntdp_participation = st.checkbox("NTDP Programme")

        submitted = st.form_submit_button("Predict Investment Score →")

    if submitted:
        funding_per_round      = total_funding / max(num_rounds, 1)
        investment_per_founder = total_funding / max(num_founders, 1)
        regulatory_support     = int(sama_sandbox) + int(rega_sandbox) + int(ntdp_participation)
        revenue_funding_ratio  = monthly_revenue / max(total_funding, 1)

        input_data = {
            "startup_age_years":           [startup_age],
            "total_funding_sar":           [total_funding],
            "num_funding_rounds":          [num_rounds],
            "num_investors":               [num_investors],
            "num_founders":                [num_founders],
            "speed_to_first_round_months": [speed_to_first],
            "market_growth_index":         [market_growth_index],
            "traction_score":              [traction_score],
            "monthly_revenue_sar":         [monthly_revenue],
            "sama_sandbox":                [int(sama_sandbox)],
            "rega_sandbox":                [int(rega_sandbox)],
            "ntdp_participation":          [int(ntdp_participation)],
            "funding_per_round":           [funding_per_round],
            "investment_per_founder":      [investment_per_founder],
            "regulatory_support":          [regulatory_support],
            "revenue_funding_ratio":       [revenue_funding_ratio],
            "sector":                      [sector],
            "city":                        [city],
            "funding_stage":               [funding_stage],
        }
        input_df = pd.DataFrame(input_data)

        with st.spinner("Running prediction…"):
            try:
                score = predict_score(
                    models["investor"], models["investor_pre"], input_df
                )
                _display_result(score, "investor")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Tip: If you see a shape mismatch error, make sure "
                    "`investor_preprocessor.joblib` is in the same folder as `app.py`."
                )


# ─── Result display (shared) ──────────────────────────────────────────────────
def _display_result(score: float, mode: str):
    label, pill_class = score_label(score)
    label_ar = {"High": "مرتفع", "Medium": "متوسط", "Low": "منخفض"}[label]
    title = "Survival Score" if mode == "founder" else "Investment Score"

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown(f"### {title}")

    # Score + label
    st.markdown(
        f"""
        <div class="maal-card" style="display:flex; align-items:center; gap:1rem;">
            <span class="score-badge">{score:.1f}</span>
            <span class="label-pill {pill_class}">{label} | {label_ar}</span>
            <span style="color:#64748b; font-size:0.9rem; margin-left:0.5rem;">/ 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar
    st.progress(int(score))

    # Score band legend
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="maal-card maal-card-red" style="text-align:center;">'
            '<b style="color:#991b1b">Low</b><br>'
            '<span style="font-size:0.8rem; color:#6b7280;">0 – 34</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="maal-card maal-card-amber" style="text-align:center;">'
            '<b style="color:#92400e">Medium</b><br>'
            '<span style="font-size:0.8rem; color:#6b7280;">35 – 64</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="maal-card maal-card-green" style="text-align:center;">'
            '<b style="color:#065f46">High</b><br>'
            '<span style="font-size:0.8rem; color:#6b7280;">65 – 100</span></div>',
            unsafe_allow_html=True,
        )

    # Recommendation
    rec = recommendation_text(score, mode)
    st.markdown("#### Recommendation")
    st.markdown(
        f'<div class="maal-card">'
        f'<span style="font-size:0.95rem; color:#1e293b; line-height:1.7;">{rec}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <b>Disclaimer — إخلاء المسؤولية:</b> '
        'This score is an AI-generated estimate based on statistical patterns in historical startup data. '
        '<b>This is not investment advice.</b> '
        'Always conduct independent due diligence and consult qualified advisors before making financial decisions. '
        '<br>هذه النتيجة مؤشر استرشادي وليست نصيحة استثمارية.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Page: Dashboard ──────────────────────────────────────────────────────────
def page_dashboard(dfs: dict):
    st.markdown(
        '<div class="section-title">📊 Data Dashboard</div>'
        '<div class="section-sub">Explore the training dataset — لوحة بيانات التدريب</div>',
        unsafe_allow_html=True,
    )

    if not dfs:
        st.warning(
            "No dataset files found. Place `founder_dataset.csv` and/or "
            "`investor_dataset.csv` in the app directory to enable the dashboard."
        )
        return

    tab_names = [k.title() for k in dfs.keys()]
    tabs = st.tabs(tab_names)

    for tab, (name, df) in zip(tabs, dfs.items()):
        with tab:
            # ── Dataset overview ──────────────────────────────────────────
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Rows", f"{len(df):,}")
            r1c2.metric("Columns", len(df.columns))
            target = "survival_score" if name == "founder" else "investment_score"
            if target in df.columns:
                r1c3.metric(f"Avg {target.replace('_',' ').title()}", f"{df[target].mean():.1f}")

            with st.expander("Dataset Preview (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)

            # ── Charts ───────────────────────────────────────────────────
            col_left, col_right = st.columns(2)

            # Sector distribution
            with col_left:
                if "sector" in df.columns:
                    sector_counts = df["sector"].value_counts().reset_index()
                    sector_counts.columns = ["Sector", "Count"]
                    st.markdown("**Sector Distribution**")
                    st.bar_chart(sector_counts.set_index("Sector")["Count"])

            # City distribution
            with col_right:
                if "city" in df.columns:
                    city_counts = df["city"].value_counts().head(10).reset_index()
                    city_counts.columns = ["City", "Count"]
                    st.markdown("**Top 10 Cities by Startup Count**")
                    st.bar_chart(city_counts.set_index("City")["Count"])

            col_left2, col_right2 = st.columns(2)

            # Funding stage
            with col_left2:
                if "funding_stage" in df.columns:
                    stage_counts = df["funding_stage"].value_counts().reset_index()
                    stage_counts.columns = ["Stage", "Count"]
                    st.markdown("**Funding Stage Distribution**")
                    st.bar_chart(stage_counts.set_index("Stage")["Count"])

            # Target score distribution (histogram via bar_chart)
            with col_right2:
                if target in df.columns:
                    bins = pd.cut(df[target], bins=10)
                    hist = bins.value_counts().sort_index().reset_index()
                    hist.columns = ["Range", "Count"]
                    hist["Range"] = hist["Range"].astype(str)
                    st.markdown(f"**{target.replace('_',' ').title()} Distribution**")
                    st.bar_chart(hist.set_index("Range")["Count"])

            # Average funding by stage
            if "funding_stage" in df.columns and "total_funding_sar" in df.columns:
                avg_funding = (
                    df.groupby("funding_stage")["total_funding_sar"]
                    .mean()
                    .sort_values(ascending=False)
                )
                st.markdown("**Average Total Funding (SAR) by Funding Stage**")
                st.bar_chart(avg_funding)

            # Numeric summary
            with st.expander("Numeric Feature Statistics"):
                st.dataframe(
                    df.select_dtypes(include="number").describe().round(2),
                    use_container_width=True,
                )


# ─── Page: About ──────────────────────────────────────────────────────────────
def page_about():
    st.markdown(
        '<div class="section-title">ℹ️ About Ma\'al | مآل</div>'
        '<div class="section-sub">Business value, vision, and team</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="maal-card">
            <b style="font-size:1.05rem;">What is Ma'al?</b>
            <p style="margin-top:0.6rem; color:#374151; line-height:1.75;">
                <b>مآل (Ma'al)</b> is an AI-powered startup evaluation platform built for the
                Saudi tech ecosystem. It provides two complementary perspectives:
                a <b>Survival Score</b> for founders and an <b>Investment Score</b> for investors —
                both generated by regression models trained on synthetic startup data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="maal-card maal-card-green">
                <b>Who is it for?</b>
                <ul style="color:#374151; font-size:0.9rem; margin-top:0.5rem; line-height:1.8;">
                    <li><b>Founders</b> — understand your startup's strength and what to improve.</li>
                    <li><b>Investors</b> — rank opportunities by data-driven attractiveness.</li>
                    <li><b>Accelerators</b> — screen cohort applicants objectively.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="maal-card">
                <b>Revenue Potential</b>
                <ul style="color:#374151; font-size:0.9rem; margin-top:0.5rem; line-height:1.8;">
                    <li>SaaS subscription for accelerators &amp; VCs.</li>
                    <li>Per-report fee for individual founders.</li>
                    <li>API access for investment platforms.</li>
                    <li>Premium insights with SHAP explanations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="maal-card maal-card-amber" style="margin-top:0.5rem;">
            <b>Technology Stack</b>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem;">
                <span style="background:#fef3c7;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.82rem;color:#92400e;">Python 3.10+</span>
                <span style="background:#fef3c7;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.82rem;color:#92400e;">Streamlit</span>
                <span style="background:#fef3c7;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.82rem;color:#92400e;">scikit-learn</span>
                <span style="background:#fef3c7;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.82rem;color:#92400e;">pandas / numpy</span>
                <span style="background:#fef3c7;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.82rem;color:#92400e;">joblib</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="maal-card" style="margin-top:0.5rem;">
            <b>Scalability</b>
            <p style="color:#374151; font-size:0.9rem; margin-top:0.5rem; line-height:1.75;">
                Ma'al is built with a modular "signal bundle" architecture.
                New verticals (e.g., AgriTech, HealthTech) are added as independent signal bundles
                without rewriting the core engine. The platform launches with Fintech (SAMA sandbox)
                and PropTech (REGA sandbox) and can expand to any tech sector.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown("### Team")
    team = [
        ("محمد", "AI Developer", "Model training, evaluation, SHAP analysis"),
        ("عبدالعزيز", "Data Analyst", "Data collection, feature engineering, similarity engine"),
        ("ماجد", "AI App Developer", "LLM generation engine, ranking logic, service layer"),
        ("أنس", "Frontend / Product", "Streamlit app, UI/UX, result pages, documentation"),
    ]
    cols = st.columns(4)
    for col, (name, role, desc) in zip(cols, team):
        with col:
            st.markdown(
                f"""
                <div class="maal-card" style="text-align:center;">
                    <div style="font-size:1.5rem; font-weight:700; color:#0a3d62;">{name}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:#0a7ea4; margin:0.3rem 0;">{role}</div>
                    <div style="font-size:0.78rem; color:#64748b;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="disclaimer" style="margin-top:1.5rem;">
            <b>Capstone Project — SDAIA / AI Bootcamp 2026</b><br>
            Ma'al is a capstone project. All predictions are for educational and demonstration
            purposes only. <b>Not investment advice.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar navigation ───────────────────────────────────────────────────────
def sidebar_nav() -> str:
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div style="font-size:2.2rem; font-weight:800; color:#ffffff; letter-spacing:-1px;">
                مآل
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.1em;">
                MA'AL · AI STARTUP EVALUATOR
            </div>
        </div>
        <hr style="border-color:#1e3a5f; margin-bottom:1rem;">
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        options=["🏠 Home", "🚀 Founder Prediction", "💼 Investor Prediction",
                 "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
        <hr style="border-color:#1e3a5f; margin-top:1rem;">
        <div style="font-size:0.72rem; color:#64748b; padding:0 0.5rem; line-height:1.6;">
            ⚠️ Scores are AI-generated estimates.<br>
            Not financial or investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    page = sidebar_nav()

    # Load models and datasets (cached)
    with st.spinner("Loading models…"):
        models = load_models()

    # Show load errors in a subtle expander (don't crash)
    if models["errors"]:
        with st.expander("⚠️ Model loading notes", expanded=False):
            for err in models["errors"]:
                st.warning(err)

    dfs = load_datasets()

    if page == "🏠 Home":
        page_home()
    elif page == "🚀 Founder Prediction":
        page_founder(models)
    elif page == "💼 Investor Prediction":
        page_investor(models)
    elif page == "📊 Dashboard":
        page_dashboard(dfs)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
