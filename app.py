
# ============================================================
# Ma'al | مآل  —  Startup Evaluation Platform
# Streamlit MVP  |  Member 2 Deliverable
# Bilingual UI: Arabic / English
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ─── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Ma'al | مآل",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #112233 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #cdd9e5 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }

    .main .block-container {
        background-color: #f7f9fc;
        padding: 2rem 2.5rem 4rem;
    }

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

    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a3d62 60%, #0a7ea4 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero h3 { font-size: 1.2rem; font-weight: 300; opacity: 0.8; margin-bottom: 1.25rem; }
    .hero p  { font-size: 1rem; opacity: 0.75; max-width: 760px; line-height: 1.7; }

    .maal-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.88rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.84rem;
        color: #9a3412;
        margin-top: 1.5rem;
    }

    #MainMenu, footer, header { visibility: hidden; }

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

# ─── Bilingual UI ─────────────────────────────────────────────────────────────
def lang() -> str:
    return st.session_state.get("lang", "ar")

def is_ar() -> bool:
    return lang() == "ar"

def t(ar: str, en: str) -> str:
    return ar if is_ar() else en

def choice_label(value: str, mapping: dict) -> str:
    return mapping.get(value, value) if is_ar() else value

SECTOR_AR = {
    "Agritech": "التقنية الزراعية",
    "Cybersecurity": "الأمن السيبراني",
    "E-commerce": "التجارة الإلكترونية",
    "Edtech": "تقنيات التعليم",
    "Fintech": "التقنية المالية",
    "HR Tech": "تقنيات الموارد البشرية",
    "Healthtech": "التقنية الصحية",
    "Logistics Tech": "تقنيات الخدمات اللوجستية",
    "Real Estate Tech": "تقنيات العقار",
    "Traveltech": "تقنيات السفر",
}
CITY_AR = {
    "Abha": "أبها", "Dammam": "الدمام", "Jeddah": "جدة", "Khobar": "الخبر",
    "Mecca": "مكة", "Medina": "المدينة", "NEOM": "نيوم", "Riyadh": "الرياض"
}
FUNDING_STAGE_AR = {
    "Growth": "النمو", "Pre-Seed": "ما قبل البذرة", "Seed": "البذرة",
    "Series A": "الجولة A", "Series B": "الجولة B", "Series C": "الجولة C"
}

# ─── Constants ────────────────────────────────────────────────────────────────
SECTORS = list(SECTOR_AR.keys())
CITIES = list(CITY_AR.keys())
FUNDING_STAGES = list(FUNDING_STAGE_AR.keys())
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
    results = {
        "founder": None, "investor": None,
        "founder_pre": None, "investor_pre": None,
        "founder_cols": None, "investor_cols": None,
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
    dfs = {}
    for name, fname in [("founder", "founder_dataset.csv"), ("investor", "investor_dataset.csv")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                dfs[name] = pd.read_csv(path)
            except Exception:
                pass
    return dfs

def predict_score(model, preprocessor, input_df: pd.DataFrame) -> float:
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
            return t(
                "شركتك تظهر مؤشرات بقاء قوية. ركّز على توسيع العمليات، تقوية العلاقات مع المستثمرين، والدخول لأسواق جديدة. المشاركة في برامج تنظيمية مثل SAMA أو REGA أو NTDP قد تدعم نموك أكثر.",
                "Your startup shows strong survival indicators. Focus on scaling operations, deepening investor relationships, and expanding to new markets. Regulatory sandbox participation can accelerate growth."
            )
        elif score >= 55:
            return t(
                "شركتك لديها أساس جيد لكن تحتاج تحسينات. ركّز على زيادة التمويل، توسيع قاعدة المستثمرين، وتقليل وقت الوصول لأول جولة تمويلية. المشاركة التنظيمية قد ترفع النتيجة بشكل واضح.",
                "Your startup has a solid foundation with room to grow. Prioritise increasing total funding, broadening your investor base, and improving speed-to-market. Regulatory participation can meaningfully boost your score."
            )
        elif score >= 35:
            return t(
                "شركتك في مرحلة حساسة. ركّز على تأمين الجولة التمويلية القادمة، إثبات المنتج مع عملاء يدفعون، وتقليل وقت الوصول لأول جولة. برامج المسرعات قد تكون مفيدة.",
                "Your startup is at a critical stage. Focus on securing the next funding round, validating your product with paying customers, and reducing time-to-first-round. Accelerator programmes may help."
            )
        else:
            return t(
                "الشركة تحتاج تحسينات كبيرة قبل أن تكون في وضع بقاء جيد. أعد تقييم نموذج العمل، قوّ الفريق، ابحث عن دعم مبكر أو منح، واستهدف البرامج التنظيمية لتحسين الأساسيات.",
                "Significant improvements are needed before this startup is positioned for survival. Re-evaluate the business model, strengthen the founding team, seek early-stage grants or angels, and target regulatory support programmes."
            )
    else:
        if score >= 75:
            return t(
                "هذه الشركة تظهر جاذبية استثمارية عالية. مؤشرات النمو، الإيرادات، والـ traction قوية. يفضل الانتقال لمرحلة الفحص النافي للجهالة قبل أي قرار.",
                "This startup scores highly on investment attractiveness. Strong traction, revenue momentum, and market growth signal a compelling opportunity. Conduct due diligence before making a decision."
            )
        elif score >= 55:
            return t(
                "فرصة واعدة لكن فيها بعض المخاطر. راجع مؤشرات الـ traction والإيرادات بعناية. قد يكون الاستثمار المرحلي أو المشروط بالإنجازات خيارًا مناسبًا.",
                "A promising opportunity with some risks. Review traction metrics and monthly revenue trends carefully. A smaller initial stake with follow-on rights may be appropriate."
            )
        elif score >= 35:
            return t(
                "الجاذبية الاستثمارية متوسطة. الشركة تحتاج إثبات سوق أقوى أو نمو إيرادات أوضح. الأفضل التفكير في تمويل مشروط بمراحل أداء محددة.",
                "Moderate investment attractiveness. The startup shows potential but needs stronger market validation or revenue growth. Consider milestone-based tranches rather than a full equity commitment."
            )
        else:
            return t(
                "الشركة غير جاهزة استثماريًا بناءً على المدخلات الحالية. تحتاج الإيرادات، الـ traction، والزخم التمويلي إلى تحسن واضح قبل تبرير الاستثمار.",
                "This startup is not yet positioned for investment based on the provided metrics. Revenue, traction, and funding momentum need substantial improvement before an investment can be justified."
            )

# ─── Page: Home ───────────────────────────────────────────────────────────────
def page_home():
    hero_subtitle = t(
        "منصة ذكية لتقييم الشركات الناشئة",
        "AI-Powered Startup Evaluation Platform"
    )
    hero_desc = t(
        "مآل منصة ذكية تساعد مؤسسي الشركات التقنية السعودية على فهم فرص بقاء شركاتهم، وتساعد المستثمرين على تقييم جاذبية الفرص الاستثمارية باستخدام نماذج تعلّم آلي تعطي درجة واضحة من 100 مع توصيات عملية.",
        "Ma'al helps Saudi tech founders understand their startup's survival potential and helps investors evaluate attractive opportunities using machine learning models that return a clear score out of 100 with practical recommendations."
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>مآل &nbsp;|&nbsp; Ma'al</h1>
            <h3>{hero_subtitle}</h3>
            <p>{hero_desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("🚀", t("وضع المؤسس", "Founder Mode"), t("أدخل بيانات شركتك واحصل على درجة البقاء من 100 مع توصيات للتحسين.", "Enter your startup details and get a Survival Score out of 100 with actionable recommendations."), "maal-card-green", "#065f46"),
        ("💼", t("وضع المستثمر", "Investor Mode"), t("قيّم الشركة من ناحية الجاذبية الاستثمارية واحصل على درجة استثمارية من 100 مع قراءة للمخاطر.", "Evaluate startups on investment attractiveness and receive an Investment Score out of 100 with risk-adjusted insights."), "", "#0a3d62"),
        ("📊", t("لوحة البيانات", "Dashboard"), t("استكشف بيانات التدريب: القطاعات، المدن، مراحل التمويل، وتوزيع الشركات.", "Explore the training dataset: sectors, cities, funding stages, and startup distributions."), "maal-card-amber", "#92400e"),
    ]
    for col, (icon, title, desc, cls, color) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""
                <div class="maal-card {cls}">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-weight:700; font-size:1.05rem; color:{color};">{title}</div>
                    <div style="font-size:0.88rem; color:#374151; margin-top:0.4rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown(t("### كيف تعمل المنصة", "### How it works"))

    steps = [
        ("1", t("أدخل البيانات", "Enter Data"), t("عبّئ بيانات الشركة من النموذج.", "Fill in the startup details using the form.")),
        ("2", t("تنبؤ بالذكاء الاصطناعي", "AI Prediction"), t("النموذج يتوقع النتيجة خلال ثوانٍ.", "The machine learning model predicts your score in seconds.")),
        ("3", t("استفد من التوصيات", "Act on Insights"), t("اقرأ التصنيف، النتيجة، والتوصية المناسبة.", "Read your label, score, and personalised recommendation.")),
    ]
    for col, (num, title, desc) in zip(st.columns(3), steps):
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
        f"""
        <div class="info-box">
            <b>{t("ابدأ الآن:", "Get started:")}</b>
            {t("اختر وضع المؤسس أو وضع المستثمر من القائمة الجانبية لتشغيل أول تقييم.", "Select Founder Prediction or Investor Prediction from the sidebar to run your first evaluation.")}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Page: Founder Prediction ─────────────────────────────────────────────────
def page_founder(models: dict):
    st.markdown(
        f'<div class="section-title">🚀 {t("تقييم المؤسس", "Founder Prediction")}</div>'
        f'<div class="section-sub">{t("توقع درجة بقاء شركتك الناشئة", "Predict your startup survival score")}</div>',
        unsafe_allow_html=True,
    )

    if models["founder"] is None:
        st.error(t(
            "نموذج المؤسس غير متوفر. تأكد من وجود founder_model.joblib و founder_preprocessor.joblib في نفس مجلد app.py.",
            "Founder model is not available. Make sure founder_model.joblib and founder_preprocessor.joblib are in the app directory."
        ))
        return

    with st.form("founder_form"):
        st.markdown(t("#### ملف الشركة", "#### Startup Profile"))
        c1, c2 = st.columns(2)
        with c1:
            sector = st.selectbox(t("القطاع", "Sector"), SECTORS, index=0, format_func=lambda x: choice_label(x, SECTOR_AR))
            city = st.selectbox(t("المدينة", "City"), CITIES, index=0, format_func=lambda x: choice_label(x, CITY_AR))
            funding_stage = st.selectbox(t("مرحلة التمويل", "Funding Stage"), FUNDING_STAGES, index=1, format_func=lambda x: choice_label(x, FUNDING_STAGE_AR))
        with c2:
            startup_age = st.slider(t("عمر الشركة بالسنوات", "Startup Age (years)"), 2, 14, 3)
            num_founders = st.slider(t("عدد المؤسسين", "Number of Founders"), 1, 4, 2)
            num_investors = st.slider(t("عدد المستثمرين", "Number of Investors"), 0, 50, 5)

        st.markdown(t("#### تفاصيل التمويل", "#### Funding Details"))
        c3, c4 = st.columns(2)
        with c3:
            total_funding = st.number_input(t("إجمالي التمويل بالريال", "Total Funding (SAR)"), min_value=100, max_value=500_000, value=10_000, step=1_000)
            num_rounds = st.slider(t("عدد الجولات التمويلية", "Number of Funding Rounds"), 1, 8, 2)
        with c4:
            speed_to_first = st.slider(t("المدة لأول جولة تمويلية بالشهور", "Speed to First Round (months)"), 1, 54, 12)

        st.markdown(t("#### المشاركة التنظيمية", "#### Regulatory Participation"))
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sama_sandbox = st.checkbox(t("بيئة SAMA التجريبية", "SAMA Fintech Sandbox"))
        with rc2:
            rega_sandbox = st.checkbox(t("بيئة REGA العقارية", "REGA PropTech Sandbox"))
        with rc3:
            ntdp_participation = st.checkbox(t("برنامج NTDP", "NTDP Programme"))

        submitted = st.form_submit_button(t("توقع درجة البقاء ←", "Predict Survival Score →"))

    if submitted:
        funding_per_round = total_funding / max(num_rounds, 1)
        investment_per_founder = total_funding / max(num_founders, 1)
        regulatory_support = int(sama_sandbox) + int(rega_sandbox) + int(ntdp_participation)

        input_df = pd.DataFrame({
            "startup_age_years": [startup_age],
            "total_funding_sar": [total_funding],
            "num_funding_rounds": [num_rounds],
            "num_investors": [num_investors],
            "num_founders": [num_founders],
            "speed_to_first_round_months": [speed_to_first],
            "sama_sandbox": [int(sama_sandbox)],
            "rega_sandbox": [int(rega_sandbox)],
            "ntdp_participation": [int(ntdp_participation)],
            "funding_per_round": [funding_per_round],
            "investment_per_founder": [investment_per_founder],
            "regulatory_support": [regulatory_support],
            "sector": [sector],
            "city": [city],
            "funding_stage": [funding_stage],
        })

        with st.spinner(t("جاري تشغيل التوقع…", "Running prediction…")):
            try:
                score = predict_score(models["founder"], models["founder_pre"], input_df)
                st.session_state["last_prediction"] = {
                    "mode": "founder",
                    "score_name": t("درجة البقاء", "Survival Score"),
                    "score": round(score, 1),
                    "inputs": {
                        t("القطاع", "Sector"): choice_label(sector, SECTOR_AR),
                        t("المدينة", "City"): choice_label(city, CITY_AR),
                        t("مرحلة التمويل", "Funding Stage"): choice_label(funding_stage, FUNDING_STAGE_AR),
                        t("عمر الشركة بالسنوات", "Startup Age (years)"): startup_age,
                        t("عدد المؤسسين", "Number of Founders"): num_founders,
                        t("عدد المستثمرين", "Number of Investors"): num_investors,
                        t("إجمالي التمويل بالريال", "Total Funding (SAR)"): total_funding,
                        t("عدد الجولات التمويلية", "Number of Funding Rounds"): num_rounds,
                        t("المدة لأول جولة بالشهور", "Speed to First Round (months)"): speed_to_first,
                    },
                }
                _display_result(score, "founder")
            except Exception as e:
                st.error(f"{t('فشل التوقع:', 'Prediction failed:')} {e}")

# ─── Page: Investor Prediction ────────────────────────────────────────────────
def page_investor(models: dict):
    st.markdown(
        f'<div class="section-title">💼 {t("تقييم المستثمر", "Investor Prediction")}</div>'
        f'<div class="section-sub">{t("توقع درجة الجاذبية الاستثمارية", "Predict investment attractiveness score")}</div>',
        unsafe_allow_html=True,
    )

    if models["investor"] is None:
        st.error(t(
            "نموذج المستثمر غير متوفر. تأكد من وجود investor_model.joblib و investor_preprocessor.joblib في نفس مجلد app.py.",
            "Investor model is not available. Make sure investor_model.joblib and investor_preprocessor.joblib are in the app directory."
        ))
        return

    with st.form("investor_form"):
        st.markdown(t("#### ملف الشركة", "#### Startup Profile"))
        c1, c2 = st.columns(2)
        with c1:
            sector = st.selectbox(t("القطاع", "Sector"), SECTORS, index=0, format_func=lambda x: choice_label(x, SECTOR_AR))
            city = st.selectbox(t("المدينة", "City"), CITIES, index=0, format_func=lambda x: choice_label(x, CITY_AR))
            funding_stage = st.selectbox(t("مرحلة التمويل", "Funding Stage"), FUNDING_STAGES, index=1, format_func=lambda x: choice_label(x, FUNDING_STAGE_AR))
        with c2:
            startup_age = st.slider(t("عمر الشركة بالسنوات", "Startup Age (years)"), 2, 14, 3)
            num_founders = st.slider(t("عدد المؤسسين", "Number of Founders"), 1, 4, 2)
            num_investors = st.slider(t("عدد المستثمرين", "Number of Investors"), 1, 50, 5)

        st.markdown(t("#### السوق والانتشار", "#### Market & Traction"))
        c3, c4 = st.columns(2)
        with c3:
            market_growth_index = st.slider(t("مؤشر نمو السوق من 0 إلى 1", "Market Growth Index (0–1)"), 0.48, 0.90, 0.78, step=0.01)
            traction_score = st.slider(t("درجة التفاعل / الانتشار من 100", "Traction Score (0–100)"), 2, 100, 50)
        with c4:
            monthly_revenue = st.number_input(t("الإيراد الشهري بالريال", "Monthly Revenue (SAR)"), min_value=3, max_value=80_000_000, value=50_000, step=5_000)

        st.markdown(t("#### تفاصيل التمويل", "#### Funding Details"))
        c5, c6 = st.columns(2)
        with c5:
            total_funding = st.number_input(t("إجمالي التمويل بالريال", "Total Funding (SAR)"), min_value=100, max_value=500_000, value=10_000, step=1_000)
            num_rounds = st.slider(t("عدد الجولات التمويلية", "Number of Funding Rounds"), 1, 8, 2)
        with c6:
            speed_to_first = st.slider(t("المدة لأول جولة تمويلية بالشهور", "Speed to First Round (months)"), 1, 54, 12)

        st.markdown(t("#### المشاركة التنظيمية", "#### Regulatory Participation"))
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sama_sandbox = st.checkbox(t("بيئة SAMA التجريبية", "SAMA Fintech Sandbox"))
        with rc2:
            rega_sandbox = st.checkbox(t("بيئة REGA العقارية", "REGA PropTech Sandbox"))
        with rc3:
            ntdp_participation = st.checkbox(t("برنامج NTDP", "NTDP Programme"))

        submitted = st.form_submit_button(t("توقع درجة الاستثمار ←", "Predict Investment Score →"))

    if submitted:
        funding_per_round = total_funding / max(num_rounds, 1)
        investment_per_founder = total_funding / max(num_founders, 1)
        regulatory_support = int(sama_sandbox) + int(rega_sandbox) + int(ntdp_participation)
        revenue_funding_ratio = monthly_revenue / max(total_funding, 1)

        input_df = pd.DataFrame({
            "startup_age_years": [startup_age],
            "total_funding_sar": [total_funding],
            "num_funding_rounds": [num_rounds],
            "num_investors": [num_investors],
            "num_founders": [num_founders],
            "speed_to_first_round_months": [speed_to_first],
            "market_growth_index": [market_growth_index],
            "traction_score": [traction_score],
            "monthly_revenue_sar": [monthly_revenue],
            "sama_sandbox": [int(sama_sandbox)],
            "rega_sandbox": [int(rega_sandbox)],
            "ntdp_participation": [int(ntdp_participation)],
            "funding_per_round": [funding_per_round],
            "investment_per_founder": [investment_per_founder],
            "regulatory_support": [regulatory_support],
            "revenue_funding_ratio": [revenue_funding_ratio],
            "sector": [sector],
            "city": [city],
            "funding_stage": [funding_stage],
        })

        with st.spinner(t("جاري تشغيل التوقع…", "Running prediction…")):
            try:
                score = predict_score(models["investor"], models["investor_pre"], input_df)
                st.session_state["last_prediction"] = {
                    "mode": "investor",
                    "score_name": t("درجة الاستثمار", "Investment Score"),
                    "score": round(score, 1),
                    "inputs": {
                        t("القطاع", "Sector"): choice_label(sector, SECTOR_AR),
                        t("المدينة", "City"): choice_label(city, CITY_AR),
                        t("مرحلة التمويل", "Funding Stage"): choice_label(funding_stage, FUNDING_STAGE_AR),
                        t("عمر الشركة بالسنوات", "Startup Age (years)"): startup_age,
                        t("عدد المؤسسين", "Number of Founders"): num_founders,
                        t("عدد المستثمرين", "Number of Investors"): num_investors,
                        t("مؤشر نمو السوق", "Market Growth Index"): market_growth_index,
                        t("درجة التفاعل", "Traction Score"): traction_score,
                        t("الإيراد الشهري بالريال", "Monthly Revenue (SAR)"): monthly_revenue,
                        t("إجمالي التمويل بالريال", "Total Funding (SAR)"): total_funding,
                        t("عدد الجولات التمويلية", "Number of Funding Rounds"): num_rounds,
                    },
                }
                _display_result(score, "investor")
            except Exception as e:
                st.error(f"{t('فشل التوقع:', 'Prediction failed:')} {e}")

# ─── Result display (shared) ──────────────────────────────────────────────────
def _display_result(score: float, mode: str):
    label, pill_class = score_label(score)
    label_show = {"High": t("مرتفع", "High"), "Medium": t("متوسط", "Medium"), "Low": t("منخفض", "Low")}[label]
    title = t("درجة البقاء", "Survival Score") if mode == "founder" else t("درجة الاستثمار", "Investment Score")

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown(f"### {title}")

    st.markdown(
        f"""
        <div class="maal-card" style="display:flex; align-items:center; gap:1rem;">
            <span class="score-badge">{score:.1f}</span>
            <span class="label-pill {pill_class}">{label_show}</span>
            <span style="color:#64748b; font-size:0.9rem; margin-left:0.5rem;">/ 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(score))

    bands = [
        ("maal-card-red", "#991b1b", t("منخفض", "Low"), "0 – 34"),
        ("maal-card-amber", "#92400e", t("متوسط", "Medium"), "35 – 64"),
        ("maal-card-green", "#065f46", t("مرتفع", "High"), "65 – 100"),
    ]
    for col, (cls, color, name, rng) in zip(st.columns(3), bands):
        with col:
            st.markdown(
                f'<div class="maal-card {cls}" style="text-align:center;"><b style="color:{color}">{name}</b><br><span style="font-size:0.8rem; color:#6b7280;">{rng}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(t("#### التوصية", "#### Recommendation"))
    st.markdown(
        f'<div class="maal-card"><span style="font-size:0.95rem; color:#1e293b; line-height:1.7;">{recommendation_text(score, mode)}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="disclaimer">
            ⚠️ <b>{t("إخلاء المسؤولية:", "Disclaimer:")}</b>
            {t("هذه النتيجة تقدير مولّد بالذكاء الاصطناعي بناءً على أنماط إحصائية في بيانات شركات ناشئة تجريبية. ليست نصيحة مالية أو استثمارية.", "This score is an AI-generated estimate based on statistical patterns in experimental startup data. It is not financial or investment advice.")}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Page: Dashboard ──────────────────────────────────────────────────────────
def _donut_chart(df: pd.DataFrame, label_col: str, title: str, top_n: int = 7):
    counts = df[label_col].value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = pd.Series({t("أخرى", "Other"): counts.iloc[top_n:].sum()})
        counts = pd.concat([top, other])
    data = counts.reset_index()
    data.columns = ["label", "count"]

    if label_col == "sector":
        data["display_label"] = data["label"].apply(lambda x: choice_label(x, SECTOR_AR))
    elif label_col == "city":
        data["display_label"] = data["label"].apply(lambda x: choice_label(x, CITY_AR))
    elif label_col == "funding_stage":
        data["display_label"] = data["label"].apply(lambda x: choice_label(x, FUNDING_STAGE_AR))
    else:
        data["display_label"] = data["label"]

    data["display_label"] = data["display_label"].replace({"Other": t("أخرى", "Other"), "أخرى": t("أخرى", "Other")})

    total = data["count"].sum()
    data["pct"] = (data["count"] / total * 100).round(1)

    pie_colors = [
        "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed",
        "#0891b2", "#db2777", "#65a30d", "#ea580c", "#475569"
    ]

    chart = (
        alt.Chart(data)
        .mark_arc(
            innerRadius=0,
            outerRadius=95,
            stroke="#ffffff",
            strokeWidth=2
        )
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color(
                "display_label:N",
                title=None,
                scale=alt.Scale(range=pie_colors),
                legend=alt.Legend(
    orient="right",
    direction="vertical",
    columns=1,
    labelLimit=300,
    labelFontSize=13,
    symbolSize=140,
    symbolType="circle",
    title=None,
    offset=20
)
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title=title),
                alt.Tooltip("count:Q", title=t("العدد", "Count")),
                alt.Tooltip("pct:Q", title=t("النسبة %", "Share %")),
            ],
        )
        .properties(width=500,
            height=320,
            title=title)
        .configure_title(
            anchor="middle",
            fontSize=16,
            fontWeight="bold",
            offset=18
        )
        .configure_view(strokeWidth=0)
        .configure_legend(
            labelFontSize=13,
            labelColor="#64748b",
            padding=12
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

def page_dashboard(dfs: dict):
    st.markdown(
        f'<div class="section-title">📊 {t("لوحة البيانات", "Data Dashboard")}</div>'
        f'<div class="section-sub">{t("استكشف بيانات التدريب", "Explore the training dataset")}</div>',
        unsafe_allow_html=True,
    )

    if not dfs:
        st.warning(t(
            "لم يتم العثور على ملفات البيانات. ضع founder_dataset.csv و investor_dataset.csv في مجلد التطبيق.",
            "No dataset files found. Place founder_dataset.csv and investor_dataset.csv in the app directory."
        ))
        return

    tab_names = [t("المؤسس", "Founder") if k == "founder" else t("المستثمر", "Investor") for k in dfs.keys()]
    tabs = st.tabs(tab_names)

    for tab, (name, df) in zip(tabs, dfs.items()):
        with tab:
            target = "survival_score" if name == "founder" else "investment_score"
            m1, m2 = st.columns(2)
            m1.metric(t("إجمالي الشركات", "Total Startups"), f"{len(df):,}")
            if target in df.columns:
                m2.metric(t("متوسط النتيجة", f"Average {target.replace('_',' ').title()}"), f"{df[target].mean():.1f} / 100")

            st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            with col_left:
                if "sector" in df.columns:
                    _donut_chart(df, "sector", t("الشركات حسب القطاع", "Startups by Sector"))
            with col_right:
                if "funding_stage" in df.columns:
                    _donut_chart(df, "funding_stage", t("الشركات حسب مرحلة التمويل", "Startups by Funding Stage"))
            if "city" in df.columns:
                _donut_chart(df, "city", t("الشركات حسب المدينة", "Startups by City"))

            with st.expander(t("👁️ عرض أول 10 صفوف من البيانات", "👁️ Preview the dataset (first 10 rows)")):
                st.dataframe(df.head(10), use_container_width=True)

# ─── Page: About ──────────────────────────────────────────────────────────────
def page_about():
    st.markdown(
        f'<div class="section-title">ℹ️ {t("عن مآل", "About Ma\'al")}</div>'
        f'<div class="section-sub">{t("كيف تعمل المنصة", "How the platform works")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="maal-card">
            <b style="font-size:1.05rem;">{t("ما هي مآل؟", "What is Ma'al?")}</b>
            <p style="margin-top:0.6rem; color:#374151; line-height:1.8;">
                {t(
                    "مآل منصة ذكية لتقييم الشركات الناشئة من زاويتين: زاوية المؤسس لفهم فرصة البقاء، وزاوية المستثمر لفهم الجاذبية الاستثمارية. يدخل المستخدم بيانات الشركة، ثم يعطي النموذج درجة واضحة من 100 مع توصية مبسطة.",
                    "Ma'al is an AI platform that evaluates startups from two angles: founder survival potential and investor attractiveness. The user enters startup details, then the model returns a clear score out of 100 with a simple recommendation."
                )}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(t("### 🧩 أقسام المنصة", "### 🧩 Platform Parts"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="maal-card maal-card-green" style="height:100%;">
                <div style="font-size:1.8rem;">🚀</div>
                <b style="font-size:1.05rem; color:#065f46;">{t("وضع المؤسس", "Founder Mode")}</b>
                <p style="color:#374151; font-size:0.9rem; line-height:1.8;">
                    {t("للمؤسسين الذين يريدون معرفة قوة شركاتهم الناشئة. يعطي درجة بقاء وتوصيات عملية للتحسين.", "For founders who want to understand how strong their startup is. It returns a Survival Score and practical improvement tips.")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="maal-card" style="height:100%; border-left-color:#0a7ea4;">
                <div style="font-size:1.8rem;">💼</div>
                <b style="font-size:1.05rem; color:#0a3d62;">{t("وضع المستثمر", "Investor Mode")}</b>
                <p style="color:#374151; font-size:0.9rem; line-height:1.8;">
                    {t("للمستثمرين الذين يريدون مقارنة الفرص بناءً على السوق، الإيرادات، والانتشار.", "For investors who want to compare opportunities based on market, revenue, and traction.")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="info-box" style="margin-top:1rem;">
            💬 <b>{t("المساعد الذكي", "AI Assistant")}</b>
            {t("يمكنه شرح النتائج والإجابة عن أسئلة ريادة الأعمال وطريقة استخدام المنصة.", "can explain results and answer questions about startups and platform usage.")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='maal-divider'>", unsafe_allow_html=True)
    st.markdown(t("### الفريق", "### Team"))
    team = [
        ("محمد", "AI Developer", "Model training & evaluation"),
        ("عبدالعزيز", "Data Analyst", "Data & feature engineering"),
        ("ماجد", "AI App Developer", "Assistant & service layer"),
        ("أنس", "Frontend / Product", "Streamlit app & UI/UX"),
    ]
    for col, (name, role, desc) in zip(st.columns(4), team):
        with col:
            st.markdown(
                f"""
                <div class="maal-card" style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:700; color:#0a3d62;">{name}</div>
                    <div style="font-size:0.78rem; font-weight:600; color:#0a7ea4; margin:0.3rem 0;">{role}</div>
                    <div style="font-size:0.76rem; color:#64748b;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="disclaimer" style="margin-top:1.5rem;">
            <b>{t("مشروع تدريبي — معسكر الذكاء الاصطناعي 2026.", "Capstone Project — AI Bootcamp 2026.")}</b>
            {t("كل النتائج تقديرية ولأغراض تعليمية فقط. ليست نصيحة مالية أو استثمارية.", "All scores are AI-generated estimates for educational purposes only. Not financial or investment advice.")}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Page: Assistant (AI Chatbot) ─────────────────────────────────────────────
def page_chatbot():
    st.markdown(
        f'<div class="section-title">💬 {t("مساعد مآل", "Ma\'al Assistant")}</div>'
        f'<div class="section-sub">{t("اسأل عن شركتك الناشئة أو عن المنصة", "Ask about your startup or the platform")}</div>',
        unsafe_allow_html=True,
    )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        st.error(t("مكتبة google-genai غير مثبّتة. ثبّتها بالأمر: pip install google-genai", "google-genai is not installed. Install it with: pip install google-genai"))
        return

    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        st.markdown(
            f"""
            <div class="info-box">
                🔑 <b>{t("مطلوب مفتاح Gemini مجاني.", "A free Gemini API key is required.")}</b>
                {t("أضفه في ملف .streamlit/secrets.toml ثم أعد تشغيل التطبيق.", "Add it to .streamlit/secrets.toml, then restart the app.")}
                <br><code>GEMINI_API_KEY = "AIza..."</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    client = genai.Client(api_key=api_key)

    SYSTEM_PROMPT = (
        "أنت مساعد مآل. أجب بنفس لغة المستخدم. المنصة تقيّم الشركات الناشئة بدرجات تقديرية وليست نصيحة مالية أو استثمارية."
        if is_ar() else
        "You are Ma'al Assistant. Answer in the user's language. The platform evaluates startups using estimated scores and does not provide financial or investment advice."
    )

    last = st.session_state.get("last_prediction")
    if last:
        inputs_text = "\n".join(f"- {k}: {v}" for k, v in last["inputs"].items())
        SYSTEM_PROMPT += (
            f"\n\nLast evaluation context:\nMode: {last['mode']}\nInputs:\n{inputs_text}\n"
            f"Model score: {last['score_name']} = {last['score']}/100.\n"
            "Do not recompute or change this score. Only explain it and give practical advice."
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if last:
        st.markdown(
            f"""
            <div class="info-box">
                📊 <b>{t("مربوط بآخر تقييم:", "Connected to last evaluation:")}</b>
                {last['score_name']} = <b>{last['score']}/100</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="info-box">
                💡 {t("شغّل تقييمًا من صفحة المؤسس أو المستثمر أولًا، أو اسأل الآن مباشرة.", "Run an evaluation from Founder or Investor first, or ask directly now.")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(t("⚙️ إعدادات المحادثة", "⚙️ Chat Settings"), expanded=False):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            model_choice = st.selectbox(t("النموذج", "Model"), ["gemini-2.5-flash", "gemini-2.5-flash-lite"], index=0)
        with col_b:
            st.write("")
            st.write("")
            if st.button(t("🗑️ محادثة جديدة", "🗑️ New Chat")):
                st.session_state.chat_messages = []
                st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(t("اكتب سؤالك هنا…", "Type your question here…")):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in st.session_state.chat_messages
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_reply = ""
            try:
                stream = client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=1024),
                )
                for chunk in stream:
                    if chunk.text:
                        full_reply += chunk.text
                        placeholder.markdown(full_reply + "▌")
                placeholder.markdown(full_reply)
            except Exception as e:
                full_reply = f"{t('⚠️ صار خطأ أثناء الاتصال بالـ API:', '⚠️ API connection error:')} {e}"
                placeholder.error(full_reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})

# ─── Sidebar navigation ───────────────────────────────────────────────────────
def sidebar_nav() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ar"

    st.sidebar.markdown(
        '<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.4rem;">Language · اللغة</div>',
        unsafe_allow_html=True,
    )
    col_ar, col_en = st.sidebar.columns(2)
    with col_ar:
        if st.button("العربية", key="btn_ar"):
            st.session_state["lang"] = "ar"
            st.rerun()
    with col_en:
        if st.button("English", key="btn_en"):
            st.session_state["lang"] = "en"
            st.rerun()

    st.sidebar.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div style="font-size:2.2rem; font-weight:800; color:#ffffff; letter-spacing:-1px;">مآل</div>
            <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.1em;">
                {t("منصة ذكية لتقييم الشركات الناشئة", "AI STARTUP EVALUATOR")}
            </div>
        </div>
        <hr style="border-color:#1e3a5f; margin-bottom:1rem;">
        """,
        unsafe_allow_html=True,
    )

    page_options = [
        ("home", f"🏠 {t('الرئيسية', 'Home')}"),
        ("founder", f"🚀 {t('تقييم المؤسس', 'Founder Prediction')}"),
        ("investor", f"💼 {t('تقييم المستثمر', 'Investor Prediction')}"),
        ("dashboard", f"📊 {t('لوحة البيانات', 'Dashboard')}"),
        ("assistant", f"💬 {t('المساعد', 'Assistant')}"),
        ("about", f"ℹ️ {t('عن مآل', 'About')}"),
    ]

    selected_label = st.sidebar.radio(
        t("التنقل", "Navigate"),
        options=[label for _, label in page_options],
        label_visibility="collapsed",
    )
    selected_key = dict((label, key) for key, label in page_options)[selected_label]

    st.sidebar.markdown(
        f"""
        <hr style="border-color:#1e3a5f; margin-top:1rem;">
        <div style="font-size:0.72rem; color:#64748b; padding:0 0.5rem; line-height:1.6;">
            ⚠️ {t("النتائج تقديرية بالذكاء الاصطناعي وليست نصيحة مالية أو استثمارية.", "Scores are AI-generated estimates. Not financial or investment advice.")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_key

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    page = sidebar_nav()

    with st.spinner(t("جاري تحميل النماذج…", "Loading models…")):
        models = load_models()

    if models["errors"]:
        with st.expander(t("⚠️ ملاحظات تحميل النماذج", "⚠️ Model loading notes"), expanded=False):
            for err in models["errors"]:
                st.warning(err)

    dfs = load_datasets()

    if page == "home":
        page_home()
    elif page == "founder":
        page_founder(models)
    elif page == "investor":
        page_investor(models)
    elif page == "dashboard":
        page_dashboard(dfs)
    elif page == "assistant":
        page_chatbot()
    elif page == "about":
        page_about()

if __name__ == "__main__":
    main()
