# Ma'al (مآل) — AI Startup Evaluation Platform

> A bilingual (Arabic/English) Streamlit platform that evaluates Saudi startups, giving **founders** a Survival Score and **investors** an Investment Score — both out of 100 — powered by machine learning and an integrated AI advisor chatbot.
>
> Capstone project — Foundational AI Bootcamp 2026 (المعسكر التأسيسي للذكاء الاصطناعي), Tuwaiq Academy.

---

| Item | What I used here | What changed across sessions |
|---|---|---|
| Final ML algorithm | **Random Forest Regressor** | You confirmed Random Forest as final in our most recent session, after earlier sessions flagged Linear Regression as the better-performing model in the notebooks. Worth having an answer ready if an evaluator asks about this discrepancy. |
| Chatbot provider | **Groq** | Built first with OpenAI (GPT-4o-mini), later corrected to Gemini (`google-genai`), then referenced as Groq in our most recent session. Confirm which one is in `app.py` right now and fix the env variable name accordingly. |
| File/folder names | `MAAL_MVP/`, `app.py`, `founder_model.joblib`, etc. | Based on what appeared in our chats — your actual repo may use slightly different names or a `models/` subfolder. |
| Model metrics | R²=0.845 (founder), R²=0.825 (investor) | Pulled from a notebook review session — re-verify against your final saved `.joblib` models, since notebooks were re-run more than once. |

Swap in the correct values and this README is ready to ship.

---

## 📌 Overview

**Ma'al** helps two types of users make faster, data-backed decisions about Saudi startups:

- **Founders** submit company details (sector, funding, team, traction, regulatory participation) and receive a **Survival Score (0–100)** with a verdict.
- **Investors** submit the same kind of company details and receive an **Investment Score (0–100)**, a classification, and a written recommendation.
- Both flows feed into an **AI Advisor** chat assistant that already knows the submitted data and the predicted score, so the user can ask follow-up questions without re-typing anything.

The platform is bilingual (Arabic UI primary, English supported) and built as a 4-person team capstone.

## 👥 Team

- Anas Alzahrani
- Abdulaziz Alsharif
- Mohammed Albassam
- Majed Alodhailah

## ✨ Features

- 🏠 **Home** — landing page, entry point to both flows
- 🚀 **Founder Prediction** — input form → Random Forest model → Survival Score
- 💼 **Investor Prediction** — input form → Random Forest model → Investment Score + classification + recommendation
- 🤖 **AI Advisor** — chat assistant grounded in the user's last submitted prediction (no need to repeat inputs)
- 📊 **Dashboard** — charts/overview of the underlying data
- ℹ️ **About** — team and project info

## 🧠 Machine Learning

| | Founder model | Investor model |
|---|---|---|
| Target | `survival_score` (0–100) | `investment_score` (0–100) |
| Algorithm | Random Forest Regressor | Random Forest Regressor |
| Training data | `founder_dataset.csv` — 10,000 synthetic rows | `investor_dataset.csv` — 5,000 synthetic rows |
| Key features | sector, city, funding stage, team size, traction, SAMA/REGA sandbox participation, NTDP participation, engineered ratios (`funding_per_round`, `investment_per_founder`, `revenue_funding_ratio`, `regulatory_support`) | similar feature set, investor-oriented framing |

> **Limitation to disclose:** both target scores are **synthetic**, generated from hand-crafted weighted formulas rather than real startup outcome data. The models learn to approximate that formula, not a real-world survival/investment pattern. This is worth stating explicitly in any report or demo, rather than letting it look like real outcome data.

## 🛠️ Tech Stack

- **Frontend/App:** Streamlit
- **ML:** scikit-learn (Random Forest), pandas, numpy, joblib
- **Chatbot:** Groq API *(confirm — see warning table above)*
- **Deployment:** Streamlit Community Cloud and Hugging Face Spaces

## 📁 Project Structure

```
MAAL_MVP/
├── app.py                          # Main Streamlit app (routing + all pages)
├── requirements.txt                # Pinned dependencies
├── founder_model.joblib            # Trained Random Forest — founder
├── founder_preprocessor.joblib     # Preprocessing pipeline — founder
├── investor_model.joblib           # Trained Random Forest — investor
├── investor_preprocessor.joblib    # Preprocessing pipeline — investor
├── founder_dataset.csv             # Synthetic training data (10,000 rows)
├── investor_dataset.csv            # Synthetic training data (5,000 rows)
├── notebooks/                      # Training/EDA notebooks
└── .streamlit/
    └── secrets.toml                # API key (local only — never commit this)
```

## 🚀 Quick Start

```bash
cd MAAL_MVP
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For the full step-by-step setup (including the API key file and common errors), see **`RUN_MANUAL.md`** (Arabic).

## 📦 Requirements (key versions)

```
streamlit
pandas
numpy
scikit-learn==1.6.1     # pinned — newer versions break the saved .joblib preprocessors
joblib
groq                    # or google-genai / openai, depending on the chatbot provider in use
```

## 🌐 Live Deployments

- Streamlit Community Cloud: *(add link)*
- Hugging Face Spaces: *(add link)*

## 📄 License / Academic Note

Built as a capstone project for the Tuwaiq Academy Foundational AI Bootcamp (2026). Not intended for production investment decisions — all scoring is based on synthetic training data.
