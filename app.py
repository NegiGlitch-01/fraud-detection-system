import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Fraud Detection | Karan",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* ---------- App background ---------- */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 229, 255, 0.08), transparent 25%),
        radial-gradient(circle at 90% 15%, rgba(255, 0, 70, 0.10), transparent 28%),
        linear-gradient(135deg, #05070b 0%, #0b0d12 48%, #250006 100%);
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------- Main spacing ---------- */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Hero ---------- */
.hero {
    padding: 34px 28px;
    border: 1px solid rgba(0, 229, 255, 0.22);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(0, 229, 255, 0.10), rgba(255, 0, 70, 0.08)),
        rgba(10, 12, 18, 0.88);
    box-shadow: 0 20px 70px rgba(0,0,0,0.35);
    text-align: center;
    margin-bottom: 22px;
}

.hero-title {
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 900;
    letter-spacing: 1px;
    margin: 0;
    background: linear-gradient(90deg, #00e5ff, #ffffff, #ff4d67);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-top: 10px;
}

.online {
    display: inline-block;
    margin-top: 16px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(0, 220, 130, 0.10);
    border: 1px solid rgba(0, 220, 130, 0.35);
    color: #55f0a8;
    font-weight: 700;
    font-size: 0.88rem;
}

/* ---------- Cards ---------- */
.section-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(16, 19, 27, 0.82);
    box-shadow: 0 12px 40px rgba(0,0,0,0.22);
    margin-bottom: 18px;
}

.section-title {
    color: #00e5ff;
    font-size: 1.25rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-caption {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-bottom: 16px;
}

/* ---------- Labels ---------- */
label {
    color: #d8faff !important;
    font-weight: 700 !important;
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #171a23 !important;
    border-radius: 10px !important;
    border-color: rgba(0,229,255,0.12) !important;
}

input {
    color: #f8fafc !important;
}

/* ---------- Analyze button ---------- */
.stButton > button {
    width: 100%;
    height: 58px;
    border: 0;
    border-radius: 14px;
    background: linear-gradient(90deg, #00a878, #00c853);
    color: white;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 0.3px;
    box-shadow: 0 10px 30px rgba(0, 200, 83, 0.18);
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px rgba(0, 200, 83, 0.28);
}

/* ---------- Result cards ---------- */
.result-safe {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(0, 180, 110, 0.18), rgba(0, 90, 60, 0.20));
    border: 1px solid rgba(0, 255, 160, 0.28);
    text-align: center;
}

.result-fraud {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255, 40, 70, 0.20), rgba(120, 0, 20, 0.22));
    border: 1px solid rgba(255, 70, 90, 0.35);
    text-align: center;
}

.result-title {
    font-size: 1.7rem;
    font-weight: 900;
    margin-bottom: 7px;
}

.result-text {
    color: #cbd5e1;
    font-size: 0.95rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090b10, #12070a);
    border-right: 1px solid rgba(0,229,255,0.10);
}

.sidebar-brand {
    text-align: center;
    padding: 10px 0 20px;
}

.sidebar-brand h2 {
    color: #00e5ff;
    margin: 0;
}

.sidebar-brand p {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #64748b;
    padding: 26px 0 8px;
    font-size: 0.88rem;
}

.footer b {
    color: #00e5ff;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
MODEL_PATH = Path("fraud_model.pkl")

if not MODEL_PATH.exists():
    st.error(
        "❌ fraud_model.pkl was not found. "
        "Place fraud_model.pkl in the same folder as app.py."
    )
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Model could not be loaded: {e}")
    st.stop()


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">💳 AI FRAUD DETECTION SYSTEM</div>
    <div class="hero-subtitle">
        Intelligent transaction screening powered by Machine Learning
    </div>
    <div class="online">● AI ENGINE ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🛡️ AI Guard</h2>
        <p>Fraud Detection Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ System Status")
    st.success("Model Loaded")
    st.info("Decision Tree Classifier")

    st.markdown("---")
    st.markdown("### 📌 Model Features")
    st.caption(
        "The current pickle was trained with the following six features:"
    )

    for feature in [
        "type",
        "amount",
        "isflaggedfraud",
        "actual_amount_orig",
        "actual_amount_dest",
        "transactionpath",
    ]:
        st.write(f"• `{feature}`")

    st.markdown("---")
    st.caption("Python • Pandas • Scikit-learn • Streamlit")


# ============================================================
# IMPORTANT MODEL NOTE
# ============================================================
st.warning(
    "⚠️ **Model compatibility note:** the current `fraud_model.pkl` was trained "
    "using encoded account IDs for `transactionpath`. The original notebook "
    "did not save the LabelEncoder mappings, so the app cannot reconstruct "
    "a real account-path encoding from new account IDs. Use the Transaction "
    "Path field for controlled testing until the encoder is saved with a "
    "retrained model."
)


# ============================================================
# TRANSACTION INPUTS
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">💸 Transaction Analysis</div>
    <div class="section-caption">
        Enter the transaction details below and let the trained model evaluate it.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">👤 Sender Details</div>
        <div class="section-caption">Information about the transaction sender.</div>
    """, unsafe_allow_html=True)

    transaction_type = st.selectbox(
        "🔄 Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        index=3,
    )

    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
        format="%.2f",
    )

    oldbalance_org = st.number_input(
        "🏦 Old Balance — Sender",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f",
    )

    newbalance_orig = st.number_input(
        "🏦 New Balance — Sender",
        min_value=0.0,
        value=900.0,
        step=100.0,
        format="%.2f",
    )

    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🏦 Receiver Details</div>
        <div class="section-caption">Information about the receiving account.</div>
    """, unsafe_allow_html=True)

    oldbalance_dest = st.number_input(
        "🏦 Old Balance — Receiver",
        min_value=0.0,
        value=500.0,
        step=100.0,
        format="%.2f",
    )

    newbalance_dest = st.number_input(
        "🏦 New Balance — Receiver",
        min_value=0.0,
        value=600.0,
        step=100.0,
        format="%.2f",
    )

    transaction_path = st.number_input(
        "🔢 Transaction Path (Encoded)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.0f",
        help=(
            "Encoded transactionpath used by the current trained model. "
            "The original notebook did not save the account LabelEncoder mappings."
        ),
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# LIVE TRANSACTION CHECK
# ============================================================
actual_amount_orig = oldbalance_org - newbalance_orig
actual_amount_dest = oldbalance_dest - newbalance_dest

st.markdown("### 📊 Derived Transaction Values")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Transaction Amount", f"₹ {amount:,.2f}")

with m2:
    st.metric("Actual Sender Change", f"₹ {actual_amount_orig:,.2f}")

with m3:
    st.metric("Actual Receiver Change", f"₹ {actual_amount_dest:,.2f}")


# ============================================================
# ADVANCED TEST CONTROL
# ============================================================
with st.expander("🧪 Advanced Model Input"):
    st.caption(
        "The current model was trained with `isflaggedfraud` as an input. "
        "This value is therefore exposed for compatibility with the existing pickle."
    )

    is_flagged = st.selectbox(
        "🚩 Is Flagged Fraud?",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
    )


# ============================================================
# PREDICTION
# ============================================================
st.markdown("---")

if st.button("🔍 ANALYZE TRANSACTION", use_container_width=True):

    # Basic validation
    validation_errors = []

    if amount < 0:
        validation_errors.append("Transaction amount cannot be negative.")

    if oldbalance_org < 0 or newbalance_orig < 0:
        validation_errors.append("Sender balances cannot be negative.")

    if oldbalance_dest < 0 or newbalance_dest < 0:
        validation_errors.append("Receiver balances cannot be negative.")

    # For the common payment/transfer/cash-out test case, sender should
    # normally have enough balance for the transaction.
    if transaction_type in ["PAYMENT", "TRANSFER", "CASH_OUT"]:
        if amount > oldbalance_org:
            validation_errors.append(
                "Transaction amount is greater than the sender's old balance."
            )

    if validation_errors:
        for error in validation_errors:
            st.warning(f"⚠️ {error}")
        st.stop()

    # LabelEncoder's alphabetical ordering used by the notebook:
    # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    type_encoding = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4,
    }

    encoded_type = type_encoding[transaction_type]

    # EXACT FEATURE NAMES USED BY THE CURRENT MODEL
    features = pd.DataFrame([{
        "type": encoded_type,
        "amount": float(amount),
        "isflaggedfraud": int(is_flagged),
        "actual_amount_orig": float(actual_amount_orig),
        "actual_amount_dest": float(actual_amount_dest),
        "transactionpath": float(transaction_path),
    }])

    try:
        prediction = int(model.predict(features)[0])

        # Model probability is shown only when the loaded classifier supports it.
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probability = float(np.max(model.predict_proba(features)[0]) * 100)
            except Exception:
                probability = None

        st.markdown("## 🤖 AI Analysis")

        if prediction == 1:
            st.markdown("""
            <div class="result-fraud">
                <div class="result-title">🚨 FRAUD TRANSACTION DETECTED</div>
                <div class="result-text">
                    The trained model classified this transaction as fraudulent.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <div class="result-title">✅ LEGITIMATE TRANSACTION</div>
                <div class="result-text">
                    The trained model classified this transaction as legitimate.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if probability is not None:
            st.metric("Model Class Probability", f"{probability:.2f}%")

        # Show exactly what was sent to the model.
        with st.expander("🔎 View Exact Model Input"):
            st.dataframe(features, use_container_width=True)

        # Prediction value
        with st.expander("🧠 Technical Prediction"):
            st.write(f"Prediction returned by model: `{prediction}`")
            st.write(
                "0 = Legitimate | 1 = Fraud"
            )

    except Exception as e:
        st.error(
            "❌ Prediction failed. The input structure does not match the "
            f"loaded model.\n\nDetails: {e}"
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🛡️ <b>AI Fraud Detection System</b><br>
    Machine Learning Project • Built with Python & Streamlit<br><br>
    Developed by <b>Karan</b>
</div>
""", unsafe_allow_html=True)
