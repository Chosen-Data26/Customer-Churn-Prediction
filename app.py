import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .risk-low    { color: #4ade80; font-size: 2rem; font-weight: 700; }
    .risk-medium { color: #facc15; font-size: 2rem; font-weight: 700; }
    .risk-high   { color: #f87171; font-size: 2rem; font-weight: 700; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 24px 0 8px;
    }
    .recommendation {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & explainer ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("Models/xgb_churn_model.pkl")
    features = joblib.load("Models/model_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

model, feature_cols, explainer = load_model()

# ── Helper functions ───────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the engineered features from notebook 03."""
    df = df.copy()
    df["Tenure_Months"]      = df["Membership_Years"] * 12
    df["Engagement_Score"]   = (
        df["Login_Frequency"] + df["Session_Duration_Avg"] +
        df["Pages_Per_Session"] + df["Mobile_App_Usage"] +
        df["Social_Media_Engagement_Score"]
    )
    df["Purchase_Intensity"] = df["Total_Purchases"] / (df["Tenure_Months"] + 1)
    df["Recency_Score"]      = 1 / (df["Days_Since_Last_Purchase"] + 1)
    df["High_Risk"]          = (
        (df["Cart_Abandonment_Rate"] > 0.5) &
        (df["Login_Frequency"] < 5)
    ).astype(int)
    return df

def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features, encode, align to model feature list."""
    df = engineer_features(df)
    df = df.drop(columns=["City"], errors="ignore")
    df = pd.get_dummies(df, drop_first=True)
    # Align to training columns — fill missing dummies with 0
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.40:
        return "Low Risk", "risk-low"
    elif prob < 0.70:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"

def retention_recommendations(shap_vals: np.ndarray, feature_names: list) -> list[str]:
    """Generate plain-English recommendations from top SHAP drivers."""
    top_idx = np.argsort(shap_vals)[::-1][:3]  # top 3 positive drivers
    recs = []
    playbook = {
        "Customer_Service_Calls":       "🔧 High support contact detected — assign a dedicated account manager or proactively resolve open issues.",
        "Cart_Abandonment_Rate":        "🛒 Frequent cart abandonment — trigger a personalised email or discount nudge within 24 hours.",
        "Days_Since_Last_Purchase":     "📅 Customer hasn't purchased recently — send a re-engagement campaign with personalised product recommendations.",
        "Engagement_Score":             "📊 Low engagement across touchpoints — consider a loyalty programme or in-app engagement reward.",
        "Lifetime_Value":               "💰 LTV pattern flags churn risk — review account health and consider a proactive retention call.",
        "Discount_Usage_Rate":          "🏷️ High discount dependency — evaluate if pricing strategy is sustainable for this customer segment.",
        "Returns_Rate":                 "📦 High return rate — investigate product fit or quality issues for this customer.",
        "Email_Open_Rate":              "📧 Low email engagement — switch communication channel or personalise subject lines.",
        "Total_Purchases":              "🛍️ Low purchase volume — offer a bundle deal or category recommendation based on browse history.",
        "Social_Media_Engagement_Score":"📱 Low social engagement — invite customer to loyalty community or exclusive member group.",
        "Login_Frequency":              "🔑 Infrequent logins — send a 'We miss you' re-activation push notification.",
        "Mobile_App_Usage":             "📲 Low app usage — promote app-exclusive deals to drive adoption.",
    }
    for i in top_idx:
        feat = feature_names[i]
        if shap_vals[i] > 0:  # only positive SHAP = pushing toward churn
            base_feat = feat.split("_")[0] if feat not in playbook else feat
            # Try exact match first, then partial
            rec = playbook.get(feat) or next(
                (v for k, v in playbook.items() if k in feat), None
            )
            if rec:
                recs.append(rec)
    if not recs:
        recs.append("✅ No strong churn drivers detected — continue standard engagement.")
    return recs

def waterfall_chart(shap_vals, feature_names, base_value, customer_data):
    """Render a SHAP waterfall chart and return as a PNG buffer."""
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=customer_data,
        feature_names=feature_names
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.sca(ax)
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn Predictor")
    st.caption("E-Commerce Customer Retention Tool")
    st.divider()
    mode = st.radio("Mode", ["Single Customer", "Batch Upload"], label_visibility="collapsed")
    st.divider()
    st.caption("Model: XGBoost · ROC-AUC 0.928")
    st.caption("Features: 37  · Test set: 10,000")
    st.caption("Churner recall: 85%")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CUSTOMER MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Customer":
    st.title("Customer Churn Risk Assessment")
    st.caption("Fill in the customer profile below and click **Predict** to assess churn risk.")

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("customer_form"):
        # Demographics
        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age             = c1.number_input("Age",                   min_value=18,  max_value=90,   value=35)
        gender          = c2.selectbox("Gender",                   ["Male", "Female", "Other"])
        country         = c3.selectbox("Country",                  ["USA", "UK", "Canada", "Australia", "India", "Germany", "France", "Japan"])

        c4, c5 = st.columns(2)
        membership_years = c4.number_input("Membership Years",     min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        signup_quarter   = c5.selectbox("Signup Quarter",          ["Q1", "Q2", "Q3", "Q4"])

        # Engagement
        st.markdown('<div class="section-header">Engagement</div>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        login_freq      = c6.number_input("Login Frequency",       min_value=0,   max_value=46,   value=10)
        session_dur     = c7.number_input("Avg Session Duration",  min_value=1.0, max_value=76.0, value=25.0, step=0.5)
        pages_per_sess  = c8.number_input("Pages Per Session",     min_value=1.0, max_value=25.0, value=8.0,  step=0.1)

        c9, c10, c11 = st.columns(3)
        mobile_usage    = c9.number_input("Mobile App Usage",      min_value=0.0, max_value=62.0, value=15.0, step=0.5)
        social_score    = c10.number_input("Social Media Score",   min_value=0.0, max_value=100.0,value=25.0, step=0.5)
        email_open      = c11.number_input("Email Open Rate (%)",  min_value=0.0, max_value=92.0, value=20.0, step=0.5)

        # Purchase behaviour
        st.markdown('<div class="section-header">Purchase Behaviour</div>', unsafe_allow_html=True)
        c12, c13, c14 = st.columns(3)
        total_purchases = c12.number_input("Total Purchases",      min_value=0.0, max_value=130.0, value=12.0, step=1.0)
        avg_order_val   = c13.number_input("Avg Order Value ($)",  min_value=26.0,max_value=9667.0,value=120.0,step=1.0)
        days_since      = c14.number_input("Days Since Last Purchase", min_value=0, max_value=287, value=20)

        c15, c16, c17 = st.columns(3)
        cart_abandon    = c15.number_input("Cart Abandonment Rate",min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        discount_rate   = c16.number_input("Discount Usage Rate",  min_value=0.0, max_value=100.0, value=40.0, step=0.5)
        returns_rate    = c17.number_input("Returns Rate (%)",     min_value=0.0, max_value=100.0, value=5.0,  step=0.5)

        c18, c19 = st.columns(2)
        wishlist        = c18.number_input("Wishlist Items",       min_value=0,   max_value=28,   value=4)
        payment_div     = c19.number_input("Payment Method Diversity", min_value=1, max_value=5,  value=2)

        # Account
        st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
        c20, c21, c22 = st.columns(3)
        lifetime_val    = c20.number_input("Lifetime Value ($)",   min_value=0.0, max_value=9000.0,value=1200.0,step=10.0)
        credit_bal      = c21.number_input("Credit Balance ($)",   min_value=0.0, max_value=7200.0,value=1800.0,step=10.0)
        service_calls   = c22.number_input("Customer Service Calls",min_value=0,  max_value=21,   value=3)

        c23, = st.columns(1)
        reviews         = c23.number_input("Product Reviews Written", min_value=0, max_value=21,  value=2)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if submitted:
        raw = pd.DataFrame([{
            "Age": age, "Gender": gender, "Country": country,
            "Membership_Years": membership_years, "Signup_Quarter": signup_quarter,
            "Login_Frequency": login_freq, "Session_Duration_Avg": session_dur,
            "Pages_Per_Session": pages_per_sess, "Mobile_App_Usage": mobile_usage,
            "Social_Media_Engagement_Score": social_score, "Email_Open_Rate": email_open,
            "Total_Purchases": total_purchases, "Average_Order_Value": avg_order_val,
            "Days_Since_Last_Purchase": days_since, "Cart_Abandonment_Rate": cart_abandon,
            "Discount_Usage_Rate": discount_rate, "Returns_Rate": returns_rate,
            "Wishlist_Items": wishlist, "Payment_Method_Diversity": payment_div,
            "Lifetime_Value": lifetime_val, "Credit_Balance": credit_bal,
            "Customer_Service_Calls": service_calls, "Product_Reviews_Written": reviews,
        }])

        X_input  = prepare_input(raw)
        prob     = float(model.predict_proba(X_input)[0, 1])
        label, css = risk_label(prob)
        sv       = explainer.shap_values(X_input)[0]

        st.divider()

        # ── Metrics row ────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Risk Category", label)
        m3.metric("Confidence", f"{max(prob, 1-prob):.1%}")

        # ── Risk badge ─────────────────────────────────────────────────────────
        st.markdown(f'<div class="{css}">{label} — {prob:.1%} probability of churning</div>', unsafe_allow_html=True)

        st.divider()

        left, right = st.columns([1.1, 1])

        # ── SHAP waterfall ─────────────────────────────────────────────────────
        with left:
            st.subheader("Why is this customer at risk?")
            buf = waterfall_chart(sv, list(X_input.columns), explainer.expected_value, X_input.values[0])
            st.image(buf, use_container_width=True)

        # ── Recommendations ────────────────────────────────────────────────────
        with right:
            st.subheader("Retention Recommendations")
            recs = retention_recommendations(sv, list(X_input.columns))
            for rec in recs:
                st.markdown(f'<div class="recommendation">{rec}</div>', unsafe_allow_html=True)

            st.divider()
            st.subheader("Top Risk Drivers")
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            driver_df = pd.DataFrame({
                "Feature" : [feature_cols[i] for i in top_idx],
                "SHAP"    : [round(sv[i], 4) for i in top_idx],
                "Effect"  : ["↑ Churn" if sv[i] > 0 else "↓ Churn" for i in top_idx]
            })
            st.dataframe(driver_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Batch Churn Scoring")
    st.caption("Upload a CSV of customers to score all at once.")

    # Template download
    template_cols = [
        "Age","Gender","Country","Membership_Years","Signup_Quarter",
        "Login_Frequency","Session_Duration_Avg","Pages_Per_Session",
        "Mobile_App_Usage","Social_Media_Engagement_Score","Email_Open_Rate",
        "Total_Purchases","Average_Order_Value","Days_Since_Last_Purchase",
        "Cart_Abandonment_Rate","Discount_Usage_Rate","Returns_Rate",
        "Wishlist_Items","Payment_Method_Diversity","Lifetime_Value",
        "Credit_Balance","Customer_Service_Calls","Product_Reviews_Written"
    ]
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        data=csv_template,
        file_name="churn_input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        raw_batch = pd.read_csv(uploaded)
        st.write(f"**{len(raw_batch):,} customers loaded.**")
        st.dataframe(raw_batch.head(5), use_container_width=True)

        if st.button("🔍 Score All Customers", use_container_width=True):
            with st.spinner("Scoring..."):
                X_batch = prepare_input(raw_batch)
                probs   = model.predict_proba(X_batch)[:, 1]

                raw_batch["Churn_Probability"] = probs.round(4)
                raw_batch["Risk_Category"]     = [risk_label(p)[0] for p in probs]

            st.divider()

            # ── Summary metrics ─────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers",  f"{len(raw_batch):,}")
            m2.metric("High Risk",        f"{(raw_batch['Risk_Category'] == 'High Risk').sum():,}")
            m3.metric("Medium Risk",      f"{(raw_batch['Risk_Category'] == 'Medium Risk').sum():,}")
            m4.metric("Avg Churn Prob",   f"{probs.mean():.1%}")

            # ── Risk distribution chart ─────────────────────────────────────────
            st.subheader("Risk Distribution")
            risk_counts = raw_batch["Risk_Category"].value_counts().reindex(["High Risk","Medium Risk","Low Risk"]).fillna(0)
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ["#f87171","#facc15","#4ade80"]
            ax.barh(risk_counts.index, risk_counts.values, color=colors)
            ax.set_xlabel("Number of Customers")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Full results table ───────────────────────────────────────────────
            st.subheader("Scored Results")
            st.dataframe(
                raw_batch.sort_values("Churn_Probability", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # ── Download ─────────────────────────────────────────────────────────
            st.download_button(
                "⬇️ Download Scored CSV",
                data=raw_batch.to_csv(index=False),
                file_name="churn_scores.csv",
                mime="text/csv",
                use_container_width=True
            )
