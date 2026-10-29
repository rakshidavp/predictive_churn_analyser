
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

try:
    from groq import Groq
except ImportError:
    Groq = None


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Predictive Churn Analyser",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# GREEN GLASSMORPHISM DESIGN
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 5%, #153c30 0, transparent 35%),
            radial-gradient(circle at 90% 90%, #173b2e 0, transparent 35%),
            #081510;
        color: #f0f7f2;
    }

    [data-testid="stSidebar"] {
        background: rgba(13, 35, 26, 0.96);
        border-right: 1px solid rgba(170, 230, 190, 0.15);
    }

    [data-testid="stSidebar"] * {
        color: #e8f5ec;
    }

    h1, h2, h3 {
        color: #f0fff4 !important;
        font-weight: 800 !important;
    }

    p, label, span {
        color: #d6e8dc;
    }

    div[data-testid="stMetric"] {
        background: rgba(28, 67, 48, 0.42);
        border: 1px solid rgba(154, 230, 174, 0.20);
        border-radius: 16px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b8d9c4 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #8ef0ad !important;
    }

    .hero {
        background: linear-gradient(
            120deg,
            rgba(42, 112, 75, 0.50),
            rgba(18, 49, 35, 0.55)
        );
        border: 1px solid rgba(155, 240, 180, 0.22);
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #a4f4ba;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #d5e9dc;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 700;
        color: #a4f4ba;
        margin: 18px 0 12px;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #258653, #48bd76);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #31a765, #65d98f);
        color: white;
    }

    .footer {
        text-align: center;
        color: #91b4a0;
        font-size: 12px;
        margin-top: 40px;
        padding: 16px;
        border-top: 1px solid rgba(160, 230, 180, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# CONSTANTS
# ==================================================

FEATURES = [
    "MonthlyBilling",
    "UsageTrendSpeed",
    "CustomerAnnoyanceLevel",
    "AutoRenewalTurnedOff",
]

CURRENCY = "₹"


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def section_heading(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def make_demo_data(number_of_customers=200):
    """Create demonstration customer data."""

    rng = np.random.default_rng(42)

    billing = rng.uniform(300, 5000, number_of_customers)
    usage = rng.uniform(-1.0, 1.0, number_of_customers)
    annoyance = rng.integers(1, 11, number_of_customers)
    renewal_off = rng.integers(0, 2, number_of_customers)

    risk_score = (
        (usage < -0.25).astype(int) * 0.25
        + (annoyance >= 7).astype(int) * 0.30
        + (renewal_off == 1).astype(int) * 0.35
        + (billing > 3500).astype(int) * 0.10
        + rng.uniform(0, 0.25, number_of_customers)
    )

    leaving_soon = (risk_score >= 0.45).astype(int)

    return pd.DataFrame({
        "CustomerID": [
            f"CUST-{i + 1001}"
            for i in range(number_of_customers)
        ],
        "CustomerName": [
            f"Customer {i + 1}"
            for i in range(number_of_customers)
        ],
        "MonthlyBilling": np.round(billing, 2),
        "UsageTrendSpeed": np.round(usage, 2),
        "CustomerAnnoyanceLevel": annoyance,
        "AutoRenewalTurnedOff": renewal_off,
        "LeavingSoon": leaving_soon,
    })


def normalise_target(series):
    """Convert common churn labels into numeric 0/1."""

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")

        if set(numeric.dropna().unique()).issubset({0, 1}):
            return numeric

    positive = {
        "1", "yes", "true", "churn", "churned",
        "leaving", "leaving soon", "cancelled", "canceled"
    }

    negative = {
        "0", "no", "false", "stay", "staying",
        "active", "not churned", "retained", "not leaving"
    }

    def convert(value):
        if pd.isna(value):
            return np.nan

        label = str(value).strip().lower()

        if label in positive:
            return 1

        if label in negative:
            return 0

        return np.nan

    return series.apply(convert)


def prepare_data(raw_data):
    """Validate customer data before model training."""

    df = raw_data.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "CustomerID" not in df.columns:
        df["CustomerID"] = [
            f"CUST-{i + 1001}" for i in range(len(df))
        ]

    if "CustomerName" not in df.columns:
        df["CustomerName"] = [
            f"Customer {i + 1}" for i in range(len(df))
        ]

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
        )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    df = df.dropna(subset=FEATURES).copy()

    if df.empty:
        raise ValueError(
            "No usable rows remain after checking the required numeric columns."
        )

    if "LeavingSoon" not in df.columns:
        score = (
            (df["UsageTrendSpeed"] < -0.25).astype(int) * 0.25
            + (df["CustomerAnnoyanceLevel"] >= 7).astype(int) * 0.30
            + (df["AutoRenewalTurnedOff"] == 1).astype(int) * 0.35
            + (df["MonthlyBilling"] > 3500).astype(int) * 0.10
        )

        df["LeavingSoon"] = (score >= 0.45).astype(int)

        st.warning(
            "Your CSV has no LeavingSoon column. "
            "Demonstration labels were created using simple rules. "
            "They are not verified historical churn outcomes."
        )

    else:
        before = len(df)

        df["LeavingSoon"] = normalise_target(
            df["LeavingSoon"]
        )

        df = df.dropna(subset=["LeavingSoon"]).copy()

        if len(df) < before:
            st.warning(
                f"{before - len(df)} rows with unrecognised "
                "LeavingSoon labels were excluded."
            )

        df["LeavingSoon"] = df["LeavingSoon"].astype(int)

    if df.empty:
        raise ValueError(
            "No usable rows remain after preparing the churn target."
        )

    if df["LeavingSoon"].nunique() < 2:
        raise ValueError(
            "LeavingSoon must contain both 0 and 1 labels "
            "to train the model."
        )

    return df


def train_model(df):
    """Train the Random Forest churn model."""

    X = df[FEATURES]
    y = df["LeavingSoon"].astype(int)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)
    churn_index = list(model.classes_).index(1)

    results = df.copy()

    results["ChurnProbability"] = probabilities[:, churn_index]

    results["ChurnProbabilityPercent"] = (
        results["ChurnProbability"] * 100
    ).round(1)

    results["PredictedChurn"] = (
        results["ChurnProbability"] >= 0.50
    ).astype(int)

    results["RiskLevel"] = results["ChurnProbability"].apply(
        lambda p: (
            "High" if p >= 0.65
            else "Medium" if p >= 0.35
            else "Low"
        )
    )

    return model, results


def retention_action(row):
    """Suggest a retention action based on customer signals."""

    actions = []

    if row["UsageTrendSpeed"] < -0.25:
        actions.append(
            "Offer a personalised re-engagement campaign"
        )

    if row["AutoRenewalTurnedOff"] == 1:
        actions.append(
            "Contact the customer about renewal concerns"
        )

    if row["CustomerAnnoyanceLevel"] >= 7:
        actions.append(
            "Arrange priority customer support"
        )

    if row["MonthlyBilling"] > 3500:
        actions.append(
            "Review plan value and suitable plan options"
        )

    if not actions:
        actions.append(
            "Maintain engagement and monitor customer activity"
        )

    return "; ".join(actions)


def get_groq_api_key():
    """Get the API key from environment variables or Streamlit secrets."""

    key = os.getenv("GROQ_API_KEY", "")

    if key:
        return key

    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def create_fallback_strategy(customer, signals):
    """Generate a retention plan without the Groq API."""

    risk = customer["RiskLevel"]
    name = customer["CustomerName"]

    if risk == "High":
        priority = "Urgent: contact the customer within 24–48 hours."
    elif risk == "Medium":
        priority = "Contact the customer within the next week."
    else:
        priority = "Preventive: continue engagement and monitor changes."

    signal_text = (
        "\n".join(f"- {signal}" for signal in signals)
        if signals
        else "- No major warning signals were identified."
    )

    return f"""## Retention plan for {name}

### 1. Priority
{priority}

### 2. Signals identified
{signal_text}

### 3. Recommended actions
1. Contact the customer through their preferred communication channel.
2. Ask about their experience and any issues they have faced.
3. Offer support that matches the identified problem.
4. Explain relevant plan options without pressuring the customer.
5. Record feedback and schedule a follow-up.

### 4. Follow-up plan
- First contact: discuss the customer's experience.
- After 3–7 days: check whether the issue has been resolved.
- After 30 days: review engagement and account activity.

### 5. Success measures
- Customer response rate
- Issue resolution rate
- Renewal or retention rate
- Customer satisfaction feedback

**Note:** This suggested plan does not guarantee that the customer will stay.
"""


def generate_ai_strategy(customer, signals):
    """Generate a Groq AI plan or use a fallback strategy."""

    api_key = get_groq_api_key()

    if Groq is None or not api_key:
        return (
            create_fallback_strategy(customer, signals),
            "Rule-based fallback",
        )

    try:
        client = Groq(api_key=api_key)

        summary = {
            "Customer ID": str(customer["CustomerID"]),
            "Monthly billing": float(customer["MonthlyBilling"]),
            "Usage trend": float(customer["UsageTrendSpeed"]),
            "Annoyance level": int(
                customer["CustomerAnnoyanceLevel"]
            ),
            "Auto-renewal turned off": int(
                customer["AutoRenewalTurnedOff"]
            ),
            "Predicted churn risk": (
                f"{float(customer['ChurnProbability']) * 100:.1f}%"
            ),
            "Risk category": str(customer["RiskLevel"]),
            "Signals": signals,
        }

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer-retention analyst. Create a "
                        "practical, ethical, customer-friendly plan using "
                        "only supplied facts. Do not invent customer history, "
                        "discounts, or outcomes. Include priority, risks, "
                        "3–5 actions, follow-up timing, and measurable "
                        "success indicators. Use Markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create a retention strategy for this customer:\n"
                        + str(summary)
                    ),
                },
            ],
        )

        answer = response.choices[0].message.content

        if not answer:
            raise ValueError(
                "The AI service returned an empty response."
            )

        return answer, "Groq AI"

    except Exception as error:
        st.warning(
            "Groq could not generate a strategy. "
            f"A fallback plan is shown instead. Details: {error}"
        )

        return (
            create_fallback_strategy(customer, signals),
            "Rule-based fallback",
        )


# ==================================================
# SESSION STATE
# ==================================================

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "strategy_text" not in st.session_state:
    st.session_state["strategy_text"] = None

if "strategy_source" not in st.session_state:
    st.session_state["strategy_source"] = None

if "strategy_customer_id" not in st.session_state:
    st.session_state["strategy_customer_id"] = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🌿 CHURN ANALYSER")
    st.caption("Customer intelligence dashboard")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            st.session_state["uploaded_data"] = pd.read_csv(
                uploaded_file
            )
            st.success("CSV uploaded successfully.")
        except Exception as error:
            st.error(f"Could not read the CSV: {error}")

    if st.button("Use demo dataset", use_container_width=True):
        st.session_state["uploaded_data"] = None
        st.session_state["strategy_text"] = None
        st.session_state["strategy_source"] = None
        st.session_state["strategy_customer_id"] = None
        st.rerun()

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Customer Analytics",
            "Predict Churn",
            "Customer Records",
            "AI Retention Strategy",
        ],
    )

    st.divider()
    st.caption("Predict • Analyse • Retain")


# ==================================================
# LOAD DATA AND TRAIN MODEL
# ==================================================

try:
    if st.session_state["uploaded_data"] is None:
        raw_data = make_demo_data()
        using_demo_data = True
    else:
        raw_data = st.session_state["uploaded_data"]
        using_demo_data = False

    data = prepare_data(raw_data)
    model, results = train_model(data)

except Exception as error:
    st.error(f"Unable to prepare the dataset: {error}")
    st.info(
        "Check that your CSV includes MonthlyBilling, UsageTrendSpeed, "
        "CustomerAnnoyanceLevel, and AutoRenewalTurnedOff. For training, "
        "LeavingSoon must contain both 0 and 1 labels."
    )
    st.stop()


# ==================================================
# SHARED SUMMARY VALUES
# ==================================================

total_customers = len(results)

high_risk_count = int(
    (results["RiskLevel"] == "High").sum()
)

medium_risk_count = int(
    (results["RiskLevel"] == "Medium").sum()
)

low_risk_count = int(
    (results["RiskLevel"] == "Low").sum()
)

predicted_churn_count = int(
    results["PredictedChurn"].sum()
)

average_risk = float(
    results["ChurnProbability"].mean() * 100
)

revenue_exposure = float(
    results.loc[
        results["RiskLevel"] == "High",
        "MonthlyBilling",
    ].sum()
)

if using_demo_data:
    st.info(
        "You are viewing demonstration data. Upload a suitable "
        "customer CSV to analyse your own records."
    )


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">PREDICTIVE CHURN ANALYSER</div>
            <div class="hero-subtitle">
                Understand customer behaviour, identify churn risk,
                and plan proactive retention actions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Customers", f"{total_customers:,}")

    with c2:
        metric_card("High-Risk Customers", f"{high_risk_count:,}")

    with c3:
        metric_card("Predicted Churn", f"{predicted_churn_count:,}")

    with c4:
        metric_card("Average Risk", f"{average_risk:.1f}%")

    section_heading("Churn Risk Distribution")

    risk_counts = results["RiskLevel"].value_counts().reindex(
        ["Low", "Medium", "High"],
        fill_value=0,
    )

    chart_col, info_col = st.columns([1.5, 1])

    with chart_col:
        st.bar_chart(
            risk_counts,
            x_label="Risk Level",
            y_label="Customers",
        )

    with info_col:
        st.markdown("### Risk summary")
        st.write(f"🟢 **Low risk:** {low_risk_count}")
        st.write(f"🟡 **Medium risk:** {medium_risk_count}")
        st.write(f"🔴 **High risk:** {high_risk_count}")

        st.markdown("### Estimated monthly billing at high risk")
        st.subheader(f"{CURRENCY}{revenue_exposure:,.2f}")

        st.caption(
            "This is the sum of monthly billing for high-risk records, "
            "not confirmed lost revenue."
        )

    section_heading("Customers Requiring Attention")

    high_risk_customers = results[
        results["RiskLevel"] == "High"
    ].copy()

    if not high_risk_customers.empty:

        high_risk_customers["SuggestedAction"] = (
            high_risk_customers.apply(retention_action, axis=1)
        )

        # Corrected sorting: use the column that actually exists.
        high_risk_customers = high_risk_customers.sort_values(
            "ChurnProbabilityPercent",
            ascending=False,
        )

        st.dataframe(
            high_risk_customers[
                [
                    "CustomerID",
                    "CustomerName",
                    "MonthlyBilling",
                    "ChurnProbabilityPercent",
                    "RiskLevel",
                    "SuggestedAction",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.success(
            "No high-risk customers were identified by the current model."
        )


# ==================================================
# PAGE 2: CUSTOMER ANALYTICS
# ==================================================

elif page == "Customer Analytics":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">CUSTOMER ANALYTICS</div>
            <div class="hero-subtitle">
                Explore billing, engagement signals, and churn-risk patterns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Customer Risk Breakdown")

    risk_counts = results["RiskLevel"].value_counts().reindex(
        ["Low", "Medium", "High"],
        fill_value=0,
    )

    st.bar_chart(
        risk_counts,
        x_label="Risk Level",
        y_label="Customers",
    )

    left, right = st.columns(2)

    with left:
        section_heading("Monthly Billing")
        st.bar_chart(
            results.set_index("CustomerID")["MonthlyBilling"],
            x_label="Customer ID",
            y_label="Monthly billing",
        )

    with right:
        section_heading("Customer Annoyance Level")
        annoyance_counts = (
            results["CustomerAnnoyanceLevel"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            annoyance_counts,
            x_label="Annoyance level",
            y_label="Customers",
        )

    section_heading("Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    st.bar_chart(
        importance.set_index("Feature"),
        x_label="Feature",
        y_label="Model importance",
    )

    st.caption(
        "Feature importance indicates contribution to this fitted model; "
        "it does not prove causation."
    )

    section_heading("Explore Records")

    risk_filter = st.selectbox(
        "Filter by risk category",
        ["All", "High", "Medium", "Low"],
    )

    filtered_results = results.copy()

    if risk_filter != "All":
        filtered_results = filtered_results[
            filtered_results["RiskLevel"] == risk_filter
        ]

    st.dataframe(
        filtered_results[
            [
                "CustomerID",
                "CustomerName",
                "MonthlyBilling",
                "UsageTrendSpeed",
                "CustomerAnnoyanceLevel",
                "AutoRenewalTurnedOff",
                "ChurnProbabilityPercent",
                "RiskLevel",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# PAGE 3: PREDICT CHURN
# ==================================================

elif page == "Predict Churn":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">PREDICT CUSTOMER CHURN</div>
            <div class="hero-subtitle">
                Enter customer details to estimate the likelihood of churn.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Customer Details")

    with st.form("prediction_form"):

        customer_name = st.text_input(
            "Customer name",
            value="New Customer",
        )

        monthly_billing = st.number_input(
            "Monthly billing (₹)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
        )

        usage_trend = st.slider(
            "Usage trend speed",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Negative values indicate declining usage in this demo.",
        )

        annoyance = st.slider(
            "Customer annoyance level",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

        auto_renewal_off = st.selectbox(
            "Has the customer turned off auto-renewal?",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            use_container_width=True,
        )

    if submitted:

        new_customer = pd.DataFrame([{
            "MonthlyBilling": monthly_billing,
            "UsageTrendSpeed": usage_trend,
            "CustomerAnnoyanceLevel": annoyance,
            "AutoRenewalTurnedOff": auto_renewal_off,
        }])

        churn_index = list(model.classes_).index(1)

        probability = float(
            model.predict_proba(new_customer)[0][churn_index]
        )

        risk_level = (
            "High" if probability >= 0.65
            else "Medium" if probability >= 0.35
            else "Low"
        )

        result_col, action_col = st.columns(2)

        with result_col:
            section_heading("Prediction Result")

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.1f}%",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if risk_level == "High":
                st.error(f"Risk category: {risk_level}")
            elif risk_level == "Medium":
                st.warning(f"Risk category: {risk_level}")
            else:
                st.success(f"Risk category: {risk_level}")

        with action_col:
            section_heading("Suggested Next Step")

            new_row = pd.Series({
                "MonthlyBilling": monthly_billing,
                "UsageTrendSpeed": usage_trend,
                "CustomerAnnoyanceLevel": annoyance,
                "AutoRenewalTurnedOff": auto_renewal_off,
            })

            st.write(retention_action(new_row))

        st.caption(
            "This prediction uses the current fitted model. Demo data "
            "or rule-generated labels are not a validated business forecast."
        )


# ==================================================
# PAGE 4: CUSTOMER RECORDS
# ==================================================

elif page == "Customer Records":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">CUSTOMER RECORDS</div>
            <div class="hero-subtitle">
                Review and export customer details with their estimated risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Customer Details Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        search_text = st.text_input(
            "Search by customer ID or name",
            placeholder="Enter a customer ID or name",
        )

    with col2:
        selected_risk = st.selectbox(
            "Select risk level",
            ["All", "High", "Medium", "Low"],
        )

    records = results.copy()

    records["SuggestedAction"] = records.apply(
        retention_action,
        axis=1,
    )

    if search_text.strip():
        term = search_text.strip().lower()

        records = records[
            records["CustomerID"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
            |
            records["CustomerName"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
        ]

    if selected_risk != "All":
        records = records[
            records["RiskLevel"] == selected_risk
        ]

    visible_columns = [
        "CustomerID",
        "CustomerName",
        "MonthlyBilling",
        "UsageTrendSpeed",
        "CustomerAnnoyanceLevel",
        "AutoRenewalTurnedOff",
        "ChurnProbabilityPercent",
        "RiskLevel",
        "SuggestedAction",
    ]

    st.dataframe(
        records[visible_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered customer records",
        data=records[visible_columns].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_churn_records.csv",
        mime="text/csv",
    )


# ==================================================
# PAGE 5: AI RETENTION STRATEGY
# ==================================================

elif page == "AI Retention Strategy":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">AI RETENTION STRATEGY</div>
            <div class="hero-subtitle">
                Generate personalised actions to help retain at-risk customers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Choose a Customer")

    customer_options = results[
        [
            "CustomerID",
            "CustomerName",
            "RiskLevel",
            "ChurnProbabilityPercent",
        ]
    ].copy()

    customer_options["Display"] = customer_options.apply(
        lambda row: (
            f"{row['CustomerID']} | {row['CustomerName']} | "
            f"{row['RiskLevel']} risk | "
            f"{row['ChurnProbabilityPercent']:.1f}%"
        ),
        axis=1,
    )

    selected_display = st.selectbox(
        "Customer ID, name and risk",
        customer_options["Display"].tolist(),
    )

    selected_id = customer_options.loc[
        customer_options["Display"] == selected_display,
        "CustomerID",
    ].iloc[0]

    customer = results[
        results["CustomerID"].astype(str) == str(selected_id)
    ].iloc[0]

    risk_col, billing_col, renewal_col = st.columns(3)

    with risk_col:
        metric_card(
            "Churn Probability",
            f"{customer['ChurnProbabilityPercent']:.1f}%",
        )

    with billing_col:
        metric_card(
            "Monthly Billing",
            f"{CURRENCY}{customer['MonthlyBilling']:,.2f}",
        )

    with renewal_col:
        metric_card(
            "Risk Category",
            str(customer["RiskLevel"]),
        )

    signals = []

    if customer["UsageTrendSpeed"] < -0.25:
        signals.append("Usage trend is declining.")

    if customer["AutoRenewalTurnedOff"] == 1:
        signals.append("Auto-renewal has been turned off.")

    if customer["CustomerAnnoyanceLevel"] >= 7:
        signals.append("Customer annoyance level is high.")

    if customer["MonthlyBilling"] > 3500:
        signals.append(
            "Monthly billing is relatively high; check perceived plan value."
        )

    section_heading("Customer Signals")

    if signals:
        for signal in signals:
            st.write(f"• {signal}")
    else:
        st.write(
            "No major warning signals were identified by the simple rules."
        )

    st.caption(
        "Signals are based on supplied columns and simple thresholds; "
        "review them alongside real customer feedback."
    )

    if st.button(
        "Generate Retention Strategy",
        use_container_width=True,
    ):

        with st.spinner("Preparing a personalised retention plan..."):
            strategy_text, strategy_source = generate_ai_strategy(
                customer,
                signals,
            )

        st.session_state["strategy_text"] = strategy_text
        st.session_state["strategy_source"] = strategy_source
        st.session_state["strategy_customer_id"] = str(selected_id)

    if (
        st.session_state.get("strategy_text")
        and st.session_state.get("strategy_customer_id") == str(selected_id)
    ):

        section_heading("Recommended Retention Plan")

        st.caption(
            "Strategy generated using: "
            f"{st.session_state.get('strategy_source', 'Unknown')}"
        )

        st.markdown(
            st.session_state["strategy_text"]
        )

        st.download_button(
            "Download Retention Strategy",
            data=st.session_state["strategy_text"],
            file_name=f"retention_strategy_{selected_id}.md",
            mime="text/markdown",
        )

    elif st.session_state.get("strategy_text"):
        st.info(
            "Select the customer and click Generate Retention Strategy "
            "to create a plan for that customer."
        )

    with st.expander("How does this feature work?"):
        st.write(
            "The app checks customer signals and churn probability. "
            "If Groq is installed and GROQ_API_KEY is configured, it "
            "requests a tailored plan from the AI model. Otherwise, "
            "it generates a rule-based fallback plan."
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        Predictive Churn Analyser · Customer Analytics & Retention
    </div>
    """,
    unsafe_allow_html=True,
)
