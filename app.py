
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Predictive Churn Analyser",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 2. CUSTOM MAROON AND BEIGE THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #FDFBF7 !important;
    color: #2D221E !important;
}

[data-testid="stSidebar"] {
    background-color: #4A0000 !important;
}

[data-testid="stSidebar"] * {
    color: #FDFBF7 !important;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
}

h1 {
    color: #800000 !important;
    font-size: 2.6rem !important;
    border-bottom: 3px solid #800000;
    padding-bottom: 12px;
}

h2 {
    color: #4A0000 !important;
}

h3 {
    color: #6B5B55 !important;
}

[data-testid="stMetricValue"] {
    color: #800000 !important;
    font-weight: 900 !important;
}

[data-testid="stMetricLabel"] {
    color: #6B5B55 !important;
    font-weight: 700 !important;
}

.strategy-card {
    border-left: 5px solid #800000;
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 5px;
    margin-bottom: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}

.section-card {
    background-color: #F5EFEB;
    border: 1px solid #E6DCD5;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.stButton > button {
    background-color: #800000 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 4px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. GENERATE DEMONSTRATION CUSTOMER DATA
# =========================================================
@st.cache_data
def load_base_data():
    rng = np.random.default_rng(42)
    n = 120

    df = pd.DataFrame({
        "CustomerID": [
            f"CUST-{1000 + i}" for i in range(n)
        ],
        "CustomerName": [
            f"Enterprise Partner Group {i}"
            for i in range(n)
        ],
        "MonthlyBilling": rng.uniform(
            3000, 30000, n
        ).round(2),
        "PastHistoricalUsage": rng.uniform(
            40, 100, n
        ).round(2),
        "CurrentRecentUsage": rng.uniform(
            5, 95, n
        ).round(2),
        "TotalPastComplaints": rng.integers(
            1, 15, n
        ),
        "RecentWeeklyComplaints": rng.integers(
            0, 6, n
        ),
        "AutoRenewalTurnedOff": rng.choice(
            [0, 1], n, p=[0.75, 0.25]
        )
    })

    df["UsageTrendSpeed"] = (
        df["CurrentRecentUsage"]
        / (df["PastHistoricalUsage"] + 1)
    ).round(2)

    df["CustomerAnnoyanceLevel"] = (
        df["RecentWeeklyComplaints"]
        - df["TotalPastComplaints"] / 4
    ).round(2)

    # Demonstration labels generated from simple rules.
    df["LeavingSoon"] = (
        (df["UsageTrendSpeed"] < 0.5)
        | (df["AutoRenewalTurnedOff"] == 1)
        | (df["CustomerAnnoyanceLevel"] > 1.2)
    ).astype(int)

    return df


# Store data in the current Streamlit session.
if "customer_db" not in st.session_state:
    st.session_state.customer_db = load_base_data()

df = st.session_state.customer_db.copy()

FEATURE_COLUMNS = [
    "MonthlyBilling",
    "UsageTrendSpeed",
    "CustomerAnnoyanceLevel",
    "AutoRenewalTurnedOff"
]


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def prepare_customer_features(customer):
    """Build a single-row DataFrame for the model."""
    return pd.DataFrame([{
        "MonthlyBilling": float(
            customer["MonthlyBilling"]
        ),
        "UsageTrendSpeed": float(
            customer["UsageTrendSpeed"]
        ),
        "CustomerAnnoyanceLevel": float(
            customer["CustomerAnnoyanceLevel"]
        ),
        "AutoRenewalTurnedOff": int(
            customer["AutoRenewalTurnedOff"]
        )
    }], columns=FEATURE_COLUMNS)


def train_model(customer_df):
    X = customer_df[FEATURE_COLUMNS]
    y = customer_df["LeavingSoon"]

    classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    classifier.fit(X, y)
    return classifier


def get_churn_probability(classifier, customer):
    features = prepare_customer_features(customer)
    classes = list(classifier.classes_)

    if 1 not in classes:
        return 0.0

    churn_column = classes.index(1)

    return float(
        classifier.predict_proba(features)[0, churn_column]
    )


# Train the model before displaying any screen.
model = train_model(df)


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("## MANAGEMENT CONTROLS")
st.sidebar.markdown("---")

screen = st.sidebar.radio(
    "SELECT VIEW",
    [
        "Company Overview Dashboard",
        "Individual Customer Health Audit",
        "Onboard New Customer Account"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Predictive Churn Analyser")
st.sidebar.caption(
    f"Customer records in session: {len(df)}"
)


# =========================================================
# 6. COMPANY OVERVIEW DASHBOARD
# =========================================================
if screen == "Company Overview Dashboard":

    st.title("PREDICTIVE CHURN ANALYSER")
    st.markdown("### Company Overview Dashboard")

    st.write(
        "Monitor customer accounts, estimated churn risk, "
        "and monthly revenue exposure from one dashboard."
    )

    df["RiskScore"] = (
        df.apply(
            lambda row: get_churn_probability(model, row),
            axis=1
        ) * 100
    ).round(1)

    df["RevenueExposure"] = (
        df["RiskScore"] / 100 * df["MonthlyBilling"]
    ).round(2)

    total_customers = len(df)
    total_mrr = df["MonthlyBilling"].sum()
    total_revenue_at_risk = df["RevenueExposure"].sum()
    high_risk_count = int(
        (df["RiskScore"] >= 70).sum()
    )

    st.header("1. Business Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Monthly Billing",
        f"₹ {total_mrr:,.2f}"
    )

    col3.metric(
        "Estimated Revenue at Risk",
        f"₹ {total_revenue_at_risk:,.2f}"
    )

    col4.metric(
        "High-Risk Accounts",
        f"{high_risk_count}"
    )

    st.markdown("---")
    st.header("2. Critical Customer Priorities")

    st.write(
        "The following accounts have the highest estimated "
        "monthly revenue exposure."
    )

    pipeline = (
        df.sort_values(
            "RevenueExposure",
            ascending=False
        )[
            [
                "CustomerID",
                "CustomerName",
                "MonthlyBilling",
                "RiskScore",
                "RevenueExposure"
            ]
        ]
        .head(10)
        .rename(columns={
            "CustomerID": "Customer ID",
            "CustomerName": "Company Name",
            "MonthlyBilling": "Monthly Billing (₹)",
            "RiskScore": "Churn Risk (%)",
            "RevenueExposure": "Estimated Revenue at Risk (₹)"
        })
        .reset_index(drop=True)
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.header("3. Customer Risk Distribution")

    risk_groups = pd.cut(
        df["RiskScore"],
        bins=[-0.01, 30, 70, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    risk_counts = risk_groups.value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )

    st.bar_chart(risk_counts)

    st.caption(
        "Risk estimates are produced by a demonstration "
        "model trained on synthetic data and rule-generated labels."
    )


# =========================================================
# 7. INDIVIDUAL CUSTOMER HEALTH AUDIT
# =========================================================
elif screen == "Individual Customer Health Audit":

    st.title("PREDICTIVE CHURN ANALYSER")
    st.markdown("### Individual Customer Health Audit")

    st.write(
        "Select a customer to view their profile, behaviour, "
        "churn risk, revenue exposure, and recommended actions."
    )

    st.sidebar.markdown("### CUSTOMER SELECTION")

    search_id = st.sidebar.selectbox(
        "Choose Customer ID",
        df["CustomerID"].tolist()
    )

    customer_data = df.loc[
        df["CustomerID"] == search_id
    ].iloc[0]

    prob = get_churn_probability(model, customer_data)
    monthly_billing = float(
        customer_data["MonthlyBilling"]
    )
    cash_exposed = prob * monthly_billing

    # SECTION 1: CUSTOMER PROFILE
    st.markdown("---")
    st.header("1. Customer Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        st.markdown("#### Basic Customer Information")

        st.write(
            f"**Customer ID:** {customer_data['CustomerID']}"
        )
        st.write(
            f"**Company Name:** {customer_data['CustomerName']}"
        )

    with profile_col2:
        st.markdown("#### Account Information")

        st.write(
            f"**Monthly Billing:** ₹ {monthly_billing:,.2f}"
        )

        renewal_status = (
            "Disabled"
            if int(customer_data["AutoRenewalTurnedOff"]) == 1
            else "Enabled"
        )

        st.write(
            f"**Automatic Renewal:** {renewal_status}"
        )

    # SECTION 2: CHURN RISK
    st.markdown("---")
    st.header("2. Churn Risk and Financial Summary")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Estimated Churn Risk",
        f"{prob * 100:.1f}%"
    )

    risk_col2.metric(
        "Monthly Revenue",
        f"₹ {monthly_billing:,.2f}"
    )

    risk_col3.metric(
        "Estimated Revenue at Risk",
        f"₹ {cash_exposed:,.2f}"
    )

    if prob >= 0.7:
        st.error(
            "HIGH RISK: Prioritise a human review of this account."
        )
    elif prob >= 0.3:
        st.warning(
            "MEDIUM RISK: Monitor customer usage and feedback."
        )
    else:
        st.success(
            "LOW RISK: Continue regular customer engagement."
        )

    # SECTION 3: CUSTOMER BEHAVIOUR
    st.markdown("---")
    st.header("3. Customer Behaviour Analysis")

    behaviour_col1, behaviour_col2 = st.columns(2)

    with behaviour_col1:
        st.markdown("#### A. Product Usage Details")

        st.write(
            f"**Past Historical Usage:** "
            f"{customer_data['PastHistoricalUsage']:.2f}"
        )

        st.write(
            f"**Current Recent Usage:** "
            f"{customer_data['CurrentRecentUsage']:.2f}"
        )

        st.write(
            f"**Usage Trend Speed:** "
            f"{customer_data['UsageTrendSpeed']:.2f}"
        )

        st.caption(
            "A lower usage trend indicates that recent usage "
            "is lower relative to historical usage."
        )

    with behaviour_col2:
        st.markdown("#### B. Customer Complaint Details")

        st.write(
            f"**Total Past Complaints:** "
            f"{int(customer_data['TotalPastComplaints'])}"
        )

        st.write(
            f"**Recent Weekly Complaints:** "
            f"{int(customer_data['RecentWeeklyComplaints'])}"
        )

        st.write(
            f"**Complaint Indicator:** "
            f"{customer_data['CustomerAnnoyanceLevel']:.2f}"
        )

        st.caption(
            "Review recent complaints and resolve outstanding issues."
        )

    # SECTION 4: RETENTION PROJECTION
    st.markdown("---")
    st.header("4. Illustrative 12-Month Retention Projection")

    months = np.arange(1, 13)

    staying_probability = [
        max(
            0.0,
            min(1.0, (1.0 - prob) ** (month / 6))
        )
        for month in months
    ]

    projection = pd.DataFrame({
        "Month": months,
        "Illustrative Staying Probability": staying_probability
    }).set_index("Month")

    st.line_chart(projection)

    st.caption(
        "Illustrative curve only; it is not a trained survival "
        "model or a validated forecast of future retention."
    )

    # SECTION 5: RETENTION PLAN
    st.markdown("---")
    st.header("5. Recommended Customer Retention Actions")

    if float(customer_data["UsageTrendSpeed"]) < 0.6:

        st.markdown("""
        <div class="strategy-card">
            <h4 style="color:#800000;">Priority: Declining Product Usage</h4>
            <strong>Recommended Action:</strong>
            Contact the customer to understand the decline. Offer a product
            walkthrough, check for technical difficulties, and schedule a follow-up.
        </div>
        """, unsafe_allow_html=True)

    elif float(
        customer_data["CustomerAnnoyanceLevel"]
    ) > 1.2:

        st.markdown("""
        <div class="strategy-card">
            <h4 style="color:#800000;">Priority: Customer Complaints</h4>
            <strong>Recommended Action:</strong>
            Review unresolved complaints, assign a support contact, and
            follow up after the issues have been addressed.
        </div>
        """, unsafe_allow_html=True)

    elif int(customer_data["AutoRenewalTurnedOff"]) == 1:

        st.markdown("""
        <div class="strategy-card">
            <h4 style="color:#800000;">Priority: Renewal Disabled</h4>
            <strong>Recommended Action:</strong>
            Ask about the customer's renewal preferences and explain
            relevant options without applying pressure.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="strategy-card">
            <h4 style="color:#800000;">Priority: Regular Engagement</h4>
            <strong>Recommended Action:</strong>
            Continue regular communication, monitor usage trends, and
            provide appropriate customer support.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 8. ONBOARD NEW CUSTOMER ACCOUNT
# =========================================================
elif screen == "Onboard New Customer Account":

    st.title("PREDICTIVE CHURN ANALYSER")
    st.markdown("### New Customer Account Onboarding")

    st.write(
        "Enter the new customer's details to add an account "
        "to this session and estimate its churn risk."
    )

    with st.form("new_customer_form"):

        st.subheader("A. Customer Information")

        customer_name = st.text_input(
            "Customer / Company Name",
            placeholder="Enter company name"
        )

        monthly_billing = st.number_input(
            "Monthly Billing (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        st.subheader("B. Product Usage Details")

        past_usage = st.slider(
            "Past Historical Usage",
            min_value=1.0,
            max_value=100.0,
            value=75.0
        )

        current_usage = st.slider(
            "Current Recent Usage",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        st.subheader("C. Customer Support Details")

        past_complaints = st.number_input(
            "Total Past Complaints",
            min_value=0,
            value=2,
            step=1
        )

        weekly_complaints = st.number_input(
            "Recent Weekly Complaints",
            min_value=0,
            value=1,
            step=1
        )

        st.subheader("D. Renewal Preferences")

        renewal_off = st.selectbox(
            "Is Automatic Renewal Turned Off?",
            ["No", "Yes"]
        )

        submitted = st.form_submit_button(
            "Add Customer and Estimate Risk",
            use_container_width=True
        )

    if submitted:

        if not customer_name.strip():
            st.error(
                "Please enter a customer or company name."
            )

        elif monthly_billing <= 0:
            st.error(
                "Monthly billing must be greater than zero."
            )

        else:
            existing_numbers = []

            for customer_id in (
                st.session_state.customer_db["CustomerID"]
            ):
                try:
                    existing_numbers.append(
                        int(customer_id.split("-")[1])
                    )
                except (IndexError, ValueError):
                    continue

            next_number = max(
                existing_numbers,
                default=999
            ) + 1

            new_id = f"CUST-{next_number}"

            renewal_value = (
                1 if renewal_off == "Yes" else 0
            )

            usage_speed = (
                current_usage / (past_usage + 1)
            )

            annoyance = (
                weekly_complaints - past_complaints / 4
            )

            leaving_soon = int(
                (usage_speed < 0.5)
                or (renewal_value == 1)
                or (annoyance > 1.2)
            )

            new_customer = {
                "CustomerID": new_id,
                "CustomerName": customer_name.strip(),
                "MonthlyBilling": float(monthly_billing),
                "PastHistoricalUsage": float(past_usage),
                "CurrentRecentUsage": float(current_usage),
                "TotalPastComplaints": int(past_complaints),
                "RecentWeeklyComplaints": int(weekly_complaints),
                "AutoRenewalTurnedOff": renewal_value,
                "UsageTrendSpeed": round(usage_speed, 2),
                "CustomerAnnoyanceLevel": round(annoyance, 2),
                "LeavingSoon": leaving_soon
            }

            # Add the customer to the session database.
            st.session_state.customer_db = pd.concat(
                [
                    st.session_state.customer_db,
                    pd.DataFrame([new_customer])
                ],
                ignore_index=True
            )

            # Retrain using the updated session data.
            df = st.session_state.customer_db.copy()
            model = train_model(df)

            new_probability = get_churn_probability(
                model,
                new_customer
            )

            st.success(
                f"Customer {new_id} was added to this session."
            )

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "New Customer ID",
                new_id
            )

            result_col2.metric(
                "Estimated Churn Risk",
                f"{new_probability * 100:.1f}%"
            )

            st.info(
                "This customer is stored in the current Streamlit "
                "session. For permanent storage, connect a database "
                "such as Supabase."
            )


# =========================================================
# 9. FOOTER AND MODEL LIMITATION
# =========================================================
st.markdown("---")

st.caption(
    "Prototype note: Customer data is synthetic and churn labels "
    "are generated from simple rules. Risk scores and revenue exposure "
    "are illustrative, not validated real-world forecasts. Test with "
    "genuine historical data before using the model for business decisions."
)
