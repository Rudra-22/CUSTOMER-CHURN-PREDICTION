# =============================================
#  WEB APP — CUSTOMER CHURN PREDICTION
#  Run this with:  streamlit run app.py
#
#  This app has 4 simple pages:
#  Page 1 — Home
#  Page 2 — EDA Charts
#  Page 3 — Model Results
#  Page 4 — Predict a Customer
# =============================================

# ── Import libraries ──────────────────────────

import streamlit as st   # builds the web app
import pandas as pd      # for data tables
import matplotlib.pyplot as plt
import seaborn as sns
import pickle            # to load saved models
import os

from sklearn.metrics import roc_curve, confusion_matrix

# =============================================
#  APP SETTINGS
# =============================================
st.set_page_config(
    page_title = "Churn Prediction App",
    page_icon  = "📉",
    layout     = "wide"
)

# =============================================
#  LOAD ALL SAVED FILES
#  (these were created by running chunks 1-5)
# =============================================
@st.cache_data
def load_everything():

    # Load raw data
    df_raw = pd.read_csv('clean_data.csv')

    # Load processed data
    df_pro = pd.read_csv('processed_data.csv')

    # Load model results
    results    = pickle.load(open('all_results.pkl',      'rb'))
    best_name  = pickle.load(open('best_name.pkl',        'rb'))
    X_test     = pickle.load(open('X_test.pkl',           'rb'))
    y_test     = pickle.load(open('y_test.pkl',           'rb'))
    scaler     = pickle.load(open('scaler.pkl',           'rb'))
    feat_cols  = pickle.load(open('feature_columns.pkl',  'rb'))

    return df_raw, df_pro, results, best_name, X_test, y_test, scaler, feat_cols

# Show loading message
with st.spinner("Loading data and models..."):
    try:
        df_raw, df_pro, results, best_name, X_test, y_test, scaler, feat_cols = load_everything()
        loaded = True
    except:
        loaded = False

# Show error if files not found
if not loaded:
    st.error("Files not found! Please run all 5 chunks first.")
    st.code("python 1_load_data.py\npython 2_eda.py\npython 3_features.py\npython 4_train.py\npython 5_evaluate.py")
    st.stop()

# Pre-calculate stats used across pages
total    = len(df_raw)
churned  = (df_raw['Churn'] == 'Yes').sum()
retained = (df_raw['Churn'] == 'No').sum()
churn_pct = round(churned / total * 100, 1)

# =============================================
#  SIDEBAR — NAVIGATION MENU
# =============================================
st.sidebar.title("📉 Churn Prediction  BY RUDRA THAKKAR")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to Page",
    [
        "🏠  Home",
        "📊  EDA Charts",
        "🤖  Model Results",
        "🔮  Predict Customer",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** IBM Telco")
st.sidebar.markdown(f"**Customers:** {len(df_raw):,}")
st.sidebar.markdown(f"**Best Model:** {best_name}")
st.sidebar.markdown(f"**AUC Score:** {results[best_name]['auc']}")


# =============================================
#  PAGE 1 — HOME
# =============================================
if "Home" in page:

    # Title
    st.title("📉 Customer Churn Prediction")
    st.subheader("for Subscription Businesses")
    st.markdown("---")

    # What is churn?
    st.markdown("""
    ### 📖 What is Customer Churn?
    **Churn** means a customer **stops using** your service.

    For example:
    - A Jio customer switches to Airtel → **Churned!**
    - A Netflix user cancels subscription → **Churned!**

    **Our Goal:** Predict WHO will churn BEFORE they leave,
    so we can take action to keep them! 🎯
    """)

    st.markdown("---")

    # 4 KPI boxes
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Customers",  f"{total:,}")
    col2.metric("✅ Retained",         f"{retained:,}")
    col3.metric("❌ Churned",          f"{churned:,}")
    col4.metric("📊 Churn Rate",       f"{churn_pct}%")

    st.markdown("---")

    # Project steps
    st.markdown("### 🗺️ How This Project Works")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.info("**Step 1**\n\nLoad Data")
    c2.info("**Step 2**\n\nAnalyze Data")
    c3.info("**Step 3**\n\nBuild Features")
    c4.info("**Step 4**\n\nTrain Models")
    c5.info("**Step 5**\n\nPredict Churn")

    st.markdown("---")

    # Raw data preview
    st.markdown("### 🔍 Raw Data Preview")
    st.dataframe(df_raw.head(10), width='stretch')


# =============================================
#  PAGE 2 — EDA CHARTS
# =============================================
elif "EDA" in page:

    st.title("📊 Exploratory Data Analysis")
    st.markdown("These charts help us **understand the data** before building the model.")
    st.markdown("---")

    # ── Row 1 ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Churn Count")
        st.markdown("How many customers churned vs stayed?")

        churn_count = df_raw['Churn'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(['Retained', 'Churned'],
               churn_count.values,
               color=['green', 'red'],
               width=0.4)
        ax.set_ylabel('Count')
        for i, v in enumerate(churn_count.values):
            ax.text(i, v + 30, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"💡 {churn_pct}% of customers have churned!")

    with col2:
        st.subheader("2️⃣ Tenure Distribution")
        st.markdown("How long do churned vs retained customers stay?")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df_raw[df_raw['Churn'] == 'Yes']['tenure'],
                bins=20, alpha=0.7, color='red',   label='Churned')
        ax.hist(df_raw[df_raw['Churn'] == 'No']['tenure'],
                bins=20, alpha=0.7, color='green', label='Retained')
        ax.set_xlabel('Tenure (Months)')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("💡 New customers (0-12 months) churn the most!")

    st.markdown("---")

    # ── Row 2 ─────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3️⃣ Contract Type")
        st.markdown("Which contract type has highest churn?")

        ct = df_raw.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).round(1)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(ct.index, ct.values,
               color=['red', 'orange', 'green'],
               width=0.4)
        ax.set_ylabel('Churn Rate (%)')
        for i, v in enumerate(ct.values):
            ax.text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("💡 Month-to-month customers churn 42.7%!")

    with col4:
        st.subheader("4️⃣ Monthly Charges")
        st.markdown("Do higher charges cause more churn?")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot(
            [df_raw[df_raw['Churn'] == 'No']['MonthlyCharges'],
             df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges']],
            labels    = ['Retained', 'Churned'],
            patch_artist = True,
            boxprops  = dict(facecolor='lightblue')
        )
        ax.set_ylabel('Monthly Charges ($)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("💡 Churned customers pay more monthly!")

    st.markdown("---")

    # ── Row 3 ─────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("5️⃣ Internet Service")
        st.markdown("Which internet type has most churn?")

        is_ = df_raw.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).round(1)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(is_.index, is_.values,
               color=['blue', 'red', 'green'],
               width=0.4)
        ax.set_ylabel('Churn Rate (%)')
        for i, v in enumerate(is_.values):
            ax.text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("💡 Fiber optic customers churn the most (41.9%)!")

    with col6:
        st.subheader("6️⃣ Payment Method")
        st.markdown("Does payment method affect churn?")

        pm = df_raw.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).round(1)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(pm.index, pm.values, color='purple', height=0.4)
        ax.set_xlabel('Churn Rate (%)')
        for i, v in enumerate(pm.values):
            ax.text(v + 0.3, i, f'{v}%', va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("💡 Electronic check users churn most (45.3%)!")


# =============================================
#  PAGE 3 — MODEL RESULTS
# =============================================
elif "Model" in page:

    st.title("🤖 Model Training Results")
    st.markdown("We trained **4 models** and picked the best one.")
    st.markdown("---")

    # ── Model Metrics Table ────────────────────
    st.subheader("📋 All Model Scores")

    rows = []
    for name, res in results.items():
        rows.append({
            'Model':    name,
            'Accuracy': f"{res['accuracy']}%",
            'AUC':      res['auc'],
            'F1 Score': res['f1'],
            'Best?':    '✅ YES' if name == best_name else ''
        })

    st.dataframe(
        pd.DataFrame(rows),
        width='stretch'
    )

    st.info(f"""
    **What do these scores mean?**
    - **Accuracy** = How many predictions were correct
    - **AUC** = How well model separates churners from non-churners (higher = better)
    - **F1 Score** = Balance of precision and recall
    """)

    st.markdown("---")

    # ── Two Charts ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 AUC Score Comparison")

        names  = list(results.keys())
        aucs   = [results[n]['auc'] for n in names]
        colors = ['gold' if n == best_name else 'steelblue' for n in names]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(range(len(names)), aucs, color=colors, width=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(['LR', 'DT', 'RF', 'GB'], fontsize=11)
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel('AUC Score')
        ax.set_title('Gold bar = Best Model')
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    str(val), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📉 ROC Curves")

        fig, ax = plt.subplots(figsize=(6, 4))
        chart_colors = ['green', 'blue', 'orange', 'red']
        for (name, res), color in zip(results.items(), chart_colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            lw = 3 if name == best_name else 1.5
            ax.plot(fpr, tpr, color=color, lw=lw,
                    label=f"{name} ({res['auc']})")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Higher curve = Better model')
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Confusion Matrix ───────────────────────
    st.subheader(f"🎯 Confusion Matrix — {best_name}")
    st.markdown("Shows what the model predicted vs what actually happened.")

    col3, col4 = st.columns(2)

    with col3:
        cm = confusion_matrix(y_test, results[best_name]['y_pred'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        st.markdown("**What each box means:**")
        st.success(f"✅ **True Negative ({tn})** → Correctly said Retained")
        st.error(f"❌ **False Positive ({fp})** → Said Churn, was Retained")
        st.warning(f"⚠️ **False Negative ({fn})** → Missed real Churners")
        st.success(f"✅ **True Positive ({tp})** → Correctly caught Churners")

    st.markdown("---")

    # ── Feature Importance ─────────────────────
    st.subheader("🔍 Top 10 Features That Predict Churn")

    rf_model = results['Random Forest']['model']
    feat_imp = pd.DataFrame({
        'Feature':    X_test.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(feat_imp['Feature'][::-1],
            feat_imp['Importance'][::-1],
            color='steelblue', height=0.6)
    ax.set_xlabel('Importance Score')
    ax.set_title('More important = predicts churn better')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================
#  PAGE 4 — PREDICT A CUSTOMER
# =============================================
elif "Predict" in page:

    st.title("🔮 Predict Churn for a Customer")
    st.markdown("Fill in the details below and click **Predict**.")
    st.markdown("---")

    # ── Input Form ─────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        gender     = st.selectbox("Gender",
                                  ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen",
                                  ["No", "Yes"])
        partner    = st.selectbox("Has Partner",
                                  ["Yes", "No"])
        dependents = st.selectbox("Has Dependents",
                                  ["No", "Yes"])
        tenure     = st.slider("Tenure (months)", 1, 72, 12)

    with col2:
        st.markdown("**📱 Services Used**")
        phone    = st.selectbox("Phone Service",
                                ["Yes", "No"])
        lines    = st.selectbox("Multiple Lines",
                                ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service",
                                ["Fiber optic", "DSL", "No"])
        sec      = st.selectbox("Online Security",
                                ["No", "Yes", "No internet service"])
        backup   = st.selectbox("Online Backup",
                                ["No", "Yes", "No internet service"])
        device   = st.selectbox("Device Protection",
                                ["No", "Yes", "No internet service"])
        tech     = st.selectbox("Tech Support",
                                ["No", "Yes", "No internet service"])
        tv       = st.selectbox("Streaming TV",
                                ["No", "Yes", "No internet service"])
        movies   = st.selectbox("Streaming Movies",
                                ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**💳 Billing Info**")
        contract  = st.selectbox("Contract Type",
                                 ["Month-to-month",
                                  "One year",
                                  "Two year"])
        paperless = st.selectbox("Paperless Billing",
                                 ["Yes", "No"])
        payment   = st.selectbox("Payment Method",
                                 ["Electronic check",
                                  "Mailed check",
                                  "Bank transfer (automatic)",
                                  "Credit card (automatic)"])
        monthly   = st.slider("Monthly Charges ($)",
                              18.0, 120.0, 65.0)
        total     = st.slider("Total Charges ($)",
                              18.0, 9000.0,
                              float(monthly * tenure))

    st.markdown("---")

    # ── Predict Button ─────────────────────────
    if st.button("🔮 Predict Now"):

        # ── Build input row ────────────────────
        inp = {
            'gender':           gender,
            'SeniorCitizen':    1 if senior == 'Yes' else 0,
            'Partner':          partner,
            'Dependents':       dependents,
            'tenure':           tenure,
            'PhoneService':     phone,
            'MultipleLines':    lines,
            'InternetService':  internet,
            'OnlineSecurity':   sec,
            'OnlineBackup':     backup,
            'DeviceProtection': device,
            'TechSupport':      tech,
            'StreamingTV':      tv,
            'StreamingMovies':  movies,
            'Contract':         contract,
            'PaperlessBilling': paperless,
            'PaymentMethod':    payment,
            'MonthlyCharges':   monthly,
            'TotalCharges':     total,
        }

        # ── Feature engineering (same as chunk 3) ──
        row = pd.DataFrame([inp])

        row['AvgChargesPerMonth'] = (
            row['TotalCharges'] / max(row['tenure'].values[0], 1)
        )
        row['NumServices'] = sum([
            sec == 'Yes', backup == 'Yes', device == 'Yes',
            tech == 'Yes', tv == 'Yes', movies == 'Yes'
        ])
        row['ContractRisk'] = {
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }.get(contract, 2)

        # ── Binary encode ──────────────────────
        binary_map = {
            'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0
        }
        for col in ['gender', 'Partner', 'Dependents',
                    'PhoneService', 'PaperlessBilling']:
            row[col] = binary_map.get(str(row[col].values[0]), 0)

        # ── One-hot encode ─────────────────────
        row = pd.get_dummies(row, columns=[
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ])

        # ── Align columns ──────────────────────
        for col in feat_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feat_cols]

        # ── Scale ──────────────────────────────
        scale_cols = ['tenure', 'MonthlyCharges',
                      'TotalCharges', 'AvgChargesPerMonth']
        existing = [c for c in scale_cols if c in row.columns]
        row[existing] = scaler.transform(row[existing])

        # ── Predict ────────────────────────────
        best_model = results[best_name]['model']
        proba      = round(
            best_model.predict_proba(row)[0][1] * 100, 1
        )

        # ── Show Result ────────────────────────
        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                label = "Churn Probability",
                value = f"{proba}%"
            )
            st.progress(int(proba))

        with r2:
            if proba >= 70:
                st.error("🔴 CRITICAL RISK")
                st.error("Very likely to leave!")
            elif proba >= 50:
                st.warning("🟠 HIGH RISK")
                st.warning("At risk — act soon!")
            elif proba >= 30:
                st.warning("🟡 MEDIUM RISK")
                st.warning("Monitor this customer.")
            else:
                st.success("🟢 LOW RISK")
                st.success("Customer looks healthy!")

        with r3:
            st.markdown("**Recommended Action:**")
            if proba >= 70:
                st.markdown("📞 Call customer immediately")
                st.markdown("🎁 Offer loyalty discount")
                st.markdown("📑 Propose annual contract")
            elif proba >= 50:
                st.markdown("📧 Send retention email")
                st.markdown("📦 Offer bundle deal")
            elif proba >= 30:
                st.markdown("📬 Add to email campaign")
                st.markdown("🏆 Enroll in loyalty program")
            else:
                st.markdown("🔗 Invite to referral program")
                st.markdown("⬆️ Upsell premium plan")

# ── Footer ────────────────────────────────────
st.markdown("---")
st.caption(
    "Customer Churn Prediction | "
    "IBM Telco Dataset | "
    "Built with Streamlit + Scikit-learn"
)