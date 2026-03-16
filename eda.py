# =============================================
#  CHUNK 2 — EXPLORATORY DATA ANALYSIS (EDA)
#  Run this to see charts about the data
# =============================================

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the clean data from chunk 1
df = pd.read_csv('clean_data.csv')
print("Data loaded! Now making charts...")

# ── CHART 1: Churn Count ─────────────────────
print("\n Making Chart 1 — Churn Count...")

churn_count = df['Churn'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(['Retained', 'Churned'],
        churn_count.values,
        color=['green', 'red'],
        width=0.4)
plt.title('How Many Customers Churned?')
plt.ylabel('Number of Customers')
for i, v in enumerate(churn_count.values):
    plt.text(i, v + 30, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('chart1_churn_count.png')
plt.show()
print(" Saved chart1_churn_count.png")

# ── CHART 2: Tenure ──────────────────────────
print("\n Making Chart 2 — Tenure Distribution...")

plt.figure(figsize=(7, 4))
plt.hist(df[df['Churn'] == 'Yes']['tenure'],
         bins=20, alpha=0.7, color='red', label='Churned')
plt.hist(df[df['Churn'] == 'No']['tenure'],
         bins=20, alpha=0.7, color='green', label='Retained')
plt.title('How Long Were Customers With Us?')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('chart2_tenure.png')
plt.show()
print(" Saved chart2_tenure.png")

# ── CHART 3: Monthly Charges ─────────────────
print("\n Making Chart 3 — Monthly Charges...")

plt.figure(figsize=(6, 4))
plt.boxplot(
    [df[df['Churn'] == 'No']['MonthlyCharges'],
     df[df['Churn'] == 'Yes']['MonthlyCharges']],
    labels=['Retained', 'Churned'],
    patch_artist=True
)
plt.title('Monthly Charges — Churned vs Retained')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig('chart3_monthly_charges.png')
plt.show()
print(" Saved chart3_monthly_charges.png")

# ── CHART 4: Contract Type ───────────────────
print("\n Making Chart 4 — Contract Type...")

contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)

plt.figure(figsize=(6, 4))
plt.bar(contract_churn.index,
        contract_churn.values,
        color=['red', 'orange', 'green'],
        width=0.4)
plt.title('Churn Rate by Contract Type (%)')
plt.ylabel('Churn Rate (%)')
for i, v in enumerate(contract_churn.values):
    plt.text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('chart4_contract.png')
plt.show()
print(" Saved chart4_contract.png")

# ── CHART 5: Internet Service ────────────────
print("\n Making Chart 5 — Internet Service...")

internet_churn = df.groupby('InternetService')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)

plt.figure(figsize=(6, 4))
plt.bar(internet_churn.index,
        internet_churn.values,
        color=['blue', 'red', 'green'],
        width=0.4)
plt.title('Churn Rate by Internet Service (%)')
plt.ylabel('Churn Rate (%)')
for i, v in enumerate(internet_churn.values):
    plt.text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('chart5_internet.png')
plt.show()
print(" Saved chart5_internet.png")

# ── CHART 6: Payment Method ──────────────────
print("\n Making Chart 6 — Payment Method...")

payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)

plt.figure(figsize=(8, 4))
plt.barh(payment_churn.index,
         payment_churn.values,
         color='purple',
         height=0.4)
plt.title('Churn Rate by Payment Method (%)')
plt.xlabel('Churn Rate (%)')
for i, v in enumerate(payment_churn.values):
    plt.text(v + 0.3, i, f'{v}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('chart6_payment.png')
plt.show()
print(" Saved chart6_payment.png")

# ── Print Key Findings ───────────────────────
print("\n" + "=" * 40)
print("  KEY FINDINGS FROM EDA")
print("=" * 40)
print(f"\n  Churn Rate       : {(df['Churn']=='Yes').mean()*100:.1f}%")
print(f"  Month-to-Month   : {contract_churn['Month-to-month']}% churn")
print(f"  Two Year Contract: {contract_churn['Two year']}% churn")
print(f"  Fiber Optic      : {internet_churn['Fiber optic']}% churn")
print(f"  Elec. Check Pay  : {payment_churn['Electronic check']}% churn")
print("\n  EDA Complete! Run 3_features.py next")