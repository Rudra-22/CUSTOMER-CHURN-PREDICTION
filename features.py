# =============================================
#  CHUNK 3 — FEATURE ENGINEERING
#  We create new useful columns and
#  prepare data so models can understand it
# =============================================

print("Starting features.py")

# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Step 2: Load clean data
df = pd.read_csv('clean_data.csv')
print("Data loaded! Now engineering features...")

# =============================================
#  PART A — CREATE NEW FEATURES
# =============================================
print("\n PART A: Creating New Features...")

# New Feature 1: Average charges per month
# Why? Tells us if customer is paying more than usual
df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
print(" Created: AvgChargesPerMonth")

# New Feature 2: Number of services customer uses
# Why? More services = less likely to churn
service_columns = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df['NumServices'] = df[service_columns].apply(
    lambda row: sum(v == 'Yes' for v in row), axis=1
)
print(" Created: NumServices")

# New Feature 3: Contract risk score
# Why? Month-to-month = high risk, Two year = low risk
contract_map = {
    'Month-to-month': 3,   # High risk
    'One year':        2,   # Medium risk
    'Two year':        1    # Low risk
}
df['ContractRisk'] = df['Contract'].map(contract_map)
print(" Created: ContractRisk")

print("\n New Features Preview:")
print(df[['AvgChargesPerMonth', 'NumServices', 'ContractRisk']].head())

# =============================================
#  PART B — ENCODE COLUMNS
# =============================================
print("\n PART B: Encoding Columns...")

# Remove customerID — not useful for prediction
df = df.drop(columns=['customerID'])

# Binary Encode: Convert Yes/No to 1/0
# Why? Models need numbers not text
binary_columns = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'PaperlessBilling', 'Churn'
]
for col in binary_columns:
    df[col] = LabelEncoder().fit_transform(df[col])
    print(f" Binary Encoded: {col}")

# One-Hot Encode: Create separate columns for each category
# Why? So model doesn't think Fiber > DSL > No (they are just different)
multi_columns = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]
df = pd.get_dummies(df, columns=multi_columns, drop_first=True)
print(f" One-Hot Encoded: {len(multi_columns)} columns")

# =============================================
#  PART C — SCALE NUMBERS
# =============================================
print("\n PART C: Scaling Numbers...")

# StandardScaler: Makes all numbers on same scale
# Why? So tenure(72) doesn't overpower gender(0/1)
scaler = StandardScaler()
scale_columns = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth'
]
df[scale_columns] = scaler.fit_transform(df[scale_columns])
print(f" Scaled: {scale_columns}")

# =============================================
#  PART D — SPLIT INTO X and y
# =============================================
print("\n PART D: Splitting Features and Target...")

# X = input features (what model learns from)
# y = target (what model predicts)
X = df.drop(columns=['Churn'])
y = df['Churn']

print(f"\n X shape (features)   : {X.shape}")
print(f" y shape (target)     : {y.shape}")
print(f" Total Features       : {X.shape[1]}")
print(f" Churn = 1 (Yes)      : {y.sum():,}")
print(f" Churn = 0 (No)       : {(y == 0).sum():,}")

# =============================================
#  SAVE FOR NEXT STEP
# =============================================
df.to_csv('processed_data.csv', index=False)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("\n Saved: processed_data.csv")
print(" Saved: scaler.pkl")
print("\n Feature Engineering Done! Run 4_train.py next")