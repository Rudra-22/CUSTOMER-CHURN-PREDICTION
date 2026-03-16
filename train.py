# =============================================
#  CHUNK 4 — TRAIN ML MODELS
#  We train 4 different models and compare
# =============================================

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pickle

# Step 2: Load processed data
df = pd.read_csv('processed_data.csv')

X = df.drop(columns=['Churn'])
y = df['Churn']

print("=" * 40)
print("  STEP 4 — TRAINING MODELS")
print("=" * 40)
print(f"\n Total Samples : {len(X):,}")

# =============================================
#  PART A — TRAIN TEST SPLIT
# =============================================
print("\n Splitting data into Train and Test...")

# 80% for training, 20% for testing
# stratify=y means both sets have same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.2,       # 20% for testing
    random_state= 42,        # same split every time
    stratify    = y          # keep churn ratio same
)

print(f" Training Samples : {len(X_train):,} (80%)")
print(f" Testing Samples  : {len(X_test):,}  (20%)")

# =============================================
#  PART B — DEFINE MODELS
# =============================================
print("\n Defining 4 Models...")

models = {

    # Model 1: Logistic Regression
    # Simple linear model — fast and easy to understand
    'Logistic Regression': LogisticRegression(
        max_iter     = 1000,
        random_state = 42,
        class_weight = 'balanced'   # handles churn imbalance
    ),

    # Model 2: Decision Tree
    # Makes decisions like a flowchart
    'Decision Tree': DecisionTreeClassifier(
        max_depth    = 8,           # don't go too deep
        random_state = 42,
        class_weight = 'balanced'
    ),

    # Model 3: Random Forest
    # Many decision trees voting together
    'Random Forest': RandomForestClassifier(
        n_estimators = 100,         # 100 trees
        random_state = 42,
        class_weight = 'balanced',
        n_jobs       = -1           # use all CPU cores
    ),

    # Model 4: Gradient Boosting
    # Learns from mistakes step by step
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators  = 100,
        learning_rate = 0.05,
        max_depth     = 4,
        random_state  = 42
    ),
}

# =============================================
#  PART C — TRAIN EACH MODEL
# =============================================
print("\n Training Models...\n")

results = {}

for name, model in models.items():

    print(f" Training {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred  = model.predict(X_test)           # 0 or 1
    y_proba = model.predict_proba(X_test)[:, 1]  # probability

    # Calculate scores
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    auc      = round(roc_auc_score(y_test, y_proba), 4)
    f1       = round(f1_score(y_test, y_pred), 4)

    # Save results
    results[name] = {
        'model':    model,
        'accuracy': accuracy,
        'auc':      auc,
        'f1':       f1,
        'y_pred':   y_pred,
        'y_proba':  y_proba,
    }

    print(f"   Accuracy : {accuracy}%")
    print(f"   AUC      : {auc}")
    print(f"   F1 Score : {f1}")
    print()

# =============================================
#  PART D — COMPARE MODELS
# =============================================
print("=" * 40)
print("  MODEL COMPARISON")
print("=" * 40)

print(f"\n {'Model':<25} {'Accuracy':>10} {'AUC':>8} {'F1':>8}")
print("-" * 55)
for name, res in results.items():
    print(f" {name:<25} {res['accuracy']:>9}% {res['auc']:>8} {res['f1']:>8}")

# Find best model by AUC score
best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n Best Model : {best_name}")
print(f" Best AUC   : {results[best_name]['auc']}")

# =============================================
#  SAVE MODELS
# =============================================
print("\n Saving models...")

# Save all results
pickle.dump(results,   open('all_results.pkl',  'wb'))
pickle.dump(best_name, open('best_name.pkl',    'wb'))
pickle.dump(X_test,    open('X_test.pkl',       'wb'))
pickle.dump(y_test,    open('y_test.pkl',       'wb'))
pickle.dump(X.columns.tolist(), open('feature_columns.pkl', 'wb'))

print(" Saved: all_results.pkl")
print(" Saved: best_name.pkl")
print("\n Training Done! Run 5_evaluate.py next")