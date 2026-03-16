# =============================================
#  CHUNK 5 — EVALUATE & VISUALIZE RESULTS
#  Final step — see how good our model is
# =============================================

# Step 1: Import libraries
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve
)

# Step 2: Load saved results
results   = pickle.load(open('all_results.pkl',  'rb'))
best_name = pickle.load(open('best_name.pkl',    'rb'))
X_test    = pickle.load(open('X_test.pkl',       'rb'))
y_test    = pickle.load(open('y_test.pkl',       'rb'))

best = results[best_name]

print("=" * 40)
print("  STEP 5 — MODEL EVALUATION")
print("=" * 40)

# =============================================
#  PART A — BEST MODEL RESULTS
# =============================================
print(f"\n Best Model  : {best_name}")
print(f" Accuracy    : {best['accuracy']}%")
print(f" AUC Score   : {best['auc']}")
print(f" F1 Score    : {best['f1']}")

# Classification Report
# Shows precision, recall, f1 for each class
print("\n" + "=" * 40)
print("  CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(
    y_test,
    best['y_pred'],
    target_names=['Retained', 'Churned']
))

# =============================================
#  PART B — CONFUSION MATRIX EXPLAINED
# =============================================
cm = confusion_matrix(y_test, best['y_pred'])
tn = cm[0][0]  # Correctly said Retained
fp = cm[0][1]  # Said Churn but was Retained (mistake)
fn = cm[1][0]  # Said Retained but was Churn (bad mistake!)
tp = cm[1][1]  # Correctly said Churned

print("=" * 40)
print("  CONFUSION MATRIX BREAKDOWN")
print("=" * 40)
print(f"\n True Negative  (TN): {tn}  → Correctly predicted Retained")
print(f" False Positive (FP): {fp}  → Said Churn, was actually Retained")
print(f" False Negative (FN): {fn}   → Missed real Churners ⚠️")
print(f" True Positive  (TP): {tp}  → Correctly predicted Churned ✅")

# =============================================
#  PART C — CHARTS
# =============================================
print("\n Making evaluation charts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Model Evaluation — Best Model: {best_name}',
             fontsize=13, fontweight='bold')

# ── Chart 1: AUC Score Comparison ────────────
names  = list(results.keys())
aucs   = [results[n]['auc'] for n in names]
colors = ['gold' if n == best_name else 'steelblue' for n in names]

bars = axes[0].bar(range(len(names)), aucs, color=colors, width=0.5)
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(['LR', 'DT', 'RF', 'GB'], fontsize=11)
axes[0].set_ylim(0.6, 1.0)
axes[0].set_title('AUC Score Comparison\n(Gold = Best Model)')
axes[0].set_ylabel('AUC Score')
for bar, val in zip(bars, aucs):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        str(val), ha='center', fontweight='bold', fontsize=10
    )

# ── Chart 2: ROC Curves ───────────────────────
chart_colors = ['green', 'blue', 'orange', 'red']
for (name, res), color in zip(results.items(), chart_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    lw = 3 if name == best_name else 1.5
    axes[1].plot(fpr, tpr, color=color, lw=lw,
                  label=f"{name} ({res['auc']})")
axes[1].plot([0, 1], [0, 1], '--', color='gray', lw=1)
axes[1].set_title('ROC Curves\n(Higher = Better)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=7)

# ── Chart 3: Confusion Matrix ────────────────
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    ax=axes[2],
    xticklabels=['Retained', 'Churned'],
    yticklabels=['Retained', 'Churned'],
    annot_kws={'size': 14, 'weight': 'bold'}
)
axes[2].set_title(f'Confusion Matrix\n{best_name}')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('chart_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: chart_evaluation.png")

# =============================================
#  PART D — FEATURE IMPORTANCE
# =============================================
print("\n Making feature importance chart...")

rf_model  = results['Random Forest']['model']
feat_imp  = pd.DataFrame({
    'Feature':    X_test.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(9, 5))
plt.barh(feat_imp['Feature'][::-1],
         feat_imp['Importance'][::-1],
         color='steelblue', height=0.6)
plt.xlabel('Importance Score')
plt.title('Top 10 Features That Predict Churn')
plt.tight_layout()
plt.savefig('chart_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: chart_feature_importance.png")

# =============================================
#  PART E — BUSINESS SUMMARY
# =============================================
df_raw = pd.read_csv('clean_data.csv')
probas = best['model'].predict_proba(
    pd.read_csv('processed_data.csv').drop(columns=['Churn'])
)[:, 1]

print("\n" + "=" * 40)
print("  BUSINESS SUMMARY")
print("=" * 40)
print(f"\n Total Customers     : {len(df_raw):,}")
print(f" Predicted Churners  : {int((probas>=0.5).sum()):,}")
print(f" Revenue at Risk     : ${(probas * df_raw['MonthlyCharges']).sum():,.0f}/mo")

print(f"\n Risk Segments:")
print(f"  Low    (<30%)  : {int((probas<0.3).sum()):,} customers")
print(f"  Medium (30-50%): {int(((probas>=0.3)&(probas<0.5)).sum()):,} customers")
print(f"  High   (50-70%): {int(((probas>=0.5)&(probas<0.7)).sum()):,} customers")
print(f"  Critical (>70%): {int((probas>=0.7).sum()):,} customers")

print("\n" + "=" * 40)
print("  PROJECT COMPLETE!")
print("=" * 40)
print(f"\n Best Model : {best_name}")
print(f" AUC Score  : {best['auc']}")
print(f" F1 Score   : {best['f1']}")
print(f" Accuracy   : {best['accuracy']}%")
print("\n All charts saved in your folder!")