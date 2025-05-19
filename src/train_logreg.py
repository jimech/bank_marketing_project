import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score, precision_recall_curve
)

# 1. Load cleaned dataset
df = pd.read_csv("data/bank-clean.csv")

# 2. Load preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# 3. Split into features and target
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# 4. Train/test split with stratification (preserves class ratio)
# stratify = y ensures that the 11% ' yes' RAttio is preserved in both train and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5 Transform features using the saved preprocessor
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 6 Train the baseline logistic regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_transformed, y_train)

# 7. Predict on test set
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]

# 8. Metrics to evalute the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
apr = average_precision_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# 9. PR Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(4, 3))
plt.plot(rec_curve, prec_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("LogReg Precision-Recall curve")
plt.savefig("results/pr_curve_logreg.png", dpi=120, bbox_inches="tight")
plt.close()

# 10. Markdown report for slides later
metrics_md = (
    "# Baseline Logistic Regression\n\n"
    f"**Class balance**\n\n"
    f"- Train positives: {y_train.mean():.3%}\n"
    f"- Test  positives: {y_test.mean():.3%}\n\n"
    "## Metrics (test set)\n\n"
    "| Metric | Value |\n"
    "| ------ | ----- |\n"
    f"| Accuracy | {accuracy:.3f} |\n"
    f"| Precision | {precision:.3f} |\n"
    f"| Recall | {recall:.3f} |\n"
    f"| F1 | {f1:.3f} |\n"
    f"| ROC-AUC | {roc_auc:.3f} |\n"
    f"| PR-AUC | {apr:.3f} |\n\n"
    "![PR curve](../results/pr_curve_logreg.png)\n\n"
    "### Confusion matrix\n\n"
    f"```\n{conf_matrix}\n```"
)
with open("docs/baseline_metrics.md", "w") as f:
    f.write(metrics_md)

# 11. Save model
joblib.dump(model, "models/logreg.pkl")
print("✅ Trained model saved to 'models/logreg.pkl'")
