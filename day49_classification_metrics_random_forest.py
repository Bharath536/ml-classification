import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Create Customer Churn Dataset
# ==========================================

data = {
    "age": [
        22, 25, 28, 30, 32, 35, 38, 40, 42, 45,
        23, 27, 31, 34, 37, 41, 44, 48, 50, 52,
        24, 29, 33, 36, 39, 43, 46, 49, 51, 55,
        26, 30, 35, 40, 45, 50, 54, 58, 60, 62,
        21, 28, 32, 38, 43, 47, 53, 57, 61, 65
    ],
    "monthly_charges": [
        25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
        28, 33, 38, 43, 48, 53, 58, 63, 68, 73,
        27, 37, 42, 47, 52, 57, 62, 67, 72, 77,
        32, 39, 46, 54, 61, 69, 75, 80, 85, 90,
        24, 36, 44, 51, 59, 66, 74, 81, 88, 95
    ],
    "tenure_months": [
        2, 3, 5, 6, 8, 10, 12, 15, 18, 20,
        4, 6, 7, 9, 11, 13, 16, 19, 21, 24,
        3, 5, 8, 10, 14, 17, 20, 22, 25, 27,
        2, 4, 7, 9, 12, 15, 18, 21, 23, 26,
        3, 6, 9, 13, 16, 19, 22, 24, 28, 30
    ],
    "support_calls": [
        5, 4, 5, 3, 4, 3, 2, 2, 1, 1,
        6, 5, 4, 5, 3, 2, 3, 1, 2, 1,
        7, 5, 4, 3, 4, 2, 2, 1, 1, 0,
        6, 5, 4, 3, 2, 2, 1, 1, 0, 0,
        7, 6, 5, 4, 3, 2, 1, 1, 0, 0
    ],
    "churn": [
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["churn"].value_counts())


# ==========================================
# 2. Features and Target
# ==========================================

X = df.drop("churn", axis=1)
y = df["churn"]


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Decision Tree
# ==========================================

decision_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)


# ==========================================
# 5. Decision Tree Metrics
# ==========================================

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, zero_division=0)
dt_recall = recall_score(y_test, dt_predictions, zero_division=0)
dt_f1 = f1_score(y_test, dt_predictions, zero_division=0)

print("\n===================================")
print("DECISION TREE METRICS")
print("===================================")

print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1-Score  : {dt_f1:.4f}")


# ==========================================
# 6. Decision Tree Confusion Matrix
# ==========================================

dt_cm = confusion_matrix(y_test, dt_predictions)

print("\nDecision Tree Confusion Matrix:")
print(dt_cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "decision_tree_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. Random Forest
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)


# ==========================================
# 8. Random Forest Metrics
# ==========================================

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, zero_division=0)
rf_recall = recall_score(y_test, rf_predictions, zero_division=0)
rf_f1 = f1_score(y_test, rf_predictions, zero_division=0)

print("\n===================================")
print("RANDOM FOREST METRICS")
print("===================================")

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1-Score  : {rf_f1:.4f}")


# ==========================================
# 9. Random Forest Confusion Matrix
# ==========================================

rf_cm = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 10. Classification Report
# ==========================================

print("\n===================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ==========================================
# 11. Feature Importances
# ==========================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n===================================")
print("TOP 5 IMPORTANT FEATURES")
print("===================================")

print(feature_importance.head(5))

feature_importance.to_csv(
    "feature_importances.csv",
    index=False
)


# ==========================================
# 12. Feature Importance Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# 13. Tune n_estimators
# ==========================================

n_estimators_values = [50, 100, 200]

rf_results = []

for n in n_estimators_values:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    rf_results.append({
        "n_estimators": n,
        "accuracy": accuracy
    })


results_df = pd.DataFrame(rf_results)

print("\n===================================")
print("N_ESTIMATORS EXPERIMENT")
print("===================================")

print(results_df)

results_df.to_csv(
    "random_forest_n_estimators.csv",
    index=False
)


# ==========================================
# 14. n_estimators vs Accuracy Plot
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    results_df["n_estimators"],
    results_df["accuracy"],
    marker="o"
)

plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")

plt.title(
    "Random Forest: n_estimators vs Accuracy"
)

plt.xticks(n_estimators_values)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "n_estimators_vs_accuracy.png",
    dpi=300
)

plt.close()


# ==========================================
# 15. Model Comparison
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        dt_precision,
        rf_precision
    ],
    "Recall": [
        dt_recall,
        rf_recall
    ],
    "F1-Score": [
        dt_f1,
        rf_f1
    ]
})

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(comparison)

comparison.to_csv(
    "model_comparison_day49.csv",
    index=False
)


# ==========================================
# 16. Completion
# ==========================================

print("\n===================================")
print("DAY 49 COMPLETED SUCCESSFULLY!")
print("===================================")