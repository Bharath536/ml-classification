import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# 1. Load Feature Engineered Dataset
df = pd.read_csv("feature_engineered_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["churn"].value_counts())


# 2. Remove categorical text column
# age_group is already represented by age_group_encoded
X = df.drop(["churn", "age_group"], axis=1)
y = df["churn"]


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Normal Random Forest Pipeline
normal_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

normal_pipeline.fit(X_train, y_train)
normal_pred = normal_pipeline.predict(X_test)


# 5. Random Forest with class_weight='balanced'
balanced_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ))
])

balanced_pipeline.fit(X_train, y_train)
balanced_pred = balanced_pipeline.predict(X_test)


# 6. SMOTE + Random Forest Pipeline
smote_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

smote_pipeline.fit(X_train, y_train)
smote_pred = smote_pipeline.predict(X_test)


# 7. Evaluation Function
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0)
    }


# 8. Compare All Three Models
results = []

results.append(
    evaluate_model(
        "Normal Random Forest",
        y_test,
        normal_pred
    )
)

results.append(
    evaluate_model(
        "Random Forest - Balanced",
        y_test,
        balanced_pred
    )
)

results.append(
    evaluate_model(
        "SMOTE + Random Forest",
        y_test,
        smote_pred
    )
)


comparison = pd.DataFrame(results)


# 9. Display Results
print("\n" + "=" * 70)
print("DAY 53 MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# 10. Save Results
comparison.to_csv(
    "day53_imbalance_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("day53_imbalance_comparison.csv")


# 11. Explanation
print("\nWhy Accuracy Can Be Misleading:")

print(
    "In an imbalanced dataset, the majority class can dominate "
    "the accuracy score. A model may achieve high accuracy while "
    "performing poorly on the minority class. Precision, Recall, "
    "and especially F1-score provide a better view of minority "
    "class performance."
)


print("\nDay 53 completed successfully!")