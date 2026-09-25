import pandas as pd
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

df = pd.read_csv("feature_engineered_dataset.csv")

print("\nDataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

# age_group is a text/categorical column.
# We already have age_group_encoded, so we can remove age_group.

X = df.drop(["churn", "age_group"], axis=1)
y = df["churn"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. GRADIENT BOOSTING CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING CLASSIFIER")
print("=" * 70)

gb_model = GradientBoostingClassifier(
    random_state=42
)

start_time = time.time()

gb_model.fit(X_train, y_train)

gb_training_time = time.time() - start_time

gb_pred = gb_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print(f"Accuracy: {gb_accuracy:.4f}")
print(f"F1 Score: {gb_f1:.4f}")
print(f"Training Time: {gb_training_time:.4f} seconds")


# ============================================================
# 5. XGBOOST BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST BASE MODEL")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

start_time = time.time()

xgb_model.fit(X_train, y_train)

xgb_training_time = time.time() - start_time

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"F1 Score: {xgb_f1:.4f}")
print(f"Training Time: {xgb_training_time:.4f} seconds")


# ============================================================
# 6. GRIDSEARCHCV FOR XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST GRIDSEARCHCV")
print("=" * 70)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(
        max_depth=3,
        random_state=42,
        eval_metric="logloss"
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

start_time = time.time()

grid_search.fit(X_train, y_train)

grid_search_time = time.time() - start_time

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(f"{grid_search.best_score_:.4f}")

print(f"\nGridSearch Time: {grid_search_time:.4f} seconds")


# ============================================================
# 7. TUNED XGBOOST PERFORMANCE
# ============================================================

best_xgb = grid_search.best_estimator_

best_xgb_pred = best_xgb.predict(X_test)

tuned_xgb_accuracy = accuracy_score(
    y_test,
    best_xgb_pred
)

tuned_xgb_f1 = f1_score(
    y_test,
    best_xgb_pred
)

print("\nTuned XGBoost Test Performance:")
print(f"Accuracy: {tuned_xgb_accuracy:.4f}")
print(f"F1 Score: {tuned_xgb_f1:.4f}")


# ============================================================
# 8. BEFORE VS AFTER TUNING
# ============================================================

print("\n" + "=" * 70)
print("BEFORE VS AFTER HYPERPARAMETER TUNING")
print("=" * 70)

tuning_comparison = pd.DataFrame({
    "Model": [
        "XGBoost Before Tuning",
        "XGBoost After Tuning"
    ],
    "Accuracy": [
        round(xgb_accuracy, 4),
        round(tuned_xgb_accuracy, 4)
    ],
    "F1_Score": [
        round(xgb_f1, 4),
        round(tuned_xgb_f1, 4)
    ]
})

print(tuning_comparison.to_string(index=False))

tuning_comparison.to_csv(
    "day54_tuning_comparison.csv",
    index=False
)


# ============================================================
# 9. SIX MODEL LEADERBOARD
# ============================================================

print("\n" + "=" * 70)
print("6-MODEL LEADERBOARD")
print("=" * 70)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "XGBoost": best_xgb
}


results = []


for name, model in models.items():

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "F1_Score": round(f1, 4),
        "Training_Time": round(training_time, 4)
    })


# ============================================================
# 10. CREATE LEADERBOARD DATAFRAME
# ============================================================

leaderboard = pd.DataFrame(results)

leaderboard = leaderboard.sort_values(
    by="F1_Score",
    ascending=False
)

print("\nFinal Leaderboard:")
print(
    leaderboard.to_string(index=False)
)


# ============================================================
# 11. SAVE LEADERBOARD
# ============================================================

leaderboard.to_csv(
    "day54_model_leaderboard.csv",
    index=False
)

print("\nLeaderboard saved:")
print("day54_model_leaderboard.csv")


# ============================================================
# 12. SAVE GRIDSEARCH RESULTS
# ============================================================

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results = grid_results[
    [
        "param_n_estimators",
        "param_learning_rate",
        "mean_test_score",
        "rank_test_score"
    ]
]

grid_results.to_csv(
    "day54_gridsearch_results.csv",
    index=False
)

print("\nGridSearch results saved:")
print("day54_gridsearch_results.csv")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 54 SUMMARY")
print("=" * 70)

print("\nGradient Boosting:")
print(f"Accuracy: {gb_accuracy:.4f}")
print(f"F1 Score: {gb_f1:.4f}")

print("\nXGBoost Before Tuning:")
print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"F1 Score: {xgb_f1:.4f}")

print("\nXGBoost After Tuning:")
print(f"Accuracy: {tuned_xgb_accuracy:.4f}")
print(f"F1 Score: {tuned_xgb_f1:.4f}")

print("\nBest XGBoost Parameters:")
print(grid_search.best_params_)

print("\nAll output files created successfully.")