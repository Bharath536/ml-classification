import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# DAY 52 - FEATURE SCALING, KNN & SVM
# ============================================================

print("=" * 60)
print("DAY 52 - FEATURE SCALING, KNN & SVM")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("feature_engineered_dataset.csv")

print("\nDataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

# Remove target column and text-based categorical column
X = df.drop(
    ["churn", "age_group"],
    axis=1
)

y = df["churn"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ============================================================
# 4. STANDARD SCALER
# ============================================================

standard_scaler = StandardScaler()

# Fit ONLY on training data
X_train_standard = standard_scaler.fit_transform(X_train)

# Transform test data using the already fitted scaler
X_test_standard = standard_scaler.transform(X_test)

print("\nStandardScaler applied successfully.")
print("Scaler fitted only on training data.")


# ============================================================
# 5. MINMAX SCALER
# ============================================================

minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

print("MinMaxScaler applied successfully.")


# ============================================================
# 6. SHOW SCALING EXAMPLES
# ============================================================

print("\nOriginal Training Data:")
print(X_train.head())

print("\nStandardScaler Data:")
print(
    pd.DataFrame(
        X_train_standard,
        columns=X.columns
    ).head()
)

print("\nMinMaxScaler Data:")
print(
    pd.DataFrame(
        X_train_minmax,
        columns=X.columns
    ).head()
)


# ============================================================
# 7. KNN WITHOUT SCALING
# ============================================================

print("\n" + "=" * 60)
print("KNN WITHOUT SCALING")
print("=" * 60)

knn_without_scaling_results = []

for k in [1, 5, 10, 20]:

    start_time = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train,
        y_train
    )

    predictions = knn.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    training_time = time.time() - start_time

    knn_without_scaling_results.append({
        "k": k,
        "accuracy": accuracy,
        "training_time": training_time
    })

    print(
        f"k={k:<2} | "
        f"Accuracy={accuracy:.4f} | "
        f"Time={training_time:.6f}s"
    )


# ============================================================
# 8. KNN WITH STANDARD SCALING
# ============================================================

print("\n" + "=" * 60)
print("KNN WITH STANDARD SCALING")
print("=" * 60)

knn_scaled_results = []

for k in [1, 5, 10, 20]:

    start_time = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train_standard,
        y_train
    )

    predictions = knn.predict(
        X_test_standard
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    training_time = time.time() - start_time

    knn_scaled_results.append({
        "k": k,
        "accuracy": accuracy,
        "training_time": training_time
    })

    print(
        f"k={k:<2} | "
        f"Accuracy={accuracy:.4f} | "
        f"Time={training_time:.6f}s"
    )


# ============================================================
# 9. KNN ACCURACY COMPARISON GRAPH
# ============================================================

without_scaling_df = pd.DataFrame(
    knn_without_scaling_results
)

scaled_knn_df = pd.DataFrame(
    knn_scaled_results
)

plt.figure(figsize=(8, 5))

plt.plot(
    without_scaling_df["k"],
    without_scaling_df["accuracy"],
    marker="o",
    label="Without Scaling"
)

plt.plot(
    scaled_knn_df["k"],
    scaled_knn_df["accuracy"],
    marker="o",
    label="StandardScaler"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy: With vs Without Scaling")
plt.xticks([1, 5, 10, 20])
plt.legend()
plt.grid(True)

plt.savefig(
    "knn_scaling_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved: knn_scaling_comparison.png")


# ============================================================
# 10. SVM - LINEAR KERNEL
# ============================================================

print("\n" + "=" * 60)
print("SVM - LINEAR KERNEL")
print("=" * 60)

start_time = time.time()

svm_linear = SVC(
    kernel="linear"
)

svm_linear.fit(
    X_train_standard,
    y_train
)

svm_linear_predictions = svm_linear.predict(
    X_test_standard
)

svm_linear_time = time.time() - start_time

print(
    "Accuracy:",
    f"{accuracy_score(y_test, svm_linear_predictions):.4f}"
)

print(
    "Training Time:",
    f"{svm_linear_time:.6f}s"
)


# ============================================================
# 11. SVM - RBF KERNEL
# ============================================================

print("\n" + "=" * 60)
print("SVM - RBF KERNEL")
print("=" * 60)

start_time = time.time()

svm_rbf = SVC(
    kernel="rbf"
)

svm_rbf.fit(
    X_train_standard,
    y_train
)

svm_rbf_predictions = svm_rbf.predict(
    X_test_standard
)

svm_rbf_time = time.time() - start_time

print(
    "Accuracy:",
    f"{accuracy_score(y_test, svm_rbf_predictions):.4f}"
)

print(
    "Training Time:",
    f"{svm_rbf_time:.6f}s"
)


# ============================================================
# 12. FIVE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FIVE MODEL COMPARISON")
print("=" * 60)


models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000
        ),
        X_train_standard,
        X_test_standard
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        X_train,
        X_test
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        X_train,
        X_test
    ),

    "KNN": (
        KNeighborsClassifier(
            n_neighbors=5
        ),
        X_train_standard,
        X_test_standard
    ),

    "SVM": (
        SVC(
            kernel="rbf"
        ),
        X_train_standard,
        X_test_standard
    )
}


results = []


# ============================================================
# 13. TRAIN AND EVALUATE ALL MODELS
# ============================================================

for model_name, (
    model,
    train_data,
    test_data
) in models.items():

    start_time = time.time()

    model.fit(
        train_data,
        y_train
    )

    predictions = model.predict(
        test_data
    )

    training_time = time.time() - start_time

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "Training Time (s)": training_time
    })


# ============================================================
# 14. MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    results
)

print("\n")
print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE MODEL COMPARISON
# ============================================================

comparison_df.to_csv(
    "day52_model_comparison.csv",
    index=False
)

print(
    "\nSaved: day52_model_comparison.csv"
)


# ============================================================
# 16. SAVE KNN RESULTS
# ============================================================

knn_comparison_df = pd.DataFrame({

    "k": [1, 5, 10, 20],

    "Without Scaling Accuracy":
        without_scaling_df["accuracy"].values,

    "StandardScaler Accuracy":
        scaled_knn_df["accuracy"].values
})

knn_comparison_df.to_csv(
    "day52_knn_comparison.csv",
    index=False
)

print(
    "Saved: day52_knn_comparison.csv"
)


# ============================================================
# 17. BEST MODEL BY ACCURACY
# ============================================================

best_accuracy_row = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("HIGHEST ACCURACY MODEL")
print("=" * 60)

print(
    "Model:",
    best_accuracy_row["Model"]
)

print(
    "Accuracy:",
    f"{best_accuracy_row['Accuracy']:.4f}"
)


# ============================================================
# 18. CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
Feature scaling is important for algorithms that depend
on distances or feature margins.

KNN was tested with k values of 1, 5, 10 and 20,
both with and without StandardScaler.

SVM was tested using both linear and RBF kernels.

The five classification models compared were:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. KNN
5. SVM

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Training Time

StandardScaler was fitted only on the training data
to prevent data leakage.

Model selection should consider all evaluation metrics
and the requirements of the specific machine learning
problem rather than accuracy alone.
""")


print("\n" + "=" * 60)
print("DAY 52 COMPLETED SUCCESSFULLY!")
print("=" * 60)