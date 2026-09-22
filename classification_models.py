import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Create a sample customer churn dataset
# --------------------------------------------------

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

print("\nDataset shape:")
print(df.shape)

print("\nChurn distribution:")
print(df["churn"].value_counts())


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------

X = df.drop("churn", axis=1)
y = df["churn"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Logistic Regression
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(random_state=42)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Accuracy:")
print(f"{logistic_accuracy:.2%}")


# --------------------------------------------------
# 5. Decision Tree Classifier
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy:.2%}")


# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        logistic_accuracy,
        tree_accuracy
    ]
})

print("\nModel Comparison:")
print(comparison)


# --------------------------------------------------
# 7. Visualize Decision Tree
# --------------------------------------------------

plt.figure(figsize=(16, 9))

plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Stay", "Churn"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Classifier - Max Depth 3")
plt.tight_layout()

plt.savefig("decision_tree.png")

plt.show()


# --------------------------------------------------
# 8. Max Depth Experiment
# --------------------------------------------------

depth_results = []

for depth in [3, 5, 10]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    depth_results.append({
        "max_depth": depth,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })


depth_df = pd.DataFrame(depth_results)

print("\nMax Depth Experiment:")
print(depth_df.to_string(index=False))


# --------------------------------------------------
# 9. Save experiment results
# --------------------------------------------------

depth_df.to_csv(
    "max_depth_experiment.csv",
    index=False
)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nFiles saved successfully.")