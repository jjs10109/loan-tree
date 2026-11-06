import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def report(name, model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(name)
    print(f"TN: {tn:.3f}")
    print(f"FP: {fp:.3f}")
    print(f"FN: {fn:.3f}")
    print(f"TP: {tp:.3f}")
    print(f"accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"precision: {precision_score(y, y_pred):.3f}")
    print(f"recall: {recall_score(y, y_pred):.3f}")


def value(tn, fp, fn, tp):
    return 2000 * tn - 5000 * fn


def should_act(applicant):
    row = pd.DataFrame([applicant], columns=X_train.columns)
    probabilities = v2.predict_proba(row)[0]
    probability = probabilities[list(v2.classes_).index(1)]
    decision = "decline" if probability > 2000 / 7000 else "approve"
    return decision, probability


df = pd.read_csv("loan_applications.csv")
y = df["default"]
X = df.drop(columns="default")

features = ["credit_score", "late_payments_12mo"]
X_v1 = X[features]

v1 = DecisionTreeClassifier(max_depth=3, random_state=42)
v1.fit(X_v1, y)

print(export_text(v1, feature_names=features))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    v1,
    filled=True,
    feature_names=features,
    class_names=["no default", "default"],
    fontsize=9,
    ax=ax,
)
fig.savefig("tree_v1.png", bbox_inches="tight")
plt.close(fig)

report("v1", v1, X_v1, y)

v2 = DecisionTreeClassifier(max_depth=3, random_state=42)
v2.fit(X, y)

print(export_text(v2, feature_names=list(X.columns)))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    v2,
    filled=True,
    feature_names=list(X.columns),
    class_names=["no default", "default"],
    fontsize=9,
    ax=ax,
)
fig.savefig("tree_v2.png", bbox_inches="tight")
plt.close(fig)

report("v2", v2, X, y)

v3 = DecisionTreeClassifier(random_state=42)
v3.fit(X, y)

report("v3", v3, X, y)
print(v3.get_depth())
print(v3.get_n_leaves())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

v1 = DecisionTreeClassifier(max_depth=3, random_state=42)
v1.fit(X_train[features], y_train)
report("v1 train", v1, X_train[features], y_train)
report("v1 test", v1, X_test[features], y_test)

v2 = DecisionTreeClassifier(max_depth=3, random_state=42)
v2.fit(X_train, y_train)
print(export_text(v2, feature_names=list(X.columns)))
report("v2 train", v2, X_train, y_train)
report("v2 test", v2, X_test, y_test)

v3 = DecisionTreeClassifier(random_state=42)
v3.fit(X_train, y_train)
report("v3 train", v3, X_train, y_train)
report("v3 test", v3, X_test, y_test)

majority = y_test.mode()[0]
print(f"test majority-class accuracy: {(y_test == majority).mean():.3f}")

predictions = {
    "baseline": y_test * 0,
    "v1": v1.predict(X_test[features]),
    "v2": v2.predict(X_test),
    "v3": v3.predict(X_test),
}
for name, y_pred in predictions.items():
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    total = value(tn, fp, fn, tp)
    print(f"{name} total value: {total:.3f}")
    print(f"{name} value per applicant: {total / 120:.3f}")

applicants = {
    "A": {
        "credit_score": 655,
        "late_payments_12mo": 2,
        "annual_income": 48000,
        "loan_amount": 18000,
        "employment_years": 3.0,
        "debt_to_income_pct": 36.0,
        "age": 29,
        "homeowner": 0,
    },
    "B": {
        "credit_score": 735,
        "late_payments_12mo": 0,
        "annual_income": 95000,
        "loan_amount": 12000,
        "employment_years": 8.0,
        "debt_to_income_pct": 22.0,
        "age": 44,
        "homeowner": 1,
    },
    "C": {
        "credit_score": 768,
        "late_payments_12mo": 1,
        "annual_income": 55000,
        "loan_amount": 9000,
        "employment_years": 10.0,
        "debt_to_income_pct": 29.0,
        "age": 43,
        "homeowner": 0,
    },
}
for name, applicant in applicants.items():
    decision, probability = should_act(applicant)
    print(f"{name} probability: {probability:.3f}")
    print(f"{name} decision: {decision}")
