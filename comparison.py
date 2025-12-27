import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data.csv", names=column_names)
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)]
)
log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
log_prec = precision_score(y_test, log_pred, average="weighted")
log_rec = recall_score(y_test, log_pred, average="weighted")
log_f1 = f1_score(y_test, log_pred, average="weighted")
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average="weighted")
rf_rec = recall_score(y_test, rf_pred, average="weighted")
rf_f1 = f1_score(y_test, rf_pred, average="weighted")
print("MODEL COMPARISON – Car Evaluation Dataset")
print("\nLogistic Regression:")
print(f"Accuracy : {log_acc:.4f}")
print(f"Precision: {log_prec:.4f}")
print(f"Recall   : {log_rec:.4f}")
print(f"F1-score : {log_f1:.4f}")
print("\nRandom Forest:")
print(f"Accuracy : {rf_acc:.4f}")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall   : {rf_rec:.4f}")
print(f"F1-score : {rf_f1:.4f}")
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
log_values = [log_acc, log_prec, log_rec, log_f1]
rf_values = [rf_acc, rf_prec, rf_rec, rf_f1]
x = range(len(metrics))
plt.figure(figsize=(9, 6))
plt.bar(x, log_values, width=0.35, label="Logistic Regression")
plt.bar([i + 0.35 for i in x], rf_values, width=0.35, label="Random Forest")
plt.xticks([i + 0.17 for i in x], metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Logistic Regression vs Random Forest – Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
