import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
column_names = [
    "buying", "maint", "doors",
    "persons", "lug_boot", "safety",
    "class"
]
df = pd.read_csv("data.csv", names=column_names)
X = df.drop("class", axis=1)
y = df["class"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ]
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression – Car Evaluation Dataset")
print("-------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))   # Classification report WITHOUT support
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report (Precision, Recall, F1-score only):")
for label, metrics in report.items():
    if label not in ["accuracy", "macro avg", "weighted avg"]:
        print(
            f"{label}: "
            f"Precision={metrics['precision']:.2f}, "
            f"Recall={metrics['recall']:.2f}, "
            f"F1-score={metrics['f1-score']:.2f}"
        )
print(f"\nOverall Accuracy: {report['accuracy']:.2f}")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
precision = report["macro avg"]["precision"]
recall = report["macro avg"]["recall"]
f1_score = report["macro avg"]["f1-score"]
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1_score]
plt.figure(figsize=(7, 5))
plt.bar(metrics, values, color=["skyblue", "lightgreen", "orange", "pink"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Performance Metrics")
plt.tight_layout()
plt.show()


