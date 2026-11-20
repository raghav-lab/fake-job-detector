import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Read dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Replace missing values
df.fillna("", inplace=True)

# Combine important text columns
df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Features and Labels
X = df["text"]
y = df["fraudulent"]

print(f"Total Jobs : {len(df)}")

print("\nReal Jobs :", (y == 0).sum())
print("Fake Jobs :", (y == 1).sum())

print("\nCreating TF-IDF Features...")

# Convert text to vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

print("TF-IDF Completed!")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Prediction
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/fake_job_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")