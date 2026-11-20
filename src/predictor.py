import joblib

# Load trained model and vectorizer
model = joblib.load("models/fake_job_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_job(job_text):
    """
    Predict whether a job is Fake or Real.
    """

    # Convert text to TF-IDF
    vector = vectorizer.transform([job_text])

    # Prediction
    prediction = model.predict(vector)[0]

    # Probability
    probability = model.predict_proba(vector)[0][1]

    return prediction, probability