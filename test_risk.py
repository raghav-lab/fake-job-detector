from src.predictor import predict_job
from src.risk_score import *

text = """
Work from home.

Earn money every week.

No experience required.

WhatsApp us now.

Limited seats.
"""

prediction, probability = predict_job(text)

keywords = find_suspicious_keywords(text)

score = calculate_risk_score(probability, keywords)

level = get_risk_level(score)

print("\nPrediction :", "Fake" if prediction == 1 else "Real")

print("Probability :", round(probability * 100, 2), "%")

print("Fraud Risk Score :", score)

print("Risk Level :", level)

print("\nSuspicious Keywords")

for k in keywords:
    print("•", k)

print("\nSafety Tips")

for tip in get_safety_tips():
    print("✓", tip)