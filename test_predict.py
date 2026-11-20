from src.predictor import predict_job

text = """
Work from home.
Earn ₹50,000 every week.
No experience required.
WhatsApp us now.
"""

prediction, probability = predict_job(text)

print("Prediction :", prediction)
print("Probability :", probability)