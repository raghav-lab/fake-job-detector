import re

SUSPICIOUS_KEYWORDS = [
    "earn money",
    "quick money",
    "work from home",
    "no experience",
    "urgent",
    "whatsapp",
    "telegram",
    "investment",
    "click here",
    "easy income",
    "weekly payment",
    "limited seats",
]


def analyze_job(text):

    text_lower = text.lower()

    indicators = []

    score = 0

    # -----------------------
    # Keyword Detection
    # -----------------------
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in text_lower:
            indicators.append(f"Keyword detected: {keyword}")
            score += 8

    # -----------------------
    # Too many exclamation marks
    # -----------------------
    if text.count("!") >= 3:
        indicators.append("Too many exclamation marks")
        score += 5

    # -----------------------
    # Unrealistic salary
    # -----------------------
    salary_pattern = r"\₹?\s?\d{5,}"

    if re.search(salary_pattern, text):
        indicators.append("Very high salary mentioned")
        score += 10

    # -----------------------
    # Short Description
    # -----------------------
    if len(text) < 120:
        indicators.append("Very short job description")
        score += 5

    return score, indicators


def calculate_risk_score(probability, indicators):

    score = probability * 100

    score += len(indicators) * 4

    score = min(score, 100)

    return round(score, 2)


def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 70:
        return "MEDIUM"

    return "HIGH"


def get_safety_tips():

    return [
        "Verify company website.",
        "Never pay registration fees.",
        "Verify recruiter email.",
        "Check company on LinkedIn.",
        "Do not share personal documents immediately."
    ]