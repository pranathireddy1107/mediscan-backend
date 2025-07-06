from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Better dataset
symptoms = [
    "fever, cold, cough",              # Flu
    "chest pain, shortness of breath", # Heart Attack
    "headache, dizziness",             # Migraine
    "sneezing, runny nose, itchy eyes",# Allergy
    "vomiting, nausea, stomach ache",  # Food Poisoning
    "joint pain, stiffness",           # Arthritis
    "back pain, leg numbness",         # Sciatica
    "irregular heartbeat, heart pain", # Heart Issue
    "sore throat, cough, fatigue",     # Flu
    "rash, fever, red spots",          # Measles
]

labels = [
    "Flu",
    "Heart Attack",
    "Migraine",
    "Allergy",
    "Food Poisoning",
    "Arthritis",
    "Sciatica",
    "Heart Issue",
    "Flu",
    "Measles"
]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("✅ Model retrained and saved as model.pkl")
