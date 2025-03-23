import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Load Model & Vectorizer
with open("models/rf_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to Predict Spam/Ham
def predict_sms(text):
    text_cleaned = preprocess_text(text)  # Error occurs here
    text_tfidf = vectorizer.transform([text_cleaned])
    prediction = classifier.predict(text_tfidf)
    return "Spam" if prediction == 1 else "Ham"

test_examples = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize reward!",
        "Hey, are we still meeting for coffee at 5 PM? Let me know!",
        "Congratulations! You've been selected for a free iPhone. Call now to claim your prize!",
        "Don't forget to pick up milk on your way home.",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
    ]
print("\nTesting example messages:")
print("-" * 60)
    
for i, message in enumerate(test_examples, 1):
    # Preprocess the message
    processed_message = preprocess_text(message)
        
    # Transform using the loaded vectorizer
    features = vectorizer.transform([processed_message])
        
    #Predict using the loaded classifier
    prediction = classifier.predict(features)[0]
        
        # Print results
    print(f"Example {i}:")
    print(f"Message: {message}")
    if prediction == 1:
        print(f"Classification: Spam")
    else:
        print(f"Classification: Ham")
    print("-" * 60)