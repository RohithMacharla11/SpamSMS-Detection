from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
print("Libraries Loaded")

# Load Trained Model & Vectorizer
with open("models/rf_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)
print("Models Loaded")

# Flask App
app = Flask(__name__)

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# API Endpoint for Prediction
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Message is required"}), 400

        text = data["message"]
        cleaned_text = clean_text(text)
        text_vectorized = loaded_vectorizer.transform([cleaned_text])
        prediction = loaded_model.predict(text_vectorized)[0]

        result = "Spam" if prediction == 1 else "Ham"
        return jsonify({"message": text, "prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
