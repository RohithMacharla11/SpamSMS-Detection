# SMS Spam Detection

## Overview
The **SMS Spam Detection** project is a machine learning-based web application that classifies SMS messages as **Spam** or **Ham** (Not Spam). The project utilizes various machine learning models to analyze text messages and predict whether they contain spam content. A Flask-based web interface allows users to input messages and receive real-time predictions.

## Features
- User-friendly web interface for SMS classification.
- Multiple machine learning models trained for spam detection.
- Uses **TF-IDF vectorization** for text preprocessing.
- Supports **Naïve Bayes, Logistic Regression, SVM, Random Forest, LSTM, and GRU** models.
- Visual representations of model performance.
- Interactive frontend with **HTML, CSS, and JavaScript**.
- Flask-powered backend for real-time predictions.

## Project Structure
```
SMS-Spam-Detection/
│── app.py                     # Main Flask application
│── training.py                 # Script for training ML models
│── testing.py                  # Script for testing trained models
│── SMS_SPAN_DETECTION.ipynb    # Jupyter Notebook for data analysis & model evaluation
│── requirements.txt            # Dependencies for the project
│── roadMap.txt                 # Project development plan
│  
├── data/  
│   ├── spam.csv                # Dataset containing spam/ham messages  
│  
├── models/                     # Pre-trained models & vectorizer  
│   ├── gru_model.keras         # GRU deep learning model  
│   ├── lstm_model.keras        # LSTM deep learning model  
│   ├── lr_model.pkl            # Logistic Regression model  
│   ├── nb_model.pkl            # Naïve Bayes model  
│   ├── rf_model.pkl            # Random Forest model  
│   ├── svm_model.pkl           # SVM model  
│   ├── processed_data.pkl      # Preprocessed dataset  
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer for text transformation  
│  
├── plots/                      # Model performance visualizations  
│   ├── Accuracies.png          # Accuracy comparison of models  
│   ├── ML_MetricsComparrision.png  # ML model metric comparison  
│   ├── spamVSham.png           # Spam vs Ham distribution plot  
│  
├── static/                     # Static frontend assets  
│   ├── script.js               # Handles form submission & API calls  
│   ├── style.css               # Stylesheet for frontend  
│  
├── templates/                  # HTML templates for Flask  
│   ├── index.html              # Main user interface  
│  
└── __pycache__/                 # Compiled Python cache files  
```

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.7+** installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Flask Application
```bash
python app.py
```
The application will start on `http://127.0.0.1:5000/`.

### Step 4: Access the Web Interface
Open a browser and go to:  
```
http://127.0.0.1:5000/
```

## Usage
1. Enter an SMS message in the input field.
2. Click the **Check** button.
3. The system will classify the message as **Spam** or **Ham**.

## Machine Learning Models Used
- **Naïve Bayes** (`nb_model.pkl`)
- **Logistic Regression** (`lr_model.pkl`)
- **Support Vector Machine (SVM)** (`svm_model.pkl`)
- **Random Forest** (`rf_model.pkl`)
- **LSTM (Long Short-Term Memory)** (`lstm_model.keras`)
- **GRU (Gated Recurrent Unit)** (`gru_model.keras`)
- **TF-IDF Vectorization** (`tfidf_vectorizer.pkl`) for feature extraction

## Data Preprocessing
- Stopword removal
- Tokenization
- Lowercasing text
- TF-IDF vectorization
- Handling class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**

## Model Performance
Model performance has been evaluated using various metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Plots are available in the `plots/` directory for better visualization.

## Troubleshooting
- If the `predict` endpoint is not responding, check `script.js` to ensure the form submission is handled correctly.
- Run `python training.py` if you need to retrain models.
- If dependencies fail, manually install them:
  ```bash
  pip install flask nltk scikit-learn imbalanced-learn tensorflow
  ```

## Future Enhancements
- **Deploy on a cloud platform (AWS/GCP/Heroku).**
- **Improve accuracy using ensemble learning.**
- **Enhance frontend UI with Bootstrap or React.**

## Contributors
- **Your Name** - *Developer & ML Engineer*
- **Other Contributors** - *Additional roles*

## License
This project is licensed under the **MIT License**.

---
📌 **Feel free to contribute and improve the model performance!** 🚀

