# 🐦 Tweet Sentiment Analysis using Machine Learning

This project performs **sentiment analysis on Twitter tweets** using **Machine Learning techniques** to classify tweets as **Positive** or **Negative**.  
It demonstrates a complete **end-to-end NLP pipeline**, including text preprocessing, feature extraction, handling class imbalance, model training, evaluation, and prediction.

---

## 📌 Project Overview

Social media platforms like Twitter generate massive amounts of text data every day.  
Understanding public sentiment from this data is valuable for:
- Brand monitoring
- Customer feedback analysis
- Opinion mining
- Social media analytics

This project builds a **sentiment classification system** using classical ML models.

---

## 🧠 Machine Learning Models Used

- **Logistic Regression**


To handle class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** is applied.

---

## 📂 Dataset

- **Source**: Twitter sentiment dataset  
- **Format**: CSV  
- **Columns**:
  - `tweet` → Text of the tweet
  - `label` → Sentiment label  
    - `0` → Positive  
    - `1` → Negative

---

## ⚙️ Project Pipeline

1. Data loading and inspection  
2. Text preprocessing  
   - Lowercasing  
   - Punctuation removal  
   - Stopword removal (negations preserved)  
3. Feature extraction using **TF-IDF Vectorizer**
4. Handling class imbalance using **SMOTE**
5. Model training (Logistic Regression & Naive Bayes)
6. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
7. Prediction on custom tweets

---

## 📊 Results & Observations

- **Naive Bayes performed better** than Logistic Regression for this dataset
- Naive Bayes showed:
  - Higher accuracy
  - Better detection of negative sentiment
- This behavior is expected for **text-based classification tasks**

---

## ▶️ Author:
Utkal Singh

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/TWEET-SENTIMENT-ANALYSIS-USING-ML.git
cd TWEET-SENTIMENT-ANALYSIS-USING-ML
