"""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk

from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
tweets_df = pd.read_csv("twitter.csv")
print(tweets_df.head())
tweets_df.info()
print(tweets_df.describe())
tweets_df = tweets_df.drop(['id'], axis=1)
sns.heatmap(tweets_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()
sns.countplot(x='label', data=tweets_df)
plt.title("Label Distribution")
plt.show()
tweets_df['length'] = tweets_df['tweet'].apply(len)
print(tweets_df.describe())
all_text = " ".join(tweets_df['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(all_text))
plt.axis("off")
plt.show()
negative_text = " ".join(tweets_df[tweets_df['label'] == 1]['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud(background_color="white").generate(negative_text))
plt.axis("off")
plt.show()
def message_cleaning(message):
    # Remove punctuation
    no_punct = ''.join(char for char in message if char not in string.punctuation)
    
    # Remove stopwords
    clean_words = [
        word for word in no_punct.split()
        if word.lower() not in STOPWORDS
    ]
    
    return clean_words
vectorizer = CountVectorizer(analyzer=message_cleaning)
X = vectorizer.fit_transform(tweets_df['tweet'])

y = tweets_df['label']

print("Feature matrix shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Logistic Regression model trained")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

test_tweets = [
    "I love this product, it is amazing!",
    "Worst experience ever, totally disappointed",
    "Not bad but could be better",
    "Absolutely terrible service and very unhappy"
]

test_vectorized = vectorizer.transform(test_tweets)
predictions = model.predict(test_vectorized)

for tweet, label in zip(test_tweets, predictions):
    sentiment = "Positive" if label == 0 else "Negative"
    print(f"Tweet: {tweet}")
    print(f"Prediction: {sentiment}\n")
"""
"""
# ===================== IMPORT LIBRARIES =====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk

from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===================== DOWNLOAD STOPWORDS =====================
nltk.download('stopwords')

# ===================== LOAD DATA =====================
tweets_df = pd.read_csv("twitter.csv")
tweets_df = tweets_df.drop(['id'], axis=1)

print(tweets_df.head())
print(tweets_df.info())

# ===================== DATA VISUALIZATION =====================
sns.countplot(x='label', data=tweets_df)
plt.title("Label Distribution")
plt.show()

# ===================== WORD CLOUD (ALL TWEETS) =====================
all_text = " ".join(tweets_df['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(all_text))
plt.axis("off")
plt.show()

# ===================== WORD CLOUD (NEGATIVE TWEETS) =====================
negative_text = " ".join(tweets_df[tweets_df['label'] == 1]['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud(background_color="white").generate(negative_text))
plt.axis("off")
plt.show()

# ===================== NLP CLEANING (IMPORTANT FIX) =====================
STOPWORDS = set(stopwords.words('english'))

# KEEP NEGATION WORDS (CRITICAL)
NEGATION_WORDS = {"not", "no", "nor", "never"}
STOPWORDS = STOPWORDS - NEGATION_WORDS

def message_cleaning(message):
    # remove punctuation
    no_punct = ''.join(char for char in message if char not in string.punctuation)
    
    # lowercase + remove stopwords (except negations)
    clean_words = [
        word.lower()
        for word in no_punct.split()
        if word.lower() not in STOPWORDS
    ]
    
    return clean_words

# ===================== FEATURE EXTRACTION =====================
vectorizer = CountVectorizer(
    analyzer=message_cleaning,
    ngram_range=(1, 2),   # UNIGRAM + BIGRAM
    min_df=2
)

X = vectorizer.fit_transform(tweets_df['tweet'])
y = tweets_df['label']

print("Feature matrix shape:", X.shape)

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===================== BEST MODEL: LOGISTIC REGRESSION =====================
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Model trained successfully")

# ===================== MODEL EVALUATION =====================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===================== TEST WITH NEW CUSTOM TWEETS =====================
test_tweets = [
    "I love this product, it is amazing!",
    "Worst experience ever, totally disappointed",
    "Not bad but could be better",
    "Absolutely terrible service and very unhappy"
]

test_vectorized = vectorizer.transform(test_tweets)
predictions = model.predict(test_vectorized)

print("\nCUSTOM TEST RESULTS:\n")
for tweet, label in zip(test_tweets, predictions):
    sentiment = "Positive" if label == 0 else "Negative"
    print(f"Tweet: {tweet}")
    print(f"Prediction: {sentiment}\n")
"""
# ===================== IMPORT LIBRARIES =====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk

from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===================== DOWNLOAD STOPWORDS =====================
nltk.download('stopwords')

# ===================== LOAD DATA =====================
tweets_df = pd.read_csv("twitter.csv")
tweets_df = tweets_df.drop(['id'], axis=1)

print(tweets_df.head())
print(tweets_df.info())

# ===================== DATA VISUALIZATION =====================
sns.countplot(x='label', data=tweets_df)
plt.title("Label Distribution")
plt.show()

# ===================== WORD CLOUD (ALL TWEETS) =====================
all_text = " ".join(tweets_df['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(all_text))
plt.axis("off")
plt.show()

# ===================== WORD CLOUD (NEGATIVE TWEETS) =====================
negative_text = " ".join(tweets_df[tweets_df['label'] == 1]['tweet'])

plt.figure(figsize=(20,20))
plt.imshow(WordCloud(background_color="white").generate(negative_text))
plt.axis("off")
plt.show()

# ===================== NLP CLEANING (IMPORTANT FIX) =====================
STOPWORDS = set(stopwords.words('english'))

# KEEP NEGATION WORDS (CRITICAL)
NEGATION_WORDS = {"not", "no", "nor", "never"}
STOPWORDS = STOPWORDS - NEGATION_WORDS

def message_cleaning(message):
    # remove punctuation
    no_punct = ''.join(char for char in message if char not in string.punctuation)
    
    # lowercase + remove stopwords (except negations)
    clean_words = [
        word.lower()
        for word in no_punct.split()
        if word.lower() not in STOPWORDS
    ]
    
    return clean_words

# ===================== FEATURE EXTRACTION =====================
vectorizer = CountVectorizer(
    analyzer=message_cleaning,
    ngram_range=(1, 2),   # UNIGRAM + BIGRAM
    min_df=2
)

X = vectorizer.fit_transform(tweets_df['tweet'])
y = tweets_df['label']

print("Feature matrix shape:", X.shape)

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===================== BEST MODEL: LOGISTIC REGRESSION =====================
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Model trained successfully")

# ===================== MODEL EVALUATION =====================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===================== TEST WITH NEW CUSTOM TWEETS =====================
test_tweets = [
    "I love this product, it is amazing!",
    "Worst experience ever, totally disappointed",
    "Not bad but could be better",
    "Absolutely terrible service and very unhappy"
]

test_vectorized = vectorizer.transform(test_tweets)
predictions = model.predict(test_vectorized)

print("\nCUSTOM TEST RESULTS:\n")
for tweet, label in zip(test_tweets, predictions):
    sentiment = "Positive" if label == 0 else "Negative"
    print(f"Tweet: {tweet}")
    print(f"Prediction: {sentiment}\n")
