import os
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Scarica risorse solo se mancano
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# === Percorso dati ===
base_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(base_dir, '../data/processed/cleaned_shoes_data.csv'))
output_path = os.path.abspath(os.path.join(base_dir, '../data/processed/sentiment_shoes_data.csv'))

# === Caricamento dati ===
df = pd.read_csv(data_path)

# === Pulizia testo ===
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # link
    text = re.sub(r'[^a-z\s]', '', text)             # solo lettere
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].astype(str).apply(clean_text)

# === Analisi Sentiment ===
sia = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    label = 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
    return pd.Series([score, label])

df[['sentiment_score', 'sentiment_label']] = df['clean_review'].apply(classify_sentiment)

# === Salvataggio ===
df.to_csv(output_path, index=False)
print("✅ Sentiment analysis completata. File salvato in:", output_path)
