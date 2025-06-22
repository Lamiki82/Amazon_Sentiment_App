import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
import os

# === Imposta percorso ===
base_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(base_dir, '../data/raw/Shoes_Data.csv'))
output_path = os.path.abspath(os.path.join(base_dir, '../data/processed/cleaned_shoes_data.csv'))

# === Carica il dataset ===
df = pd.read_csv(data_path)

# === Pulizia colonne numeriche ===
df['price'] = df['price'].replace(r'[^\d.]', '', regex=True).astype(float)
df['rating'] = df['rating'].str.extract(r'(\d\.\d)').astype(float)
df['total_reviews'] = df['total_reviews'].str.replace(r'[^\d]', '', regex=True).astype(int)

# === Esplodi liste di recensioni ===
df['review_list'] = df['reviews'].str.split(r'\|\|')
df['review_rating_list'] = df['reviews_rating'].str.split(r'\|\|')
df_exploded = df.explode(['review_list', 'review_rating_list']).reset_index(drop=True)

# === Rinomina colonne ===
df_exploded.rename(columns={
    'review_list': 'review',
    'review_rating_list': 'review_rating'
}, inplace=True)

# === Pulizia dei testi ===
df_exploded['review'] = df_exploded['review'].astype(str).str.strip().replace('', np.nan)
df_exploded['review_rating'] = df_exploded['review_rating'].str.extract(r'(\d\.\d)').astype(float)

# === Rimozione righe con valori nulli fondamentali ===
df_exploded.dropna(subset=['review', 'review_rating', 'price', 'rating'], inplace=True)

# === Rimuove recensioni troppo corte o non informative ===
df_exploded = df_exploded[df_exploded['review'].str.len() > 10]

# === Rimozione duplicati ===
df_exploded.drop_duplicates(subset=['title', 'review'], inplace=True)

# === Salva il dataset pulito ===
df_exploded.to_csv(output_path, index=False)
print("✅ Dataset pulito salvato in:", output_path)
