import streamlit as st
st.set_page_config(page_title="Sentiment Analysis - Amazon Shoes", layout="wide")
def load_external_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import necessary libraries
from models.prediction_module import (simulate_sentiment_prediction,generate_trend_forecast)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import base64
from io import BytesIO
from fpdf import FPDF
import numpy as np
from datetime import datetime, timedelta
import random
import tempfile

import base64

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


nltk.download('stopwords')

# === CONFIGURAZIONE STREAMLIT ===
load_external_css("assets/Style.css")
# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/sentiment_shoes_data.csv")
    df['Shoe Type'] = df['Shoe Type'].fillna('Unknown').str.lower().str.strip()
    modello_map = {
        "men": "uomo",
        "women": "donna",
        "mens": "uomo",
        "womens": "donna",
        "ladies": "donna",
        "boys": "uomo",
        "girls": "donna"
    }
    df["modello"] = df["Shoe Type"].map(modello_map).fillna("altro")
    df['titolo'] = df['title'].fillna("Non specificato").astype(str)
    return df

df = load_data()

# === HEADER ===
st.markdown("""
    <h1 style='text-align:center; color: #4CAF50; font-size: 36px;'>Recensioni 👟 scarpe: Amazon</h1>
    <p style='text-align: center;'>Analisi delle recensioni.</p>
    <hr style='margin-top: -10px;'>
""", unsafe_allow_html=True)
st.markdown(
        """
        <div style='text-align: center; font-size:18px; color:#4CAF50; font-weight: bold;'>
        Esplora l'andamento del sentiment dei clienti e simula scenari futuri.<br>
        __________________________________________________________________________<br>
        <br>
        <br>
        </div>
        """,
        unsafe_allow_html=True
    )
# === FOOTER ===
def show_footer():
    st.markdown("***")
    st.markdown(""" **like this?** Follow me on
                [Linkedin](https://www.linkedin.com/in/michela-bernardini-38053a3a).""")

# === SIDEBAR FILTRI ===
st.sidebar.header("🔍 filtra visualizzazione")
modello_filter = st.sidebar.multiselect("Filtra per modello", options=sorted(df["modello"].unique()), default=sorted(df["modello"].unique()))
sentiment_filter = st.sidebar.multiselect("Filtra per sentiment", options=["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])
title_filter = st.sidebar.multiselect("Filtra per articolo (titolo)", options=sorted(df["titolo"].unique()), default=[])
sentiment_wordcloud = st.sidebar.radio("Visualizza WordCloud per sentiment", options=["positive", "neutral", "negative"])
show_wordcloud = st.sidebar.checkbox("Mostra Wordcloud & Conteggio parole")

if st.sidebar.button("📤 Esporta CSV filtrato"):
    st.download_button(
        label="Scarica dati",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='filtrato_sentiment_shoes.csv',
        mime='text/csv'
    )

note = st.sidebar.text_area("📝 Bloc-notes personale", "Aggiungi le tue osservazioni personali qui...")

# === APPLICAZIONE FILTRI ===
if modello_filter:
    df = df[df["modello"].isin(modello_filter)]
if sentiment_filter:
    df = df[df["sentiment_label"].isin(sentiment_filter)]
if title_filter:
    df = df[df["titolo"].isin(title_filter)]

# === METRICHE ===
col1, col2, col3 = st.columns(3)
col1.metric("Totale recensioni", len(df))
col2.metric("Media rating", round(df["rating"].mean(), 2) if not df.empty else "N/A")
col3.metric("Sentiment positivo", f"{(df['sentiment_label'] == 'positive').mean() * 100:.1f}%" if not df.empty else "N/A")

# === GRAFICI ===
if not df.empty:
    st.subheader("📊 Distribuzione del Sentiment")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="sentiment_label", palette={"positive": "#3498DB", "neutral": "#95A5A6", "negative": "#E74C3C"}, ax=ax)
    st.pyplot(fig)
    st.info("La maggior parte delle recensioni risulta essere positiva, con una quota minore di opinioni negative o neutrali. Questo suggerisce un buon apprezzamento generale da parte degli utenti.")

    st.subheader("⭐ Distribuzione Rating")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["rating"], bins=5, kde=False, color="#3498DB", ax=ax2)
    st.pyplot(fig2)
    st.success("I rating si concentrano maggiormente sulle fasce più alte (4 e 5 stelle), confermando il trend positivo osservato anche nel sentiment testuale.")

    st.subheader("👟 Top Modelli con Sentiment Positivo")
    top_df = df[df['sentiment_label'] == 'positive']
    top_models = top_df['Shoe Type'].value_counts().head(10)
    if not top_models.empty:
        fig3, ax3 = plt.subplots()
        top_models.plot(kind='barh', color="#2ECC71", ax=ax3)
        ax3.set_xlabel("Numero recensioni positive")
        ax3.set_ylabel("Modello")
        st.pyplot(fig3)
        st.info("Questi modelli risultano essere i più apprezzati dai clienti in termini di sentiment positivo.")
    else:
        st.warning("⚠️ Nessun modello con recensioni positive nei dati filtrati.")

    st.subheader("🏆 Top Articoli con Sentiment Positivo")
    top_pos_titles = top_df['titolo'].value_counts().head(10)
    if not top_pos_titles.empty:
        fig_pos_titles, ax_pos_titles = plt.subplots()
        top_pos_titles.plot(kind='barh', color="#27AE60", ax=ax_pos_titles)
        ax_pos_titles.set_xlabel("Numero recensioni positive")
        ax_pos_titles.set_ylabel("Titolo")
        st.pyplot(fig_pos_titles)
        st.success("Questi articoli hanno ricevuto il maggior numero di feedback positivi, indicando alta soddisfazione.")
    else:
        st.info("Nessun articolo positivo da mostrare.")

    st.subheader("📉 Top Articoli con Sentiment Negativo")
    neg_df = df[df['sentiment_label'] == 'negative']
    top_neg_titles = neg_df['titolo'].value_counts().head(10)
    if not top_neg_titles.empty:
        fig4, ax4 = plt.subplots()
        top_neg_titles.plot(kind='barh', color="#E74C3C", ax=ax4)
        ax4.set_xlabel("Numero recensioni negative")
        ax4.set_ylabel("Titolo")
        st.pyplot(fig4)
        st.warning("Questi articoli hanno ricevuto un numero elevato di feedback negativi: potrebbero necessitare miglioramenti.")
    else:
        st.info("Nessun articolo negativo da mostrare.")
else:
    st.warning("⚠️ Nessun dato disponibile con i filtri selezionati.")

# === WORDCLOUD ===
sentiment_df = df[df["sentiment_label"] == sentiment_wordcloud]
text = " ".join(sentiment_df["clean_review"].dropna().tolist())

if show_wordcloud:
    with st.expander("☁️ Visualizza WordCloud e parole comuni"):
        if text.strip():
            st.subheader(f"☁️ WordCloud - Sentiment: {sentiment_wordcloud.capitalize()}")
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=set(stopwords.words("english"))).generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

            word_freq = Counter(text.split())
            common_words = dict(word_freq.most_common(15))
            if common_words:
                st.subheader("📈 Frequenza parole comuni")
                fig_bar, ax_bar = plt.subplots()
                sns.barplot(x=list(common_words.values()), y=list(common_words.keys()), palette="Blues_d", ax=ax_bar)
                st.pyplot(fig_bar)
            else:
                st.warning("Nessuna parola disponibile per il grafico.")
        else:
            st.warning(f"Nessuna recensione '{sentiment_wordcloud}' disponibile per visualizzare la Wordcloud.")

# === ANTEPRIMA TABELLA ===
if not df.empty:
    st.subheader("📄 Anteprima recensioni")
    st.dataframe(df[["review", "review_rating", "sentiment_score", "sentiment_label", "titolo", "modello"]].head(20))

# === LINK ALLA DASHBOARD ===
st.markdown("Vuoi una versione interattiva di questa analisi? Dai un'occhiata alla dashboard pubblicata su Tableau!")
st.markdown("""
    <a href="https://public.tableau.com/views/sentimentrecensioniamazonshoes/sentimentdash?:language=it-IT&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
       target="_blank" style="display: inline-block; padding: 0.5em 1em; font-size: 1.1em; color: white; background-color: #4CAF50; border-radius: 8px; text-decoration: none;">
        🔗 Dashboard Interattiva
    </a>
""", unsafe_allow_html=True)

# === CONCLUSIONI FINALI ===
st.markdown("---")
st.subheader("📌 Conclusioni Dinamiche")

summary = ""
sentiment_dist = df['sentiment_label'].value_counts(normalize=True).to_dict()
if sentiment_dist.get("positive", 0) > 0.6:
    summary += "\n- Le recensioni mostrano un forte orientamento positivo."
elif sentiment_dist.get("negative", 0) > 0.3:
    summary += "\n- Sono presenti numerose critiche: si consiglia una revisione dei prodotti segnalati."
else:
    summary += "\n- Il sentiment è distribuito in modo bilanciato."

if 'donna' in modello_filter and 'uomo' not in modello_filter:
    summary += "\n- Il focus attuale è su articoli per donna."
elif 'uomo' in modello_filter and 'donna' not in modello_filter:
    summary += "\n- Il focus attuale è su articoli da uomo."
else:
    summary += "\n- L'analisi copre modelli sia per uomo che donna."

if len(title_filter) == 1:
    summary += f"\n- L'articolo analizzato è: '{title_filter[0]}'."
elif len(title_filter) > 1:
    summary += f"\n- Sono stati analizzati {len(title_filter)} articoli specifici."

st.code(summary.strip(), language="markdown")

st.markdown("---")
st.subheader("📎 Note personali")
st.text_area("Appunti dell'utente", note, height=150)

# === PREVISIONE DEL SENTIMENT FUTURO ===
st.markdown("---")
st.subheader("🧠 Previsione Sentiment Futuro")
st.markdown("Simula l'evoluzione del sentiment sulla base di scenari ipotetici e scarica un report PDF della previsione.")

# Input utente per scenario e mesi
scenario = st.selectbox(
    "📌 Seleziona uno scenario ipotetico:",
    ['Nessuna modifica', 'Aumento prezzo', 'Diminuzione rating', 'Maggiore promozione']
)

forecast_months = st.slider(
    "📆 Mesi da prevedere:",
    min_value=3,
    max_value=24,
    value=12,
    step=1
)

# Genera previsione solo se ci sono dati
if not df.empty and st.button("🚀 Genera Previsione"):
    forecast_df, forecast_fig, commentary = generate_trend_forecast(df, scenario, forecast_months)

    st.pyplot(forecast_fig)
    st.success("✅ Previsione generata con successo!")
    st.markdown(f"**🧠 Commento automatico:** {commentary}")

    with st.expander("📋 Visualizza dati previsionali"):
        st.dataframe(forecast_df)
