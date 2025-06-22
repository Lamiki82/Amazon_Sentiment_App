import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from fpdf import FPDF
from io import BytesIO
import random

# === SIMULA UNA PREVISIONE DI SENTIMENT ===
def simulate_sentiment_prediction(data, scenario):
    base_score = data['sentiment_label'].value_counts(normalize=True).get('positive', 0.5)
    scenario_impact = {
        'Nessuna modifica': 0.0,
        'Aumento prezzo': -0.05,
        'Diminuzione rating': -0.10,
        'Maggiore promozione': +0.08,
    }
    noise = random.uniform(-0.03, 0.03)
    adjusted_score = max(0.0, min(1.0, base_score + scenario_impact.get(scenario, 0.0) + noise))
    return adjusted_score

# === GENERA LA TABELLA E IL GRAFICO DI PREVISIONE ===
def generate_trend_forecast(data, scenario, forecast_months=12):
    today = datetime.today()
    months = pd.date_range(today, periods=forecast_months, freq='MS')
    scores = []
    upper_bounds = []
    lower_bounds = []

    base = simulate_sentiment_prediction(data, scenario)
    for i in range(forecast_months):
        score = base + random.uniform(-0.05, 0.05)
        score = max(0, min(score, 1))
        scores.append(score)
        upper_bounds.append(min(1.0, score + 0.1))
        lower_bounds.append(max(0.0, score - 0.1))

    df_forecast = pd.DataFrame({
        'Mese': months.strftime('%b %Y'),
        'Sentiment Previsto': scores,
        'Upper Bound': upper_bounds,
        'Lower Bound': lower_bounds
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_forecast['Mese'], df_forecast['Sentiment Previsto'], label='Previsione', color='blue', marker='o')
    ax.fill_between(df_forecast['Mese'], df_forecast['Lower Bound'], df_forecast['Upper Bound'],
                    color='blue', alpha=0.2, label='Margine di errore (±10%)')
    ax.set_title(f"Trend Sentiment Previsto ({scenario})")
    ax.set_xlabel("Mese")
    ax.set_ylabel("% Sentiment Positivo")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    commentary = f"Con lo scenario '{scenario}', ci si aspetta che il sentiment positivo resti tra {int(min(lower_bounds)*100)}% e {int(max(upper_bounds)*100)}% nei prossimi {forecast_months} mesi."

    return df_forecast, fig, commentary

# === CREA UN FILE PDF CON LA PREVISIONE ===
def generate_pdf_forecast(data, scenario, forecast_months=12):
    forecast_df, fig, commentary = generate_trend_forecast(data, scenario, forecast_months)

    # Salva grafico in memoria
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Crea PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="📊 Report Previsione Sentiment", ln=True, align='C')
    pdf.ln(10)

    pdf.multi_cell(0, 10, txt=f"Scenario scelto: {scenario}\n\n{commentary}")

    pdf.image(img_buf, x=10, y=50, w=180)

    # Aggiungi tabella dati
    pdf.add_page()
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "Mese", 1)
    pdf.cell(50, 10, "Sentiment Previsto", 1)
    pdf.cell(50, 10, "Margine Min", 1)
    pdf.cell(50, 10, "Margine Max", 1)
    pdf.ln()
    pdf.set_font("Arial", '', 10)
    for i, row in forecast_df.iterrows():
        pdf.cell(40, 10, row['Mese'], 1)
        pdf.cell(50, 10, f"{row['Sentiment Previsto']:.2f}", 1)
        pdf.cell(50, 10, f"{row['Lower Bound']:.2f}", 1)
        pdf.cell(50, 10, f"{row['Upper Bound']:.2f}", 1)
        pdf.ln()

    # Converti in BytesIO
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output
