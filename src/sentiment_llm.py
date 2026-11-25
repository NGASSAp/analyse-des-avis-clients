import json
from pathlib import Path
import time
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FICHIER = BASE_DIR / "results" / "resultats_test.csv"
SORTIE = BASE_DIR / "results" / "predictions_llm.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"  # URL de l'API Ollama
MODELE = "qwen3:1.7b"  # Nom du modèle Ollama à utiliser

# chargement des données de test
df = pd.read_csv(FICHIER, sep=",", encoding="utf-8")

# sélection de 30 avis pour le premier test
df_test = df.head(30).copy()

# fonction pour interroger le modèle Ollama
def interroger_modele(texte):
    # construction du prompt pour le modèle
    prompt = f"""
You are a customer review sentiment classifier.

Classify the overall sentiment of the following
English clothing review.

Allowed labels:
- positif: overall positive satisfaction
- negatif: overall negative dissatisfaction
- neutre: mixed or neutral opinion

Consider the entire review, including complaints
appearing after initial compliments.

Return only a valid JSON object:
{{"sentiment": "positif"}}

Customer review:
{texte}
"""
    # construction du payload pour la requête POST
    payload = {
        "model": MODELE,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 100
        }
    }

# envoi de la requête POST à l'API Ollama
    reponse = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )
    # vérification du statut de la réponse
    reponse.raise_for_status()
    contenu = reponse.json()["response"]
    resultat = json.loads(contenu)
    sentiment = resultat["sentiment"]
    if sentiment not in ["positif", "negatif", "neutre"]:
        raise ValueError(
            f"Sentiment invalide : {sentiment}"
        )

    return sentiment

# analyse des avis et stockage des résultats
predictions = []
temps_execution = []
for index, row in df_test.iterrows():
    debut = time.perf_counter()
    texte = row["avis"]
    try:
        sentiment = interroger_modele(texte)
    except Exception as e:
        sentiment = f"Erreur: {str(e)}"
    fin = time.perf_counter()
    temps_execution.append(fin - debut)
    predictions.append(sentiment)
    print(f"Avis {index + 1}/{len(df_test)} : Sentiment prédit = {sentiment}")

# ajouter les résultats
df_test["sentiment_pred"] = predictions
df_test["temps_execution"] = temps_execution

# enregistrement des résultats dans un fichier CSV
SORTIE.parent.mkdir(parents=True, exist_ok=True)
df_test.to_csv(SORTIE, index=False, sep=",", encoding="utf-8")
print("\n Analyse terminée")
print("Résultats enregistrés :", SORTIE)
