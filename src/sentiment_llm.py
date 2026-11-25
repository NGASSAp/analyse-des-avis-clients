import json
from pathlib import Path
import time
import requests
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
FICHIER = BASE_DIR / "results" / "resultats_test.csv"
SORTIE = BASE_DIR / "results" / "predictions_llm.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"  # URL de l'API Ollama
MODELE = "qwen3:1.7b"  # Nom du modèle Ollama à utiliser

# chargement des données de test
df = pd.read_csv(FICHIER, sep=",", encoding="utf-8")

# sélection de 30 avis pour le premier test
df_test = df.head(30).rename(columns={"sentiment_pred": "sentiment_pred_ml"}).copy()

# fonction pour interroger le modèle Ollama
def interroger_modele(texte):
    # construction du prompt pour le modèle
    
    prompt = f"""
    You are an expert in sentiment analysis of
    customer reviews about clothing products.

    Your task is to classify the OVERALL customer
    satisfaction into exactly one category.

    Classification rules:

    POSITIF:
    - The customer is generally satisfied.
    - Positive feedback dominates.
    - Minor complaints do not change an overall
    positive opinion.

    NEGATIF:
    - The customer is generally dissatisfied.
    - Complaints dominate the review.
    - Strong dissatisfaction, returning the item,
    or inability to use the product are signs
    of negative sentiment.

    NEUTRE:
    - The review expresses balanced positive and
    negative opinions without a clear conclusion.
    - The overall satisfaction is ambiguous.
    - Do not classify every review containing
    a complaint as negative.

    Important:
    - Read the entire review before deciding.
    - Pay attention to phrases such as
    "however", "but", "overall" and "unfortunately".
    - Focus on the final overall satisfaction.
    - Return exactly one label.

    Return only a valid JSON object:
    {{"sentiment": "positif"}}

    Review:
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
df_test["sentiment_pred_llm"] = predictions
df_test["temps_execution"] = temps_execution

# enregistrement des résultats dans un fichier CSV
SORTIE.parent.mkdir(parents=True, exist_ok=True)
df_test.to_csv(SORTIE, index=False, sep=",", encoding="utf-8")
print("\n Analyse terminée")
print("Résultats enregistrés :", SORTIE)

# conservation des réponses valides
sentiments_valides = ["positif", "negatif", "neutre"]
evaluation = df_test[df_test["sentiment_pred_llm"].isin(sentiments_valides)]
print("\n Réponses valides :", len(evaluation), "/", len(df_test))
if len(evaluation) == 0:
    raise ValueError("Aucune réponse valide pour l'évaluation.")
y_reel = evaluation["sentiment_reel"]
y_ml = evaluation ["sentiment_pred_ml"]
y_llm = evaluation["sentiment_pred_llm"]

# évaluation des performances des modèles ML et LLM
print("Accuracy du modèle ML :", accuracy_score(y_reel, y_ml))
print("Rapport de classification du modèle ML :")
print(classification_report(y_reel, y_ml))

print("Accuracy du modèle LLM :", accuracy_score(y_reel, y_llm))
print("Rapport de classification du modèle LLM :")
print(classification_report(y_reel, y_llm))