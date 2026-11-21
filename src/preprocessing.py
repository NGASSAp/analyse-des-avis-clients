from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "raw" / "Womens Clothing E-Commerce Reviews.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "Womens Clothing E-Commerce Reviews-selected-columns.csv"
COLONNES = {"Clothing ID": "id", "Review Text": "avis", "Rating": "note"}

df = pd.read_csv(INPUT_FILE, usecols=list(COLONNES), encoding="utf-8-sig")
df = df.rename(columns=COLONNES)[list(COLONNES.values())]
df["note"] = pd.to_numeric(df["note"], errors="coerce")
df["avis"] = df["avis"].str.strip()

print("Nombre de lignes initial :", len(df))
print("Valeurs manquantes :")
print(df.isna().sum())

df = df.dropna(subset=["avis", "note"])
df = df.loc[df["avis"].ne("") & df["note"].isin(range(1, 6))].copy()
df["note"] = df["note"].astype(int)
df = df.drop_duplicates(subset=["avis", "note"]).copy()


def sentiment(note):
    if note >= 4:
        return "positif"
    if note == 3:
        return "neutre"
    return "negatif"


df["sentiment"] = df["note"].map(sentiment)
df["longueur_avis"] = df["avis"].str.len()

repartition = df["sentiment"].value_counts()
print("Nombre de lignes après nettoyage :", len(df))
print("Répartition des sentiments :")
print(repartition)
print("Statistiques des longueurs des avis :")
print(df["longueur_avis"].describe())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("Fichier nettoyé enregistré :", OUTPUT_FILE)

plt.figure(figsize=(8, 6))
plt.bar(repartition.index, repartition.values)
plt.title("Répartition des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre d'avis")
plt.show()
