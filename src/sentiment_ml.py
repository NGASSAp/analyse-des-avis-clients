import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

fichier = BASE_DIR / "data/processed/Womens Clothing E-Commerce Reviews-selected-columns.csv"

df = pd.read_csv(fichier, encoding="utf-8-sig")

# Vérifier les données nécessaires
df = df.dropna(subset=["avis", "note"]).copy()
df["avis"] = df["avis"].astype(str).str.strip()
df["note"] = pd.to_numeric(df["note"], errors="coerce")
df = df[df["avis"] != ""].copy()
df = df[df["note"].between(1, 5)].copy()

# Transformer les notes en sentiments
def determiner_sentiment(note):
    if note <= 2:
        return "negatif"
    elif note == 3:
        return "neutre"
    else:
        return "positif"

df["sentiment"] = df["note"].apply(determiner_sentiment)

print("\nRépartition des classes :")
print(df["sentiment"].value_counts())

x = df["avis"]
y = df["sentiment"]

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("\nDimensions de l'ensemble d'entraînement :", x_train.shape)
print("Dimensions de l'ensemble de test :", x_test.shape)

