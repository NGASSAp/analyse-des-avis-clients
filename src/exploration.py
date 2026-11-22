import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
fichier = BASE_DIR / "data" / "processed" / "Womens Clothing E-Commerce Reviews-selected-columns.csv"

df = pd.read_csv(fichier, encoding="utf-8-sig")

# Nombre de lignes et de colonnes
print("Dimensions du dataset :", df.shape)

# Informations sur les colonnes
print("\nInformations générales :")
df.info()

# Statistiques descriptives
print("\nStatistiques :")
print(df.describe(include="all"))

# répartition des notes
plt.figure(figsize=(8, 6))
df["note"].value_counts().sort_index().plot(kind="bar")
plt.title("Répartition des notes")
plt.xlabel("Note")
plt.ylabel("Nombre d'avis")
plt.show()

# nombres de mots dans chaque avis
df["nombre_mots"] = (df["avis"].astype(str).str.split().str.len())
print("\nStatistiques sur le nombre de mots :")
print(df["nombre_mots"].describe())

plt.figure(figsize=(8, 6))
plt.hist(df["nombre_mots"], bins=30, edgecolor="black")
plt.title("Répartition du nombre de mots dans les avis")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre d'avis")
plt.show()

# Mots courants à exclure pour cette première analyse
mots_vides = [ "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on", "at", "for", "with", "this", "that", "these", "those", "it", "its", "i", "my", "me", "you", "your", "we", "they", "he", "she", "as", "by", "from"]

vectorizer = CountVectorizer( stop_words=mots_vides, max_features=30)

# création de la matrice de fréquence des mots
matrice = vectorizer.fit_transform(df["avis"])

# création d'un DataFrame pour visualiser les mots et leurs fréquences
frequences = matrice.sum(axis=0).A1
mots = vectorizer.get_feature_names_out()
df_mots = pd.DataFrame({ "mot": mots, "frequence": frequences })

# tri des mots par fréquence décroissante
df_mots = df_mots.sort_values( "frequence", ascending=False )
print(df_mots)