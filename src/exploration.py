import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
plt.hist(df["note"], bins=6, edgecolor="black")
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
plt.grid(True)
plt.show()
