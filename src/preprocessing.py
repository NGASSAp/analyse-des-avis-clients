import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#chargement des données
chemin_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "avis_clients.csv"
OUTPUT_FILE = chemin_csv.parents[1] / "processed" / "avis_clients_nettoyes.csv"
with chemin_csv.open(encoding="cp1252", newline="") as fichier:
    lecteur = csv.reader(fichier)
    colonnes = next(lecteur)
    lignes = []
    for ligne in lecteur:
        if not ligne:
            continue
        # Certaines lignes contiennent un enregistrement CSV entier dans un seul champ.
        if len(ligne) == 1:
            ligne = next(csv.reader(ligne))
        if len(ligne) != len(colonnes):
            raise ValueError(f"Ligne CSV mal form?e ? la ligne {lecteur.line_num}")
        lignes.append(ligne)

df = pd.DataFrame(lignes, columns=colonnes)
df["id"] = pd.to_numeric(df["id"])
df["note"] = pd.to_numeric(df["note"], errors="coerce")
df["avis"] = df["avis"].replace("", pd.NA)
print(df.head())
print("Dimensions initiales : ",df.shape)
print("Valeurs manquantes : ",df.isnull().sum())

#nettoyage des avis
df_copy = df.copy()
df_copy = df_copy.dropna(subset = ["avis", "note"]) # suppression des valeurs manquantes
df_copy["avis"] = df_copy["avis"].astype(str).str.strip() # c'est pour convertir les avis en chaines de caractères et supprimer les espaces en début et fin de chaine
df_copy = df_copy[df_copy["avis"] != ""] # suppression des avis vides
df_copy = df_copy[df_copy["note"].between(0, 5)] # suppression des notes hors de l'intervalle [0, 5]
df_copy["note"] = df_copy["note"].astype(int) # conversion des notes en entiers
df_copy = df_copy.drop_duplicates(subset = ["avis", "note"]) # suppression des doublons

# affectation d'un sentiment en fonction de la note
def sentiment(note):   
    if note >= 4:
        return "positif"
    elif note == 3:
        return "neutre"
    else:
        return "negatif"

# application de la fonction sentiment sur la colonne "note"
df_copy["sentiment"] = df_copy["note"].apply(sentiment) 

# longueur des avis
df_copy["longueur_avis"] = df_copy["avis"].str.len()

print("Repartition des sentiments :")
print(df_copy["sentiment"].value_counts())
print("Statistiques des longueurs des avis :")
print(df_copy["longueur_avis"].describe())

# visuazlisation de la répartition des sentiments
plt.figure(figsize=(8, 6))
plt.bar(df_copy["sentiment"].value_counts().index, df_copy["sentiment"].value_counts().values)
plt.title("Répartition des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre d'avis")
plt.show()

# 8. Sauvegarde
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_copy.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("\nFichier nettoyé enregistré :", OUTPUT_FILE)