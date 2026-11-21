import pandas as pd
import matplotlib.pyplot as plt

#chargement des données
df = pd.read_csv("C:\\Users\\pasca\\Desktop\\analyse-avis-clients-llm\\data\\raw\\avis_clients.csv", encoding = "cp1252")
print("Dimensions initiales : ",df.shape)
print("Valeurs manquantes : ",df.isnull().sum())

#nettoyage des avis
df_copy = df.copy()
df_copy = df_copy.dropna(subset = ["avis", "note"]) # suppression des valeurs manquantes
df_copy["avis"] = df_copy["avis"].astype(str).str.strip() # c'est pour convertir les avis en chaines de caractères et supprimer les espaces en début et fin de chaine
df_copy = df_copy[df_copy["avis"] != ""] # suppression des avis vides
df_copy = df_copy[df_copy["note"].between(0, 5)] # suppression des notes hors de l'intervalle [0, 5]
df_copy["note"] = pd.to_numeric(df_copy["note"], errors="coerce") # conversion des notes en nombres, les valeurs non convertibles seront remplacées par NaN
df_copy["note"] = df_copy["note"].astype(int) # conversion des notes en entiers
df_copy = df_copy.drop_duplicates(subset = ["avis", "note"]) # suppression des doublons
