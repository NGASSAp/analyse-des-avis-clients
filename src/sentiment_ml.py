import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

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

# Créer un pipeline de traitement du texte et de classification 
modele = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("regression_logistique", LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000))])
parametres = {"tfidf__max_features": [10000, 20000], "tfidf__ngram_range": [(1, 1), (1, 2)], "regression_logistique__C": [0.1, 1, 10], "regression_logistique__class_weight": [None, "balanced"]}

# Effectuer une recherche par grille pour trouver les meilleurs hyperparamètres
recherche = GridSearchCV(estimator=modele, param_grid=parametres, scoring="f1_macro", cv=3, n_jobs=-1)

# Entraîner le modèle
modele.fit(x_train, y_train)

print("\nModèle entraîné avec succès !")

recherche.fit(x_train, y_train)
print("\nMeilleurs hyperparamètres trouvés :", recherche.best_params_)

# prédictions
y_pred = modele.predict(x_test)

# dataframe pour les résultats
resultats = pd.DataFrame({"avis": x_test, "sentiment_reel": y_test, "sentiment_pred_ml": y_pred})
print("\nExemples de résultats :")
print(resultats.head())

# sélection des erreurs de prédiction
erreurs = resultats[resultats["sentiment_reel"] != resultats["sentiment_pred_ml"]]
print("\nNombre d'erreurs de prédiction :", len(erreurs))
print("\nExemples d'erreurs de prédiction :")
for index, ligne in erreurs.head(5).iterrows():
    print("\n" + "=" * 60)
    print("Index :", index)
    print("Avis :", ligne["avis"])
    print("Sentiment réel :", ligne["sentiment_reel"])
    print("Sentiment prédit :", ligne["sentiment_pred_ml"])

# accurary
print("Accuracy :", accuracy_score(y_test, y_pred))

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# matrice de confusion
ConfusionMatrixDisplay.from_estimator(modele, x_test, y_test, display_labels=modele.classes_, cmap="Blues", xticks_rotation=45)
plt.title("Matrice de confusion")
plt.show()

# créer un répertoire pour sauvegarder le modèle si nécessaire
modele_dir = BASE_DIR / "models"
results_dir = BASE_DIR / "results"
modele_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# sauvegarder le modèle
modele_path = modele_dir / "sentiment_model.pkl"
joblib.dump(modele, modele_path)

# sauvegarder les données de test et les résultats
resultats = pd.DataFrame({"avis": x_test, "sentiment_reel": y_test, "sentiment_pred": y_pred})
resultats.to_csv(results_dir / "resultats_test.csv", index=False, encoding="utf-8-sig")

# sauvegarder les erreurs de prédiction
rapport = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rapport).transpose().to_csv(results_dir / "rapport_classification.csv", index=True, encoding="utf-8-sig")

print("\nModèle et résultats sauvegardés avec succès !")
