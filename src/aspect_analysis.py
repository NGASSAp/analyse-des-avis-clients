import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"
MODELE = "qwen3:1.7b"

def analyse_aspect(texte):
    prompt = f"""
    You are an expert in sentiment analysis of
    customer reviews about clothing products.

    Identify the aspects discussed in the review and classify the sentiment for each one.

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
    Rules:
- Do not invent aspects.
- An aspect can be positive even when the overall
  review is negative.
- Focus on the customer's opinion.
- Use only the specified sentiment labels.
- Return valid JSON only.

Expected format:
{{
  "aspects": [
    {{
      "aspect": "design",
      "sentiment": "positif"
    }}
  ]
}}

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
            "num_predict": 512
        }
    }

    reponse = requests.post(OLLAMA_URL, json=payload, timeout=180)
    reponse.raise_for_status()
    return json.loads(reponse.json()["response"])

# premier test
texte_test = """ I love the design of this dress, but the fabric
feels cheap and the sizing is terrible.
"""

resultat = analyse_aspect(texte_test)

print(json.dumps(resultat, indent=4, ensure_ascii=False))
