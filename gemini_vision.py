from google import genai
from pydantic import BaseModel

# 1. On initialise l'IA (Mets TA clé API à la place de TA_CLE_API_ICI)
client = genai.Client(api_key="AIzaSyArRR7eTVn12B4fz4gx-p37KATgqIUMmrI")

# 2. Le format spécial pour les coordonnées (ymin, xmin, ymax, xmax)
class BoiteCoordonnees(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int

class ZoneDetectee(BaseModel):
    nom_zone: str
    coordonnees: BoiteCoordonnees

class RapportDrone(BaseModel):
    style_jardin: str
    zones_plantables: list[ZoneDetectee] # C'est ici qu'on aura la liste des boîtes !

def analyser_precision(image_path: str):
    print(f"🚀 Analyse de précision en cours sur {image_path}...")
    
    # On envoie l'image
    photo = client.files.upload(file=image_path)
    
    # 3. La consigne de "Drone" pour l'IA
    consigne = """
    Agis comme un drone de cartographie. 
    Détecte toutes les zones plantables (herbe, terre vide, parterres).
    Pour chaque zone, donne-moi son nom et ses coordonnées exactes 
    sous forme de boîte [ymin, xmin, ymax, xmax] (échelle de 0 à 1000).
    """
    
    # 4. L'appel à l'IA
    reponse = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[photo, consigne],
        config={
            "response_mime_type": "application/json",
            "response_schema": RapportDrone,
        }
    )
    
    print("✅ Détection terminée !")
    return reponse.text

# --- TEST ---
resultat = analyser_precision("jardin.jpg")
print(resultat)

# Sauvegarde de la preuve
with open("analyse_precision.json", "w", encoding="utf-8") as f:
    f.write(resultat)
print("📁 Le fichier 'analyse_precision.json' a été créé à gauche !")