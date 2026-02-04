import pandas as pd
from sentence_transformers import SentenceTransformer

from config_fr import get_or_create_collection, generate_answer, EMBEDDING_MODEL_NAME

# Chargement et indexation de la base de données
print('📚 Chargement et indexation de la base de données des muffins...')
df_muffin = pd.read_csv('./data/muffin_dataset.csv') # Full database

db = get_or_create_collection(df_muffin, db_path='./chromadb_full',verbose=True)

# Requête utilisateur
query = """
J'ai des framboises et du chocolat blanc à la maison, t'aurais une recette de muffin pour moi ?
"""

# Recherche dans la base de données
print('🔍 Recherche de la recette la plus pertinente...')
model_embed = SentenceTransformer(EMBEDDING_MODEL_NAME)
results = db.query(query_embeddings=model_embed.encode([query]).tolist(), n_results=1)

# Génération de la réponse
print('👨‍🍳 Le Chef pâtissier analyse la base de données...')
final_answer = generate_answer(query, results)

print('-' * 30)
print(final_answer)
print('-' * 30)