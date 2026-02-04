from tabnanny import verbose
import chromadb
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import uuid

# --- CONFIGURATION FRANÇAISE ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError('⚠️ Clé API non trouvée ! Veuillez vérifier votre fichier .env.')
client_groq = Groq(api_key=GROQ_API_KEY)

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
COLLECTION_NAME = 'royaume_du_muffin'


def get_or_create_collection(df, db_path='./chromadb', verbose=False, batch_size=5000):
    '''
    Récupère ou crée la collection ChromaDB.
    '''
    client = chromadb.PersistentClient(path=db_path)
    existing_collections = [col.name for col in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        if verbose:
            print(f'📂 Collection {COLLECTION_NAME} trouvée dans {db_path}, Chargement...')
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    
    if verbose:
        print(f'📂 Collection {COLLECTION_NAME} non trouvée, Création...')
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = (df['title'] + ' : ' + df['ingredients']).tolist()
    metadatas = df.to_dict(orient='records')
    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    embeddings = model.encode(documents, show_progress_bar=verbose).tolist()

    collection = client.create_collection(name=COLLECTION_NAME)

    if verbose:
        print(f'⚙️ Indexation des données dans ChromaDB ({len(ids)} entrées)...')
    for i in range(0, len(ids), batch_size):
        batch_size_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_size_end],
            documents=documents[i:batch_size_end],
            embeddings=embeddings[i:batch_size_end],
            metadatas=metadatas[i:batch_size_end]
        )
        if verbose:
            print(f'   ✅ Lot {i // batch_size + 1} inséré...')

    return collection


def generate_answer(query, search_results):
    '''
    Génère une réponse basée sur la demande de l'utilisateur et les résultats de recherche.
    '''
    top_recipe = search_results['metadatas'][0][0]
    title = top_recipe['title']
    ingredients = top_recipe['ingredients']
    directions = top_recipe['directions']

    # Prompt
    prompt = f"""
    Tu es un chef pâtissier expert spécialisé dans les muffins et obsédé par ces délicieuses pâtisseries.
    Un utilisateur t'a posé la question suivante : "{query}"
    Tu connais cette recette de muffin (en anglais) :

    Titre : {title}
    Ingrédients : {ingredients}
    Instructions : {directions}

    Dans le cas où cette recette ne correspond pas à la question ou que l'utilisateur demande quelque chose hors sujet, réponds poliment et brièvement que tu ne peux pas aider avec cette demande spécifique et coupe court en ne mentionnant pas la recette trouvée.
    Ne cherche pas d'autres recettes, utilise uniquement celle fournie ci-dessus.

    Réponds de manière chaleureuse et enthousiaste en français, en donnant des conseils utiles et des instructions claires pour préparer ces muffins.
    Traduis la recette et les termes culinaires en français de manière appropriée (ex: oz en grammes, cups en cuillères à soupe, etc...).
    Assure-toi que ta réponse est concise, engageante et facile à suivre avec éventuellement une blague ou une anecdote sur les muffins en lien avec la question de l'utilisateur.
    Termine ta réponse en proposant à l'utilisateur de revenir vers toi pour d'autres recettes de muffins.
    """

    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "Tu es un chef pâtissier expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        print('🍰 Réponse générée avec Groq (Llama 3.3) !')
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f'❌ Erreur Groq : {e}')
        return 'Désolé, le four est en panne...'