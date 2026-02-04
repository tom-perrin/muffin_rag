from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from config_fr import get_or_create_collection, generate_answer, EMBEDDING_MODEL_NAME

app = Flask(__name__)

# Chargement du cerveau (une seule fois au démarrage)
df = pd.read_csv('./data/muffin_dataset.csv')
db = get_or_create_collection(df, db_path='./chromadb_full')
model_embed = SentenceTransformer(EMBEDDING_MODEL_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')
    # RAG Logic
    query_vector = model_embed.encode([user_query]).tolist()
    results = db.query(query_embeddings=query_vector, n_results=1)
    
    answer = generate_answer(user_query, results)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)