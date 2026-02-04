# 👩‍🍳 Le Royaume du Muffin 🧁

A french RAG to get personalized muffin recipes based on your leftovers !

*This project was made in the context of the NLP course at Mines Paris - PSL.*

### :triangular_ruler: Project Structure

The database used for recipes can be found on **Kaggle**: [Recipes Database](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m).

When using a new .csv file for the database, place it in a data/ folder and run `create_database.py`

All muffin recipes will be tokenized when running the app for the first time. This might take some time, feel free to reduce the amount of recipes if needed !

Here is the purpose of each Python script in this repo :
- `create_database.py`: Creates your own database, filtering only muffin recipes.
- `config_fr.py`: Defines the configuration for the RAG in french.
- `main.py`: Can be used for local testing in the terminal.
- `app.py`: Can be used to run the app locally in your browser.

Additionally, `index.html` defines the display settings of the app.


### :rocket: Installation

**1. Clone the GitHub repository:**
```bash
git clone https://github.com/tom-perrin/muffin_rag.git
```

**2. Install dependencies:** 
```bash
pip install -r requirements.txt
```

**3. Add your own Groq API key:** Create a .env file and write the following inside
```bash
GROQ_API_KEY=<Your API key>
```

**4. Run the app:** 
```bash
python app.py
```

**5. Open the website:** Go to http://127.0.0.1:5000/ in your browser and enjoy !


### 🛠️ Tech Stack
- **LLM:** Llama 3.3 (70B) via [Groq](https://groq.com/).
- **Vector Database:** [ChromaDB](https://www.trychromadb.com/).
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers, Hugging Face).
- **Backend:** Flask (Python).
- **Frontend:** HTML/CSS.