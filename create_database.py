import pandas as pd

DATA_PATH = './data/full_dataset.csv'

def load_data(path=DATA_PATH):
    '''
    Charge toutes les données des recettes situées à path.
    '''
    df = pd.read_csv(path)
    return df

def filter_only_muffins(df):
    '''
    Filtre strict pour ne garder que les recettes de muffins.
    '''
    keywords = ['muffin', 'cupcake']
    muffin_mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
    return df[muffin_mask]

# --- CHARGEMENT DES DONNEES DEPUIS LE DATASET RECIPENLG ---
df = load_data()
df_muffins = filter_only_muffins(df)
df_muffins.to_csv('./data/muffin_dataset.csv')