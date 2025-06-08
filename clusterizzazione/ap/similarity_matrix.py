#importo le librerie
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe e i vincoli
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/normalization/risultati/analysis_df.pkl")
df = pd.read_pickle(pkl_path)

pkl_path = os.path.join(cartella_corrente, 'preparazione', 'must_link.pkl')
ml = pickle.load(open(pkl_path, 'rb'))
pkl_path = os.path.join(cartella_corrente, 'preparazione', 'cannot_link.pkl')
cl = pickle.load(open(pkl_path, 'rb'))

#seleziono solo le colonne numeriche
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res', ]
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]
df = df[colonne_da_includere].select_dtypes(include='number')

#creo la matrice delle distanze e quella di similarita come il suo opposto
distance_matrix = euclidean_distances(df)
similarity_matrix_base = -distance_matrix

#mediana da impostare sulle diagonali
off_diagonal_similarities = similarity_matrix_base[np.triu_indices(similarity_matrix_base.shape[0], k=1)]
median_preference = np.median(off_diagonal_similarities)

#matrice delle distanze contenenti i vincoli
constrained_similarity_matrix = np.copy(similarity_matrix_base)
for mli in ml:
    for i, j in mli:
        constrained_similarity_matrix[i, j] = 0.0
        constrained_similarity_matrix[j, i] = 0.0
for cli in cl:
    for i, j in cli:
        constrained_similarity_matrix[i, j] = -1e8
        constrained_similarity_matrix[j, i] = -1e8

#esporto la matrice di similarita
file_path = os.path.join(cartella_corrente, 'preparazione', 'similarity_matrix.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(constrained_similarity_matrix, file)
file_path = os.path.join(cartella_corrente, 'preparazione', 'median_distance.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(off_diagonal_similarities, file)