import numpy as np
from scipy.linalg import eigh
import os
import pandas as pd
import pickle
import geopandas as gp

#funzione per creare liste di indici con links di elementi dalle marici dei links
def nodes_with_links(links):
    #indici delle righe (quindi elementi) della matrice con almeno un elemento diverso da zero, se avessi fatto con le colonne (axis=0) ottenevo lo stesso risultato, matrice simmetrica
    return np.any(links > 0, axis=1).nonzero()[0].tolist()

#funzione per riproiettare i punti
def learn_projection(X, M, C, alpha=0.01):
    #dimensioni dei dati, m elementi n componenti
    n, m = X.shape
    #costruzione dei Laplaciani
    D_M = np.diag(np.sum(M, axis=1))
    L_M = D_M - M
    D_C = np.diag(np.sum(C, axis=1))
    L_C = D_C - C
    #matrici B e P (P non singolare)
    P = X @ L_M @ X.T + alpha * np.eye(n)
    B = X @ L_C @ X.T
    #autovalori autovettori
    eigvals, eigvecs = eigh(B, P)
    #ordinamento decrescente rispetto agli autovalori
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    #restituisce la matrice di proiezione ω (dimensione n x n)
    omega = eigvecs[:, :n]
    return omega

#funzioni per trovare i neighboors con must e cannot links
def get_tau_pi(Y,M,u,k):
    #estraggo il vettore u e trovo gli indici di quelli collegati, se non ci sono ritorno lista vuota
    y_u = Y[:, u]
    must_link_indices = np.where(M[u] > 0)[0]
    if len(must_link_indices) == 0: return []
    #calcolo la distanza dei collegati come norma della differenza tra i vari vettori e il vettore u, faccio la media e calcolo le distanze con tutti i punti
    distances_to_ml = np.linalg.norm(Y[:, must_link_indices] - y_u[:, np.newaxis], axis=0)
    avg_ml_distance = np.mean(distances_to_ml)
    all_distances = np.linalg.norm(Y - y_u[:, np.newaxis], axis=0)
    #maschera per escludere i valori collegati, u stesso e quelli lontani
    mask_valid = (M[u] == 0)
    mask_valid[u] = False
    mask_closer_than_avg = all_distances < avg_ml_distance
    final_mask = mask_valid & mask_closer_than_avg
    #applicazione e ordinamento crescente di quelli validi
    candidate_indices = np.where(final_mask)[0]
    sorted_candidates = candidate_indices[np.argsort(all_distances[candidate_indices])]
    #ritorno i massimo k piu vicini
    return sorted_candidates[:k].tolist()
def get_kappa_pi(Y, C, u, k, mu):
    #estraggo il vettore u, estraggo gli indici degli elementi collegati con cannot, se sono 0 ritorno una lista vuota
    y_u = Y[:, u]
    cannot_link_indices = np.where(C[u] > 0)[0]
    if len(cannot_link_indices) == 0:   return []
    #calcolo la distanza minima tra u e questi elementi
    distances_to_cl = np.linalg.norm(Y[:, cannot_link_indices] - y_u[:, np.newaxis], axis=0)
    min_distance_cl = np.min(distances_to_cl)
    #calcolo la distanza con tutti gli elementi e creo una maschera prendendo gli elemnti piu vicii del minimo cannot e della media degli ml
    all_distances = np.linalg.norm(Y - y_u[:, np.newaxis], axis=0)
    mask_closer_than_min = all_distances < min_distance_cl
    mask_closer_than_mu = all_distances < mu
    final_mask = mask_closer_than_min & mask_closer_than_mu
    #la applico, ordino e ritorno i massimo k piu vicini
    candidate_indices = np.where(final_mask)[0]
    sorted_candidates = candidate_indices[np.argsort(all_distances[candidate_indices])]
    return sorted_candidates[:k].tolist(), cannot_link_indices

#funzioni per nuovi must e cannot links
def update_must_link_constraints(Y, ml, M, beta, k):
    #itero per gli elementi con Must Link
    for u in ml:
        #trovo i neighboors
        pi_u = get_tau_pi(Y, M, u, k)
        #aggiorno la matrice
        for j in pi_u:
            M[u, j] = min(1 - beta, M[u, j] + beta)
            M[j, u] = M[u, j]
    return M
def update_cannot_link_constraints(Y, cl, C, beta, k, mu):
    #itero per gli elementi con Cannot Link
    for u in cl:
        #trovo i neighboors
        pi_u,C_u = get_kappa_pi(Y, C, u, k, mu)
        #aggiorno la matrice
        for i in C_u:
            for j in pi_u:
                C[i, j] = min(1 - beta, C[i, j] + beta)
                C[j, i] = C[i, j]
    return C

#funzione per regolarizzare conflitti tra M e C
def regularize_matrices(M, C):
    mask_M_is_1 = (M == 1)
    C[mask_M_is_1] = 0
    mask_C_is_1 = (C == 1)
    M[mask_C_is_1] = 0
    mask=(M * C > 0)
    somma_masked = M[mask] + C[mask]
    C[mask] = np.minimum(C[mask], (C[mask] / somma_masked))
    M[mask] = np.minimum(M[mask], (M[mask] / somma_masked))
    return M, C

#funzione che controlla se il numero di vincoli generati sia sufficiente
def check_convergence(M, C, percent_threshold):
    total_elements = M.shape[0] * M.shape[1]
    nonzero_count = np.count_nonzero(M + C)
    percent_full = 100 * nonzero_count / total_elements
    return True if percent_full >= percent_threshold else False

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe e lo trasformo in una np.matrix
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/risultati/analysis_df.pkl")
df = pd.read_pickle(pkl_path)
X = np.matrix(df.T.values)

#importa M e C, devono essere np.matrix dimensione mxm con m numero di isole
pkl_path = os.path.join(cartella_corrente, "must_link.pkl")
M = pickle.load(pkl_path)
pkl_path = os.path.join(cartella_corrente, "cannot_link.pkl")
C = pickle.load(pkl_path)
#applico l'algoritmo per aumentare il numero di vincoli dopo averne definito i parametri
max_iter=100
percent_treshold=20
beta=0.2
k=3
#creo liste elementi con links
ml_nodes=nodes_with_links(M)
cl_nodes=nodes_with_links(C)
#itero fino a max_iterazioni o matrici vincoli abbastanza dense
for _ in range(max_iter):
    # E-step: trova ω* ottimizzando la funzione obiettivo
    omega_star = learn_projection(X, M, C)
    #proiezione di X nel nuovo spazio
    Y = omega_star.T @ X
    #M-step, aggiornamento vincoli must-link e cannot-link e relative liste
    M = update_must_link_constraints(Y, ml_nodes, M, beta, k)
    ml_nodes=nodes_with_links(M)
    #elementi collegati da un mustlink, np.triu considera solo la parte sopra visto che è simmetrica
    i_indices, j_indices = np.where(np.triu(M, k=1) > 0)
    #calcolo le ditanze tra elementi collegati e la media
    distances = np.linalg.norm(Y[:, i_indices] - Y[:, j_indices], axis=0)
    mu=np.mean(distances)
    C = update_cannot_link_constraints(Y, cl_nodes, C, beta, k, mu)
    cl_nodes = nodes_with_links(C)

    #regolarizzazione
    M, C = regularize_matrices(M, C)
    #criterio di arresto
    if check_convergence(M, C, percent_treshold):
        break

n_clust_max=20
n_clust_min=0

N = M.shape[0]
#insieme degli indici degli elementi non ancora assegnati e di quelli assegnati (inizialmente vuoto)
X_idx = set(range(N))
assigned=set()
#creo i clusters come lista di liste
clusters = []
#must-link gradi
degrees = np.sum(M, axis=1)
#itero fino ad aver assegnato tutti gli elementi a un cluster
while X_idx:
    #elemento con grado piu alto
    xi = max(X_idx, key=lambda idx: degrees[idx])
    #set di elementi connessi a xi con must e cannot non precedentemente assegnati
    Kh = {xi} | {j for j in range(N) if M[xi, j] != 0 and j not in assigned}
    ICh = {j for j in range(N) if C[xi, j] != 0}
    #creo il cluster
    changed = True
    while changed:
        Kh1=Kh.copy()
        #itero per i vicini di xi
        for xj in list(Kh):
            #se in un'ietrazione prima un elemento aveva xj tra i cannot link xj è stato rimosso da Kh ma il ciclo for ancora non lo riconosce
            if xj in ICh:
                continue
            #elementi collegati sufficientemente forte con must o cannot
            Mj = {j for j in range(N) if M[xj, j] >= 0.5}
            Cj = {j for j in range(N) if C[xj, j] >= 0.5}
            if len(Mj & ICh) >= len(Mj & Kh):
                Kh1.discard(xj)
            else:
                ICh.update(Cj)
                Kh = (Kh | Mj) - ICh
        if Kh1==Kh:
            changed=False
        else:
            changed=True
            Kh=Kh1.copy()
    
    #aggiungo il set alla lista di clusters e aggiorno i set di elementi non assegnati e assegnati
    clusters.append(Kh)
    X_idx -= Kh
    assigned += Kh

#importo il geodataframe
gdf_path=os.path.join(cartella_progetto, 'data/isole_filtrate/finali/isole_arro4.gpkg')
gdf=gp.read_file(gdf_path)
output_path=os.path.join(cartella_corrente, 'geodataframe_clusterizzati')
os.makedirs(output_path, exist_ok=True)

K = len(clusters)
#ripeto per numero di clusters diversi riducendolo di volta in volta
while K>n_clust_min:
    #se ho meno clusters del numero massimo esporto
    if K<n_clust_max:
        #creo una copia del dataframe, aggiungo la colonna dei clusters e della geometria prima di esportare
        df1=df.copy()
        df1['Cluster_label']=0
        for i in range(K):
            for j in clusters[i]:
                df1.iloc[j]['Cluster_label']=i
        df1 = pd.merge(df1, gdf[['ALL_Uniq', 'geometry']], on='ALL_Uniq', how='left')
        gdf1=gp.GeoDataFrame(df1, geometry='geometry', crs="EPSG:4326")
        folder_out=os.path.join(output_path, f'{K}')
        os.makedirs(folder_out, exist_ok=True)
        out_file=os.path.join(folder_out, 'isole_clusters.gpkg')
        gdf1.to_file(out_file, driver="GPKG")
        for j in range(K):
            gdf2=gdf1[(gdf1['Cluster_Label']==j)]
            out_file=os.path.join(folder_out, f'isole_clusters_{j}.gpkg')
            gdf2.to_file(out_file, driver="GPKG")

    #creo la matrice di somiglianza
    r_matrix = np.full((K, K), np.inf)
    for i in range(K):
        for j in range(i + 1, K):
            Ki, Kj = clusters[i], clusters[j]
            #coppie miste
            Ki_Kj = [(a, b) for a in Ki for b in Kj]
            #vincoli tra coppie miste
            num_cannot = sum(C[a, b] > 0 for a, b in Ki_Kj)
            num_must = sum(M[a, b] > 0 for a, b in Ki_Kj)
            if num_must > 0:
                r_matrix[i, j] = num_cannot / num_must
    #sosituisco i valori inf con valori grandi, per poter prendere il minore
    r_matrix[np.isinf(r_matrix)] = np.max(r_matrix[np.isfinite(r_matrix)]) * 10
    #valore minimo
    min_idx = np.unravel_index(np.argmin(r_matrix), r_matrix.shape)
    i, j = min_idx
    #unisco i clusters, li rimuovo e aggiungo quello unito
    new_cluster = clusters[i] + clusters[j]
    clusters = [clusters[idx] for idx in range(K) if idx not in (i, j)]
    clusters.append(new_cluster)
    K=len(clusters)
    