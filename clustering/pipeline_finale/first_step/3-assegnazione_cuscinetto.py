#importo le librerie
import os
import pandas as pd
import numpy as np
import random
import math

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, "dataframes/df_raw_constraints.pkl")
df = pd.read_pickle(pkl_path)

soglie_den=[50,350]
soglie_consumption=[2*(10**6), 15*(10**6), 100*(10**6)]

n = len(df[df['cluster']==-1])
print(f'isole non assegnate con soglie rigide: {n}')

#assegno le isole senza must link
for i, isl in df.iterrows():
    if isl.cluster != -1:
        continue
    dens_list, consumption_list = isl.dens_cluster_list, isl.consumption_cluster_list
    if len(dens_list) == 2 and len(isl.must) == 0:
        if isl.Densità_pop < soglie_den[dens_list[0]]:
            del dens_list[1]
        else:
            del dens_list[0]
    if len(consumption_list) == 2 and len(isl.must) == 0:
        if isl.consumption < soglie_consumption[consumption_list[0]]:
            del consumption_list[1]
        else:
            del consumption_list[0]
    if len(dens_list)==1 and len(consumption_list)==1:
        value = dens_list[0] * (len(soglie_consumption) + 1) + consumption_list[0]
        df.loc[i,'cluster']=value

n = len(df[df['cluster']==-1])
print(f"isole non assegnate considerando l'assenza di must link: {n}")

condizione1 = df['dens_cluster_list'].apply(len) == 1
condizione2 = df['consumption_cluster_list'].apply(len) == 2
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti alla zona cuscinetto solo per i consumi (due possibili assegnazioni): {numero}')

condizione1 = df['dens_cluster_list'].apply(len) == 2
condizione2 = df['consumption_cluster_list'].apply(len) == 1
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti alla zona cuscinetto solo per la densità (due possibili assegnazioni): {numero}')

condizione1 = df['dens_cluster_list'].apply(len) == 2
condizione2 = df['consumption_cluster_list'].apply(len) == 2
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti a entrambe le zone cuscinetto (quattro possibili assegnazioni): {numero}')

#per ogni elemento creo la lista di possibili cluster, non relativi alle singole variabili, metto l'etichetta del nome complessivo
df['cluster_list'] = [[] for _ in range(len(df))]
for i,isl in df.iterrows():
    if isl.cluster != -1:
        df.loc[i,'cluster_list'].append(isl.cluster)
    if (len(isl.dens_cluster_list) == 2) and (len(isl.consumption_cluster_list) == 1):
        value = isl.dens_cluster_list[0] * 4 + isl.consumption_cluster_list[0]
        df.loc[i,'cluster_list'].append(value)
        value +=4
        df.loc[i,'cluster_list'].append(value)
    if (len(isl.dens_cluster_list) == 1) and (len(isl.consumption_cluster_list) == 2):
        value = isl.dens_cluster_list[0] * 4 + isl.consumption_cluster_list[0]
        df.loc[i,'cluster_list'].append(value)
        value +=1
        df.loc[i,'cluster_list'].append(value)
    if (len(isl.dens_cluster_list) == 2) and (len(isl.consumption_cluster_list) == 2):
        value = isl.dens_cluster_list[0] * 4 + isl.consumption_cluster_list[0]
        df.loc[i,'cluster_list'].append(value)
        value +=1
        df.loc[i,'cluster_list'].append(value)
        value += 3
        df.loc[i,'cluster_list'].append(value)
        value += 1
        df.loc[i,'cluster_list'].append(value)

#funzione che parte da un insieme di liste di must link e ritorna un gruppo di elementi collegati
def find_groups(vincoli):
    n = len(vincoli)
    visited = [False] * n
    groups = []
    def dfs(i, current):
        visited[i] = True
        current.append(i)
        for j in vincoli[i]:
            if not visited[j]:
                dfs(j, current)
    for i in range(n):
        if not visited[i]:
            current = []
            dfs(i, current)
            groups.append(current)
    return groups
def assegnamento_ottimizzato(possibili_clusters, assegnato, vincoli):
    #gruppi di elementi collegati
    groups = find_groups(vincoli)
    #creo una copia
    assegnamenti = assegnato[:]
    #itero e cerco assegnazioni comuni, se presenti le eseguo
    for group in groups:
        possibili = [set(possibili_clusters[i]) for i in group]
        intersezione = set.intersection(*possibili)
        if not intersezione:
            continue
        if len(intersezione)>1:
            print('più di un cluster possibile')    
        assegnamento = intersezione.pop()
        for i in group:
            if assegnamento not in possibili_clusters[i]:
                print('prob')
            if assegnamenti[i] != -1 and assegnamenti[i] != assegnamento:
                print('prob')
            df.loc[i, 'cluster'] = assegnamento
assegnamento_ottimizzato(df['cluster_list'], df['cluster'], df['must'])
n = len(df[df['cluster'] == -1])
print(f'elementi non assegnati dopo della propagazione sul grafo (nessun vincolo violato con questa assegnazione): {n}')

#ricerca locale
#funzione che prende in input gli assegnamenti e i vincoli e calcola le violazioni
def compute_violations(assignments, must_link):
    violations = 0
    for i in range(len(assignments)):
        for j in must_link[i]:
            if assignments[i] != assignments[j]:
                violations += 1
    return violations
def local_search(assignments, possible_assignments, not_assigned_indexes, must_link, max_iter=10000):
    #itero per elementi non assegnati scegliendo un'assegnazione casuale tra le possibili
    for i in not_assigned_indexes:
        assignments[i] = random.choice(possible_assignments[i])
    best_assignments = assignments.copy()
    best_violations = compute_violations(assignments, must_link)
    for _ in range(max_iter):
        improved = False
        #itero in ordine casuale, cambiando l'ordine di volta in volta
        for i in random.sample(not_assigned_indexes, len(not_assigned_indexes)):
            current_group = assignments[i]
            #parametri ricerca locale, per valutare miglioramenti
            best_local = assignments[i]
            min_local_violations = best_violations
            #itero per i possibili assegnamenti
            for g in possible_assignments[i]:
                if g == current_group:
                    continue
                assignments[i] = g
                v = compute_violations(assignments, must_link)
                if v < min_local_violations:
                    best_local = g
                    min_local_violations = v
                    improved = True
            assignments[i] = best_local
            best_assignments = assignments.copy()
            best_violations = min_local_violations
        if not improved:
            break
    return best_assignments,best_violations

#ripetizione ricerca locale
min = 100000
for i in range(200):
    score = (local_search(list(df['cluster']), list(df['cluster_list']), list(df[df['cluster'] == -1].index), list(df['must']))[1])
    if score < min:
        min = score
print(f"L'algoritmo di ricerca locale arriva a una soluzione con {min} violazioni")

#simulated annealing
def simulated_annealing(assignments, possible_assignments, not_assigned_indexes, must_link,
                        max_iter=10000, T_start=100.0, T_end=1e-3, alpha=0.995):
    #assegnazione iniziale casuale e calcolo violazioni
    for i in not_assigned_indexes:
        assignments[i] = random.choice(possible_assignments[i])
    #valori correnti
    current_assignments = assignments.copy()
    current_violations = compute_violations(current_assignments, must_link)
    #usati come iniziali valori ottimi
    best_assignments = current_assignments.copy()
    best_violations = current_violations
    #inizializzazione temperatura
    T = T_start
    #itero
    for _ in range(max_iter):
        #condizione arresto
        if T < T_end:
            break
        # scelta elemento casuale da modificare
        i = random.choice(not_assigned_indexes)
        current_group = current_assignments[i]
        # cambiamento casuale
        new_group = random.choice([g for g in possible_assignments[i] if g != current_group])
        current_assignments[i] = new_group
        new_violations = compute_violations(current_assignments, must_link)
        delta = new_violations - current_violations
        #accettare miglioramento per situazione corrente e migliore
        if delta <= 0:
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_assignments = current_assignments.copy()
        #accettare peggioramento per situazione corrente con probabilità decrescente
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_violations = new_violations
            else:
                current_assignments[i] = current_group
        #riduco la temperatura
        T *= alpha
    return best_assignments, best_violations

labels, score = simulated_annealing(list(df['cluster']), list(df['cluster_list']), list(df[df['cluster'] == -1].index), list(df['must']))
print(f"L'algoritmo di ricerca locale arriva a una soluzione con {score} violazioni")
df['cluster'] = labels
df.drop(['dens_cluster_list', 'consumption_cluster_list', 'must', 'cluster_list'], axis=1, inplace=True)
#esportazione
output_folder = os.path.join(cartella_corrente, 'dataframes')
output_path = os.path.join(output_folder, 'df_raw_first_step.csv')
df.to_csv(output_path, index=False, encoding='utf-8')

#importo il dataframe normalizzato
csv_path = os.path.join(cartella_progetto, "exploratory_data_analisys\df_norm.csv")
df = pd.read_csv(csv_path)
df['cluster'] = labels
output_folder = os.path.join(cartella_corrente, 'dataframes')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_norm_first_step.csv')
df.to_csv(output_path, index=False, encoding='utf-8')