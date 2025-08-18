#importo le librerie
import os
import pandas as pd
import pickle
import numpy as np
import random
import math

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, 'dataframes', 'df_raw_constraints.pkl')
df = pd.read_pickle(pkl_path)

df['clusters_list']= [[] for _ in range(len(df))]
df['clusters'] = -1
soglie_den=[50,350]
soglie_consumption=[2*(10**6), 15*(10**6), 100*(10**6)]

for i in range(len(soglie_den)+1):
    lower1_d = 0 if i == 0 else soglie_den[i-1]*1.1
    upper1_d = np.inf if i == len(soglie_den) else soglie_den[i]*0.9
    for j in range(len(soglie_consumption)+1):
        value = i * (len(soglie_consumption) + 1) + j
        df1 = df[(df['Densità_pop'] >= lower1_d) & (df['Densità_pop'] <= upper1_d)]
        lower1_c = 0 if j == 0 else soglie_consumption[j-1]*1.1
        upper1_c = np.inf if j == len(soglie_consumption) else soglie_consumption[j]*0.9
        df2 = df1[(df1['consumption'] >= lower1_c) & (df1['consumption'] <= upper1_c)]
        for ind in df2.index:
            df.loc[ind, 'clusters_list'].append(value)
        if j != len(soglie_consumption):
            lower2_c = soglie_consumption[j]*0.9
            upper2_c = soglie_consumption[j]*1.1
            df2 = df1[(df1['consumption'] > lower2_c) & (df1['consumption'] < upper2_c)]
            for ind in df2.index:
                df.loc[ind, 'clusters_list'].append(value)
                df.loc[ind, 'clusters_list'].append(value+1)
        if i != len(soglie_den):
            lower2_d = soglie_den[i]*0.9
            upper2_d = soglie_den[i]*1.1
            df1 = df[(df['Densità_pop'] > lower2_d) & (df['Densità_pop'] < upper2_d)]
            df2 = df1[(df1['consumption'] >= lower1_c) & (df1['consumption'] <= upper1_c)]
            for ind in df2.index:
                df.loc[ind, 'clusters_list'].append(value)
                df.loc[ind, 'clusters_list'].append(value+4)
            if j != len(soglie_consumption):
                df2 = df1[(df1['consumption'] > lower2_c) & (df1['consumption'] < upper2_c)]
                for ind in df2.index:
                    df.loc[ind, 'clusters_list'].append(value)
                    df.loc[ind, 'clusters_list'].append(value+1)
                    df.loc[ind, 'clusters_list'].append(value+4)
                    df.loc[ind, 'clusters_list'].append(value+5)

for i, isl in df.iterrows():
    clusters_list = isl.clusters_list
    must = isl.must
    if len(clusters_list) == 1:
        df.loc[i, 'clusters'] = clusters_list[0]
    if len(clusters_list) == 2:
        if len(must) == 0 or (len(must) == 1 and list(must)[0] == i):
            if clusters_list[1]-clusters_list[0] == 1:
                if isl.consumption < soglie_consumption[clusters_list[0]%4]:
                    df.loc[i, 'clusters'] = clusters_list[0]
                    df.loc[i, 'clusters_list'].remove(clusters_list[1])
                else:
                    df.loc[i, 'clusters'] = clusters_list[1]
                    df.loc[i, 'clusters_list'].remove(clusters_list[0])
            elif clusters_list[1]-clusters_list[0] == 4:
                if isl.Densità_pop < soglie_den[clusters_list[0]//4]:
                    df.loc[i, 'clusters'] = clusters_list[0]
                    df.loc[i, 'clusters_list'].remove(clusters_list[1])
                else:
                    df.loc[i, 'clusters'] = clusters_list[1]
                    df.loc[i, 'clusters_list'].remove(clusters_list[0])
    if len(clusters_list) == 4:
        if len(must) == 0 or (len(must) == 1 and list(must)[0] == i):
            if isl.Densità_pop < soglie_den[clusters_list[0]//4]:
                df.loc[i, 'clusters_list'].remove(clusters_list[3])
                df.loc[i, 'clusters_list'].remove(clusters_list[2])
            else:
                df.loc[i, 'clusters_list'].remove(clusters_list[1])
                df.loc[i, 'clusters_list'].remove(clusters_list[0])
            if isl.consumption < soglie_consumption[clusters_list[0]%4]:
                df.loc[i, 'clusters_list'].remove(clusters_list[1])
            else:
                df.loc[i, 'clusters_list'].remove(clusters_list[0])
            df.loc[i, 'clusters'] = df.loc[i, 'clusters_list'][0]

len_df = len(df[df['clusters'] == -1])
print(f'elementi non assegnati prima della propagazione sul grafo (nessun vincolo violato con questa assegnazione): {len_df}')

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
    groups = find_groups(vincoli)
    assegnamenti = assegnato[:]
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
            df.loc[i, 'clusters'] = assegnamento
assegnamento_ottimizzato(df['clusters_list'], df['clusters'], df['must'])
len_df = len(df[df['clusters'] == -1])
print(f'elementi non assegnati prima della propagazione sul grafo (nessun vincolo violato con questa assegnazione): {len_df}')

#ricerca locale e simulated annealing
def compute_violations(assignments, must_link):
    violations = 0
    for i in range(len(assignments)):
        for j in must_link[i]:
            if assignments[i] != assignments[j]:
                violations += 1
    return violations
def local_search(assignments, possible_assignments, not_assigned_indexes, must_link, max_iter=10000):
    for i in not_assigned_indexes:
        assignments[i] = random.choice(possible_assignments[i])
    best_assignments = assignments.copy()
    best_violations = compute_violations(assignments, must_link)
    for _ in range(max_iter):
        improved = False
        #itero in ordine casuale, cambiando l'ordine di volta in volta
        for i in random.sample(not_assigned_indexes, len(not_assigned_indexes)):
            current_group = assignments[i]
            best_local = assignments[i]
            min_local_violations = best_violations
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
    if i%10 == 0:
        print(i)
    score = (local_search(list(df['clusters']), list(df['clusters_list']), list(df[df['clusters'] == -1].index), list(df['must']))[1])
    if score < min:
        print('miglioramento')
        min = score
        
def simulated_annealing(assignments, possible_assignments, not_assigned_indexes, must_link,
                        max_iter=10000, T_start=100.0, T_end=1e-3, alpha=0.995):
    for i in not_assigned_indexes:
        assignments[i] = random.choice(possible_assignments[i])
    current_assignments = assignments.copy()
    current_violations = compute_violations(current_assignments, must_link)

    best_assignments = current_assignments.copy()
    best_violations = current_violations

    T = T_start

    for _ in range(max_iter):
        if T < T_end:
            break

        # Scegli un elemento casuale da modificare
        i = random.choice(not_assigned_indexes)
        current_group = current_assignments[i]
        new_group = random.choice([g for g in possible_assignments[i] if g != current_group])

        # Applica la mossa
        current_assignments[i] = new_group
        new_violations = compute_violations(current_assignments, must_link)

        delta = new_violations - current_violations

        if delta <= 0:
            # Miglioramento → accetta
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_assignments = current_assignments.copy()
        else:
            # Peggioramento → accetta con probabilità e^(-delta/T)
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_violations = new_violations
            else:
                # Annulla la mossa
                current_assignments[i] = current_group

        # Raffreddamento
        T *= alpha

    return best_assignments, best_violations

def constraint(dataframe, labels):
    violations1 = 0
    violations2 = 0
    tot_cannot = 0
    tot_must = 0
    dataframe['clusters'] = labels
    for i, isl in dataframe.iterrows():
        cannot = isl.cannot
        must = isl.must
        df1 = dataframe[dataframe['clusters'] == isl.clusters]
        violations1 += len(df1[df1.index.isin(cannot)])
        df1 = dataframe[dataframe['clusters'] != isl.clusters]
        violations2 += len(df1[df1.index.isin(must)])
        tot_cannot += len(cannot)
        tot_must +=len(must)
    return violations1, violations2

labels = simulated_annealing(list(df['clusters']), list(df['clusters_list']), list(df[df['clusters'] == -1].index), list(df['must']))
if (constraint(df, labels[0])) == (0, min):
    df['clusters'] = labels[0]

output_folder = os.path.join(cartella_corrente, 'dataframes')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_raw_first_step.pkl')
df.to_pickle(output_path)

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, 'dataframes', 'df_norm_constraints.pkl')
df = pd.read_pickle(pkl_path)
df['clusters'] = labels[0]
output_folder = os.path.join(cartella_corrente, 'dataframes')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_norm_first_step.pkl')
df.to_pickle(output_path)