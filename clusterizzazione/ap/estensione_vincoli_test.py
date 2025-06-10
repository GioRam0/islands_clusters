#importo le librerie
import os
import pickle
from collections import defaultdict
from itertools import combinations

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo i vincoli
pkl_path = os.path.join(cartella_corrente, 'preparazione', 'must_link.pkl')
ml = pickle.load(open(pkl_path, 'rb'))
pkl_path = os.path.join(cartella_corrente, 'preparazione', 'cannot_link.pkl')
cl = pickle.load(open(pkl_path, 'rb'))

ml_names=['geot', 'hydro', 'off', 'wind', 'solar', 'nores']
cl_names={0:  'densita', 1: 'gdp proc', 2: 'dimensione', 3: 'wind', 4: 'solar', 5: 'nores'}

def estensione(ml, cont):
    #creo un grafo per le componenti connesse
    grafo = defaultdict(set)
    for a, b in ml:
        grafo[a].add(b)
        grafo[b].add(a)
    #insieme di elementi visitati e di gruppi di elementi connessi
    visitato = set()
    componenti = []
    #funzione per esplorare i nodi connessi a un nodo
    def dfs_iterativo(start):
        stack = [start]
        componente = []
        while stack:
            nodo = stack.pop()
            if nodo not in visitato:
                visitato.add(nodo)
                componente.append(nodo)
                stack.extend(grafo[nodo] - visitato)
        return componente
    #applico la funzione ai vari nodi e inserisco nella lista componenti un lista di elementi connessi
    for nodo in grafo:
        if nodo not in visitato:
            componente=dfs_iterativo(nodo)
            componenti.append(componente)
    #creo un nuovo set con i vincoli
    nuova_ml = set()
    for comp in componenti:
        for a, b in combinations(comp, 2):
            nuova_ml.add((min(a, b), max(a, b)))
    print(f'ml {ml_names[cont]}')
    print(f'ml originale {len(ml)}')
    print(f'nuovo ml {len(list(nuova_ml))}')
    print(f'nuovi vincoli {len(list(nuova_ml))-len(ml)}')
    for i in range(len(cl)):
        print(f'cl {cl_names[i]} interseca estensione {len(nuova_ml & set(cl[i]))} volte')
    print(' ')
    return list(nuova_ml)

k=0
for mli in ml:
    estensione(mli,k)
    k+=1