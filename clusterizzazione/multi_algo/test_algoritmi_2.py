import numpy as np
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch, BisectingKMeans, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_kernels

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]

# Dizionario degli algoritmi di clustering disponibili, inserisci qua le options come dizionario
algorithms = {
    'KMeans': KMeans,
    'AgglomerativeClustering': AgglomerativeClustering, #connectivity per implementare vincoli
    'SpectralClustering': SpectralClustering,
    'Birch': Birch,
    'BisectingKMeans': BisectingKMeans
}
#'FeatureAgglomeration': FeatureAgglomeration prova a afare qualcosa

# Dizionario delle metriche di valutazione clustering
evaluation_metrics = {
    'silhouette_score': silhouette_score,
    'calinski_harabasz_score': calinski_harabasz_score,
    'davies_bouldin_score': davies_bouldin_score
}

X = df[colonne_includere].values
results = pd.DataFrame(columns=["algorithm", "options", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"])

print(f"Running KMeans...")
for n in [5,6,7,8,9,10,11,12]:
    for init in ["k-means++", "random"]:
        for algo in ['lloyd', 'elkan']:
            for state in [10, 23, 35, 42, 55, 67, 78, 84]:
                model = KMeans(n_clusters=n, init=init, algorithm=algo, random_state=state)
                labels = model.fit_predict(X)
                options={'n_clusters': n, 'init': init, 'algorithm': algo, 'rand_state': state}
                scores = []
                for metric_name, metric_func in evaluation_metrics.items():
                    try:
                        score = metric_func(X, labels)
                        scores.append(score)
                    except Exception as e:
                        scores.append(None)
                        print(f"  {metric_name}: Error ({e})")
                results.loc[len(results)] = ['KMeans', options, scores[0], scores[1], scores[2]]

print(f"Running Agglomerative Clustering...")
for n in range(5,50):
    for link in ['average']:
        metric_list=['euclidean', 'l1', 'l2', 'manhattan'] if link!='ward' else ['euclidean']
        for metr in metric_list:
            model = AgglomerativeClustering(n_clusters=n, linkage=link, metric=metr)
            labels = model.fit_predict(X)
            options={'n_clusters': n, 'linkage': link, 'metric': metr}
            scores = []
            for metric_name, metric_func in evaluation_metrics.items():
                try:
                    score = metric_func(X, labels)
                    scores.append(score)
                except Exception as e:
                    scores.append(None)
                    print(f"  {metric_name}: Error ({e})")
            results.loc[len(results)] = ['Agglomerative', options, scores[0], scores[1], scores[2]]

print(f"Running SpectralClustering...")
for n in [5,6,7,8,9,10,11,12]:
    print(n)
    for affinity in ['rbf', 'nearest_neighbors', 'linear', 'poly', 'sigmoid', 'cosine', 'laplacian']:
        print(affinity)
        assign_labels_list = ['kmeans', 'discretize', 'cluster_qr']
        n_neighbors_list = [5,10,20,30,40,50] if affinity == 'nearest_neighbors' else [None]
        for assign_labels in assign_labels_list:
            print(assign_labels)
            for n_neighbors in n_neighbors_list:
                print(n_neighbors)
                for state in [10, 23, 35, 42, 55, 67, 78, 84]:
                    print(state)
                    try:
                        if affinity == 'nearest_neighbors':
                            model = SpectralClustering(n_clusters=n, affinity=affinity, n_neighbors=n_neighbors,
                                                      assign_labels=assign_labels, random_state=state)
                            options = {'n_clusters': n, 'affinity': affinity, 'n_neighbors': n_neighbors,
                                       'assign_labels': assign_labels, 'rand_state': state}
                        else:
                            model = SpectralClustering(n_clusters=n, affinity=affinity,
                                                      assign_labels=assign_labels, random_state=state)
                            options = {'n_clusters': n, 'affinity': affinity,
                                       'assign_labels': assign_labels, 'rand_state': state}
                        labels = model.fit_predict(X)
                        scores = []
                        for metric_name, metric_func in evaluation_metrics.items():
                            try:
                                score = metric_func(X, labels)
                                scores.append(score)
                            except Exception as e:
                                scores.append(None)
                                print(f"  {metric_name}: Error ({e})")
                        results.loc[len(results)] = ['SpectralClustering', options, scores[0], scores[1], scores[2]]
                    except Exception as e:
                        print(f"  SpectralClustering: Error ({e})")

print(f"Running Birch...")
for n in [5,6,7,8,9,10,11,12]:
    for threshold in [0.01,0.05,0.1,0.2,0.5,1,2,5,10]:
        for branching_factor in [10,20,30,40,50,100,200,300,400,500,1000]:
            model = Birch(n_clusters=n, threshold=threshold, branching_factor=branching_factor)
            labels = model.fit_predict(X)
            if len(list(set(labels)))!=n:
                continue
            options = {'n_clusters': n, 'threshold': threshold, 'branching_factor': branching_factor}
            scores = []
            for metric_name, metric_func in evaluation_metrics.items():
                try:
                    score = metric_func(X, labels)
                    scores.append(score)
                except Exception as e:
                    scores.append(None)
                    print(f"  {metric_name}: Error ({e})")
            results.loc[len(results)] = ['Birch', options, scores[0], scores[1], scores[2]]

print(f"Running BisectingKMeans...")
for n in [5,6,7,8,9,10,11,12]:
    for init in ['k-means++', 'random']:
        for algo in ['lloyd', 'elkan']:
            for strat in ['biggest_inertia','largest_cluster']:
                for state in [10, 23, 35, 42, 55, 67, 78, 84]:
                    model = BisectingKMeans(n_clusters=n, init=init, algorithm=algo, bisecting_strategy=strat, random_state=state)
                    labels = model.fit_predict(X)
                    options={'n_clusters': n, 'init': init, 'algorithm': algo, 'bisecting_strategy': strat, 'rand_state': state}
                    scores = []
                    for metric_name, metric_func in evaluation_metrics.items():
                        try:
                            score = metric_func(X, labels)
                            scores.append(score)
                        except Exception as e:
                            scores.append(None)
                            print(f"  {metric_name}: Error ({e})")
                    results.loc[len(results)] = ['BisectingKMeans', options, scores[0], scores[1], scores[2]]

# Ordina i risultati per ciascuna metrica e mostra i primi 3 algoritmi per ciascuna
for metric in ["silhouette_score", "calinski_harabasz_score"]:
    print(f"\nTop 20 algorithms by {metric}:")
    print(results.sort_values(by=metric, ascending=False)[["algorithm", "options", metric]].head(20))

print(f"\nTop 20 algorithms by davies_bouldin_score (lower is better):")
print(results.sort_values(by="davies_bouldin_score", ascending=True)[["algorithm", "options", "davies_bouldin_score"]].head(20))

# Seleziona gli algoritmi top in almeno due metriche
top_n = 20
top_silhouette = set(results.sort_values(by="silhouette_score", ascending=False).head(top_n).index)
top_calinski = set(results.sort_values(by="calinski_harabasz_score", ascending=False).head(top_n).index)
top_davies = set(results.sort_values(by="davies_bouldin_score", ascending=True).head(top_n).index)

# Algoritmi che sono top in almeno due metriche
top_two_or_more = (top_silhouette & top_calinski) | (top_silhouette & top_davies) | (top_calinski & top_davies)

if not top_two_or_more:
    print("No algorithms are top performers in at least two metrics.")
else:
    print("Algorithms top in at least two metrics:")
    print(results.loc[list(top_two_or_more)].sort_values(
        by=["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"],
        ascending=[False, False, True]
    )[["algorithm", "options", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]])

output_path = os.path.join(cartella_corrente, 'algorithm_df.pkl')
df.to_pickle(output_path)