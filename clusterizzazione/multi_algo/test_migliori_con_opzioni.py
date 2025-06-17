import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans, Birch, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)

colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]

algorithms = {
    "KMeans": KMeans,
    "BisectingKMeans": BisectingKMeans,
    "Birch": Birch,
    "AgglomerativeClustering": AgglomerativeClustering,
    "SpectralClustering": SpectralClustering
}
evaluation_metrics = {
    "silhouette_score": silhouette_score,
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score
}
columns = ['algorithm', 'options'] + list(evaluation_metrics.keys())
results = pd.DataFrame(columns=columns)

X = df[colonne_includere].values

print('Kmeans...')
for n in range(5, 13):
    print(n)
    for init in ['k-means++', 'random']:
        for algo in ['lloyd', 'elkan']:
            for state in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
                options = {
                    'n_clusters': n,
                    'init': init,
                    'algorithm': algo,
                    'random_state': state
                }
                kmeans = KMeans(n_clusters=n, init=init, random_state=state, algorithm=algo)
                labels = kmeans.fit_predict(X)
                scores = []
                for metric_name, metric_func in evaluation_metrics.items():
                    try:
                        score = metric_func(X, labels)
                    except Exception as e:
                        print(f"Error calculating {metric_name} for options {options}: {e}")
                        score = np.nan
                    scores.append(score)
                results.loc[len(results)] = ['KMeans', options] + scores

print('BisectingKMeans...')
for n in range(5, 13):
    print(n)
    for init in ['k-means++', 'random']:
        for algo in ['lloyd', 'elkan']:
            for state in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
                options = {
                    'n_clusters': n,
                    'init': init,
                    'algorithm': algo,
                    'random_state': state
                }
                bisect_kmeans = BisectingKMeans(n_clusters=n, init=init, algorithm=algo, random_state=state)
                labels = bisect_kmeans.fit_predict(X)
                scores = []
                for metric_name, metric_func in evaluation_metrics.items():
                    try:
                        score = metric_func(X, labels)
                    except Exception as e:
                        print(f"Error calculating {metric_name} for options {options}: {e}")
                        score = np.nan
                    scores.append(score)
                results.loc[len(results)] = ['BisectingKMeans', options] + scores

print('Birch...')
for n in range(5, 13):
    print(n)
    for threshold in [0.01,0.02,0.05,0.1,0.2,0.3,0.5, 1.0, 1.5,2,5,10]:
        for branching_factor in [10, 20, 30, 40, 50, 75, 100, 150, 200, 500]:
            options = {
                'n_clusters': n,
                'threshold': threshold,
                'branching_factor': branching_factor
            }
            birch = Birch(n_clusters=n, threshold=threshold, branching_factor=branching_factor)
            labels = birch.fit_predict(X)
            if len(list(set(labels)))>=n-2:
                scores = []
                for metric_name, metric_func in evaluation_metrics.items():
                    try:
                        score = metric_func(X, labels)
                    except Exception as e:
                        print(f"Error calculating {metric_name} for options {options}: {e}")
                        score = np.nan
                    scores.append(score)
                results.loc[len(results)] = ['Birch', options] + scores

print('AgglomerativeClustering...')
for n in range(5, 50):
    print(n)
    for linkage in ['ward', 'complete', 'average', 'single']:
        # 'ward' linkage only supports 'euclidean' affinity
        metric_list = ['euclidean'] if linkage == 'ward' else ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        for metric in metric_list:
            options = {
                'n_clusters': n,
                'linkage': linkage,
                'metric': metric
            }
            try:
                agglo = AgglomerativeClustering(n_clusters=n, linkage=linkage, metric=metric)
                labels = agglo.fit_predict(X)
                scores = []
                for metric_name, metric_func in evaluation_metrics.items():
                    try:
                        score = metric_func(X, labels)
                    except Exception as e:
                        print(f"Error calculating {metric_name} for options {options}: {e}")
                        score = np.nan
                    scores.append(score)
                results.loc[len(results)] = ['AgglomerativeClustering', options] + scores
            except Exception as e:
                print(f"Error fitting AgglomerativeClustering for options {options}: {e}")

print('SpectralClustering...')
for n in range(5, 13):
    print(n)
    for affinity in ['rbf', 'nearest_neighbors', 'cosine', 'linear', 'polynomial', 'poly', 'sigmoid']:
        print(affinity)
        # Only set n_neighbors for 'nearest_neighbors' affinity
        n_neighbors_list = [5, 10, 15, 20, 50, 100] if affinity == 'nearest_neighbors' else [None]
        for n_neighbors in n_neighbors_list:
            print(n_neighbors)
            for assign_label in ['kmeans', 'discretize', 'cluster_qr']:
                print(assign_label)
                for state in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
                    options = {
                        'n_clusters': n,
                        'affinity': affinity,
                        'assign_labels': assign_label,
                        'random_state': state
                    }
                    if n_neighbors is not None:
                        options['n_neighbors'] = n_neighbors
                    try:
                        spectral_kwargs = dict(
                            n_clusters=n,
                            affinity=affinity,
                            assign_labels=assign_label,
                            random_state=state
                        )
                        if n_neighbors is not None:
                            spectral_kwargs['n_neighbors'] = n_neighbors
                        spectral = SpectralClustering(**spectral_kwargs)
                        labels = spectral.fit_predict(X)
                        scores = []
                        for metric_name, metric_func in evaluation_metrics.items():
                            try:
                                score = metric_func(X, labels)
                            except Exception as e:
                                print(f"Error calculating {metric_name} for options {options}: {e}")
                                score = np.nan
                            scores.append(score)
                        results.loc[len(results)] = ['SpectralClustering', options] + scores
                    except Exception as e:
                        print(f"Error fitting SpectralClustering for options {options}: {e}")

# Export results dataframe to CSV and pickle
output_dir = os.path.join(cartella_corrente, "results")
os.makedirs(output_dir, exist_ok=True)
results.to_csv(os.path.join(output_dir, "clustering_results.csv"), index=False)
results.to_pickle(os.path.join(output_dir, "clustering_results.pkl"))
print("Results exported to", output_dir)