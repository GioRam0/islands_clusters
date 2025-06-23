import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, MeanShift, SpectralClustering, Birch, OPTICS, AffinityPropagation, FeatureAgglomeration, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)

#suddivisione colonne
colonne_includere=['Densità_pop', 'gdp_pro_capite', 'temp', 'gdp_pop_urban_merged', 'geothermal_potential', 'hydro']
colonne_esplorabili=['superficie_res']
colonne_risorse=['evi', 'eolico', 'offshore', 'solar_pow']
colonne_varianza=['eolico_std', 'solar_seas_ind']
colonne_etichette=['Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_identificative=['ALL_Uniq', 'Name_USGSO']

# Dizionario degli algoritmi di clustering disponibili
algorithms = {
    'KMeans': KMeans,
    'AgglomerativeClustering': AgglomerativeClustering,
    'DBSCAN': DBSCAN,
    'HDBSCAN': HDBSCAN,
    'OPTICS': OPTICS,
    'MeanShift': MeanShift,
    'SpectralClustering': SpectralClustering,
    'Birch': Birch,
    'GaussianMixture': GaussianMixture,
    'AffinityPropagation': AffinityPropagation,
    'BisectingKMeans': BisectingKMeans
}
#'FeatureAgglomeration': FeatureAgglomeration prova a afare qualcosa

# Dizionario delle metriche di valutazione clustering
evaluation_metrics = {
    'silhouette_score': silhouette_score,
    'calinski_harabasz_score': calinski_harabasz_score,
    'davies_bouldin_score': davies_bouldin_score
}

#tento con questi numeri di clusters
n_clust=[5,10,20,30,40]

results = pd.DataFrame(columns=["algorithm", "n_clusters", "option", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"])

def results_creation(name,n,labels,option=None):
    if n<200 and n>4:
        print(f"Results for {name}, {n} clusters, {option} option:")
        scores = []
        for metric_name, metric_func in evaluation_metrics.items():
            try:
                score = metric_func(X, labels)
                scores.append(score)
                print(f"  {metric_name}: {score:.3f}")
            except Exception as e:
                scores.append(None)
                print(f"  {metric_name} {n} clusters: Error ({e})")
        results.loc[len(results)] = [name, n, option, scores[0], scores[1], scores[2]]
    else:
        print(f"Results for {name}, {n} clusters, {option} option:")
        print('Too many clusters')
#itero per gli algoritmi
for name, algo in algorithms.items():
    print(f"Running {name}...")
    X = df[colonne_includere].values
    #imposto iperparametri e itero per numero diverso di clusters se possibile
    if name in ['KMeans', 'MiniBatchKMeans', 'AgglomerativeClustering', 'SpectralClustering', 'Birch', 'GaussianMixture', 'BisectingKMeans']:
        for n in n_clust:
            if name == 'KMeans' or name == 'MiniBatchKMeans':
                model = algo(n_clusters=n, random_state=42)
                labels = model.fit_predict(X)
                results_creation(name, n, labels)
            elif name == 'AgglomerativeClustering':
                linkage = ['ward', 'complete', 'average', 'single']
                for method in linkage:
                    model = algo(n_clusters=n, linkage=method)
                    labels = model.fit_predict(X)
                    results_creation(name, n, labels, method)
            elif name == 'Birch':
                model = algo(n_clusters=n)
                labels = model.fit_predict(X)
                results_creation(name, n, labels)
            elif name == 'SpectralClustering':
                affinity = ['nearest_neighbors', 'rbf']
                for aff in affinity:
                    model = algo(n_clusters=n, assign_labels='discretize', random_state=42, affinity=aff)
                    labels = model.fit_predict(X)
                    results_creation(name, n, labels, aff)
            elif name == 'GaussianMixture':
                model = algo(n_components=n, random_state=42)
                labels = model.fit_predict(X)
                results_creation(name, n, labels)
            elif name == 'BisectingKMeans':
                bisecting_strategy = ["biggest_inertia", "largest_cluster"]
                for strategy in bisecting_strategy:
                    model = algo(n_clusters=n, random_state=42, bisecting_strategy=strategy)
                    labels = model.fit_predict(X)
                    results_creation(name, n, labels, strategy)
    else:
        if name == 'DBSCAN':
            eps_range = np.arange(0.2, 1.2, 0.1)
            for eps in eps_range:
                model=algo(eps=eps)
                labels=model.fit_predict(X)
                n=max(labels+1)
                results_creation(name, n, labels, eps)
        elif name == 'HDBSCAN':
            cluster_selection_method=['eom', 'leaf']
            for method in cluster_selection_method:
                model = algo(cluster_selection_method=method)
                labels=model.fit_predict(X)
                n=max(labels+1)
                results_creation(name, n, labels, method)
        elif name == 'AffinityPropagation':
            #non capisco perche senza opzioni ritorna 126 clusters con multipliers pari a 1 too many
            multipliers = [0.3, 0.5, 1, 2, 3, 5]
            model = algo()
            labels = model.fit_predict(X)
            n = max(labels) + 1
            results_creation(name, n, labels)
            for multi in multipliers:
                preference = - np.median(X) * multi
                model = algo(preference=preference)
                labels = model.fit_predict(X)
                n = max(labels) + 1
                results_creation(name, n, labels, multi)
        else:
            model = algo()
            labels = model.fit_predict(X)
            n=max(labels+1)
            results_creation(name, n, labels)

print(results)

# Ordina i risultati per ciascuna metrica e mostra i primi 3 algoritmi per ciascuna
for metric in ["silhouette_score", "calinski_harabasz_score"]:
    print(f"\nTop 5 algorithms by {metric}:")
    print(results.sort_values(by=metric, ascending=False)[["algorithm", "n_clusters", "option", metric]].head(5))

print(f"\nTop 5 algorithms by davies_bouldin_score (lower is better):")
print(results.sort_values(by="davies_bouldin_score", ascending=True)[["algorithm", "n_clusters", "option", "davies_bouldin_score"]].head(5))