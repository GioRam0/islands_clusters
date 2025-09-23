import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe normalizzato
csv_path = os.path.join(cartella_corrente, 'results/dataframes/df_norm_final.csv')
df = pd.read_csv(csv_path)
colonne1 = ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential', 'temp']
colonne2 = ['Densità_pop', 'gdp_cons_pop_urban_merged']

#heatmap dei cluster di primo livello e relative variabili
df1 = df[colonne2+['cluster']]
cluster_means = df1.groupby("cluster").mean().T
#normalizzazione medie
cluster_means_norm = cluster_means.apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis=1
)
#plot della heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_norm, cmap="viridis", cbar=True)
plt.title("Heatmap delle medie per cluster (normalizzate per feature)")
plt.ylabel("Feature")
plt.xlabel("Cluster")
plt.tight_layout()
#esportazione
output_path = os.path.join(cartella_corrente, 'results/heatmap')
os.makedirs(output_path, exist_ok=True)
heatmap_path = os.path.join(output_path, f'heatmap_primo_livello.png')
plt.savefig(heatmap_path)
plt.close()

#heatmap sui cluster di primo livello con clustering di secondo e variabili relative
n_clust_first = df['cluster'].max()+1
#calcolo delle medie e normalizzazione
means = df.groupby(['cluster', 'cluster_finali'])[colonne1].mean().reset_index()
means_norm = means.copy()
means_norm[colonne1] = means[colonne1].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
)
#itero per realizzare le varie heatmap
for cl in range(n_clust_first):
    subset = means_norm[means_norm['cluster'] == cl]
    subset = subset.set_index('cluster_finali')[colonne1]
    plt.figure(figsize=(10, 6))
    sns.heatmap(subset, vmin=0, vmax=1, cmap="viridis", cbar=True)
    plt.tight_layout()
    # esportazione
    output_path = os.path.join(cartella_corrente, 'results/heatmap/second_step')
    os.makedirs(output_path, exist_ok=True)
    heatmap_path = os.path.join(output_path, f'heatmap_cluster_{cl}.png')
    plt.savefig(heatmap_path)
    plt.close()

#heatmap complessiva con dendogramma dei due livelli
#calcolo le medie le ordino e le normalizzo
grouped = df[colonne2+colonne1+['cluster','cluster_finali']].groupby(["cluster", "cluster_finali"]).mean()
ordered = grouped.sort_index(level=[0,1])
ordered_norm = ordered.copy()
ordered_norm[colonne2 + colonne1] = ordered[colonne2 + colonne1].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
)
fig = plt.figure(figsize=(20, 28))
#dendogramma su due livelli
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.05)
ax_tree = plt.subplot(gs[0])
ax_tree.set_xlim(0, 1)
ax_tree.set_ylim(0, len(ordered))
y_ticks = []
labels = []
y_pos = 0
#itero per i cluster
for lvl1, subdf in ordered.groupby(level=0):
    n_sub = len(subdf)
    mid = y_pos + n_sub/2
    #ramo livello 1
    ax_tree.plot([0.7, 0.4], [mid, mid], color="black")
    ax_tree.text(0.2, mid, f"{lvl1}", va="center", ha="right", fontsize=24)
    #itero per i sottocluster
    for lvl2 in subdf.index.get_level_values(1):
        leaf_y = y_pos + 0.5
        #rami livello 2
        ax_tree.plot([0.7, 0.7], [mid, leaf_y], color="black")
        ax_tree.plot([0.7, 1], [leaf_y, leaf_y], color="black")
        labels.append(f"{lvl1}.{lvl2}")
        y_ticks.append(leaf_y)
        y_pos += 1
ax_tree.set_xticks([])
ax_tree.set_yticks([])
ax_tree.invert_yaxis()
ax_tree.set_yticklabels(ax_tree.get_yticklabels(), fontsize=30)
ax_tree.text(
    -0.2,
    len(ordered_norm)/2,
    "Cluster",
    va="center", ha="center", 
    rotation=90,
    fontsize=30
)
#heatmap
ax_heat = plt.subplot(gs[1])
sns.heatmap(
    ordered_norm,
    cmap="viridis",
    cbar=True,
    ax=ax_heat,
    yticklabels=[f"{i[0]}.{i[1]}" for i in ordered.index]
)
ax_heat.set_ylabel("")
ax_heat.set_xlabel("Features", fontsize=30)
ax_heat.set_xticklabels(ax_heat.get_xticklabels(), fontsize=18, rotation=35, ha="right")
ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=18)
#esportazione
output_path = os.path.join(cartella_corrente, 'results/heatmap')
heatmap_path = os.path.join(output_path, f'heatmap_secondo_livello_con_dendogramma.png')
plt.savefig(heatmap_path)
plt.close()