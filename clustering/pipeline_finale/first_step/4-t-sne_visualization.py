import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_corrente, 'dataframes/df_norm_first_step.csv')
df = pd.read_csv(csv_path)

#colonne cui applicare la proiezione
colonne = ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std", "offshore", "evi", "hydro", "geothermal_potential"]
#cartella di esportazione
tsne_fold = os.path.join(cartella_corrente, 'tsne_visualizations_colors')
os.makedirs(tsne_fold, exist_ok=True)
#diversi valori di perplexity
for i in range(20, 41, 10):
    tsne = TSNE(n_components=2, perplexity=i, random_state=42)
    data = df[colonne].values
    tsne_result = tsne.fit_transform(data)
    #visualizzazione
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        hue=df['cluster'],
        palette='viridis',
        legend='full',
        s=50
    )
    #esportazione
    plt.title(f"t-SNE projection (Perplexity: {i})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(tsne_fold, f'tsne_visualization_perp_{i}.png'))
    plt.close()