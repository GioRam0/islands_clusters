#importo le librerie
import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_corrente, 'dataframes/df_norm_first_step.csv')
df = pd.read_csv(csv_path)
colonne = [["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res"],
            ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std"],
            ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std", "offshore"],
            ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std", "offshore", "evi"],
            ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std", "offshore", "evi", "hydro"],
            ["Densità_pop", "gdp_cons_pop_urban_merged", "eolico", "solar_pow", "superficie_res", "solar_seas_ind", "eolico_std", "offshore", "evi", "hydro", "geothermal_potential"]
        ]
list_names = ['basic','varianze','offshore','evi','hydro','geothermal_potential']
tsne_fold = os.path.join(cartella_corrente, 'tsne_visualizations_colors1')
os.makedirs(tsne_fold, exist_ok=True)
for i in range(30, 51, 5):
    tsne_fold1 = os.path.join(tsne_fold, f'perplexity_{i}')
    os.makedirs(tsne_fold1, exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=i, random_state=42)
    for j in range(len(colonne)):
        data = df[colonne[j]].values
        tsne_result = tsne.fit_transform(data)
        # Visualizzazione
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            hue=df['clusters'],
            palette='viridis',
            legend='full',
            s=50  # Dimensione dei punti, puoi aggiustarla
        )
        plt.title(f"t-SNE projection (Perplexity: {i})")
        plt.grid(True, linestyle='--', alpha=0.6) # Aggiungi una griglia per maggiore chiarezza
        plt.tight_layout() # Adatta i margini per evitare il taglio di etichette
        plt.savefig(os.path.join(tsne_fold1, f'tsne_visualization_{list_names[j]}_perp_{i}.png'))
        plt.close() # Chiudi la figura per liberare memoria (importante nei loop)