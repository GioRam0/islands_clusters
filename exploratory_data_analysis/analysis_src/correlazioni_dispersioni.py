import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#dataframe con statistiche descrittive delle varie colonne
descr=df[colonne_da_includere].select_dtypes(include='number').describe()

#creo ed esporto mappa di correlazione ed heatmap
output_folder = os.path.join(ris_folder, "correlazioni_dispersioni")
os.makedirs(output_folder, exist_ok=True)
correlation_matrix = df[colonne_da_includere].select_dtypes(include='number').corr(numeric_only=True)
output_path=os.path.join(output_folder, "matrice_correlazione.xlsx")
correlation_matrix.to_excel(output_path)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
output_path=os.path.join(output_folder, "correlation_heatmap.png")
plt.savefig(output_path)
plt.close()

#creo ed esporto i grafici a dipersione per alcune features
colonne_dispersioni=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico', 'gdp', 'gdp_pro_capite', 'temp', 'prec', 'solar_pow']
plt.figure(figsize=(30, 30))
sns.pairplot(df[colonne_dispersioni].select_dtypes(include='number'))
output_path=os.path.join(output_folder,'pairplot_gdp.png')
plt.savefig(output_path)
plt.close()