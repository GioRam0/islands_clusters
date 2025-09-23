import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_norm.csv')
df = pd.read_csv(csv_path)
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#creo ed esporto mappa di correlazione ed heatmap
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
output_folder = os.path.join(ris_folder, "correlazioni_dispersioni")
os.makedirs(output_folder, exist_ok=True)
correlation_matrix = df[colonne_da_includere].select_dtypes(include='number').corr(numeric_only=True)
output_path=os.path.join(output_folder, "matrice_correlazione.xlsx")
correlation_matrix.to_excel(output_path)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("")
plt.tight_layout()
output_path=os.path.join(output_folder, "correlation_heatmap.png")
plt.savefig(output_path)
plt.close()

#creo ed esporto i grafici a dipersione per alcune features
colonne_dispersioni=['superficie_res', 'Densità_pop', 'eolico', 'temp', 'solar_pow', 'gdp_cons_pop_urban_merged']
plt.figure(figsize=(30, 30))
sns.pairplot(df[colonne_dispersioni].select_dtypes(include='number'))
output_path=os.path.join(output_folder,'pairplot_gdp.png')
plt.savefig(output_path)
plt.close()