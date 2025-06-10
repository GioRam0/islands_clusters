#importo le librerie
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/dimensions_reduction/risultati/analisys_df.pkl")
df2 = pd.read_pickle(pkl_path)
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/raw/risultati/analisys_df.pkl")
df3 = pd.read_pickle(pkl_path)
max_sol=df2[df2['Solar_etichetta']=='L']['solar_pow'].min()
min_sol=df2[df2['Solar_etichetta']=='M']['solar_pow'].min()
def calcolo_param(mini,maxi):
    tau   = (maxi+mini)/2
    delta = 0.1
    alpha = 2 * np.log((1-delta)/delta)
    return tau,alpha


colonne_da_includere=['solar_pow', 'eolico', 'offshore', 'geothermal_potential', 'hydro']
colonne_etichette=['Solar_etichetta', 'Wind_class']
df=df[colonne_da_includere+colonne_etichette]
df2=df2[colonne_da_includere+colonne_etichette]

#tutti i valori partono da zero e una buona parte arriva a 1 per un migliore confronto
df['eolico']=df['eolico']-df['eolico'].min()
min_eol_alto=df[df['Wind_class']==7]['eolico'].min()
df['eolico']=df['eolico']/min_eol_alto

df['solar_pow']=df['solar_pow']-df['solar_pow'].min()
min_sol_alto=df[df['Solar_etichetta']=='L']['solar_pow'].min()
df['solar_pow']=df['solar_pow']/min_sol_alto

#valore equivalente a 10 GW, 258 isole ne hanno di piu
df['offshore']=df['offshore']/2.3

#valore equivalente a 3GWH annuali, 280 isole circa ne hanno di piu
df['hydro']=df['hydro']/2.25

#valore equivalente a 1000 per il df raw, 50 isole circa ne hanno di piu
df['geothermal_potential']=df['geothermal_potential']/4.5

#sole da solo
#print(len(df[(df['Solar_etichetta']=='L') & (df['Wind_class']<=3) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#vento da solo
#print(len(df[(df['Solar_etichetta']!='L') & (df['Wind_class']>5) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#offshore da solo
#print(len(df[(df['Solar_etichetta']!='L') & (df['Wind_class']<=3) & (df['offshore']>2) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#vento e offshore
#print(len(df[(df['Solar_etichetta']!='L') & (df['Wind_class']>5) & (df['offshore']>2) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#vento sole e offshore
#print(len(df[(df['Solar_etichetta']=='L') & (df['Wind_class']>5) & (df['offshore']>2) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#vento e sole
#print(len(df[(df['Solar_etichetta']=='L') & (df['Wind_class']>5) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]))
#solo hydro
#print(len(df[(df['Solar_etichetta']!='L') & (df['Wind_class']<=3) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']>1)]))
#geot
#print(len(df[(df['geothermal_potential']>1000)]))
#nessuna risorsa
#print(len(df[(df['Solar_etichetta']!='L') & (df['Wind_class']<=3) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]))

#funzione per calcolare la distanza come media pesata di coseno e euclidea
def hybrid_distance_matrix(X, alpha=0.5):
    D_euclidean = euclidean_distances(X)
    D_cosine = cosine_distances(X)
    return alpha * D_euclidean + (1 - alpha) * D_cosine

#funzione per ricavare il centroide da un gruppo ideale realizzato
def get_median_index(dataframe,list):
    #somma colonne
    row_sums = dataframe[list].sum(axis=1)
    #mediana
    median_of_sums = row_sums.median()
    #differenza mediana e suo minimo
    absolute_differences = abs(row_sums - median_of_sums)
    min_abs_difference = absolute_differences.min()
    #restituisco l'indice dove questa diffreenza è pari al minimo
    return int(dataframe.loc[absolute_differences == min_abs_difference].index[0])

#centri iniziali
initial_medoids = []
#ipotizzo i vari gruppi e ne individuo il centroide
#centro solo solare
df1=df[(df['Solar_etichetta']=='L') & (df['Wind_class']<=2) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,['solar_pow']))
#centro solo vento (per solar etichetta==s troppo stringente, !=l troppo poco, cosi valore intermedio)
df1=df[(df['solar_pow']<0.65) & (df['Wind_class']>5) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,['solar_pow']))
#centro vento e offshore
df1=df[(df['Solar_etichetta']=='S') & (df['Wind_class']>5) & (df['offshore']>1) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,['offshore', 'eolico']))
#centro vento sole e offshore
df1=df[(df['Solar_etichetta']=='L') & (df['Wind_class']>5) & (df['offshore']>1) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,['offshore', 'eolico', 'solar_pow']))
#centro vento e sole
df1=df[(df['Solar_etichetta']=='L') & (df['Wind_class']>5) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,['solar_pow','eolico']))
#centro solo hydro
df1=df[(df['solar_pow']<0.65) & (df['Wind_class']<=3) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']>1)]
initial_medoids.append(get_median_index(df1,['hydro']))
#centro solo geot
df1=df[(df['geothermal_potential']>0.8) & (df['Wind_class']<=3) & (df['offshore']==0) & (df['hydro']==0) & (df['solar_pow']<0.7)]
initial_medoids.append(get_median_index(df1,['geothermal_potential']))
#nessuna risorsa (con solar stesso discorso di hydro e solare)
df1=df[(df['solar_pow']<0.65) & (df['Wind_class']<=3) & (df['offshore']==0) & (df['geothermal_potential']==0) & (df['hydro']==0)]
initial_medoids.append(get_median_index(df1,colonne_da_includere))


import numpy as np

alpha = 5.0
tau = np.median(X, axis=0)      # soglia per ogni feature
T  = 0.5                     # temperatura

# 1) soft-threshold
F = 1 / (1 + np.exp(-alpha*(X - tau)))

# 2) definisci prototipi (es. per d=2)
prototypes = np.array([
    [1,0],
    [0,1],
    [1,1],
    [0,0],
])

# 3) score e softmax
scores = F @ prototypes.T       # shape (n_samples, n_prototypes)
exp_scores = np.exp(scores / T)
U = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # membership

# 4) assegnamento crisp (se serve)
labels = np.argmax(U, axis=1)

