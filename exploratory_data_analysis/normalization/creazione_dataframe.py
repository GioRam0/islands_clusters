#importo le librerie
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "../dimensions_reduction/risultati")
pkl_path = os.path.join(folder_path, "analisys_df.pkl")
df = pd.read_pickle(pkl_path)

#il suo valore minimo è 3, lo porto a 0
df['IslandArea'] = df['IslandArea']-3

standscaler=StandardScaler(with_mean=False)
#solar applico standardscaler
df['solar_pow']=standscaler.fit_transform(df[['solar_pow']])

#features cui applicare solo robustscaler
robust_features = ['temp', 'superficie_res', 'eolico_std', 'evi']
robscaler=RobustScaler()
for col in robust_features:
    df[col]=robscaler.fit_transform(df[[col]])

#features cui applicare yeo-johnson e robustscaler
yeo_features = ['gdp_pop_urban_merged', 'gdp_pro_capite']
yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('robust_scaler', robscaler)
    ])
for col in yeo_features:
    df[col] = yeo_pipeline.fit_transform(df[[col]])

#eolico stessa cosa ma con standard scaler
yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('standard_scaler', standscaler)
    ])
df['eolico'] = yeo_pipeline.fit_transform(df[['eolico']])

#features cui applicare log1p e robustscaler
log_robust_features = ['IslandArea', 'Densità_pop', 'solar_seas_ind']
log_pipeline = Pipeline([
        ('log_transformer', FunctionTransformer(np.log1p, validate=True)),
        ('robust_scaler', robscaler)
    ])
for col in log_robust_features:
    df[col] = log_pipeline.fit_transform(df[[col]])

#features cui applicare log1p e standardscaler, solo sui valori diversi da zero in quanto molti
zeri_log=['offshore', 'hydro']
for col in zeri_log:
    df[col] = np.log1p(df[col])
    df[col] = standscaler.fit_transform(df[[col]])

#alla colonna geothermal applico yeo-johnson e robustscaler, ma solo sui valori diversi da zero
yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('standard_scaler', standscaler)
    ])
zero_mask = df['geothermal_potential'] <= 0
df.loc[zero_mask, 'geothermal_potential'] = np.nan
df['geothermal_potential'] = yeo_pipeline.fit_transform(df[['geothermal_potential']])
df.loc[zero_mask, 'geothermal_potential'] = 0

#esportazione
output_folder = os.path.join(cartella_corrente, 'risultati')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'analisys_df.pkl')
df.to_pickle(output_path)