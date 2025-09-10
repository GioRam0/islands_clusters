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
folder_path = os.path.join(cartella_corrente, "..")
pkl_path = os.path.join(folder_path, "df_dim_reduction.csv")
df = pd.read_csv(pkl_path)

#features cui applicare solo robustscaler
robust_features = ['solar_pow','temp', 'eolico_std', 'evi']
robscaler=RobustScaler()
for col in robust_features:
    df[col]=robscaler.fit_transform(df[[col]])

#features cui applicare yeo-johnson e robustscaler
yeo_features = ['gdp_cons_pop_urban_merged', 'eolico']
yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('robust_scaler', robscaler)
    ])
for col in yeo_features:
    df[col] = yeo_pipeline.fit_transform(df[[col]])

#features cui applicare log1p e robustscaler
log_robust_features = ['superficie_res', 'Densità_pop', 'solar_seas_ind']
log_pipeline = Pipeline([
        ('log_transformer', FunctionTransformer(np.log1p, validate=True)),
        ('robust_scaler', robscaler)
    ])
for col in log_robust_features:
    df[col] = log_pipeline.fit_transform(df[[col]])

#features cui applicare log1p e robustscaler senza sottrazione della media, solo sui valori diversi da zero in quanto molti
standsclaer=StandardScaler(with_mean=False)
zeri_log=['offshore', 'hydro']
for col in zeri_log:
    df[col] = np.log1p(df[col])
    df[col] = robscaler.fit_transform(df[[col]])

#alla colonna geothermal applico yeo-johnson e standard scaler per non sottrarre la media, ma solo sui valori diversi da zero
yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('standard_scaler', standsclaer)
    ])
zero_mask = df['geothermal_potential'] <= 0
df.loc[zero_mask, 'geothermal_potential'] = np.nan
df['geothermal_potential'] = yeo_pipeline.fit_transform(df[['geothermal_potential']])
df.loc[zero_mask, 'geothermal_potential'] = 0
df['geothermal_potential'] = df['geothermal_potential']

#esportazione
output_folder = os.path.join(cartella_corrente, '..')
output_path = os.path.join(output_folder, 'df_norm.csv')
df.to_csv(output_path, index=False, encoding='utf-8')