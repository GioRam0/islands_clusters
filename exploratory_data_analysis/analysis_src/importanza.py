import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#dataframe con statistiche descrittive delle varie colonne
descr=df[colonne_da_includere].select_dtypes(include='number').describe()

#seleziono le features da me ritenute principali e calcolo l'importanza delle altre
target_features=['IslandArea', 'Popolazione', 'solar_pow', 'eolico', 'gdp']
output_folder = os.path.join(ris_folder, "importanza")
os.makedirs(output_folder, exist_ok=True)

for target_feature in target_features:
    #seleziono le colonne
    X = df[colonne_da_includere].select_dtypes(include='number').drop(columns=target_features)
    y = df[target_feature]
    #train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    #calcolo l'importanza delle features, creo un dataframe ordinato, creo un grafico e lo esporto
    importances = model.feature_importances_
    features = X.columns
    feat_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(data=feat_importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f"Importanza delle features rispetto a '{target_feature}'")
    plt.tight_layout()
    output_path=os.path.join(output_folder, f'feature_importance_for_{target_feature}.png')
    plt.savefig(output_path)
    plt.close()