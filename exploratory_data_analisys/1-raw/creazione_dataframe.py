import geopandas as gp
import numpy as np
import pickle
import os

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo le isole ed elimino la colonna geometrica
isl_path=os.path.join(cartella_progetto, "data/isole_filtrate/finali", "isole_arro4.gpkg")
gdf = gp.read_file(isl_path)
df = gdf.drop(columns=['geometry'])
del gdf

#importo i vari dizionari pkl contenenti i dati tramite una funzione
pkl_folder=os.path.join(cartella_progetto, "data/dati_finali")

#funzione che prende in input il nome del file, importa il dizionario e crea la colonna nel dataframe, da applicare ai vari files
def create_dict(str):
    df[str]=float(0)
    nome=str+'.pkl'
    percorso_pkl=os.path.join(pkl_folder1, nome)
    with open(percorso_pkl, 'rb') as file:
        return pickle.load(file)

#applico la funzione alle varie cartelle    
pkl_folder1=os.path.join(pkl_folder, "biomassa")
evi=create_dict('evi')
evi_nodata=create_dict("evi_nodata")

pkl_folder1=os.path.join(pkl_folder, "eolico")
eolico = create_dict("eolico")
eolico_nodata=create_dict("eolico_nodata")
eolico_std=create_dict("eolico_std")
offshore=create_dict("offshore")

pkl_folder1=os.path.join(pkl_folder, "gdp_consumption_2019")
gdp_2019=create_dict("gdp_2019")
gdp_2019_nodata=create_dict("gdp_2019_nodata")
consumption=create_dict("consumption")
cons_nodata=create_dict("cons_nodata")

pkl_folder1=os.path.join(pkl_folder, "geotermico")
geothermal_potential=create_dict("geothermal_potential")

pkl_folder1=os.path.join(pkl_folder, "hydro")
hydro=create_dict("hydro")

pkl_folder1=os.path.join(pkl_folder, "metereologici")
temp=create_dict("temp")
temp_nodata=create_dict("temp_nodata")
prec=create_dict("prec")
prec_nodata=create_dict("prec_nodata")
hdd=create_dict("hdd")
hdd_nodata=create_dict("hdd_nodata")
cdd=create_dict("cdd")
cdd_nodata=create_dict("cdd_nodata")

pkl_folder1=os.path.join(pkl_folder, "solare")
solar_pow= create_dict("solar_pow")
solar_seas_ind= create_dict("solar_seas_ind")
solar_nodata= create_dict("solar_nodata")

pkl_folder1=os.path.join(pkl_folder, "urban")
urban_area=create_dict("urban_area")
urban_area_rel=create_dict("urban_area_rel")

pkl_folder1=os.path.join(pkl_folder, "superficie_res")
superficie_res=create_dict("superficie_res")
ele_max=create_dict("ele_max")

#creo un dizionario che per ogni colonna nodata valuta conta le righe in cui il booleano non corrisponde con dato isnan
#il primo elemento della lista rappresenta le righe nan e 0, il secondo not nan e 1
diz={}
diz['evi']=[0,0]
diz['eolico']=[0,0]
diz['gdp_2019']=[0,0]
diz['consumption']=[0,0]
diz['temp']=[0,0]
diz['prec']=[0,0]
diz['hdd']=[0,0]
diz['cdd']=[0,0]
diz['solar']=[0,0]

#riempio le nuove colonne del dataframe
for i,isl in df.iterrows():
    codice=isl.ALL_Uniq
    df.loc[i,'evi']=evi[codice]
    df.loc[i,'evi_nodata']=evi_nodata[codice]
    if np.isnan(evi[codice]) and evi_nodata[codice]==0:
        diz['evi'][0]+=1
    if (not np.isnan(evi[codice])) and evi_nodata[codice]==1:
        diz['evi'][1]+=1
    df.loc[i,'eolico']=eolico[codice]
    df.loc[i,'eolico_nodata']=eolico_nodata[codice]
    if np.isnan(eolico[codice]) and eolico_nodata[codice]==0:
        diz['eolico'][0]+=1
    if (not np.isnan(eolico[codice])) and eolico_nodata[codice]==1:
        diz['eolico'][1]+=1
    df.loc[i,'eolico_std']=eolico_std[codice]
    df.loc[i,'offshore']=offshore[codice]
    df.loc[i,'gdp_2019']=gdp_2019[codice]
    df.loc[i,'gdp_2019_nodata']=gdp_2019_nodata[codice]
    if np.isnan(gdp_2019[codice]) and gdp_2019_nodata[codice]==0:
        diz['gdp_2019'][0]+=1
    if (not np.isnan(gdp_2019[codice])) and gdp_2019_nodata[codice]==1:
        diz['gdp_2019'][1]+=1
    df.loc[i,'consumption']=consumption[codice]
    df.loc[i,'cons_nodata']=cons_nodata[codice]
    if np.isnan(consumption[codice]) and cons_nodata[codice]==0:
        diz['consumption'][0]+=1
    if (not np.isnan(consumption[codice])) and cons_nodata[codice]==1:
        diz['consumption'][1]+=1
    df.loc[i,'geothermal_potential']=geothermal_potential[codice]
    df.loc[i,'hydro']=hydro[codice]
    df.loc[i,'temp']=temp[codice]
    df.loc[i,'temp_nodata']=temp_nodata[codice]
    if np.isnan(temp[codice]) and temp_nodata[codice]==0:
        diz['temp'][0]+=1
    if (not np.isnan(temp[codice])) and temp_nodata[codice]==1:
        diz['temp'][1]+=1
    df.loc[i,'prec']=prec[codice]
    df.loc[i,'prec_nodata']=prec_nodata[codice]
    if np.isnan(prec[codice]) and prec_nodata[codice]==0:
        diz['prec'][0]+=1
    if (not np.isnan(prec[codice])) and prec_nodata[codice]==1:
        diz['prec'][1]+=1
    df.loc[i,'hdd']=hdd[codice]
    df.loc[i,'hdd_nodata']=hdd_nodata[codice]
    if np.isnan(hdd[codice]) and hdd_nodata[codice]==0:
        diz['hdd'][0]+=1
    if (not np.isnan(hdd[codice])) and hdd_nodata[codice]==1:
        diz['hdd'][1]+=1
    df.loc[i,'cdd']=cdd[codice]
    df.loc[i,'cdd_nodata']=cdd_nodata[codice]
    if np.isnan(cdd[codice]) and cdd_nodata[codice]==0:
        diz['cdd'][0]+=1
    if (not np.isnan(cdd[codice])) and cdd_nodata[codice]==1:
        diz['cdd'][1]+=1
    df.loc[i,'solar_pow']=solar_pow[codice]
    df.loc[i,'solar_seas_ind']=solar_seas_ind[codice]
    df.loc[i,'solar_nodata']=solar_nodata[codice]
    if np.isnan(solar_pow[codice]) and solar_nodata[codice]==0:
        diz['solar']+=1
    if (not np.isnan(solar_pow[codice])) and solar_nodata[codice]==1:
        diz['solar']+=1
    df.loc[i,'urban_area']=urban_area[codice]
    df.loc[i,'urban_area_rel']=urban_area_rel[codice]
    df.loc[i,'superficie_res']=superficie_res[codice]
    df.loc[i,'ele_max']=ele_max[codice]

#stampo il dizionario che riporta le isole con non-corrispndenza tra valore nan e booleano nodata
print(diz)

#elimino le righe con dati non completi
print(f"tutte le isole sono {len(df)}")
df=df.dropna()
print(f"le isole con dati completi sono {len(df)}")
print(' ')

#elimino le isole con valori gdp e consumption non validi (sono due, le stesse per entrambe le variabili, se non dovessero essere già state eliminate in quanto contenenti nan su altre colonne)
indici_da_eliminare = df[df['gdp_2019_nodata'] == 1].index
df = df.drop(indici_da_eliminare)
indici_da_eliminare = df[df['consumption'] == 1].index
df = df.drop(indici_da_eliminare)
print(f"le isole con dati completi sono {len(df)}")
print(' ')

#elimino le colonne nodata e shape_leng, inutili a questo punto
df = df.drop(columns=['evi_nodata', 'eolico_nodata',  'cons_nodata', 'gdp_2019_nodata', 'temp_nodata', 'prec_nodata', 'hdd_nodata', 'cdd_nodata', 'solar_nodata'])
df=df.drop(columns=['Shape_Leng'])

#definisco una funzione che crea delle etichette per una feature input
def etichettatura(valore, soglie, etichette):
    if valore < soglie[0]:
        return etichette[0]
    for i in range(len(soglie) - 1):
        if soglie[i] <= valore < soglie[i+1]:
            return etichette[i+1]
    if valore >= soglie[-1]:
        return etichette[-1]

#soglie e nome etichette per solar power e gdp procapite
soglie_den=[50,350]
soglie_solar=[3.5,4.5]
etichette=['S','M','L']
df['Densità_pop_etichetta']=df['Densità_pop'].apply(etichettatura, args=(soglie_den, etichette))
for etic in etichette:
    leng=len(df[(df['Densità_pop_etichetta']==etic)])
    print(f'Ci sono {leng} isole con etichetta {etic} per la densità abitativa')
print(' ')
df['Solar_etichetta']=df['solar_pow'].apply(etichettatura, args=(soglie_solar, etichette))
for etic in etichette:
    leng=len(df[(df['Solar_etichetta']==etic)])
    print(f'Ci sono {leng} isole con etichetta {etic} per la potenza solare')
print(' ')

#etichette vento
soglie_wind_power=[100, 150, 200, 250, 300, 400]
wind_classes=[1,2,3,4,5,6,7]
soglie_wind_cubo = [elemento / 1.225 for elemento in soglie_wind_power]
df['Wind_class']=df['eolico'].apply(etichettatura, args=(soglie_wind_cubo, wind_classes))
for classe in wind_classes:
    leng=len(df[(df['Wind_class']==classe)])
    print(f'Ci sono {leng} isole con classe di vento {classe}')
print(' ')

#etichette energy consumption
soglie_consumption=[2, 15, 100]
soglie_consumption = [x * 10**6 for x in soglie_consumption] #passaggio da kWh a GWh
etichette_consumption=['XS','S','M','L']
df['consumption_etichetta']=df['consumption'].apply(etichettatura, args=(soglie_consumption, etichette_consumption))
for etic in etichette_consumption:
    leng=len(df[(df['consumption_etichetta']==etic)])
    print(f'Ci sono {leng} isole con etichetta consumptions {etic}')
print(' ')

#isole senza superficie per le rinnovabili
df['NO_res'] = np.where(df['superficie_res'] == 0, 1, 0)
leng=len(df[(df['NO_res']==1)])
print(f'Ci sono {leng} isole con senza superficie agibile per rinnovabili')

#resetto gli indici a causa degli elementi eliminati
df = df.reset_index(drop=True)

#esportazione
output_folder = os.path.join(cartella_corrente, '..')
output_path = os.path.join(output_folder, 'df_raw.csv')
df.to_csv(output_path, index=False, encoding='utf-8')

#{'evi': [0, 0], 'eolico': [0, 0], 'gdp_2019': [0, 2], 'consumption': [0, 2], 'temp': [0, 0], 'prec': [0, 0], 'hdd': [0, 0], 'cdd': [0, 0], 'solar': [0, 0]}
#tutte le isole sono 3134
#le isole con dati completi sono 2012
#
#le isole con dati completi sono 2012
#
#Ci sono 885 isole con etichetta S per la densità abitativa
#Ci sono 880 isole con etichetta M per la densità abitativa
#Ci sono 247 isole con etichetta L per la densità abitativa
#
#Ci sono 404 isole con etichetta S per la potenza solare
#Ci sono 1395 isole con etichetta M per la potenza solare
#Ci sono 213 isole con etichetta L per la potenza solare
#
#Ci sono 863 isole con classe di vento 1
#Ci sono 234 isole con classe di vento 2
#Ci sono 165 isole con classe di vento 3
#Ci sono 143 isole con classe di vento 4
#Ci sono 123 isole con classe di vento 5
#Ci sono 159 isole con classe di vento 6
#Ci sono 325 isole con classe di vento 7
#
#Ci sono 1214 isole con etichetta consumptions XS
#Ci sono 448 isole con etichetta consumptions S
#Ci sono 226 isole con etichetta consumptions M
#Ci sono 124 isole con etichetta consumptions L
#
#Ci sono 43 isole con senza superficie agibile per rinnovabili