import os
import geopandas as gp
import pickle

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "clusters_stat")
os.makedirs(ris_folder, exist_ok=True)

colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res', 'geometry', 'Cluster_Label']

vincoli_folder=os.path.join(cartella_corrente, "vincoli")
percorso_pkl=os.path.join(vincoli_folder,"soft_must_link.pkl")
with open(percorso_pkl, 'rb') as file:
    sml = pickle.load(file)
percorso_pkl=os.path.join(vincoli_folder,"soft_cannot_link.pkl")
with open(percorso_pkl, 'rb') as file:
    scl = pickle.load(file)
percorso_pkl=os.path.join(vincoli_folder,"soft_must_link_weights.pkl")
with open(percorso_pkl, 'rb') as file:
    sml_weights = pickle.load(file)
percorso_pkl=os.path.join(vincoli_folder,"soft_cannot_link_weights.pkl")
with open(percorso_pkl, 'rb') as file:
    scl_weights = pickle.load(file)


#funzione che crea statistiche per i singoli clusters
def vincoli_violati(i):
    gdf_path=os.path.join(cartella_corrente, f"geodataframe_clusterizzati/{i}/isole_clusters.gpkg")
    gdf=gp.read_file(gdf_path)
    #somma e numero dei pesi di tutti gli sml violati
    sml_violazioni_totali=0
    sml_violazioni_numero=0
    #somma e numero dei pesi di tutti gli scl violati
    scl_violazioni_totali=0
    scl_violazioni_numero=0
    #considera se una coppia compare sia in sml e scl
    violazioni_complessive=0
    violazioni_complessive_numero=0
    #lista delle coppie gia controllate
    registro=[]
    for i in range(len(sml)):
        coppia=sml[i]
        if gdf.loc[coppia[0],'Cluster_label']!=gdf.loc[coppia[1],'Cluster_label']:
            sml_violazioni_totali+=sml_weights[i]
            sml_violazioni_numero+=1
            #se la coppia è gia stata controllata per il contatore totale proseguo, non ripeto
            if coppia in registro:
                continue
            registro.append(coppia)
            if sml.count(coppia)==1:
                total_score=sml_weights[i]
            if sml.count(coppia)>1:
                total_score=0
                #controllo solo le coppie successive, se fosse gia comparsa prima sarebbe stata inserita nel registro e quindi non ricontrollata
                for j in range(i, len(sml)):
                    coppia1=sml[j]
                    if coppia1==coppia:
                        total_score+=sml_weights[j]
            if coppia in scl:
                for j in range(len(scl)):
                    coppia1=scl[j]
                    if coppia1==coppia:
                        total_score-=scl_weights[j]
            #le isole non stanno nello stesso cluster, se il peso totale è positivo considero il vincolo totale come violato
            if total_score>0:
                violazioni_complessive+=total_score
                violazioni_complessive_numero+=1

    for i in range(len(scl)):
        coppia=scl[i]
        if gdf.loc[coppia[0],'Cluster_label']==gdf.loc[coppia[1],'Cluster_label']:
            scl_violazioni_totali+=scl_weights[i]
            scl_violazioni_numero+=1
            if coppia in registro:
                continue
            registro.append(coppia)
            if scl.count(coppia)==1:
                total_score=-scl_weights[i]
            if scl.count(coppia)>1:
                total_score=0
                #controllo solo le coppie successive, se fosse gia comparsa prima sarebbe stata inserita nel registro e quindi non ricontrollata
                for j in range(i, len(scl)):
                    coppia1=scl[j]
                    if coppia1==coppia:
                        total_score-=scl_weights[j]
            if coppia in sml:
                for j in range(len(sml)):
                    coppia1=sml[j]
                    if coppia1==coppia:
                        total_score+=scl_weights[j]
            #le isole stanno nello stesso cluster, se il peso totale è negativo considero il vincolo totale come violato
            if total_score<0:
                violazioni_complessive-=total_score
                violazioni_complessive_numero+=1
    print(f'vincoli sml violati: {sml_violazioni_numero}')
    print(f'pesi totali vincoli sml violati: {sml_violazioni_totali}')
    print(f'vincoli scl violati: {scl_violazioni_numero}')
    print(f'pesi totali vincoli scl violati: {scl_violazioni_totali}')
    print(f'vincoli soft complessivi violati: {violazioni_complessive}')
    print(f'pesi totali vincoli soft violati: {violazioni_complessive}')    
for i in range(1,11):
    vincoli_violati(i)

def confronto_isole(i):
    gdf_path=os.path.join(cartella_corrente, f"geodataframe_clusterizzati/{i}/isole_clusters.gpkg")
    gdf=gp.read_file(gdf_path)
    isole_simili_divise=0
    isole_simili_insieme=0
    for k,(ind, isl) in enumerate(gdf.iterrows(), 1):
        #riprova con valori diversi creando contatori diversi
        #dovrebbero venire tutti pari perche ogni coppia è contata due volte
        #capisci come inserire varianza di sole e vento
        #isole eoliche
        if isl.Wind_class>=5 and isl.etichetta_solare=='S' and isl.hydro<soglia and gdf['NO_res']==0:
            gdf1=gdf[(gdf['Wind_class']>=5) and (gdf['etichetta_solare']=='S') and (gdf['hydro']<soglia) and (gdf['Densità_pop_etichetta']==isl.Densità_pop_etichetta) and (gdf['NO_res']==0)]
            for k1,(ind1, isl1) in enumerate(gdf1.iterrows(), 1):
                if 0.6<isl.GDP/isl1.GDP<(1/0.6) and 0.6<isl.Popolazione/isl1.Popolazione<(1/0.6):
                    if isl1.Cluster_label==isl.Cluster_label:
                        isole_simili_insieme+=1
                    else:
                        isole_simili_divise+=1
        #isole solari
        if isl.Wind_class<=4 and isl.etichetta_solare=='L' and isl.hydro<soglia and gdf['NO_res']==0:
            gdf1=gdf[(gdf['Wind_class']<=4) and (gdf['etichetta_solare']=='L') and (gdf['hydro']<soglia) and (gdf['Densità_pop_etichetta']==isl.Densità_pop_etichetta) and (gdf['NO_res']==0)]
            for k1,(ind1, isl1) in enumerate(gdf1.iterrows(), 1):
                if 0.6<isl.GDP/isl1.GDP<(1/0.6) and 0.6<isl.Popolazione/isl1.Popolazione<(1/0.6):
                    if isl1.Cluster_label==isl.Cluster_label:
                        isole_simili_insieme+=1
                    else:
                        isole_simili_divise+=1
        #isole idroelettriche
        if isl.Wind_class<=4 and isl.etichetta_solare=='S' and isl.hydro>=soglia:
            gdf1=gdf[(gdf['Wind_class']<=4) and (gdf['etichetta_solare']=='S') and (gdf['hydro']>=soglia) and (gdf['Densità_pop_etichetta']==isl.Densità_pop_etichetta)]
            for k1,(ind1, isl1) in enumerate(gdf1.iterrows(), 1):
                if 0.6<isl.GDP/isl1.GDP<(1/0.6) and 0.6<isl.Popolazione/isl1.Popolazione<(1/0.6):
                    if isl1.Cluster_label==isl.Cluster_label:
                        isole_simili_insieme+=1
                    else:
                        isole_simili_divise+=1
        print(f'isole simili negli stessi clusters {isole_simili_insieme}')
        print(f'isole simili in clusters separati {isole_simili_divise}')
for i in range(11):
    confronto_isole(i)