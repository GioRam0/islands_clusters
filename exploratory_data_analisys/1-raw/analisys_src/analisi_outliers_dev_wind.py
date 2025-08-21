#script per analizzare alti valori di dev_stand della potenza eolica, vorremmo capire se dovuti a potenza bassa e in caso non problematici
import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "results")
csv_path=os.path.join(ris_folder, 'analisys_df.csv')
df = pd.read_csv(csv_path)
print('tutti i dati')
print(f'dev_stand max {df["eolico_std"].mean()}')
print(f'dev_stand min {df["eolico_std"].min()}')
print(f'dev_stand max {df["eolico_std"].max()}')
print(' ')
for i in range(1,8):
    df1=df[(df['Wind_class']==i)]
    print(f'classe {i}')
    print(f'dev_stand media {df1["eolico_std"].mean()}')
    print(f'dev_stand min {df1["eolico_std"].min()}')
    print(f'dev_stand max {df1["eolico_std"].max()}')
    print(' ')

#tutti i dati
#dev_stand max 0.5361465847746759
#dev_stand min 0.1100567912109582
#dev_stand max 1.63834005044413
#
#classe 1
#dev_stand media 0.6057289279808981
#dev_stand min 0.1100567912109582
#dev_stand max 1.63834005044413
#
#classe 2
#dev_stand media 0.5592488494547584
#dev_stand min 0.177010326997543
#dev_stand max 1.02412836471747
#
#classe 3
#dev_stand media 0.5229931827430727
#dev_stand min 0.1583180566144543
#dev_stand max 0.9071016464742412
#
#classe 4
#dev_stand media 0.4981014445258339
#dev_stand min 0.1189274893272428
#dev_stand max 1.1389436876768102
#
#classe 5
#dev_stand media 0.47014559669811523
#dev_stand min 0.1881365352299327
#dev_stand max 0.7839444819004479
# 
#classe 6
#dev_stand media 0.4688286434336779
#dev_stand min 0.1355024539306727
#dev_stand max 0.7921253807086825
#
#classe 7
#dev_stand media 0.41607565700644694
#dev_stand min 0.1240760362645335
#dev_stand max 1.23988058708189