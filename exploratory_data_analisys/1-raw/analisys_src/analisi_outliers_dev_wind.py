#script per analizzare alti valori di dev_stand della potenza eolica, vorremmo capire se dovuti a potenza bassa e in caso non problematici
import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
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
#dev_stand max 0.5354589806546662
#dev_stand min 0.10980237977728456
#dev_stand max 1.6383400504441301
#
#classe 1
#dev_stand media 0.6076684604522686
#dev_stand min 0.10980237977728456
#dev_stand max 1.6383400504441301
#
#classe 2
#dev_stand media 0.5538112216232856
#dev_stand min 0.17784799743585541
#dev_stand max 1.02412836471747
#
#classe 3
#dev_stand media 0.5246694607978909
#dev_stand min 0.1585723110960928
#dev_stand max 0.9071016464742411
#
#classe 4
#dev_stand media 0.4959291833211582
#dev_stand min 0.11890078650692582
#dev_stand max 1.1392821313939934
#
#classe 5
#dev_stand media 0.46854126675727187
#dev_stand min 0.1881365352299327
#dev_stand max 0.7839444819004479
#
#classe 6
#dev_stand media 0.4688293853855267
#dev_stand min 0.1371993201152755
#dev_stand max 0.7921253807086825
#
#classe 7
#dev_stand media 0.4073123396598149
#dev_stand min 0.1241670440387318
#dev_stand max 1.2487891501792525