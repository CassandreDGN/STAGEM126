#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=PLOTLY_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
import os 
import argparse
import pandas as pd 
import plotly.express as px





def extractname_frompath(organismpath) :
    filename = os.path.basename(organismpath)
    nomseul = os.path.splitext(filename)[0]
    label= nomseul.split('_')[0]

    return label

def extracting_h5embeddings(file_path,IDorganism): 

    embedding_fonction = []
    ids_fonction = []
    keys_fonctions = []


    with h5py.File(file_path, 'r') as f : #H5 is like a dictionarry with keys (proteinIDS) and binary values
        for proteinID in f.keys():#pour chaque clé dans le fichier h5 
            embedding_fonction.append(f[proteinID][:])#on ajoute à la liste 'valeursembeddings' une liste de chacune des valeurs de l'embedding 
            ids_fonction.append(IDorganism) #on ajoute un identifiant pour retenir que ces embeddings appartiennent à l'organisme 1 
            keys_fonctions.append(proteinID)

    return embedding_fonction,ids_fonction,keys_fonctions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path1", help="Chemin du premier fichier H5")
    parser.add_argument("path2", help="Chemin du deuxième fichier H5")
    parser.add_argument("output", help="Dossier d'output")

    args = parser.parse_args()
   
    h5path_organism1 = args.path1
    h5path_organism2 = args.path2 

    label1 = extractname_frompath(h5path_organism1)
    label2 = extractname_frompath(h5path_organism2)


    emb1,id1,keys1 = extracting_h5embeddings(h5path_organism1,1)  #les variables dans l'ordre du return seront stockés dans les var avant le = !! c'est du génie on en apprend tous les jours damn
    emb2,id2,keys2 = extracting_h5embeddings(h5path_organism2,2)

    valeursembeddings = emb1 + emb2
    ID = id1 + id2
    keysprot = keys1 + keys2

    X = np.array(valeursembeddings) #on transforme les listes en array pour qu'elle soit utilisable pour faire l'UMAP
    y = np.array(ID)

    X_scaled = StandardScaler().fit_transform(X) 

    reducer = umap.UMAP(n_neighbors=21, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df= pd.DataFrame({'ProteinID' : keysprot,
                    'UMAP_1' : embedding[:, 0] ,
                    'UMAP_2' : embedding[:,1] , 
                    'Organism' : [label1 if i == 1 else label2 for i in ID] })

    fig = px.scatter(df, x="UMAP_1", y="UMAP_2", color='Organism',hover_data=['ProteinID'])
    fig.update_traces(marker=dict(size=3))

    fig.write_html(f"{args.output}/comparison_{label1}_{label2}.html")




#/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path

#  ex : sbatch -p gpu --gres=gpu:1 --job-name=umap_compare --wrap="/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path"
# sbatch -p gpu --gres=gpu:1 path