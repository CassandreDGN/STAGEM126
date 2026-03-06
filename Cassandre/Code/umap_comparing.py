#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=UMAP_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
import sys
import os 


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

   
   
    h5path_organism1 = sys.argv[1]
    h5path_organism2 = sys.argv[2] 

    label1 = extractname_frompath(h5path_organism1)
    label2 = extractname_frompath(h5path_organism2)


    emb1,id1,keys1 = extracting_h5embeddings(sys.argv[1],1)  #les variables dans l'ordre du return seront stockés dans les var avant le = !! c'est du génie on en apprend tous les jours damn
    emb2,id2,keys2 = extracting_h5embeddings(sys.argv[2],2)

    valeursembeddings = emb1 + emb2
    ID = id1 + id2
    keysprot = keys1 + keys2

    X = np.array(valeursembeddings) #on transforme les listes en array pour qu'elle soit utilisable pour faire l'UMAP
    y = np.array(ID)

    X_scaled = StandardScaler().fit_transform(X) 

    reducer = umap.UMAP(n_neighbors=21, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)


    
    plt.figure(figsize=(10, 8))

    plt.scatter(embedding[y==1, 0], embedding[y==1, 1], s=0.1, c='blue', alpha=0.3, label=label1)
    plt.scatter(embedding[y==2, 0], embedding[y==2, 1], s=0.1, c='red', alpha=0.3, label=label2)

#add prot id ? plotly 3rd list w/ IDs or dataframe before plotting

    plt.legend()
    plt.savefig(f"/home/cassandre/stage/Cassandre/UMAPComparisonFig/comparison_{label1}_{label2}.png")




#/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path

#  ex : sbatch -p gpu --gres=gpu:1 --job-name=umap_compare --wrap="/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path"