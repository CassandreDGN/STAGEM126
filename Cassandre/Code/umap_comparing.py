import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
import sys

if __name__ == "__main__":

    h5_organism1 = sys.argv[1]
    h5_organism2 = sys.argv[2] 


    valeursembeddings = []   #une liste de liste contenant chaque valeur d'embedding
    ID = [] 

    with h5py.File(sys.argv[1], 'r') as f: #H5 is like a dictionarry with keys (proteinIDS) and binary values
        for proteinID in f.keys(): #pour chaque clé dans le fichier h5 
            valeursembeddings.append(f[proteinID][:]) #on ajoute à la liste 'valeursembeddings' une liste de chacune des valeurs de l'embedding 
            ID.append(1) #on ajoute un identifiant pour retenir que ces embeddings appartiennent à l'organisme 1

    with h5py.File(sys.argv[2], 'r') as f:
        for proteinID in f.keys():
            valeursembeddings.append(f[proteinID][:])
            ID.append(2) #identifiant pour embedding de l'organisme 2

    X = np.array(valeursembeddings) #on transforme les listes en array pour qu'elle soit utilisable pour faire l'UMAP
    y = np.array(ID)

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))

    plt.scatter(embedding[y==1, 0], embedding[y==1, 1], s=0.1, c='blue', alpha=0.3, label='Mouse')
    plt.scatter(embedding[y==2, 0], embedding[y==2, 1], s=0.1, c='red', alpha=0.3, label='Rat')

    plt.legend()
    plt.savefig('comparison_map2.png', dpi=300)




#/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py