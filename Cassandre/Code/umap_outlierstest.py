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
import dash
from scipy.spatial import distance
import faiss 


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

def get_outliers_list(emb1,emb2,id1,id2):

    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    dim = emb1.shape[1]
    emb1_f32 = np.ascontiguousarray(emb1.astype('float32'))
    emb2_f32 = np.ascontiguousarray(emb2.astype('float32'))

    faiss.normalize_L2(emb1_f32)     
    faiss.normalize_L2(emb2_f32)

    # 1. Calcul des similarités
    index2 = faiss.IndexFlatIP(dim)
    index2.add(emb2_f32)
    sim1_2, _ = index2.search(emb1_f32, 1)

    index1 = faiss.IndexFlatIP(dim)
    index1.add(emb1_f32)
    sim2_1, _ = index1.search(emb2_f32, 1)

    all_sims = np.concatenate([sim1_2.flatten(), sim2_1.flatten()])
    all_ids = id1 + id2 # keys1 + keys2 passés en arguments

    # 2. Calcul du cutoff
    mean_similarity = np.mean(all_sims)
    std_similarity = np.std(all_sims)
    cutoff = mean_similarity - (3 * std_similarity)

    outliers = []
    for i, sim in enumerate(all_sims):
        if sim < cutoff :
            id_protein = all_ids[i]
            outliers.append(id_protein)

    print(f"{len(outliers)} outliers. Seuil: {cutoff:.4f}")

    return outliers, all_sims

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path1", help="Chemin du premier fichier H5")
    parser.add_argument("path2", help="Chemin du deuxième fichier H5")
    parser.add_argument("output", help="Dossier d'output de UMAP")
    parser.add_argument("genelist_output", help="Dossier d'output des fichiers texte contenant les listes des gènes")

    args = parser.parse_args()

    txtoutput=args.genelist_output

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

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df= pd.DataFrame({'ProteinID' : keysprot,
                    'UMAP_1' : embedding[:, 0] ,
                    'UMAP_2' : embedding[:,1] , 
                    'Organism' : [label1 if i == 1 else label2 for i in ID] })
    
    outliers_list, closes_similarity=get_outliers_list(emb1,emb2,keys1,keys2)


    df['Status'] = 'Normal'
    df.loc[df['ProteinID'].isin(outliers_list), 'Status'] = 'Outlier'

    df['Status_Organism'] = df['Organism'] + " " + df['Status']


    beautiful_color_choices = {f"{label1} Normal": 'red', f"{label1} Outlier": 'orange', f"{label2} Normal": 'DodgerBlue', f"{label2} Outlier": 'OliveDrab'}

    fig = px.scatter(df, x="UMAP_1", y="UMAP_2", color='Status_Organism', color_discrete_map=beautiful_color_choices, hover_data=['ProteinID'])
    fig.update_traces(marker=dict(size=3),opacity=0.6) #pour réduire la taille des points et les rendre un peu transparents pour mieux voir les zones denses

    fig.write_html(f"{args.output}/umap_result_{label1}_{label2}.html")

    outliers_org1 = []
    outliers_org2 = []

    for protein_id in outliers_list:
        
        if protein_id in keys1:
            outliers_org1.append(protein_id)
            
        elif protein_id in keys2:
            outliers_org2.append(protein_id)

    def clean_uniprot(id_list):
        return [p.split('|')[1] if '|' in p else p for p in id_list]

    keys1_clean = clean_uniprot(keys1)
    keys2_clean = clean_uniprot(keys2)
    outliers_org1_clean = clean_uniprot(outliers_org1)
    outliers_org2_clean = clean_uniprot(outliers_org2)

    
    with open(txtoutput + "background_" + label1 + ".txt", "w") as f:
        f.write("\n".join(keys1_clean))
    with open(txtoutput + "outliers_" + label1 + ".txt", "w") as f:
        f.write("\n".join(outliers_org1_clean))

    with open(txtoutput + "background_" + label2 + ".txt", "w") as f:
        f.write("\n".join(keys2_clean))
    with open(txtoutput + "outliers_" + label2 + ".txt", "w") as f:
        f.write("\n".join(outliers_org2_clean))


#/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path

#  ex : sbatch -p gpu --gres=gpu:1 --job-name=umap_compare --wrap="/home/cassandre/.conda/envs/LPM_2/bin/python /home/cassandre/stage/Cassandre/Code/umap_comparing.py org1path org2path"

# sbatch  gpu --gres=gpu:1 path

# sbatch path (if not pig)