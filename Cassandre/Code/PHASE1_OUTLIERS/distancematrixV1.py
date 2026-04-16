#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=MATRIXDISTANCE_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

#this code generates a excel file with the closest neighbor of each protein and the distance / similarity

import h5py
import numpy as np
import faiss
import pandas as pd

import argparse
import os


def embedding_setup(path):
    ids = []
    embeddings = []

    with h5py.File(path, "r") as hf:
        for sequenceID in hf.keys():
            dataset = hf[sequenceID]
            # 1. On crée un array vide de la bonne taille en Float32
            # dataset.shape te donne la dimension du vecteur (ex: 1024)
            vec = np.empty(dataset.shape, dtype='float32')
            
            # 2. On force la lecture directe dedans
            dataset.read_direct(vec)
            
            ids.append(sequenceID)
            embeddings.append(vec)
            
    # On empile tout à la fin
    embed_array = np.array(embeddings).astype('float32')
    
    return ids, embed_array
# have to use this function twice : once on query once on reference


def index_search_distances(emb_query,emb_ref):
  
    
    faiss.normalize_L2(emb_query)
    faiss.normalize_L2(emb_ref) #need to normalize to do the Inner product (dot product for cosine similarity)

    d = 1024  #tell faiss how many dimension we're working with
    index = faiss.IndexFlatIP(d) #indexflat = no compression (avoid info loss)  / IP = inner or dot product = the cosine similarity
    index.add(emb_ref) #store the information of organism2 to search through

    distances, indices = index.search(emb_query, k=1)#look through organism2 to compare with organism1 prot and look for the closest neighbor only
#distances store the distances, indices = the row number that match the indice of the list of ids !! genius congrats me :)

    return distances,indices 


def extractname_frompath(organismpath) :
    filename = os.path.basename(organismpath)
    nomseul = os.path.splitext(filename)[0]
    label= nomseul.split('_')[0]

    return label


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(    )
    parser.add_argument('Path_Org1', help='Chemin du fichier H5 de organisme 2')
    parser.add_argument('Path_Org2', help='Chemin du fichier H5 de organisme 2')
    parser.add_argument('Output_Path',help='Path where the output dataframe will be created')
    args = parser.parse_args()

    path_Org1 = args.Path_Org1
    path_Org2 = args.Path_Org2
    output_path=args.Output_Path
        
    id1,emb1 = embedding_setup(path_Org1)
    id2,emb2 = embedding_setup(path_Org2)
    
    label1 = extractname_frompath(path_Org1)
    label2 = extractname_frompath(path_Org2)

    distances1_2, indices1_2= index_search_distances(emb1,emb2)
    distances2_1, indices2_1= index_search_distances(emb2, emb1)
    # run the function twice to get all the closest neighbor

    all_closest_neighbors = np.concatenate([distances1_2.flatten(),distances2_1.flatten()])


    neighbor_names_2_1 = [id1[idx[0]] for idx in indices2_1]
    neighbor_names_1_2 = [id2[idx[0]] for idx in indices1_2]


    df1 = pd.DataFrame({'proteinID' : id1, 'closest_Neighbor' : neighbor_names_1_2, 'similarité' : distances1_2.flatten(), 'species' : label1  })
    df2 = pd.DataFrame({'proteinID' : id2, 'closest_Neighbor' : neighbor_names_2_1, 'similarité' : distances2_1.flatten(), 'species' : label2  })

    df_allorganism = pd.concat([df1,df2],ignore_index=True)


    df_allorganism.to_csv(f"{output_path}/{label1}_{label2}_clo sestneighbors.csv",index=False)


    #sbatch -p gpu --gres=gpu:1 path