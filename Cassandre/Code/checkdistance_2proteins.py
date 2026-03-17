#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=MATRIXDISTANCE_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out
#SBATCH --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

#this code was made to check the distance between protein using scipy

from scipy.spatial import distance
import numpy as np
import h5py
import os   
 

def extractname_frompath(organismpath) :
    filename = os.path.basename(organismpath)
    nomseul = os.path.splitext(filename)[0]
    label= nomseul.split('_')[0]

    return label


def embedding_setup(path):
    ids = []
    embeddings = []

    with h5py.File(path, "r") as hf:
        for sequenceID in hf.keys():
            vec = hf[sequenceID][()].astype('float64')
                        
            ids.append(sequenceID)
            embeddings.append(vec)
            
    # On empile tout à la fin    
    return ids,np.array(embeddings)

def verifierdistance(prot_id1, prot_id2):
    # On trouve l'emplacement (l'index) des protéines dans nos listes
    try:
        idx1 = id1.index(prot_id1)
        idx2 = id2.index(prot_id2)
        
        # On pioche la valeur dans la matrice
        dist = dist_matrix[idx1, idx2]
        
        print(f"Distance (Scipy float64) entre {prot_id1} et {prot_id2} :")
        print(f"-> {dist:.15f}")
        print(f"-> Similarité (1 - dist) : {1 - dist:.15f}")
        
    except ValueError:
        print("L'un des IDs n'a pas été trouvé dans les listes.")



organism1 = "/home/cassandre/stage/Cassandre/Embeddings/PROTT5_AVERAGEPOOLING/chimp_embedding_protT5_uniprot_proteinembeddings.h5"
organism2 =  "/home/cassandre/stage/Cassandre/Embeddings/PROTT5_AVERAGEPOOLING/homosapiens_embedding_protT5_uniprot_proteinembeddings.h5"

id1,emb1 = embedding_setup(organism1)
id2,emb2 = embedding_setup(organism2)



dist_matrix=distance.cdist(emb1,emb2, metric = "cosine")

verifierdistance("tr|A0A2I3S1V5|A0A2I3S1V5_PANTR", "sp|Q4W4Y0|DRIP1_HUMAN")

# have to use this function twice : once on query once on reference