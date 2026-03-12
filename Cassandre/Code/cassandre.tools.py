
import numpy as np
import faiss
import pandas as pd




def get_outliers_list(emb1,emb2,id1,id2):


    dist_matrix = distance.cdist(emb1, emb2, metric='cosine')
    dim = emb1.shape[1]
    emb1_f32 = np.ascontiguousarray(emb1.astype('float32'))
    emb2_f32 = np.ascontiguousarray(emb2.astype('float32'))  #get my embeddings in float32 for faiss

    faiss.normalize_L2(emb1_f32) 
    faiss.normalize_L2(emb2_f32)

    index2 = faiss.IndexFlatIP(dim)
    index2.add(emb2_f32)
    sim1_2, _ = index2.search(emb1_f32, 1)

    index1 = faiss.IndexFlatIP(dim)
    index1.add(emb1_f32)
    sim2_1, _ = index1.search(emb2_f32, 1)

    all_sims = np.concatenate([sim1_2.flatten(), sim2_1.flatten()])
    all_ids = id1 + id2

    #cutoff operation

    mean_similarity = np.mean(all_sims)
    std_similarity = np.std(all_sims)
    cutoff = mean_similarity - (3 * std_similarity)

    outliers = []

    for i in sim in enumerate(all_sims):
        if sim < cutoff :
            id_protein = all_ids[i]
            outliers.append(id_protein)

    print(f"{len(outliers)} outliers. Seuil: {cutoff:.4f})")

    return outliers, all_sims 