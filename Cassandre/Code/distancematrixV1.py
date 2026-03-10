import h5py
import numpy as np
import faiss



def embedding_setup(path):
    ids = []
    embeddings=[]

    with h5py.File(path,"r") as hf :
        for sequenceID in hf.keys():
            vectors = hf[sequenceID][:]
            ids.append(sequenceID)
            embeddings.append(vectors)
        embed_array = np.array(embeddings).astype('float32')
    
    return ids, embed_array
# have to use this function twice : once on query once on reference

id1,emb1 = embedding_setup(organism1)
id2,emb2 = embedding_setup(organism2)

faiss.normalize_L2(emb1)
faiss.normalize_L2(emb2)  #need to normalize to do the Inner product (dot product for cosine similarity)

d = 1024 #tell faiss how many dimension we're working with
index2 = faiss.IndexFlatIP(d)  #indexflat = no compression (avoid info loss)  / IP = inner or dot product = the cosine similarity
index2.add(emb2) #store the information of organism2 to search through

distances,indices=index2.search(emb1, k=1) #look through organism2 to compare with organism1 prot and look for the closest neighbor only
#distances store the distances, indices the row number that match the indice of the list of ids !! genius congrats me :)


