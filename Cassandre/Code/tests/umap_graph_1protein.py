import h5py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd
import umap 



#Transforming H5 on something we can do a umap with

embedding_vectors = []
embedding_id = []

with h5py.File('/home/cassandre/stage/Cassandre/Embeddings/mus_embedding_protT5_uniprot_proteinembeddings.h5','r') as f :  #open h5 file in read mode
    for proteinID in f.keys(): 
        embeddings = f[proteinID][:] #take all the numbers of the embeddings into the variable

        embedding_vectors.append(embeddings) #throw the embeddings values into a list
        embedding_id.append(proteinID) #take the protein id to link it to the values
        
X = np.array(embedding_vectors) #turn the list of values into a numpy array to make maths on it 
 
X_scaled = StandardScaler().fit_transform(X) #idk they did that on the umap tutorial


reducer = umap.UMAP(n_neighbors=12,min_dist=0.1,metric='cosine',random_state=42)
embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1, alpha=0.5, color='blue')
plt.savefig('mus_umap_plot.png', dpi=300)