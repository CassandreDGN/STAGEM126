#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=autoPLOTLY_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

#this code create a umap of one h5 file but has no arguments coded so it has to be changed in the script
#it is colored based on the size of the sequences of the fasta file

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import umap 
from Bio import SeqIO
import plotly.express as px






fasta_path = "/home/cassandre/stage/Cassandre/Proteome/UniProt/UP000000589_10090.fa"
lengths = {}
for record in SeqIO.parse(fasta_path, "fasta"):
    clean_id = record.id.split('|')[1] if '|' in record.id else record.id
    lengths[clean_id] = len(record.seq)

#Transforming H5 on something we can do a umap with

embedding_vectors = []
matched_lengths = []
matched_ids = [] 

with h5py.File('/home/cassandre/stage/Cassandre/Embeddings/PROTT5_AVERAGEPOOLING/mus_embedding_protT5_uniprot_proteinembeddings.h5','r') as f:
    for proteinID in f.keys(): 
        clean_id = proteinID.split('|')[1] if '|' in proteinID else proteinID

        if clean_id in lengths:
            embedding_vectors.append(f[proteinID][:])
            matched_lengths.append(lengths[clean_id])
            matched_ids.append(proteinID) 


X = np.array(embedding_vectors)
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_neighbors=12, min_dist=0.1, metric='cosine', random_state=42)
embedding = reducer.fit_transform(X_scaled)


df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'Protein_ID': matched_ids, 
    'Length': matched_lengths,
    'Log10_Length': np.log10(matched_lengths)
})




fig = px.scatter(
    df, 
    x='UMAP1', 
    y='UMAP2',
    color='Log10_Length',
    hover_data=['Protein_ID'],
    color_continuous_scale='turbo',

    title='Mouse Protein Embeddings UMAP protT5 (Colored by Length)')


fig.update_traces(marker=dict(size=3))

output_html = '/home/cassandre/stage/Cassandre/UMAPComparisonFig/ORDERBY_SIZE/mus_umap_interactive.html'
fig.write_html(output_html)