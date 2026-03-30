#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=protT5_embedding
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm 


input_parquet = ""
output_h5 = " "

df = pd.read_parquet(input_parquet)

with h5py.File(output_h5, "w") as hf : 
    for _, row in tqdm(df.iterrows(), total = len(df)):
        seq_id = row['seq_id']
        embedding = row.drop('seq_id').values.astype(np.float32)
        hf.create_dataset(seq_id, data=embedding)