"""
Embed protein sequences from a FASTA file using ESM models.

This script reads protein sequences from a FASTA file, generates embeddings
using a specified ESM (Evolutionary Scale Modeling) model, and saves the
results to a Parquet file.

Command-line Arguments:
    -i, --input (str): Path to the input FASTA file containing protein sequences.
                       The file must be readable.
    -o, --output (str): Path to the output Parquet file to save the embeddings.
    -m, --model (str): Name of the ESM model to use for embedding.
                       Available options: esmc_300m, esmc_600m

Raises:
    AssertionError: If the input FASTA file is not readable or if the specified
                    model is not in the list of available models.

Output Format:
    The output Parquet file contains the following columns:
    - seq_id (str): The sequence identifier from the FASTA file
    - position (int): The position index within the sequence
    - Numbered columns (0 to N): Embedding for each position

Notes:
    - The script automatically uses CUDA GPU if available, otherwise falls back
      to CPU.
    - Each embedding is converted to a NumPy array and squeezed to remove
      singleton dimensions.
    - If no embeddings are generated, the script exits with code 1.
"""
import os
import sys
import argparse
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import numpy as np

AVAILABLE_MODELS = ["esmc_300m", "esmc_600m"]


parser = argparse.ArgumentParser(
    description="Embed protein sequences from a FASTA file."
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the input FASTA file containing protein sequences.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output Parquet file to save the embeddings.",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help=f"Name of the ESM model to use for embedding ({', '.join(AVAILABLE_MODELS)}).",
)
args = parser.parse_args()

assert os.path.isfile(args.input) and os.access(args.input, os.R_OK), (
    f"FASTA file '{args.input}' is not readable."
)
assert args.model in AVAILABLE_MODELS, (
    f"Model '{args.model}' is not available. Choose from: {', '.join(AVAILABLE_MODELS)}."
)

FASTA_FILE = args.input
OUT_FILE = args.output
MODEL = args.model


def embed_protein(sequence: str):
    protein = ESMProtein(sequence=sequence) #transforme str en objet que ESMC peut comprendre 
    protein_tensor = client.encode(protein) #transforme sequence en tokens
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True) 
    )#donne les tokens au modèle, sequence = resultat pour chq aa, return_embeddings = on veut les embeddings 

    full_embeddings = logits_output.embeddings

    mean_embeddings = torch.mean(full_embeddings,dim=1) 


    return logits_output.embeddings #renvoie uniquement les embeddings


has_cuda = torch.cuda.is_available()
client = ESMC.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")

embeddings = {}

for record in tqdm(SeqIO.parse(FASTA_FILE, "fasta")):  #loop for everyline in the fasta
    seq_id = record.id #recup de l'identifiants 
    sequence = str(record.seq) #transforme chaine d'AA en string
    embeddings[seq_id] = embed_protein(sequence) #on passe la seq dans le modèle, qui renvoie une matrice représentant la protéine

embeddings_np = {k: v.cpu().numpy().squeeze() for k, v in embeddings.items()} #passe les données du GPU au CPU, convertir les tensors en numpy, nettoyage des dimensions inutiles

if len(embeddings_np) == 0:
    print("No embeddings were generated. Exiting.")
    sys.exit(1)

arrays = np.vstack([arr for arr in embeddings_np.values()])  #prend tts les lignes des matrices et les mets en vertical stack = une matrice ou chaque ligne coorespond à un AA
keys = np.repeat(
    list(embeddings_np.keys()), [arr.shape[0] for arr in embeddings_np.values()]
) #on prend la liste des ID, on compte le nb d'aa dans chq prot : repete le nom de la proteine autant de fois qu'elle a d'AA 
position = np.concatenate([np.arange(arr.shape[0]) for arr in embeddings_np.values()]) #genere une suite de chiffre pour donner une position aux AA

df = pd.DataFrame(arrays, columns=[str(i) for i in range(arrays.shape[1])]) #crée une df avec matrice des aa, nomme les colonnes 
df.insert(0, "seq_id", keys) #ajoute noms des protéines
df.insert(1, "position", position) #insère position de l'AA
df.columns = df.columns.astype(str) #mets chaque colonne en str

df.to_parquet(OUT_FILE, index=True) #turn dataframe to parquet
