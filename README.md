Description and comparison of proteomes with protein Language Models

This README file will explain each code, what it was for and the input/output.

The test folder in the code repertory is just codes that may hold useful scripts but are not essential to the pipeline.

Folder embed_gen :

1. protT5_embedding.py
This code generates embeddings using ProtT5.
Input : FASTA File
Output : .h5 file = the embedding corresponding the fasta input.

Folder PHASE1_OUTLIERS : 

1. automaticplotly.py
This script generates umap plotly graph of 1 h5 file agaisnt all file in a foler
Input : Fichier H5 de référence, emplacement des fichiers à comparer,
Output : Une UMAP par paire d'organisme entre la référence et les requêtes.

2. checkdistance_2proteins.py
This code was made to cjeck the distance between 2 specific proteins using scipy.
Les fichiers sont à modifier dans le code.
Output : Un print de la valeur de la distance.

3. colorumap_size.py
This code creates a umap of one embedding. It colors the map based on the sizeof the  sequences in a fasta
The arguments have to be changed in the script.
Input : h5 embedding, fasta file of same organism
Output : UMAP html file.

4. coverage_outlierlist.ipynb
This notebook holds the code to see the overlap between 2 or 3 .txt lists. Used to compare outliers list of different settings / models.

5. distancematrixV1.py
This code generates a excel file with the closest neighbor of each protein and the cosine distance.
Input :  2 h5 embeddings, an output path
Output : .csv file

6. umap_outliers.py
This code generates a umap representing embeddings from 2 organism and seperating the outliers from the non-outliers
Input : 2 h5 embeddings, UMAP output and gene list output path.
Output : gene list and UMAP

Folder PHASE2 : 

1. goatools_try.py
This code performs an enrichment analysis using goatools.
Input: GAF annotation files, background gene lists (txt), "fail" gene lists (txt), and the GO basic OBO file.
Output: Individual CSV files containing enriched terms, fold enrichment, and corrected p-values for each analysis, along with a processing summary.

 3. ortholog_validation_ranking.py
This code evaluates and compares different protein language models. It separates orthologs by ortholog relationship and look at the distance : is the ortholog top 1, top k, top 10....
Input: Path to ortholog pairs, protein ID lists, FASTA files, and protein embeddings (H5) for multiple models
Output: TSV files with detailed ranking data, TXT files listing "failed" ortholog pairs, a comprehensive summary file (means, medians, and Mann-Whitney U tests), and PNG plots showing distance and rank distributions.

4. OrthologComparison.ipynb
This notebook interacts with the OrthoInspector API to retrieve orthology relationships and protein catalogs for specific organisms based on their taxonomic IDs.
Input: Taxonomic IDs (e.g., 9606 for Human, 559292 for Zebrafish) and API endpoints from OrthoInspector.
Output: TSV files containing lists of orthologous protein pairs and full protein ID catalogs for the specified organisms.

5. OrthologListMaker.ipynb
This notebook implements a complete validation pipeline, from processing sequence IDs to statistical testing and visualization. It compares three protein language models to see how well their embeddings distinguish orthologs from non-orthologs using cosine similarity and FAISS. It calculates performance metrics like Top-K ranks and uses Fisher’s Exact Tests to see if "outlier" proteins are statistically linked to a lack of known orthology.

6. orthoo_rank100.py
This script performs a large-scale comparative analysis of three protein language models to evaluate their ability to identify orthologous pairs across different species.
Input: Ortholog pair lists, protein TSV files, FASTA files, and H5 embeddings for ProtT5, ESM-C 300M, and ESM-C 600M.
Output:  TSV results, rank distribution plots (log scale), distance histograms, and a global summary text file.

