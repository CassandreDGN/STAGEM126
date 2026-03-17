#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=autoPLOTLY_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

#this makes one umap ploly graph of ur file VS all file in folder 
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap


def extractname_frompath(organismpath):
	filename = os.path.basename(organismpath)
	nomseul = os.path.splitext(filename)[0]
	label = nomseul.split('_')[0]
	return label


def extracting_h5embeddings(file_path, id_organism):
	embedding_fonction = []
	ids_fonction = []
	keys_fonctions = []

	with h5py.File(file_path, 'r') as f:
		for protein_id in f.keys():
			embedding_fonction.append(f[protein_id][:])
			ids_fonction.append(id_organism)
			keys_fonctions.append(protein_id)

	return embedding_fonction, ids_fonction, keys_fonctions


def compare_two_h5(h5path_organism1, h5path_organism2, output_dir):
	label1 = extractname_frompath(h5path_organism1)
	label2 = extractname_frompath(h5path_organism2)

	emb1, id1, keys1 = extracting_h5embeddings(h5path_organism1, 1)
	emb2, id2, keys2 = extracting_h5embeddings(h5path_organism2, 2)

	valeursembeddings = emb1 + emb2
	ids = id1 + id2
	keysprot = keys1 + keys2

	x = np.array(valeursembeddings)
	x_scaled = StandardScaler().fit_transform(x)

	reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
	embedding = reducer.fit_transform(x_scaled)

	df = pd.DataFrame(
		{
			'ProteinID': keysprot,
			'UMAP_1': embedding[:, 0],
			'UMAP_2': embedding[:, 1],
			'Organism': [label1 if i == 1 else label2 for i in ids],
		}
	)

	fig = px.scatter(df, x='UMAP_1', y='UMAP_2', color='Organism', hover_data=['ProteinID'])
	fig.update_traces(marker=dict(size=3))

	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, f'comparison_{label1}_{label2}.html')
	fig.write_html(output_path)
	return output_path


def find_other_h5_files(reference_h5, search_dir):
	reference_abs = os.path.abspath(reference_h5)
	other_h5 = []

	for filename in sorted(os.listdir(search_dir)):
		if not filename.lower().endswith('.h5'):
			continue
		candidate = os.path.abspath(os.path.join(search_dir, filename))
		if candidate == reference_abs:
			continue
		if os.path.isfile(candidate):
			other_h5.append(candidate)

	return other_h5


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path1', help='Chemin du fichier H5 de référence')
	parser.add_argument(
		'--folder',
		default=None,
		help='Dossier où chercher les autres fichiers .h5 (par défaut: dossier de path1)',
	)
	parser.add_argument('output', help="Dossier d'output pour les fichiers HTML")

	args = parser.parse_args()

	h5_reference = args.path1
	if not os.path.isfile(h5_reference):
		raise FileNotFoundError(f'Fichier introuvable: {h5_reference}')

	search_folder = args.folder if args.folder else os.path.dirname(os.path.abspath(h5_reference))
	if not os.path.isdir(search_folder):
		raise NotADirectoryError(f'Dossier introuvable: {search_folder}')

	targets = find_other_h5_files(h5_reference, search_folder)
	if not targets:
		print('Aucun autre fichier .h5 trouvé à comparer.')
	else:
		print(f'{len(targets)} fichier(s) .h5 à comparer avec {h5_reference}')

	for other_h5 in targets:
		try:
			generated = compare_two_h5(h5_reference, other_h5, args.output)
			print(f'Généré: {generated}')
		except Exception as exc:
			print(f'Erreur avec {other_h5}: {exc}')

