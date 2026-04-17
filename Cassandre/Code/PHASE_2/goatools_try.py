#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=goatools_test
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

import os
import glob
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
import csv
import glob


stats = {
	'total_files_processed': 0,
	'files_created': 0,
	'skip_no_genes_in_bg': 0,
	'skip_no_significant_go': 0,
	'skip_missing_data': 0
}

obo_fname = download_go_basic_obo()
godag = GODag(obo_fname)

input_path = "/home/cassandre/stage/Cassandre/results/ortho"
output_path = "/home/cassandre/stage/Cassandre/results/ortho/GO_analysis_rank100"

#put gaf files in a dictionnary to make it easier to loop on
gaf_files = { 'mouse': '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_mouse.gaf',
			  'human' : '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_human.gaf',
				'droso' : '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_fly.gaf',
				'yeast':'/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_yeast.gaf',
				'homosapiens' : '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_human.gaf',
				 'fish' : '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_zebrafish.gaf',
				  'zebrafish' : '/home/cassandre/stage/Cassandre/Code/PHASE_2/goatoolsfiles/goa_zebrafish.gaf'}

gaf_association={}

#parse through the gaf files and associate them
for org, path in gaf_files.items(): 
	reader = GafReader(path, godag=godag)
	gaf_association[org] = reader.get_ns2assc(id_type='DB_Object_ID') 

test_org = 'yeast'
if test_org in gaf_association:
	# that was just a test bc something wasn't working
	keys = list(gaf_association[test_org].get('BP', {}).keys())
	print(f"Exemple de clés chargées pour {test_org}: {keys[:5]}")

background_dict = {}  #make a dictionnary out of the background file cause i have about 800 billions

bg_search = os.path.join(input_path, "**/background_*.txt")

for bg_path in glob.glob(bg_search, recursive=True):
	file_name = os.path.basename(bg_path).lower() 
	
	# We need the model and ortho type from the background filename too
	# background_human_m_esm300m_mtm.txt
	parts = file_name.replace(".txt", "").split("_")
	# parts will be ['background', 'human', 'm', 'esm300m', 'mtm']
	
	bg_org = parts[1] # human
	bg_model = parts[3] # esm300m
	bg_ortho = parts[4] # mtm
	
	# Create a specific key: "human_esm300m_mtm"
	unique_key = f"{bg_org}_{bg_model}_{bg_ortho}"
	
	with open(bg_path, 'r') as f: 
		background_dict[unique_key] = [line.strip().split(':')[-1].upper() for line in f if line.strip()]


master_results = []

search_motif = os.path.join(input_path, "rank100/human_droso/*fail*.txt")

for file_path in glob.glob(search_motif, recursive=True):
	stats['total_files_processed'] += 1
	file_name = os.path.basename(file_path)
	
	# 1. Parse filename (e.g., ESM300M_MtM_homosapiens_fail_top100.txt)
	parts = file_name.split("_")
	model, ortho, organism = parts[0], parts[1], parts[2].lower()
	pairs_folder = os.path.basename(os.path.dirname(file_path))

	# 2. Standardize names for mapping
	org_name = "human" if "homo" in organism else "mouse" if "mus" in organism else organism
	target_bg_key = f"{org_name}_{model.lower()}_{ortho.lower()}"

	# 3. Get GAF and Background
	current_assoc = gaf_association.get(org_name)
	current_bg = None
	if target_bg_key in background_dict:
		current_bg = set(background_dict[target_bg_key])
			
	if current_bg is None or current_assoc is None:
		stats['skip_missing_data'] += 1
		continue 

	# 4. Get the study IDs from the fail file
	with open(file_path, 'r') as f:
		study_ids = [line.strip().split(':')[-1].replace('\r', '').strip().upper() for line in f if line.strip()] 

	# 5. Filter IDs against background 
	study_ids_filtered = [sid for sid in study_ids if sid in current_bg]

	if not study_ids_filtered:
		stats['skip_no_genes_in_bg'] += 1
		continue
		
	print(f"Processing: {file_name} | BG size: {len(current_bg)} | Study size: {len(study_ids_filtered)}")

	# 6. Run Study
	goeaobj = GOEnrichmentStudyNS(
		current_bg, 
		current_assoc, 
		godag,
		alpha = 0.05, 
		methods = ['fdr_bh']
		)

	results_all = goeaobj.run_study(study_ids_filtered)
	if results_all: #if it works, w
		perorganism_output = os.path.join(output_path, f"{pairs_folder}_{model}_{ortho}_{organism}_TOP100_GO.csv")		
		current_file_result = []
		
		for results in results_all:  #get all the info we need, had to calculate some ratios
			if results.p_fdr_bh < 0.05 and results.NS == 'BP' and results.enrichment == 'e':
				direction = "+" if results.enrichment == 'e' else "-"
				study_ratio = results.study_count / results.study_n
				pop_ratio = results.pop_count / results.pop_n
				
				# apparently thats useful so
				fold_enrichment = study_ratio / pop_ratio if pop_ratio > 0 else 0
				
				row = {
					'Model': model,
					'OrthoType': ortho,
					'Organism': organism,
					'GO_ID': results.goterm.id,
					'Term_Name': results.goterm.name,
					'Direction': direction,                    
					'Fold_Enrichment': fold_enrichment, 
					'p_corr': results.p_fdr_bh,
					'Sample_Count': results.study_count,
					'Total_Count': results.pop_count
				} #all the info that will be in the final csv

				master_results.append(row)
				current_file_result.append(row)

		if current_file_result:
			keys = current_file_result[0].keys()
			with open(perorganism_output, 'w', newline='') as f:
				writer = csv.DictWriter(f, fieldnames=keys)
				writer.writeheader()
				writer.writerows(current_file_result)
			stats['files_created'] += 1
		else:
			stats['skip_no_significant_go'] += 1
	else:
		stats['skip_no_significant_go'] += 1

print("\n" + "="*40)
print("GO ANALYSIS FINAL SUMMARY")
print("="*40)
print(f"Total input files scanned:      {stats['total_files_processed']}")
print(f"CSV files successfully created: {stats['files_created']}")
print(f"Skipped (Missing GAF/BG):       {stats['skip_missing_data']}")
print(f"Skipped (No genes in BG):       {stats['skip_no_genes_in_bg']}")
print(f"Skipped (No enriched terms):    {stats['skip_no_significant_go']}")
print("="*40)