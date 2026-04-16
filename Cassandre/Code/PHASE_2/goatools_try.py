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

input_path = "/home/cassandre/stage/Cassandre/ortho/comparison_results"
output_path = "/home/cassandre/stage/Cassandre/ortho/GO_analysis_output"

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

for bg_path in glob.glob(bg_search, recursive=True): #for each background file, i get the name, the pair it belongs too and mix both so i know who is who
	file_name = os.path.basename(bg_path).lower() 
	pair_folder = os.path.basename(os.path.dirname(os.path.dirname(bg_path)))
	unique_key = f"{pair_folder}_{file_name}"

	with open(bg_path, 'r') as f: 
			background_dict[unique_key] = [line.strip().split(':')[-1].upper() for line in f if line.strip()] #make the id all pretty 


master_results = []

search_motif = os.path.join(input_path, "**/*_fail.txt") #grab all my failures text file

for file_path in glob.glob(search_motif, recursive=True): #for each of my failures i again get the name and infos i'll probably need
	stats['total_files_processed'] += 1
	file_name = os.path.basename(file_path)
	parts = file_name.replace("_fail.txt", "").split("_")
	model,ortho,organism = parts[0], parts[1],parts[2].lower()

	pairs_folder = os.path.basename(os.path.dirname(file_path)) #again figure out which pair it belongs too

	current_assoc = gaf_association.get(organism) #get the gaf assoc of the organism we're working on

	search_org = organism #cause im not smart and used different names on different files
	if organism == "homosapiens":
		search_org = "human"
	if organism == "zebrafish" : search_org = "fish" 
	
	# associate file to background
	current_bg = None
	target_model, target_ortho = model.lower(), ortho.lower()
	

	search_org = "human" if organism == "homosapiens" else "fish" if organism == "zebrafish" else organism
	
	for bg_key, bg_list in background_dict.items():  
		bg_k = bg_key.lower() #key is the name of the file kinda + the pair 
		
		#take out the background and pair folder name
		bg_filename_part = bg_k.split("background_")[-1]

		cond_folder = pairs_folder.lower() in bg_k #make the conditions for the bg to be the right now
		cond_model = target_model in bg_k
		cond_ortho = target_ortho in bg_k
		
		#bg start by the organism they are made for so look them up
		name_for_bg = "human" if organism == "homosapiens" else "fish" if organism == "zebrafish" else organism
		cond_org = bg_filename_part.startswith(f"{name_for_bg}_") 
		
		if cond_folder and cond_model and cond_ortho and cond_org:  #if we fill all conditions the bg must be the right now 
			current_bg = set(bg_list) 
			break
			
	if current_bg is None:
		stats['skip_missing_data'] += 1
		continue 

	# get the id of the file we're working on
	with open(file_path, 'r') as f:
		study_ids = [line.strip().split(':')[-1].replace('\r', '').strip().upper() for line in f if line.strip()] 

	# check if the ids are in the background 
	study_ids_filtered = [sid for sid in study_ids if sid in current_bg]

	if not study_ids_filtered:
		stats['skip_no_genes_in_bg'] += 1
		continue
		
	if current_bg and current_assoc: #again, needed a safety cause i had trouble pairing the file and background
		print(f"DEBUG BG - 3 premiers IDs du background: {list(current_bg)[:3]}")
		
		assoc_keys = list(current_assoc.get('BP', {}).keys())
		print(f"DEBUG ASSOC - 3 premières clés de l'association: {assoc_keys[:3]}")

	#create the actual study using bg and file 
	goeaobj = GOEnrichmentStudyNS(
		current_bg, 
		current_assoc, 
		godag,
		alpha = 0.05, 
		methods = ['fdr_bh']
		)

	results_all = goeaobj.run_study(study_ids_filtered) #get all the info


	if results_all: #if it works, w
		perorganism_output = os.path.join(output_path, f"{pairs_folder}_{model}_{ortho}_{organism}_GO.csv")
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