#!/home/cassandre/.conda/envs/LPM_2/bin/python
#SBATCH --job-name=MATRIXDISTANCE_comparing
#SBATCH --output=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.out --error=/home/cassandre/stage/Cassandre/slurm_out/slurm-%J.err

import h5py
import numpy as np
import os
import faiss
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.spatial import distance
from scipy.stats import mannwhitneyu
import random
import argparse
import sys
import csv
import gc


def extractname_frompath(organismpath) :
    filename = os.path.basename(organismpath)
    nomseul = os.path.splitext(filename)[0]
    label= nomseul.split('_')[0]

    return label

def get_list_protein(chemin_fichier):  #get the protein list from the orthoinspector tsv file per organism
    listA = set()
    with open(chemin_fichier,'r') as f:
        for line in f:
            clean_id = line.strip()
            if clean_id:
                listA.add(clean_id)
    return listA



def embedding_setup(path):  #turn my embeddings in np.array in float32 for scipy compatibility
    ids = []
    embeddings = []

    with h5py.File(path, "r") as hf:
        for sequenceID in hf.keys():
            dataset = hf[sequenceID]
            vec = np.empty(dataset.shape, dtype='float32')
            dataset.read_direct(vec)
            
            ids.append(sequenceID)
            embeddings.append(vec)
            
    
    embed_array = np.array(embeddings).astype('float32')
    
    return ids, embed_array


def normalize_embs(emb):  #normalize the embeddings because faiss used to do it so i guess its useful
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def clean_id(long_id):  #clean the fasta header so i only keep the protein ID
    if '|' in long_id:
        return long_id.split('|')[1]
    return long_id

def get_orthology_results(matrix, h_map, y_map, paires, inv_h, inv_m):  #split the protein based on their orthology relationship types 
    raw_results = {'All': [], 'OtO': [], 'OtM': [], 'MtO': [], 'MtM': []}

    for h_id, m_id in paires:
        if h_id in h_map and m_id in y_map:
            dist = matrix[h_map[h_id], y_map[m_id]]  #distance in the matrix
            rank = (matrix[h_map[h_id], :] < dist).sum() + 1 #look up the rank of said distance across all the distances possible
            K = inv_h[h_id] 
            
            res = (rank, K, h_id, m_id)  #store information about every pair
            raw_results['All'].append(res) #contain every pair information just in case lol
            
            if inv_h[h_id] == 1 and inv_m[m_id] == 1:  #count the nb of occurences of each protein ID in the ortholog pair list from orthoinspector
                raw_results['OtO'].append(res)
            elif inv_h[h_id] > 1 and inv_m[m_id] == 1:
                raw_results['OtM'].append(res)
            elif inv_h[h_id] == 1 and inv_m[m_id] > 1:
                raw_results['MtO'].append(res)
            else:
                raw_results['MtM'].append(res)

    stats = {}
    
    for cat, r_list in raw_results.items():  #calculate median and average for each
        if len(r_list) > 0:
            ranks_only = [r[0] for r in r_list]  #take the ranks out of the tuples 
            stats[cat] = (np.mean(ranks_only), np.median(ranks_only)) #calculate the mean and median of the ranks per category
        else:
            stats[cat] = (0.0, 0.0)
    
    return stats, raw_results





def save_raw_to_tsv(raw_results, model_name, out_dir):  #to save my results as tsv, contains : category, protID 1 & 2, ranking, and nb of orthologs
    output_path = os.path.join(out_dir, f"{model_name}_full_results.tsv")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Category', 'Org1_ID', 'Org2_ID', 'Rank', 'K_Expected'])
        for cat, pairs in raw_results.items():
            if cat == 'All': continue
            for rank, K, h_id, m_id in pairs:
                writer.writerow([cat, h_id, m_id, rank, K])



def print_fixed_thresholds(list_models, category, out_file=None): #for oto and mto, check if the orthologs are in the top 1/5/10/100
    """Prints Top 1, 5, 10, and 100 percentages for absolute ranks."""
    print(f"\n{'MODÈLE':<9} | {'TOP 1':>10} | {'TOP 5':>10} | {'TOP 10':>10} | {'TOP 100':>10}", file=out_file)
    print("-" * 65, file=out_file)

    for name, raw_data in list_models:
        ranks = [r[0] for r in raw_data.get(category, [])]
        
        if not ranks:
            print(f"{name:<9} | No {category} data found", file=out_file)
            continue
            
        data = np.array(ranks)
        top1 = (data <= 1).mean() * 100
        top5 = (data <= 5).mean() * 100
        top10 = (data <= 10).mean() * 100
        top100 = (data <= 100).mean() * 100
        
        print(f"{name:<9} | {top1:>9.2f}% | {top5:>9.2f}% | {top10:>9.2f}% | {top100:>9.2f}%", file=out_file) #print bc i use it in a file after


def print_relative_thresholds(list_models, category, threshold, out_file=None): #for mtm and otm, check if the orthologs distance are in the top k/k+10/1% and top 100
    """Prints Top K, K+10, Threshold, and 100 percentages for dynamic ranks."""
    print(f"\n{'MODÈLE':<12} | {'TOP K':>9} | {'TOP K+10':>11} | {'TOP THRESH':>11} | {'TOP 100':>11}", file=out_file)
    print("-" * 65, file=out_file)

    for name, raw_data in list_models:
        pairs = raw_data.get(category, [])
        if not pairs:
            print(f"{name:<12} | No {category} data found", file=out_file)
            continue

        count_topk = count_topk10 = count_threshold = count_top100 = 0
        total = len(pairs)
        
        for r in pairs:
            rang = r[0]
            K = r[1]

            if rang <= K: count_topk += 1
            if rang <= K + 10: count_topk10 += 1
            if rang <= threshold: count_threshold += 1
            if rang <= 100: count_top100 += 1

        percentk = (count_topk / total) * 100
        percentk10 = (count_topk10 / total) * 100
        percent_thresh = (count_threshold / total) * 100
        percent100 = (count_top100 / total) * 100

        print(f"{name:<12} | {percentk:>9.2f}% | {percentk10:>11.2f}% | {percent100:>11.2f}% | {percent_thresh:>11.2f}%", file=out_file)            

def get_failures_ids(raw_data, name1, name2):  
    failure_lists = {}
    categories = ['OtO', 'MtO', 'OtM', 'MtM']

    for cat in categories:
        failure_lists[f"{cat}_{name1}_fail_top100"] = set()
        failure_lists[f"{cat}_{name2}_fail_top100"] = set()
    
    for cat in categories:
        for rank, K, id1, id2 in raw_data[cat]:
            if rank > 100:
                failure_lists[f"{cat}_{name1}_fail_top100"].add(id1)
                failure_lists[f"{cat}_{name2}_fail_top100"].add(id2)
                
    return {k: sorted(list(v)) for k, v in failure_lists.items()}

def save_fails_to_txt(fail_dict, model_name, out_dir):  #saves the protein ID of the ones that are out of the top yk what i mean save those id to a txt file 
    for list_name, ids in fail_dict.items():
        with open(os.path.join(out_dir, f"{model_name}_{list_name}.txt"), 'w') as f:
            f.write("\n".join(ids))

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    
    parser.add_argument("ortho", help="Path to orthologs list")
    parser.add_argument("org1_list", help="Path to protein tsv file for org1")
    parser.add_argument("org2_list", help="Path to protein tsv file for org2")
    parser.add_argument("org1_fasta", help="Path to org1 FASTA")
    parser.add_argument("org2_fasta", help="Path to org2 FASTA")

    parser.add_argument("--t5_1", help="Path to org1 PROTT5 embeddings")
    parser.add_argument("--t5_2", help="Path to org2 PROTT5 embeddings")
    parser.add_argument("--esm300_1", help="Path to org1 esm300m embeddings")
    parser.add_argument("--esm300_2", help="Path to org2 esm300m embeddings")
    parser.add_argument("--esm600_1", help="Path to org1 esm600m embeddings")
    parser.add_argument("--esm600_2", help="Path to org2 esm600m embeddings")
    
    parser.add_argument("--out_dir", default="results", help="Output directory")

    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
  


    # put the args in list to make it easier to use below
    models_tasks = []
    if args.t5_1 and args.t5_2: 
        models_tasks.append(("ProtT5", args.t5_1, args.t5_2))
    if args.esm300_1 and args.esm300_2:
        models_tasks.append(("ESM300M", args.esm300_1, args.esm300_2))
    if args.esm600_1 and args.esm600_2:
        models_tasks.append(("ESM600M", args.esm600_1, args.esm600_2))

    label1 = extractname_frompath(models_tasks[0][1])
    label2 = extractname_frompath(models_tasks[0][2])

    # load orthology tsv files to get the info needed 
    paires_temp = []
    inv_org1, inv_org2 = {}, {}

    with open(args.ortho, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                h, m = parts[0], parts[1]
                paires_temp.append((h, m))
                inv_org1[h] = inv_org1.get(h, 0) + 1
                inv_org2[m] = inv_org2.get(m, 0) + 1

    all_raw_results = []
    all_stats_results = []
    stats_comparison_results = []  #get some variables set up
    total_prots_target = 0

    # process embeddings to get the matrix calculation going
    for model_label, path1, path2 in models_tasks:  #loop for 3 models used
        print(f"\n--- Processing {model_label} ---") 
        
        id1, emb1 = embedding_setup(path1)
        id2, emb2 = embedding_setup(path2)
        total_prots_target = emb2.shape[0]
        
        emb1_n = normalize_embs(emb1)
        emb2_n = normalize_embs(emb2)
        
        # matrix calculation
        dist_matrix = distance.cdist(emb1_n, emb2_n, metric="cosine")

        m1 = {clean_id(pid): i for i, pid in enumerate(id1)}
        m2 = {clean_id(pid): i for i, pid in enumerate(id2)}
        
        # ortho ranks
        stats, raw = get_orthology_results(dist_matrix, m1, m2, paires_temp, inv_org1, inv_org2)
        all_raw_results.append((model_label, raw))
        all_stats_results.append((model_label, stats))
        
        # test stats
        valid_pairs = [(h, m) for h, m in paires_temp if h in m1 and m in m2] #make sure everything is in the orthology list and in the matrix
        ortho_dist = [dist_matrix[m1[h], m2[m]] for h, m in valid_pairs] #get the distance between orthologs

 
        sample_pairs = random.sample(valid_pairs, min(1000, len(valid_pairs))) #get a random sample bc it was way to heavy to do it for all

        h_nonortho_dist = [dist_matrix[m1[h], j] #get some distance for non ortho for the stat test, 
                        for h, m in sample_pairs 
                        for j in range(dist_matrix.shape[1]) 
                        if j != m2[m]]

        m_nonortho_dist = [dist_matrix[i, m2[m]] 
                        for h, m in sample_pairs 
                        for i in range(dist_matrix.shape[0]) 
                        if i != m1[h]]

        stat_h, p_h = mannwhitneyu(h_nonortho_dist, ortho_dist, alternative='greater')
        stat_m, p_m = mannwhitneyu(m_nonortho_dist, ortho_dist, alternative='greater')

        stats_comparison_results.append({
            'label': model_label,
            'p_h': p_h, 'significant_h': "VRAI" if p_h < 0.05 else "FAUX",
            'p_m': p_m, 'significant_m': "VRAI" if p_m < 0.05 else "FAUX",
            'ortho': ortho_dist,
            'h_nonortho': h_nonortho_dist,
            'm_nonortho': m_nonortho_dist,
        }) #get the results in a df 
 
        save_raw_to_tsv(raw, model_label, args.out_dir)
        save_fails_to_txt(get_failures_ids(raw, label1, label2), model_label, args.out_dir)  #save everyone)
        
        # Cleanup memory immediately apparently thats gppd 
        del id1, emb1, id2, emb2, emb1_n, emb2_n, dist_matrix  #
        gc.collect()

    # make a summary file
    one_percent_thresh = total_prots_target * 0.01  #set the threshold for the 1% ranking of the top for otm and mtm 
    summary_path = os.path.join(args.out_dir, "comparison_summary.txt") #

    with open(summary_path, 'w') as f:  #create a file with a lot of summarized info like the top %, avg and median, stat test....
        print("=== ANALYSIS COMPARISON SUMMARY ===", file=f)
        print(f"Target Database Size: {total_prots_target}", file=f)
        print(f"1% Threshold Rank: {one_percent_thresh:.2f}\n", file=f)

        # 1. distribution
        for cat in ['OtO', 'OtM', 'MtO', 'MtM']:
            print(f"--- {cat} Performance ---", file=f)
            if cat in ['OtO', 'MtO']:
                print_fixed_thresholds(all_raw_results, cat, out_file=f)
            else:
                print_relative_thresholds(all_raw_results, cat, one_percent_thresh, out_file=f)
            print("\n", file=f)

        # 2. avg & median
        print("=== MEAN AND MEDIAN RANKS ===", file=f)
        header = f"{'Category':<10} | {'Metric':<8}"
        for label, _ in all_stats_results:
            header += f" | {label:<10}"
        print(header, file=f)
        print("-" * len(header), file=f)

        for cat in ['All', 'OtO', 'OtM', 'MtO', 'MtM']:
            avg_line = f"{cat:<10} | Mean"
            med_line = f"{'':<10} | Median"
            for _, stat in all_stats_results:
                avg_line += f" | {stat[cat][0]:<10.2f}"
                med_line += f" | {stat[cat][1]:<10.2f}"
            print(avg_line, file=f)
            print(med_line, file=f)
            print("-" * len(header), file=f)
        
        print("\n=== MANN-WHITNEY U TEST (Non-Ortho > Ortho) ===", file=f)
        print(f"{'Model':<12} | {f'P-Value {label1}':<14} | {'Sig':<10} | {f'P-Value {label2}':<14} | {'Sig':<10}", file=f)
        print("-" * 65, file=f)
        for res in stats_comparison_results:
            print(f"{res['label']:<12} | {res['p_h']:<14.2e} | {res['significant_h']:<10} | {res['p_m']:<14.2e} | {res['significant_m']:<10}", file=f)

    # 1. Distance Distributions (The Yellow/Green Notebook Style)
    nb_bins = np.linspace(0, 1, 101)    
    n_models = len(stats_comparison_results)
    
    fig1, axes1 = plt.subplots(2, n_models, 
                               figsize=(6 * n_models, 10), 
                               squeeze=False)

    for i, res in enumerate(stats_comparison_results):
        # Row 0: Blue/Salmon | Row 1: Green/Gold
        row_configs = [
            {'ortho_k': 'ortho', 'non_k': 'h_nonortho', 'c': ("skyblue", "salmon"), 'lab': f"Humain vs Proteome {label2}"},
            {'ortho_k': 'ortho', 'non_k': 'm_nonortho', 'c': ("limegreen", "gold"), 'lab': f"{label2} vs Proteome Humain"}
        ]
        
        for row, cfg in enumerate(row_configs):
            ax = axes1[row, i]
            
            ortho_data = res[cfg['ortho_k']]
            non_ortho_data = res[cfg['non_k']]

            ortho_hist, _ = np.histogram(ortho_data, bins=nb_bins, density=True)
            nonortho_hist, _ = np.histogram(non_ortho_data, bins=nb_bins, density=True)
            
            ax.bar(nb_bins[:-1], ortho_hist, width=np.diff(nb_bins), alpha=0.5, 
                   label=f"Ortho (µ={np.mean(ortho_data):.3f})", color=cfg['c'][0], edgecolor="none", align="edge")
            
            ax.bar(nb_bins[:-1], nonortho_hist, width=np.diff(nb_bins), alpha=0.5, 
                   label=f"Non-Ortho (µ={np.mean(non_ortho_data):.3f})", color=cfg['c'][1], edgecolor="none", align="edge")
            
            ax.set_title(f"{res['label']}\n{cfg['lab']}", fontsize=13)
            ax.set_xlabel("Cosine Distance")
            ax.set_ylabel("Density")
            ax.legend(loc='upper left', frameon=False)

    plt.suptitle("Cosine distance distributions: orthologs vs non-orthologs", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "distance_distributions.png"), dpi=200, bbox_inches='tight')
    plt.close(fig1) # Crucial: frees up memory on the cluster

    # 2. Rank Distributions (Log Scale)
    categories = ['OtO', 'OtM', 'MtO', 'MtM']
    n_raw = len(all_raw_results)
    
    fig2, axes2 = plt.subplots(n_raw, len(categories), 
                               figsize=(5 * len(categories), 4 * n_raw), 
                               squeeze=False)

    for row, (model_label, raw) in enumerate(all_raw_results):
        for col, cat in enumerate(categories):
            ax = axes2[row, col]
            pairs = raw.get(cat, [])

            if not pairs:
                ax.set_visible(False)
                continue

            ranks = np.array([r[0] for r in pairs])
            # Log bins to handle the long tail of ranks
            bins = np.logspace(np.log10(1), np.log10(max(ranks.max(), 10)), 40)

            ax.hist(ranks, bins=bins, color="skyblue", edgecolor="white", linewidth=0.4)
            ax.set_xscale('log')
            ax.axvline(np.median(ranks), color="crimson", linestyle="--",
                       linewidth=1.2, label=f"Med = {np.median(ranks):.0f}")
            ax.set_title(f"{model_label} — {cat}", fontsize=10)
            ax.set_xlabel("Rank (Log Scale)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "rank_distributions_log.png"), dpi=200)
    plt.close(fig2)

    print(f"\n[DONE] All results saved in: {args.out_dir}")