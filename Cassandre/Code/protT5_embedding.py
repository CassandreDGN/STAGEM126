import re
import torch
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
import time
from tqdm import tqdm
import h5py
import sys



# Load model directly
#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein,
                   max_residues=4000, max_seq_len=1000, max_batch=100, use_idctq=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (identifier, seq) in tqdm(enumerate(seq_dict,1)):

        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((identifier,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            identifiers, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
        
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(identifier, seq_len))
                continue
            for batch_idx, identifier in enumerate(identifiers): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    if use_idctq:
                        protein_emb = idct_quant_prost(emb.detach().cpu().numpy().squeeze())
                        results["protein_embs"][identifier]  = protein_emb
                    else:
                        protein_emb = emb.mean(dim=0)
                        results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')

    return results

def extract_seq_FASTA(file):
    seq_dic = {}
    for record in SeqIO.parse(file, "fasta"):
        seq_dic[record.id] = str(record.seq.replace('U','X').replace('Z','X').replace('O','X'))
    return seq_dic

#@title Write embeddings to disk. { display-mode: "form" }
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None



if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Error: You must provide input and output paths.")
        print("Usage: python3 script.py <input_fasta> <output_h5>")
        sys.exit(1)        

    fasta_input = sys.argv[1]
    h5_output = sys.argv[2] 

    model, tokenizer = get_T5_model()
    PROTEOME = sys.argv[1]
    # Added the path and the H5 extension
    SAVE_F = sys.argv[2]


    proteins  = extract_seq_FASTA(PROTEOME)
    res = get_embeddings(model, tokenizer, proteins , False, True,
                   max_residues=4000, max_seq_len=1000, max_batch=100 )
    save_embeddings(res['protein_embs'], SAVE_F)

