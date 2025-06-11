import process_data as prc
import numpy as np
import utility_functions as uf
import random
import read_specific as rs
import torch
import cox_loss as cl
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import torch
from torch_geometric.nn import GraphConv, global_add_pool
from torch.nn import Linear
from torch import optim
from cox_loss import cox_loss_effron
import os
import re
import pandas as pd
import download_study as ds
from torch_geometric.data import Data, Batch
import sys
import model as m
from tqdm import tqdm  # optional, for a nice progress bar
import gc

# downlaod msk
# msk_immuno_2019
# msk_pan_2017
path, sources, urls = ds.download_study(name = 'msk_pan_2017')

# give path to process data 
data_dict = prc.read_files(path[0])

# exctract relevant data -> mutations
protein_pos = data_dict['mutation']['protein_pos']      
var_type = data_dict['mutation']['variant_type_np']   
aa_sub = data_dict['mutation']['amino_acid']      
chrom_mut = data_dict['mutation']['chromosome_np']              
var_class_mut = data_dict['mutation']['var_class_np']
fs_mut = data_dict['mutation']['frameshift']

# Verify shapes
print(f"shape mut:{protein_pos.shape}")
print(f"shape var_class_mut:{var_class_mut.shape}")
print(f"shape chrom_mut: {chrom_mut.shape}")
print(f"shape var_type: {var_type.shape}")
print(f"shape aa {aa_sub.shape}")
print(f"shape fs:{fs_mut.shape}")

#extract relevant information -> SV
chrom_sv = data_dict['sv']['chromosome']             
var_class_sv = data_dict['sv']['var_class']       
region_sites = data_dict['sv']['region_sites']
connection_type = data_dict['sv']['connection_type']
sv_length = data_dict['sv']['sv_length']

print(f"chrom sv:{chrom_sv.shape}")
print(f"shape var_sv:{var_class_sv.shape}")
print(f"regions: {region_sites.shape}")
print(f"connection: {connection_type.shape}")
print(f"sv_length: {sv_length.shape}")


# extract relevant data -> CNA
cna = data_dict['cna']['cna']
cna = np.expand_dims(cna, axis=-1)  # Resulting shape: (7702, 1181, 1)
print(f"shape cna:{cna.shape}")

# patient clincal data
osurv_data = data_dict['os_array']
print(np.isnan(osurv_data).sum())
clinical_data = data_dict['patient']
print(f"clin shape:{clinical_data.shape}")

# sample_data
sample_meta = data_dict['sample_meta']['metadata']
sample_embeddings = data_dict['sample_meta']['embeddings']

print(f"sample_meta: {len(sample_meta)}")
print(f"sample embeddings: {sample_embeddings.shape}")

# Merge on last two dimensions
var_class_mut_flat = uf.merge_last_two_dims(var_class_mut)
chrom_mut_flat = uf.merge_last_two_dims(chrom_mut)
aa_sub_flat = uf.merge_last_two_dims(aa_sub)
var_type_flat = uf.merge_last_two_dims(var_type)

# Apply merge_last_two_dims to SV arrays
chrom_sv_flat = uf.merge_last_two_dims(chrom_sv)
var_class_sv_flat = uf.merge_last_two_dims(var_class_sv)
region_sites_flat = uf.merge_last_two_dims(region_sites)

# Create a list of arrays to concatenate in the specified order
arrays_to_concat = [
    protein_pos,
    fs_mut,      
    var_class_mut_flat,  
    chrom_mut_flat,    
    var_type_flat,      
    chrom_sv_flat,
    aa_sub_flat, 
    var_class_sv_flat,
    region_sites_flat,
    sv_length,
    connection_type,
    cna      
]

# join omics layers
uf.log_memory('Before Conact')
omics = np.concatenate(arrays_to_concat, axis=2)
uf.log_memory('After Conact')

# Convert to tenors
omics_tensor = torch.tensor(omics, dtype=torch.float32)
clin_tensor = torch.tensor(clinical_data, dtype=torch.float32)
osurv_tensor = torch.tensor(osurv_data, dtype=torch.float32)
sample_embeddings_tensor = torch.tensor(sample_embeddings, dtype=torch.float32)

# Free memory by deleting individual arrays
del arrays_to_concat, protein_pos, var_type_flat, aa_sub_flat, chrom_sv_flat, 
var_class_mut_flat, omics, clinical_data, osurv_data, chrom_mut_flat, fs_mut, sample_embeddings, cna
gc.collect()

# genes with atleast one feature
genes_to_keep_mask = (omics_tensor != 0).any(dim=(0, 2))
print(len(genes_to_keep_mask))

# We will set a seed and split into training (80%), validation (10%) and testing (10%)
sample_index = data_dict['sample_index'] # get samples
gene_index = data_dict['gene_index']

# Free memory by deleting data_dict
del data_dict
gc.collect()

# Create train, validation, and test indices
uf.log_memory('Before train-test split')
random.seed(3)
sample_idx = list(sample_index.values())
random.shuffle(sample_idx)

ntrain = int(0.8 * len(sample_index))
nval = int(0.1 * len(sample_index))
train_idx = sample_idx[:ntrain]

val_idx = sample_idx[ntrain:ntrain + nval]
test_idx = sample_idx[ntrain + nval:]
uf.log_memory('After train-test split')

# Get the graph with the data.
uf.log_memory('Before Create Adj Matrix') 
adj_matrix = rs.read_reactome_new(gene_index = gene_index)
row_sums = adj_matrix.sum(axis=1).mean()
print(f"Avg Number of neighbors per gene: {row_sums}")
uf.log_memory('After Create Adj Matrix')

if adj_matrix is None:
    raise ValueError("Adjacency matrix was not returned correctly.")

print(sample_embeddings_tensor.shape[0])
# Load model and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = m.Net_omics(features_omics=omics_tensor.shape[2], features_clin=clin_tensor.shape[-1], dim=50, embedding_dim_string = sample_embeddings.shape[1], max_tokens=len(gene_index), output=2).to(device)

# Send model and adj matrix to device
model = model.to(device)
adj_matrix = adj_matrix.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Now we can make a training function!
def train_block(model, data):
    model.train()
    loss_all = 0
    c_index = 0
    j = 0
    for batch in data:
	
        batch = uf.move_batch_to_device(batch=batch, device=device) # wrapper for moving batch to gpu
        j += 1

        # set the gradients in the optimizer to zero.
        optimizer.zero_grad()
        
        # run the model
        pred = model(batch.omics, adj_matrix, batch.clin, batch.sample_meta)
        
        # calculate Cox' partial likelihood and get the autograd output.
        loss = cl.cox_loss_effron(batch.osurv, pred)
        loss.backward()
        
        # update the loss_all object
        loss_all += loss.item()
        
        # update parameters in model.
        optimizer.step()
        
        # calculate concordance index
        c_index += cl.concordance_index(batch.osurv, pred)

    return loss_all/j, c_index/j

# We will make a testing function too. 
def evaluate_model(model, data):
    """
    Evaluate the model on a validation or test set.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for validation or test data.

    Returns:
        Tuple (avg_loss, avg_c_index): Mean Cox loss and concordance index over batches.
    """
    model.eval()
    loss_all = 0
    c_index = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data:
            batch = uf.move_batch_to_device(batch, device=device)
            num_batches += 1
            pred = model(batch.omics, adj_matrix, batch.clin, batch.sample_meta)
            loss = cl.cox_loss_effron(batch.osurv, pred)
            loss_all += loss.item()
            c_index += cl.concordance_index(batch.osurv, pred)

    return loss_all / num_batches, c_index / num_batches

# get train and validation
uf.log_memory('Create batch data') 
train_data = m.CoxBatchDataset(osurv_tensor, clin_tensor, omics_tensor, sample_embeddings_tensor, batch_size=32, indices = train_idx, shuffle=True)
print("train batch complete")
val_data = m.CoxBatchDataset(osurv_tensor, clin_tensor, omics_tensor, sample_embeddings_tensor, batch_size=32, indices = val_idx, shuffle=True)

######################
# Training Process
######################

# Training parameters
epochs = 500
print(f"Number of epochs:{epochs}")

# Metrics tracking
ci_val = []
ci_train = []
loss_val = []
loss_train = []

# Initial evaluation before training
vloss, vci = evaluate_model(model, val_data)
tloss, tci = evaluate_model(model, train_data)

ci_val.append(float(vci))
ci_train.append(float(tci))
loss_val.append(float(vloss))
loss_train.append(float(tloss))

print(f"[Init] Train CI: {tci:.4f}, Loss: {tloss:.4f} | Val CI: {vci:.4f}, Loss: {vloss:.4f}")

# Training loop
for epoch in tqdm(range(1, epochs + 1), desc="Training"):
    # Training step
    tloss, tci = train_block(model, train_data)
    ci_train.append(tci)
    loss_train.append(tloss)

    # Validation step
    vloss, vci = evaluate_model(model, val_data)
    ci_val.append(vci)
    loss_val.append(vloss)

    # Print progress
    print(f"Epoch {epoch:03d} | Train CI: {tci:.4f}, Loss: {tloss:.4f} | Val CI: {vci:.4f}, Loss: {vloss:.4f}")

    # Optional: save best model
    # if vci == max(ci_val):
    #     torch.save(model.state_dict(), "best_model.pt")

# Call the plotting function after the training loop
def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

ci_train = to_numpy(ci_train)
ci_val = to_numpy(ci_val)
loss_train = to_numpy(loss_train)
loss_val = to_numpy(loss_val)

uf.plot_training_metrics(loss_train, loss_val, ci_train, ci_val, epochs)

