import read_folders as rf
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

# downlaod msk
path, sources, urls = ds.download_study(name = 'msk_immuno_2019')

# give path to process data 
data_dict = rf.read_files(path[0])

# exctract relevant data
mut_pos = data_dict['mutations']['mut_pos']      # Shape (1661, 554, 12) -> Needs reshape
var_type = data_dict['mutations']['var_type']   # Shape (1661, 554, 5, 12)
aa_sub = data_dict['mutations']['aa_sub']      # Shape (1661, 554, 48, 12) 24 aa (ref and alt)
ns = data_dict['mutations']['ns']              # Shape (1661, 554, 12) -> Needs reshape
chrom = data_dict['sv']['chrom']              # Shape (1661, 554, 46, 12)  46 = 23 * 2 (ref and alt chromosome)
var_class = data_dict['sv']['var_class']       # Shape (1661, 554, 6, 12)
clinical_data = data_dict['patient_array']
gene_list = data_dict['gene_list']
osurv_data = data_dict['os_array']

# Convert Gene List names
gene_list = uf.convert_symbols(gene_list)

# Merge on last two dimensions
aa_sub_flat = uf.merge_last_two_dims(aa_sub)     # → (1661, 554, 48*12  = 576)
chrom_flat = uf.merge_last_two_dims(chrom)      # → (1661, 554, 46*12  = 552)
var_class_flat = uf.merge_last_two_dims(var_class)  # → (1661, 554,  6*12  =  72)
var_type_flat = uf.merge_last_two_dims(var_type)

# Create a list of arrays in the desired concatenation order
arrays_to_concat = [
    mut_pos,  # Size 1 along axis 2
    var_type_flat,    # Size 5 along axis 2
    aa_sub_flat,      # Size 48 along axis 2
    ns,       # Size 1 along axis 2
    chrom_flat,       # Size 46 along axis 2
    var_class_flat    # Size 6 along axis 2
]

# join omics layers
omics = np.concatenate(arrays_to_concat, axis=2)

# genes with atleast one feature
genes_to_keep_mask = np.any(omics != 0, axis=(0, 2))
print(len(genes_to_keep_mask))

# We will set a seed and split into training (80%), validation (10%) and testing (10%)
sample_list = data_dict['sample_list'] # get samples
random.seed(3)
sample_index = [i for i in range(len(sample_list))]
random.shuffle(sample_index)

# create training, val, and test sets
ntrain = int(0.8*len(sample_list))
nval = int(0.1*len(sample_list))

train_set = sample_index[0:ntrain]
val_set = sample_index[ntrain:(ntrain+nval)]
test_set = sample_index[(ntrain+nval):]

# omics -> split
omics_train = omics[train_set]
omics_test = omics[test_set]
omics_val = omics[val_set]

# clinical -> split
clin_train = clinical_data[train_set] 
clin_test = clinical_data[test_set]
clin_val = clinical_data[val_set]

# surv -> split
osurv_train = osurv_data[train_set]
osurv_test = osurv_data[test_set]
osurv_val = osurv_data[val_set]

# Get the graph with the data. 
adj_matrix = rs.read_reactome_new(tokens = gene_list)
row_sums = adj_matrix.sum(axis=1).mean()
print(f"Avg Number of neighbors per gene: {row_sums}")

if adj_matrix is None:
    raise ValueError("Adjacency matrix was not returned correctly.")

tokens = torch.tensor(np.arange(0,len(gene_list)), dtype = torch.long)

# Load model and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = m.Net_omics(features_omics=omics_train.shape[2], features_clin=clin_train.shape[-1], dim=50, max_tokens=len(gene_list), output=2).to(device)
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
        pred = model(batch.omics, adj_matrix, batch.clin)
        
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
            pred = model(batch.omics, adj_matrix, batch.clin)
            loss = cl.cox_loss_effron(batch.osurv, pred)
            loss_all += loss.item()
            c_index += cl.concordance_index(batch.osurv, pred)

    return loss_all / num_batches, c_index / num_batches

# get train and validation 
train_data = m.CoxBatchDataset(osurv_train, clin_train, omics_train, batch_size=10, shuffle=True)
print("train batch complete")
val_data = m.CoxBatchDataset(osurv_val, clin_val, omics_val, batch_size=10, shuffle=True)

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
uf.plot_training_metrics(loss_train, loss_val, ci_train, ci_val, epochs)

