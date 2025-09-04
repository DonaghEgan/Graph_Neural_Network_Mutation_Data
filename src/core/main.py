import process_data as prc
import numpy as np
import utility_functions as uf
import random
import read_specific as rs
import cox_loss as cl
import torch
from cox_loss import cox_loss_effron
import os
import pandas as pd
import download_study as ds
from torch_geometric.data import Data, Batch
import sys
import model as m
from tqdm import tqdm  # optional, for a nice progress bar
import gc
sys.path.insert(1, '/home/degan/Graph_Neural_Network_Mutation_Data/scripts/')

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
if torch.cuda.is_available():
    # Check GPU memory usage and select the best GPU
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    # Check memory usage for each GPU and select the one with most free memory
    best_gpu = 0
    max_free_memory = 0
    
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        print(f"GPU {i} ({torch.cuda.get_device_name(i)}): Free memory = {free_memory / 1024**3:.1f} GB")
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    
    device = torch.device(f'cuda:{best_gpu}')
    print(f"Selected GPU {best_gpu} with {max_free_memory / 1024**3:.1f} GB free memory")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

model = m.Net_omics(
    features_omics=omics_tensor.shape[2], 
    features_clin=clin_tensor.shape[-1], 
    dim=48,  # Changed from 50 to 48 (48 % 4 = 0)
    embedding_dim_string=sample_embeddings_tensor.shape[1], 
    max_tokens=len(gene_index), 
    output=64,  # Changed from 2 to 64 for better performance
    dropout=0.3
).to(device)
print(f"Using device: {device}")

# Send model and adj matrix to device
model = model.to(device)
adj_matrix = adj_matrix.to(device)

# Improved optimizer: AdamW with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))

# Learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Now we can make a training function with improved loss!
def train_block(model, data, use_combined_loss=True, loss_schedule_epoch=0):
    model.train()
    loss_all = 0
    c_index = 0
    j = 0
    
    for batch in data:
        batch = uf.move_batch_to_device(batch=batch, device=device)
        j += 1

        optimizer.zero_grad()
        pred = model(batch.omics, adj_matrix, batch.clin, batch.sample_meta)
        
        if use_combined_loss and loss_schedule_epoch > 10:
            cox_loss = cl.combined_loss(batch.osurv, pred, cox_weight=0.8, ranking_weight=0.2)
        elif loss_schedule_epoch > 5:
            cox_loss = cl.weighted_cox_loss(batch.osurv, pred)
        else:
            cox_loss = cl.cox_loss_effron(batch.osurv, pred)
        
        l2_strength = max(0.005, 0.02 * (1 - loss_schedule_epoch / 100))
        l2_reg = sum(torch.norm(p, p=2) for p in model.parameters() if p.requires_grad)
        total_loss = cox_loss + l2_strength * l2_reg
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Invalid loss at batch {j}, skipping...")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Convert to Python float immediately
        loss_all += float(cox_loss.item())
        optimizer.step()
        
        # Ensure c_index calculation returns a float
        ci_batch = cl.concordance_index(batch.osurv, pred)
        c_index += float(ci_batch) if isinstance(ci_batch, torch.Tensor) else float(ci_batch)

    return float(loss_all/j), float(c_index/j)

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
            
            # Convert to Python float immediately
            loss_all += float(loss.item())
            ci_batch = cl.concordance_index(batch.osurv, pred)
            c_index += float(ci_batch) if isinstance(ci_batch, torch.Tensor) else float(ci_batch)

    return float(loss_all / num_batches), float(c_index / num_batches)

# get train and validation
uf.log_memory('Create batch data') 
train_data = m.CoxBatchDataset(osurv_tensor, clin_tensor, omics_tensor, sample_embeddings_tensor, batch_size=32, indices = train_idx, shuffle=True)
print("train batch complete")
val_data = m.CoxBatchDataset(osurv_tensor, clin_tensor, omics_tensor, sample_embeddings_tensor, batch_size=32, indices = val_idx, shuffle=True)

######################
# Training Process
######################

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Initialize early stopping
early_stopping = EarlyStopping(patience=25, verbose=True)

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

# Training loop with improved loss scheduling
for epoch in tqdm(range(1, epochs + 1), desc="Training"):
    # Training step with loss scheduling
    tloss, tci = train_block(model, train_data, use_combined_loss=True, loss_schedule_epoch=epoch)
    ci_train.append(tci)
    loss_train.append(tloss)

    # Validation step
    vloss, vci = evaluate_model(model, val_data)
    ci_val.append(vci)
    loss_val.append(vloss)

    # Learning rate scheduler step
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # Print progress
    print(f"Epoch {epoch:03d} | Train CI: {tci:.4f}, Loss: {tloss:.4f} | Val CI: {vci:.4f}, Loss: {vloss:.4f} | LR: {current_lr:.6f}")

    # Early stopping check
    early_stopping(vloss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

    # Optional: save best model based on C-index instead
    if vci == max(ci_val):
        torch.save(model.state_dict(), "best_cindex_model.pt")
        print(f"New best C-index: {vci:.4f}")

print("Training completed!")
print(f"Best validation C-index: {max(ci_val):.4f}")
print(f"Final validation loss: {min(loss_val):.4f}")

# Evaluate on test set with best model
print("\n" + "="*50)
print("FINAL TEST SET EVALUATION")
print("="*50)

# Load best model for test evaluation
model.load_state_dict(torch.load("best_model.pt"))
test_data = m.CoxBatchDataset(osurv_tensor, clin_tensor, omics_tensor, sample_embeddings_tensor, batch_size=32, indices=test_idx, shuffle=False)

test_loss, test_ci = evaluate_model(model, test_data)
print(f"Test C-index: {test_ci:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Also evaluate with best C-index model
try:
    model.load_state_dict(torch.load("best_cindex_model.pt"))
    test_loss_ci, test_ci_best = evaluate_model(model, test_data)
    print(f"Test C-index (best CI model): {test_ci_best:.4f}")
    print(f"Test Loss (best CI model): {test_loss_ci:.4f}")
except:
    print("Best C-index model not found, using early stopping model")

# Save training results to CSV for later plotting
print("\n" + "="*50)
print("SAVING TRAINING RESULTS")
print("="*50)

# Create a DataFrame with training metrics
import pandas as pd

# Ensure all lists have the same length (in case of early stopping)
actual_epochs = len(loss_train)
epoch_numbers = list(range(actual_epochs))

# Convert tensor values to Python floats/ints to ensure proper CSV format
results_df = pd.DataFrame({
    'epoch': [int(x) for x in epoch_numbers],
    'train_loss': [float(x) if isinstance(x, torch.Tensor) else float(x) for x in loss_train],
    'val_loss': [float(x) if isinstance(x, torch.Tensor) else float(x) for x in loss_val],
    'train_ci': [float(x) if isinstance(x, torch.Tensor) else float(x) for x in ci_train],
    'val_ci': [float(x) if isinstance(x, torch.Tensor) else float(x) for x in ci_val]
})

# Verify data types
print("Data types in results DataFrame:")
print(results_df.dtypes)
print("\nFirst few rows:")
print(results_df.head())

# Save to CSV with timestamp into results/training_outputs
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out_dir = os.path.join(project_root, "results", "training_outputs")
os.makedirs(out_dir, exist_ok=True)

csv_filename = os.path.join(out_dir, f"training_results_{timestamp}.csv")
results_df.to_csv(csv_filename, index=False)

print(f"Training results saved to: {csv_filename}")
print(f"Columns: {list(results_df.columns)}")
print(f"Total epochs trained: {actual_epochs}")

# Also save a summary file with key metrics
summary_data = {
    'metric': ['best_val_ci', 'best_val_loss', 'final_test_ci', 'final_test_loss', 'total_epochs'],
    'value': [
        float(max(ci_val)),
        float(min(loss_val)), 
        float(test_ci),
        float(test_loss),
        int(actual_epochs)
    ]
}

if 'test_ci_best' in locals():
    summary_data['metric'].extend(['test_ci_best_model', 'test_loss_best_model'])
    summary_data['value'].extend([float(test_ci_best), float(test_loss_ci)])

summary_df = pd.DataFrame(summary_data)
summary_filename = os.path.join(out_dir, f"training_summary_{timestamp}.csv")
summary_df.to_csv(summary_filename, index=False)

print(f"Training summary saved to: {summary_filename}")

