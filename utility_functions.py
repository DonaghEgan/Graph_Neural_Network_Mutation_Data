from torch_geometric.data import download_url, extract_zip, extract_gz
from typing import Optional, List
import matplotlib.pyplot as plt
import psutil
import os
import torch

def explore_structure(d, indent=0):
    spacing = '  ' * indent
    if isinstance(d, dict):
        for key, value in d.items():
            print(f"{spacing}{key}: {type(value).__name__}")
            explore_structure(value, indent + 1)
    elif isinstance(d, list):
        print(f"{spacing}List of {len(d)} items")
        if d:  # Only recurse if list is non-empty
            explore_structure(d[0], indent + 1)

def move_batch_to_device(batch, device):
    batch.omics = batch.omics.to(device)
    batch.clin = batch.clin.to(device)
    batch.osurv = batch.osurv.to(device)
    batch.sample_meta = batch.sample_meta.to(device)
    return batch

def merge_last_two_dims(x):
    # x.shape == (1661, 554, D, 12)
    n0, n1, D, C = x.shape
    return x.reshape(n0, n1, D*C)

def convert_symbols(gene_list: List[str], database: str = 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt', folder: str = 'temp/'):
 
    """
    Converts old gene symbols to new symbols based on input data.
    
    Args:
        input_data (str): Input string with tab-separated fields containing old and new symbols.
        
    Returns:
        dict: Mapping of old symbols to new symbol.
    """
    path = download_url(database, folder)    
    
    # Split the input line into columns
    gene_index = {}
    gene_idx = 0
    prev_gene_idx = 0
    with open(path, 'r') as fo:
        for idx, line in enumerate(fo):
            line = line.strip('\n').split('\t')
            if idx == 0: # Header
                gene_idx = line.index('symbol')
                prev_gene_idx = line.index('prev_symbol')
                continue
            prev_symbol = line[prev_gene_idx].strip('"')
            current_symbol = line[gene_idx].strip('"')
            
            # skip empty strings
            if not prev_symbol: 
                continue

            prev_symbol_split = prev_symbol.split('|')
            
            for sub_prev_symbol in prev_symbol_split:
                gene_index[sub_prev_symbol] = current_symbol
    
    converted = []
    for gene in gene_list:
        if gene in gene_index:
            converted.append(gene_index[gene])
        else:
            converted.append(gene)
  
    return converted

# After the training loop, create summary plots
def plot_training_metrics(loss_train, loss_val, ci_train, ci_val, epochs):
    """
    Generate and save plots for training and validation metrics.

    Args:
        loss_train (list): Training Cox loss per epoch.
        loss_val (list): Validation Cox loss per epoch.
        ci_train (list): Training concordance index per epoch.
        ci_val (list): Validation concordance index per epoch.
        epochs (int): Number of training epochs.
    """
    # Create a range of epochs, including initial evaluation (0)
    epoch_range = range(0, epochs + 1)

    # Plot 1: Cox Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, loss_train, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epoch_range, loss_val, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cox Loss')
    plt.title('Training and Validation Cox Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/cox_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Concordance Index
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, ci_train, label='Training CI', color='blue', linewidth=2)
    plt.plot(epoch_range, ci_val, label='Validation CI', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Concordance Index')
    plt.title('Training and Validation Concordance Index over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/concordance_index_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def log_memory(step):
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / (1024 ** 2)
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_info = f", GPU Memory: {gpu_mem_mb:.2f} MB"
    print(f"{step}: CPU Memory: {cpu_mem_mb:.2f} MB{gpu_info}")


