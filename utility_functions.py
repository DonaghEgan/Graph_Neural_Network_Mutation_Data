from torch_geometric.data import download_url, extract_zip, extract_gz
from typing import Optional, List

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

