import torch
import torch.nn as nn
from torch.nn import Linear
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    
    def __init__(self, feats_in, feats_out, max_tokens):
        """
        Initialize a GIN layer for message passing gene features with a graph structure.

        Args:
        feats_in (int): Number of input features per gene.
        eats_out (int): Number of output features per gene after processing.
        """
 
        super(GINLayer, self).__init__()

        self.max_tokens = max_tokens

        self.mlp = nn.Sequential(
            nn.Linear(feats_in, feats_out),
            nn.BatchNorm1d(feats_out),
            nn.ReLU(),
            nn.Linear(feats_out, feats_out)
        )
 
        self.eps = nn.Parameter(torch.zeros(self.max_tokens)) # explore gene-specifc eps?

    def forward(self, x, adj):
        """
        Process gene features through the GIN layer using an adjacency matrix.

        Args:
            x (torch.Tensor): Node features, shape [B, G, F] (B: batch size, G: genes, F: features).
            adj (torch.Tensor): Adjacency matrix, shape [G, G] (binary, unweighted).

        Returns:
            torch.Tensor: Processed features, shape [B, G, feats_out].
        """
        B, G, F_in = x.shape
        
        # Expand adjacency matrix to match batch dimension: [G, G] -> [B, G, G]
        adj_expand = adj.expand(B, -1, -1)

        # Aggregate neighbor features via matrix multiplication: [B, G, G] @ [B, G, F] -> [B, G, F]
        agg = adj_expand @ x

        if G != self.max_tokens:
            raise ValueError(f"Input number of genes ({G}) does not match "
                             f"layer's num_genes ({self.max_tokens})")
        
        one_plus_eps_reshaped = (1 + self.eps).view(1, self.max_tokens, 1)
        
        # Combine central node features with neighbors, scaled by (1 + eps)
        out = one_plus_eps_reshaped * x + agg
        
         # Check for numerical instability
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Warning: NaN or inf detected in out")
            print(f"x max: {x.max()}, agg max: {agg.max()}, eps: {self.eps}")

        # Apply MLP to transform combined features
        return self.mlp(out.view(-1, out.shape[-1])).view(x.shape[0], x.shape[1], -1)

class Net_omics(torch.nn.Module):
    def __init__(self, features_omics, features_clin, dim, embedding_dim_string, max_tokens, output = 2):
        """
        Initialize the GNN model for integrating omics and clinical data.
        Args:
           features_omics (int): Number of omics features per gene.
           features_clin (int): Number of clinical features per sample.
           dim (int): Hidden dimension for GIN layers.
           max_tokens (int): Number of genes (nodes) in the graph.
           embedding_dim_string: the dimension of each samples clinical information (tumor purity, cancer type etc)
           output (int, optional): Output features from omics branch. Defaults to 2.
        """

        super(Net_omics, self).__init__()
        # previosuly defined GIN layer
        self.gin1 = GINLayer(features_omics, dim, max_tokens)
        self.gin2 = GINLayer(dim, dim, max_tokens)
        self.gin3 = GINLayer(dim, 1, max_tokens)
        self.linout = Linear(max_tokens, output, bias = False)
        self.max_tokens = max_tokens
        self.features_omics = features_omics
        self.tokens = torch.tensor(np.arange(0, max_tokens))
        self.linclin = Linear(features_clin + embedding_dim_string, 1)
        self.lin3 = Linear(output, 1)
       
    def forward(self, omics, adj, clin, sample_meta):
        # get the weights for the connections through kd.
        x = self.gin1(omics, adj)
        x = self.gin2(x, adj)
        x = self.gin3(x, adj)
        x = torch.flatten(x, 1)
        x = self.linout(x)
        # Clinical + categorical branch
        combined_clin  = torch.cat([clin, sample_meta], dim = -1)
        x2 = self.linclin(combined_clin)
        x1 = self.lin3(x) + x2
        return x1

class CoxBatchDataset(IterableDataset):
    """
    Initialize a dataset for batching survival data with balanced censored/uncensored samples.

    Args:
    osurv (np.ndarray or torch.Tensor): Survival data, shape [N, 2] (time, event indicator).
    clin (np.ndarray or torch.Tensor): Clinical features, shape [N, C].
    omics (np.ndarray or torch.Tensor): Omics features, shape [N, G, F].
    batch_size (int, optional): Samples per batch. Defaults to 10.
    shuffle (bool, optional): Whether to shuffle samples. Defaults to True.
    """
    def __init__(self, osurv, clin, omics, sample_meta, batch_size=10, indices = None, shuffle=True):
        # Convert to tensors if needed
        self.osurv = osurv if isinstance(osurv, torch.Tensor) else torch.tensor(osurv, dtype=torch.float)
        self.clin = clin if isinstance(clin, torch.Tensor) else torch.tensor(clin, dtype=torch.float)
        self.omics = omics if isinstance(omics, torch.Tensor) else torch.tensor(omics, dtype=torch.float)
        self.sample_meta = sample_meta if isinstance(sample_meta, torch.Tensor) else torch.tensor(sample_meta, dtype=torch.float)
        
        # Use all indices if none provided, otherwise use the specified subset
        if indices is None:
            self.indices = list(range(self.osurv.shape[0]))
            print('Indices not provided, using all samples')
        else:
            self.indices = list(indices)

        if self.osurv.shape[0] < batch_size:
            raise ValueError("Dataset size smaller than batch size")

        # Identify dead (event=1) and censored (event=0) samples within the provided indices
        self.deads = [idx for idx in self.indices if self.osurv[idx, 1] == 1]
        self.censored = [idx for idx in self.indices if self.osurv[idx, 1] == 0]       

        if not self.deads:
            raise ValueError("No uncensored events found")
        
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # Initialize the list of all indices
        indices = self.deads + self.censored
        if self.shuffle:
            random.shuffle(self.deads)
            random.shuffle(self.censored)
            indices = self.deads + self.censored
            random.shuffle(indices)
        
        # Work with a copy of indices that we can modify
        remaining_indices = indices.copy()
        
        while remaining_indices:
            # Identify remaining uncensored and censored samples
            remaining_deads = [i for i in remaining_indices if i in self.deads]
             # Break if no uncensored samples remain
            if not remaining_deads: 
                break
            remaining_censored = [i for i in remaining_indices if i in self.censored]
    
            # Determine the current batch size
            current_batch_size = min(self.batch_size, len(remaining_indices))
            # Stop when number of samples is less than 5
            if current_batch_size < 5:
                break       
            # Decide how many uncensored samples to include
            ratio = len(remaining_deads) / len(remaining_indices)
            j_d = np.random.binomial(current_batch_size, ratio)
            j_d = max(1, min(j_d, len(remaining_deads)))  # At least 1, but not more than availabl
        
            # Select samples for the batch
            batch_indices = []
            if j_d > 0:
                selected_deads = random.sample(remaining_deads, j_d)
                batch_indices.extend(selected_deads)
                for idx in selected_deads:
                    remaining_indices.remove(idx)
            
            censored_to_select = current_batch_size - j_d
            if censored_to_select > 0 and remaining_censored:
                selected_censored = random.sample(remaining_censored, min(censored_to_select, len(remaining_censored)))
                batch_indices.extend(selected_censored)
                for idx in selected_censored:
                    remaining_indices.remove(idx)
            
            assert all(0 <= idx < self.osurv.shape[0] for idx in batch_indices), f"Indices out of bounds: {batch_indices}"
            
            # Create batch data
            batch_data = [
                Data(
                    osurv=self.osurv[i].unsqueeze(0),
                    clin=self.clin[i].unsqueeze(0),
                    omics=self.omics[i].unsqueeze(0),
                    sample_meta=self.sample_meta[i].unsqueeze(0)
                ) for i in batch_indices
            ]

            # Yield the batch if it contains samples
            if batch_data:
                yield Batch.from_data_list(batch_data)
