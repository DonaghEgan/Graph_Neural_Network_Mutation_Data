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
    
    def __init__(self, feats_in, feats_out, max_tokens, dropout=0.2):
        """
        Initialize a GIN layer for message passing gene features with a graph structure.

        Args:
        feats_in (int): Number of input features per gene.
        feats_out (int): Number of output features per gene after processing.
        max_tokens (int): Maximum number of genes/nodes.
        dropout (float): Dropout probability for regularization.
        """
 
        super(GINLayer, self).__init__()

        self.max_tokens = max_tokens

        self.mlp = nn.Sequential(
            nn.Linear(feats_in, feats_out),
            nn.LayerNorm(feats_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feats_out, feats_out),
            nn.LayerNorm(feats_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
 
        self.eps = nn.Parameter(torch.tensor(0.0))  # scalar eps per layer for stability

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
        
        # Normalize adjacency (row-normalized A -> D^{-1}A) for stable aggregation
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        adj_norm = adj / deg
        
        # Expand normalized adjacency matrix to match batch dimension: [G, G] -> [B, G, G]
        adj_expand = adj_norm.expand(B, -1, -1)

        # Aggregate neighbor features via matrix multiplication: [B, G, G] @ [B, G, F] -> [B, G, F]
        agg = adj_expand @ x

        if G != self.max_tokens:
            raise ValueError(f"Input number of genes ({G}) does not match "
                             f"layer's num_genes ({self.max_tokens})")
        
        # Combine central node features with neighbors, scaled by (1 + eps)
        out = (1 + self.eps) * x + agg
        
         # Check for numerical instability
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Warning: NaN or inf detected in out")
            print(f"x max: {x.max()}, agg max: {agg.max()}, eps: {self.eps}")

        # Apply MLP to transform combined features
        return self.mlp(out.view(-1, out.shape[-1])).view(x.shape[0], x.shape[1], -1)

class Net_omics(torch.nn.Module):
    def __init__(self, features_omics, features_clin, dim, embedding_dim_string, max_tokens, output=2, dropout=0.3):
        """
        Initialize the GNN model for integrating omics and clinical data.
        Args:
           features_omics (int): Number of omics features per gene.
           features_clin (int): Number of clinical features per sample.
           dim (int): Hidden dimension for GIN layers.
           max_tokens (int): Number of genes (nodes) in the graph.
           embedding_dim_string: the dimension of each samples clinical information (tumor purity, cancer type etc)
           output (int, optional): Output features from omics branch. Defaults to 2.
           dropout (float): Dropout probability for regularization.
        """

        super(Net_omics, self).__init__()
        
        # Input projection for omics data
        self.input_proj = nn.Linear(features_omics, dim)
        
        # Enhanced GIN layers with dropout
        self.gin1 = GINLayer(dim, dim, max_tokens, dropout)
        self.gin2 = GINLayer(dim, dim, max_tokens, dropout)
        self.gin3 = GINLayer(dim, dim, max_tokens, dropout)
        
        # Global pooling with attention
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        
        # Omics branch with residual connection
        self.omics_proj = nn.Sequential(
            nn.Linear(dim, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced clinical branch
        self.clin_proj = nn.Sequential(
            nn.Linear(features_clin + embedding_dim_string, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, output)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output * 2, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output, 1)
        )
        
        self.max_tokens = max_tokens
        self.features_omics = features_omics
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, omics, adj, clin, sample_meta):
        # Input projection
        x = self.input_proj(omics)
        
        # Store initial features for residual connection
        x_residual = x
        
        # Graph convolutions with residual connections
        x = self.gin1(x, adj)
        x = x + x_residual  # Residual connection
        
        x_residual = x
        x = self.gin2(x, adj)
        x = x + x_residual  # Residual connection
        
        x = self.gin3(x, adj)
        
        # Attention-based global pooling
        # Compute attention weights for each gene
        attention_weights = self.attention(x)  # [B, G, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights and sum
        x_pooled = (x * attention_weights).sum(dim=1)  # [B, dim]
        
        # Omics branch
        omics_out = self.omics_proj(x_pooled)
        
        # Clinical branch
        combined_clin = torch.cat([clin, sample_meta], dim=-1)
        clin_out = self.clin_proj(combined_clin)
        
        # Fusion of both branches
        fused = torch.cat([omics_out, clin_out], dim=-1)
        output = self.fusion(fused)
        
        return output

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