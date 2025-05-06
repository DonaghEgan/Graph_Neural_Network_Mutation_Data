import torch.nn as nn
from torch.nn import Linear
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
     """
     Initialize a GIN layer for message passing gene features with a graph structure.

     Args:
     feats_in (int): Number of input features per gene.
     eats_out (int): Number of output features per gene after processing.
     """

    def __init__(self, feats_in, feats_out):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feats_in, feats_out),
            nn.ReLU(),
            nn.Linear(feats_out, feats_out)
        )
 
        self.eps = nn.Parameter(torch.Tensor([0.0])) # explore gene-specifc eps?
    
    def forward(self, x, adj):
        """
        x: Node features, shape [B, G, F]
        adj: Adjacency matrix, shape [G, G] (binary, unweighted)
        """
        # Aggregate neighbors (simple sum)
        adj_expand = adj.expand(x.shape[0],-1,-1) # adj = [B, G, G]
        agg = adj_expand @ x  # [B, G, F]
        
        # Combine with central node features
        out = (1 + self.eps) * x + agg # x varies by input 
        
        # Apply MLP
        return self.mlp(out)

class Net_omics(torch.nn.Module):
     """
     Initialize the GNN model for integrating omics and clinical data.

     Args:
     features_omics (int): Number of omics features per gene.
     features_clin (int): Number of clinical features per sample.
     dim (int): Hidden dimension for GIN layers.
     max_tokens (int): Number of genes (nodes) in the graph.
     output (int, optional): Output features from omics branch. Defaults to 2.
    """
    def __init__(self, features_omics, features_clin, dim, max_tokens, output = 2):
        super(Net_omics, self).__init__()
        self.gin1 = GINLayer(features_omics, dim)
        self.gin2 = GINLayer(dim, dim)
        self.gin3 = GINLayer(dim, 1)
        self.linout = Linear(max_tokens, output, bias = False)
        self.max_tokens = max_tokens
        self.features_omics = features_omics
        self.tokens = torch.tensor(np.arange(0,max_tokens))
        self.linclin = Linear(features_clin, 1)
        self.lin3 = Linear(output, 5)
        
    def forward(self, omics, adj, clin):
        # get the weights for the connections through kd.
        x = self.gin1(omics, adj)
        x = self.gin2(x, adj)
        x = self.gin3(x, adj)
        x = torch.flatten(x, 1)
        x = self.linout(x)
        x2 = self.linclin(clin)
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
    def __init__(self, osurv, clin, omics, batch_size=10, shuffle=True):
        # Convert to tensors if needed
        self.osurv = osurv if isinstance(osurv, torch.Tensor) else torch.tensor(osurv, dtype=torch.float)
        self.clin = clin if isinstance(clin, torch.Tensor) else torch.tensor(clin, dtype=torch.float)
        self.omics = omics if isinstance(omics, torch.Tensor) else torch.tensor(omics, dtype=torch.float)
        
        if self.osurv.shape[0] < batch_size:
            raise ValueError("Dataset size smaller than batch size")
        
        self.deads = torch.where(self.osurv[:, 1] == 1)[0].tolist()
        self.censored = torch.where(self.osurv[:, 1] == 0)[0].tolist()
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
            remaining_censored = [i for i in remaining_indices if i in self.censored]
            
            # Determine the current batch size
            current_batch_size = min(self.batch_size, len(remaining_indices))
            # stop when number of samples is less than 5
            if current_batch_size < 5:
                break #      
            # Decide how many uncensored samples to include
            if remaining_deads:
                ratio = len(remaining_deads) / len(remaining_indices)
                j_d = np.random.binomial(current_batch_size, ratio)
                j_d = max(1, min(j_d, len(remaining_deads)))  # At least 1, but not more than available
            else:
                j_d = 0  # No uncensored samples left, use only censored
            
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
            
            # Create batch data
            batch_data = [
                Data(
                    osurv=self.osurv[i].unsqueeze(0),
                    clin=self.clin[i].unsqueeze(0),
                    omics=self.omics[i].unsqueeze(0)
                ) for i in batch_indices
            ]
            
            # Yield the batch if it contains samples
            if batch_data:
                yield Batch.from_data_list(batch_data)
