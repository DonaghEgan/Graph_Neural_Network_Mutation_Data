import torch

def cox_loss_effron(t_d, pred, eps=1e-7, alpha=0.1):
    """
    Enhanced Cox loss with Efron's approximation including numerical stability improvements.
    
    Args:
        t_d: Survival data [time, event]
        pred: Model predictions (log hazard ratios)
        eps: Small constant for numerical stability
        alpha: Regularization strength for extreme predictions
    """
    device = t_d.device
    t = t_d[:, 0]
    d = t_d[:, 1]
    
    # Check for valid input
    if torch.any(torch.isnan(t)) or torch.any(torch.isnan(d)) or torch.any(torch.isnan(pred)):
        print("⚠️ NaN in input data!")
        return torch.tensor(float('inf'), device=device)
    
    # Clip extreme predictions to prevent overflow
    pred_clipped = torch.clamp(pred, min=-10, max=10)
    
    # We will flatten the inputs.
    t_flat = t.flatten()
    d_flat = d.flatten()
    pred_flat = pred_clipped.flatten()
    
    # Check if we have any events
    if torch.sum(d_flat) == 0:
        print("⚠️ No events in batch!")
        return torch.tensor(0.0, device=device)
    
    # sort input based on times.
    sort_t, idx_t = t_flat.sort(descending=True)
    pred_flat = pred_flat[idx_t]
    d_flat = d_flat[idx_t]
    t_flat = t_flat[idx_t]
    
    # create a variable where we indicate the non-censored events.
    d_ind = d_flat.nonzero().flatten()
    
    if len(d_ind) == 0:
        return torch.tensor(0.0, device=device)
    
    hazard_sum_zeros = torch.zeros((d_ind.shape), dtype=pred_flat.dtype, device=device)
    
    # precompute the numerator with numerical stability
    log_num = pred_flat[d_ind].mean()
    
    # Add small regularization term to prevent extreme predictions
    reg_term = alpha * torch.mean(pred_flat ** 2)
    
    # calculate the hazard for patients whose t is greater than tu_i
    hazard_gt_mi = torch.logcumsumexp(pred_flat, dim=0)[d_ind]
    
    # number of events for each unique risk set
    tu, m_inverse, m = torch.unique_consecutive(t_flat[d_ind], return_counts=True, return_inverse=True)
    
    # position of last event (lowest duration) of each unique risk set
    m_indx = m.cumsum(axis=0) - 1
    
    # sample the cumulative sum of hazards we need.
    hazard_gt_mi = hazard_gt_mi[m_indx][m_inverse]
    
    # logsumexp of ties, duplicated within tie set
    hazard_max = pred_flat[d_ind].scatter_reduce(0, m_indx[m_inverse], pred_flat[d_ind], reduce="amax")[m_indx][m_inverse]
    recentered_scores = (pred_flat[d_ind] - hazard_max).exp()
    hazard_sum = hazard_sum_zeros.scatter_reduce(0, m_indx[m_inverse], recentered_scores, reduce="sum")[m_indx][m_inverse]
    
    # Add epsilon for numerical stability
    hazard_sum = hazard_max + (hazard_sum + eps).log()
    
    # next we need add l delete m in log space
    aux = torch.ones_like(m_inverse)
    aux[m_indx[:-1] + 1] -= m[:-1]
    event_id_in_tie = torch.cumsum(aux, dim=0) - 1
    
    # Add epsilon to prevent log(0)
    hazard_sum += torch.log(event_id_in_tie + eps) - torch.log(m[m_inverse] + eps)
    
    # we can combine it in the log_denominator term.
    log_denom = log_substract(hazard_gt_mi, hazard_sum).mean()
    
    # Final numerical stability check
    if torch.isnan(log_denom) or torch.isnan(log_num):
        print("⚠️ NaN detected!")
        print("log_denom:", log_denom)
        print("log_num:", log_num)
        print("pred range:", pred_flat.min().item(), pred_flat.max().item())
        # Return a safe fallback loss
        return torch.tensor(1.0, device=device)
    
    # Add regularization to the final loss
    loss = log_denom - log_num + reg_term
    
    return loss

def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())

def concordance_index(t_d, h, eps=1e-8):
    """
    Enhanced Harrell's concordance index with improved numerical stability and efficiency.
    
    Parameters:
    t_d (torch.Tensor): Tensor containing [time, event] 
             where event is 1 for death, 0 for censored
    h (torch.Tensor): Tensor containing hazard scores for each sample
             Higher hazard score should mean higher risk (shorter survival)
    eps (float): Small constant for numerical stability
    
    Returns:
    float: Concordance index (c-index)
    """
    device = t_d.device
    
    # Extract time and event status
    t = t_d[:, 0]  # time
    d = t_d[:, 1]  # event indicator
    h_flat = h.flatten()
    
    # Input validation
    if torch.any(torch.isnan(t)) or torch.any(torch.isnan(d)) or torch.any(torch.isnan(h_flat)):
        print("⚠️ NaN in concordance index input!")
        return torch.tensor(0.5, device=device)
    
    n = len(t)
    if n < 2:
        return torch.tensor(0.5, device=device)
    
    # Vectorized approach for better efficiency
    # Create pairwise comparison matrices
    t_i = t.unsqueeze(1)  # [n, 1]
    t_j = t.unsqueeze(0)  # [1, n]
    h_i = h_flat.unsqueeze(1)  # [n, 1]
    h_j = h_flat.unsqueeze(0)  # [1, n]
    d_i = d.unsqueeze(1)  # [n, 1]
    
    # Valid pairs: i has event and j has longer survival time
    valid_pairs = (d_i == 1) & (t_j > t_i)
    
    if not torch.any(valid_pairs):
        return torch.tensor(0.5, device=device)
    
    # Concordant pairs: higher hazard for shorter survival
    concordant = valid_pairs & (h_i > h_j)
    
    # Handle ties with small epsilon
    ties = valid_pairs & (torch.abs(h_i - h_j) < eps)
    
    # Count concordant pairs and ties (give ties 0.5 weight)
    num_concordant = torch.sum(concordant.float()) + 0.5 * torch.sum(ties.float())
    total_pairs = torch.sum(valid_pairs.float())
    
    if total_pairs == 0:
        return torch.tensor(0.5, device=device)
    
    c_index = num_concordant / total_pairs
    
    # Ensure c-index is in valid range [0, 1]
    c_index = torch.clamp(c_index, 0.0, 1.0)
    
    return c_index


def weighted_cox_loss(t_d, pred, sample_weights=None, eps=1e-7):
    """
    Weighted Cox loss for handling class imbalance in survival data.
    
    Args:
        t_d: Survival data [time, event]
        pred: Model predictions
        sample_weights: Optional weights for each sample
        eps: Numerical stability constant
    """
    device = t_d.device
    
    if sample_weights is None:
        # Automatically compute weights based on event frequency
        events = t_d[:, 1]
        n_events = torch.sum(events)
        n_censored = len(events) - n_events
        
        if n_events > 0 and n_censored > 0:
            # Weight events more if they're rare
            event_weight = len(events) / (2.0 * n_events)
            censored_weight = len(events) / (2.0 * n_censored)
            sample_weights = torch.where(events == 1, event_weight, censored_weight)
        else:
            sample_weights = torch.ones_like(events, dtype=torch.float, device=device)
    
    # Apply weights to predictions
    weighted_pred = pred * sample_weights.unsqueeze(-1)
    
    return cox_loss_effron(t_d, weighted_pred, eps)


def ranking_loss(t_d, pred, margin=1.0):
    """
    Ranking-based loss that directly optimizes concordance.
    
    Args:
        t_d: Survival data [time, event]
        pred: Model predictions
        margin: Margin for ranking loss
    """
    device = t_d.device
    t = t_d[:, 0]
    d = t_d[:, 1]
    pred_flat = pred.flatten()
    
    n = len(t)
    if n < 2:
        return torch.tensor(0.0, device=device)
    
    # Create pairwise comparisons
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for i in range(n):
        if d[i] == 1:  # Patient i had an event
            for j in range(n):
                if t[j] > t[i]:  # Patient j survived longer
                    # pred[i] should be higher than pred[j] (higher risk for shorter survival)
                    loss += torch.clamp(margin - (pred_flat[i] - pred_flat[j]), min=0)
                    count += 1
    
    return loss / max(count, 1)


def combined_loss(t_d, pred, cox_weight=0.7, ranking_weight=0.3, **kwargs):
    """
    Combined loss function using both Cox loss and ranking loss.
    
    Args:
        t_d: Survival data
        pred: Model predictions
        cox_weight: Weight for Cox loss component
        ranking_weight: Weight for ranking loss component
    """
    cox_loss = cox_loss_effron(t_d, pred, **kwargs)
    rank_loss = ranking_loss(t_d, pred)
    
    return cox_weight * cox_loss + ranking_weight * rank_loss 