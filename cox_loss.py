import torch

def cox_loss_effron(t_d, pred):
    """
    Taken from: https://github.com/lasso-net/lassonet
    Reference
    Yang, Xuelin, Louis Abraham, Sejin Kim, Petr Smirnov, Feng Ruan, Benjamin Haibe-Kains, and Robert Tibshirani. "FastCPH: Efficient Survival Analysis for Neural Networks." arXiv preprint arXiv:2208.09793 (2022)
    
    Take a dataset with log risks: pred = {pred1, ..., predn}, observed time points and events: t_d = {t_d1, ..., t_dn} where each t_di = [ti, di] where di is 1 if the event of interest occurred for patient i, and 0 otherwise. 
    Take tu = {tu1, ..., tup} as the set of p, unique uncensored values for t (i.e. whose associated d is 1), with each point in p associated to a set of data points m = {m1 , ..., mp}, m1:t==tu1
    and a risk set R = {R1, ..., Rp}, R1:t>=u1
    The general formula of Efron's approximation to Cox log-likelihood is:
     L = sum([L1:Lp])
     
     Li = sum(pred[mi]) - log( sum(l = 0:(len(mi) - 1) (exp(pred[Ri] - l/len(mi)exp(pred[mi]))) )
     # let's rename this for simplicity.
     Li = log_hazard_mi - log ( sum (l = 0:(len(mi) - 1) (hazard_gt_mi)  -  l/len(mi) hazard_mi)
     # Where hazard_gt_mi means hazard for times greater than mi. 
     # The first part of Li can be calculated at the start.
     L = sum(log_hazard_m) - sum(i in m)( log ( sum (l = 0:(len(mi) - 1) (hazard_gt_mi) -  l/len(mi) hazard_mi) ) 
    """
    device = t_d.device
    t = t_d[:, 0]
    d = t_d[:, 1]
    # We will flatten the inputs.
    t_flat = t.flatten()
    d_flat = d.flatten()
    pred_flat = pred.flatten()
    # sort input based on times.
    sort_t, idx_t = t_flat.sort(descending = True)
    pred_flat = pred_flat[idx_t]
    d_flat = d_flat[idx_t]
    t_flat = t_flat[idx_t]
    # create a variable where we indicate the non-censored events.
    d_ind = d_flat.nonzero().flatten()
    hazard_sum_zeros =  torch.zeros((d_ind.shape), dtype = pred_flat.dtype,device = device)
    # precompute the numerator
    log_num = pred_flat[d_ind].mean()
    # calculate the hazard for patients whose t is greater than tu_i
    hazard_gt_mi = torch.logcumsumexp(pred_flat, dim=0)[d_ind] # this is the cummulative sum of hazards
    # number of events for each unique risk set
    tu, m_inverse, m = torch.unique_consecutive(t_flat[d_ind], return_counts=True, return_inverse=True)
    # position of last event (lowest duration) of each unique risk set
    m_indx = m.cumsum(axis=0) - 1
    # Now we need to get the hazard sums for each m.
    # if we just use the original data we will add up extra terms we don't need.
    hazard_gt_mi = hazard_gt_mi[m_indx][m_inverse]  # sample the cummulate sum of hazards we need.
    # logsumexp of ties, duplicated within tie set
    hazard_max = pred_flat[d_ind].scatter_reduce(0, m_indx[m_inverse], pred_flat[d_ind], reduce="amax")[m_indx][m_inverse]
    recentered_scores = (pred_flat[d_ind] - hazard_max).exp()
    hazard_sum = hazard_sum_zeros.scatter_reduce(0, m_indx[m_inverse], recentered_scores, reduce="sum")[m_indx][m_inverse]
    hazard_sum = hazard_max + hazard_sum.log()
    # next we need add l delete m in log space (multiplication of l/m)
    aux = torch.ones_like(m_inverse)
    aux[m_indx[:-1] + 1] -= m[:-1]
    event_id_in_tie = torch.cumsum(aux, dim=0) - 1
    hazard_sum += torch.log(event_id_in_tie) - torch.log(m[m_inverse])
    # we can combine it in the log_denominator term. 
    log_denom = log_substract(hazard_gt_mi,hazard_sum).mean()
    if torch.isnan(log_denom) or torch.isnan(log_num):
        print("⚠️ NaN detected!")
        print("log_denom:", log_denom)
        print("log_num:", log_num)
        print(t_d, pred)
        raise ValueError("NaN encountered in Cox loss calculation.")
    # loss is negative log likelihood
    return log_denom - log_num

def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())

def concordance_index(t_d,h):

    """
    Calculate Harrell's concordance index for survival data.
    
    Parameters:
    t_d (torch.Tensor): Tensor containing [sample, os_status, os_months]
             where os_status is 1 for event (death), 0 for censored
             and os_months is the survival time
    h (torch.Tensor): Tensor containing hazard scores for each sample
             Higher hazard score should mean higher risk (shorter survival)
    
    Returns:
    float: Concordance index (c-index) or NaN if no comparable pairs exist
    """

    device = t_d.device
    
    # Extract time and event status
    t = t_d[:, 0]  # os_months is in the first column (index 0)
    d = t_d[:, 1]  # os_status is in the second column (index 1)
    h_flat = h.flatten()

    num = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    
    valid_comparison_found = False
    # For all patients with events (d=1)
    for i in range(len(t)):
        if d[i] == 1:  # Only consider pairs where first patient had an event
            # Find all patients with longer survival times
            comparable = t > t[i]
            # Check if there are any comparable patients
            if not torch.any(comparable):
                continue
            
            # Higher h should mean shorter survival time
            num += torch.sum(comparable & (h_flat[i] > h_flat))
            denom += torch.sum(comparable)
            valid_comparison_found = True
    
    # Only return the ratio if we found valid comparisons
    if valid_comparison_found and denom > 0:
        return num / denom
    else:
        # Handle the case where no valid comparisons were found
        # Return a default value that makes sense for your application
        return torch.tensor(0.5, device=device)  # or another sensible default 