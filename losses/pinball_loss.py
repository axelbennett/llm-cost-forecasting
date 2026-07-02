import torch

def pinball_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """
    pred:       [batch, horizon, n_quantiles]
    target:     [batch, horizon]
    quantiles:  list of quantile levels e.g. [0.1, 0.5, 0.9]
    """
    quantiles = torch.tensor(quantiles, dtype=torch.float32)
    
    # expand target to match pred shape [batch, horizon, n_quantiles]
    target = target.unsqueeze(-1).expand_as(pred)
    
    errors = target - pred
    
    # pinball: max(tau * e, (tau-1) * e)
    loss = torch.max(
        quantiles * errors,
        (quantiles - 1) * errors
    )
    
    return loss.mean()
