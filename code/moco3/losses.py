import torch
import torch.nn.functional as F

def ctr(q, k, temperature=0.9):
    logits = torch.mm(q, k.t()) # [N, N] pairs
    N = q.size(0)
    labels = torch.arange(N, dtype=torch.long, device=q.device) # positives are in diagonal
    loss = F.cross_entropy(logits / temperature, labels)
    
    return 2 * temperature * loss