import torch
import torch.nn.functional as F

def nt_xent_loss(target, inputs, temperature=0.6):
    embeddings = F.normalize(inputs, p=2, dim=1)
    
    cosine_sim = torch.matmul(embeddings, embeddings.t())
    cosine_sim /= temperature
    
    mask = torch.eye(cosine_sim.size(0), dtype=torch.bool, device=cosine_sim.device)
    cosine_sim = cosine_sim.masked_fill(mask, -1e9)
    
    loss = F.cross_entropy(cosine_sim, target)
    return loss.mean()
