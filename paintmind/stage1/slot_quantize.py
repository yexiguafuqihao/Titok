import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

class SlotVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, n_codebook, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.n_codebook = 2
        assert self.e_dim % self.n_codebook == 0
        self.embedding = nn.Embedding(self.n_codebook * self.n_e, self.e_dim // self.n_codebook)
        self.embedding.weight.data.normal_()

    def forward(self, z):

        bs = z.size(0)
        z = rearrange(z, 'b n (f c) -> b f n c', f = self.n_codebook)
        prototypes = rearrange(self.embedding.weight, '(b f) ... -> b f ...', b=self.n_codebook, f = self.n_e)
        prototypes = repeat(prototypes, 'f ... -> b f ...', b=bs)
        z, embedd_norm = l2norm(z), l2norm(prototypes)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = 2 - 2 * torch.einsum('bfnc, bfmc -> bfnm', z, embedd_norm)

        encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(encoding_indices)
        z_q = l2norm(z_q)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q-z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b f n c -> b n (f c)')
        return z_q, loss, encoding_indices

    def decode_from_indice(self, indices):
        z_q = self.embedding(indices)
        z_q = l2norm(z_q)
        
        return z_q