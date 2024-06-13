import torch, pdb
import torch.nn as nn
import os.path as osp
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from .layers import Encoder, Decoder
from .quantize import VectorQuantizer
from .slot_quantize import SlotVectorQuantizer
from ..engine.util import instantiate_from_config
from ..modules.ocl.perceptual_grouping import build_mlp

class VQModel(nn.Module):
    def __init__(self, n_embed, embed_dim, beta, encoder_config, decoder_config, 
                 n_slots, slot_dim, ckpt_path=None):
        super().__init__()

        self.num_slots = n_slots
        self.slots = nn.Embedding(n_slots, slot_dim)
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.query_quantisation = VectorQuantizer(2 * n_embed, embed_dim, beta)

        self.prev_quant = nn.Linear(encoder_config.params.dim, embed_dim)
        self.post_quant = nn.Linear(embed_dim, decoder_config.params.dim)  
  
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        
        bs = x.size(0)
        slots = repeat(self.slots.weight, 'f c -> b f c', b=bs)
        x = self.encoder(x, slots)
        
        slots = x[:, -self.num_slots:]
        
        slots = self.prev_quant(slots)
        queries, q_loss, q_indices = self.query_quantisation(slots)

        return queries, q_loss, q_indices
    
    def decode(self, slots):

        x = self.decoder(slots)
        return x.clamp(-1.0, 1.0)
    
    def forward(self, img, targets):
        
        queries, q_loss, q_indices = self.encode(img)

        slots = self.post_quant(queries)

        rec = self.decode(slots)

        results = {}
        if self.training:
            results.update({'q_loss': q_loss})
        return rec, results
    
    def decode_from_indice(self, indice):

        z_q = self.quantize.decode_from_indice(indice)
        img = self.decode(z_q)
        return img
    
    def from_pretrained(self, path):

        return self.load_state_dict(torch.load(path, map_location='cpu'))
    


        
