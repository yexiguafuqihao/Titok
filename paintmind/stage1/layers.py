from torch import nn
import torch, math, pdb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ..modules.mlp import SwiGLUFFNFused
from ..engine.util import instantiate_from_config
from ..modules.attention import CrossAttention, MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE


def pair(t):

    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(mlp_dim, dim)
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
    def forward(self, x, slots=None):

        x = self.attn1(self.norm1(x)) + x
        x = self.ffnet(self.norm2(x)) + x

        return x

class DecoderLayer(Layer):

    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0):
        
        super().__init__(dim, dim_head, mlp_dim, num_head, dropout)
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn2 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, slots):

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), slots) + x
        x = self.ffnet(self.norm3(x)) + x

        return x

class Transformer(nn.Module):
    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        assert layer_type in ['normal', 'dec_layer']
        layers = {'normal': Layer, 'dec_layer': DecoderLayer}
        self.layers = nn.ModuleList([layers[layer_type](dim, dim_head, mlp_dim, num_head, dropout) for i in range(depth)])
    
    def forward(self, x, slots = None):
        
        for i, layer in enumerate(self.layers):
            x = layer(x, slots)
        return x


class Encoder(nn.Module):

    def __init__(self, image_size, layer_type, num_slots,
                 patch_size, dim, depth, num_head, mlp_dim,
                 in_channels=3, out_channels=3, dim_head=64, 
                 visual_encoder_config = None, dropout=0.):

        super().__init__()
        self.backbone = None
        if visual_encoder_config is not None:
            self.backbone = instantiate_from_config(visual_encoder_config)

        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * scale)
        self.norm_pre = nn.LayerNorm(dim)
        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout)
        
        self.initialize_weights()

    def initialize_weights(self):

        if self.backbone:
            assert self.backbone.is_trainable is False
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.norm_pre.apply(self._init_weights)
        for m in self.transformer.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, slots):

        if self.backbone is not None:
            x = self.backbone(x)
        x = self.to_patch_embedding(x)
        x = torch.cat((x +  self.position_embedding, slots), dim=1)
        x = self.norm_pre(x)
        x = self.transformer(x, None)
        
        return x
       
class Decoder(nn.Module):
    def __init__(self, layer_type, image_size, patch_size, in_channels, dim, num_slots,
                 depth, num_head, mlp_dim,  out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.num_slots = num_slots
        self.num_patches = num_patches
        self.dim = dim

        self.mask_token = nn.Embedding(1, dim)
        self.position_embedding = nn.Embedding(num_patches, dim)

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_channels * patch_size * patch_size, bias=True)
        
        self.initialize_weights()

    @property
    def pos_embedding(self):
        
        return self.position_embedding

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, slots):
        
        bs = slots.size(0)
        position_embedding = repeat(self.position_embedding.weight, 'f c -> b f c', b=bs)

        mask_tokens = position_embedding + self.mask_token.weight.unsqueeze(0)

        z = torch.cat((mask_tokens, slots), dim=1)
        z = self.transformer(z)
        z = self.norm(z)

        x = z[:, :self.num_patches]
        x = self.proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_size//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        
        return x

