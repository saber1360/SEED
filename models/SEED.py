import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.seed_layers_v6 import SEED_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    
    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x
    
class channel_id_aware(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.r = 4
        self.cid = nn.Parameter(
            0.002*torch.randn(self.n_vars, self.dim, self.r)
        )
        self.linear = nn.Linear(self.r, self.dim)
    
    def forward(self, x):
        # x: [B, N, T]
        B, L, D = x.shape
        embed = F.gelu(self.linear(self.cid)) # N, D, D
        # print(embed.shape, x.shape)
        x = torch.einsum('bnpt,ntd->bnpd', x.reshape(B, self.n_vars, -1, D), embed).reshape(B, L, D)
        return x

def compute_dominant_freq_ratio(x, eps=1e-12):
    """
    x: Tensor of shape [B, N, T]
    Returns: DFR of shape [B, N]
    """
    fft = torch.fft.fft(x, dim=-1)  # [B, N, T], complex
    power = fft.real**2 + fft.imag**2  # power spectrum: [B, N, T]
    
    power = power[..., :x.shape[-1] // 2]  # Keep only positive frequencies
    total_power = power.sum(dim=-1, keepdim=False) + eps
    max_power = power.max(dim=-1).values
    
    dfr = max_power / total_power  # [B, N]
    return dfr


def compute_spectral_entropy(x, eps=1e-12, normalized=True):
    """
    x: Tensor of shape [B, N, T]
    Returns: Spectral Entropy of shape [B, N]
    """
    fft = torch.fft.fft(x, dim=-1)
    power = fft.real**2 + fft.imag**2  # [B, N, T]
    
    power = power[..., :x.shape[-1] // 2]  # Keep only positive frequencies
    psd = power / (power.sum(dim=-1, keepdim=True) + eps)  # normalize to probability distribution
    
    spectral_entropy = -torch.sum(psd * torch.log2(psd + eps), dim=-1)  # [B, N]

    if normalized:
        max_entropy = torch.log2(torch.tensor(psd.shape[-1], dtype=psd.dtype, device=psd.device))
        spectral_entropy = spectral_entropy / max_entropy

    return spectral_entropy

class MySpectralEntropy(nn.Module):
    def __init__(self, T, normalized=True):
        super().__init__()
        self.normalized = normalized
        self.scale = 0.02
        self.w = nn.Parameter(self.scale * torch.randn(1, T))  # 可学习频率通道
        
        # self.hann = torch.hann_window(T).view(1, 1, -1)  # 可选窗函数
    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        return y
    
    def forward(self, x, filter=True):
        B, N, T = x.shape
        x = x - x.mean(dim=-1, keepdim=True)  # 去均值
        # x = x * self.hann.to(x.device)  # 加窗
        if filter:
            fft = self.circular_convolution(x, self.w.to(x.device))
        else:
            fft = torch.fft.rfft(x, dim=2, norm='ortho')
        power = fft.real**2 + fft.imag**2 # torch.abs(fft) # 
        # power = power[..., :T // 2]

        # assert torch.isnan(power).sum() == 0, print(power)
        # 可学习频域调制
        weighted_power = power
        psd = weighted_power / (weighted_power.sum(dim=-1, keepdim=True) + 1e-6)
        # assert torch.isnan(psd).sum() == 0, print(psd)
        entropy = -torch.sum(psd * torch.log2(psd + 1e-6), dim=-1)
        # assert torch.isnan(entropy).sum() == 0, print(entropy)

        if self.normalized:
            max_entropy = torch.log2(torch.tensor(psd.shape[-1], dtype=psd.dtype, device=psd.device))
            entropy = entropy / max_entropy

        return entropy  # shape [B, N]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1) # L
        self.enable_env = configs.enable_env
        
        configs.n_vars = self.n_vars
        configs.dim = configs.d_model
        configs.num_patches = self.num_patches
        configs.n_blocks = configs.e_layers
        # Filter
        configs.alpha = 0.1 if configs.alpha is None else configs.alpha
        configs.top_p = 0.5 if configs.top_p is None else configs.top_p
        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        self.block = configs.e_layers
        self.backbone = SEED_Backbone(configs)
        
        
        # head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        
        # self.mappings = nn.ModuleList(
        #     [nn.Linear(configs.seq_len // self.seg_size, configs.pred_len // self.seg_size) for _ in range(self.num_map + 1)]
        # )
        
        # if configs.use_pi:
        #     # period injection
        #     period = configs.period
        #     stride = period // configs.seg_size
        #     new_weights = torch.zeros(configs.pred_len // self.seg_size, configs.seq_len // self.seg_size)
            
        #     for i in range(0, configs.pred_len // self.seg_size):
        #         for j in range(configs.seq_len // self.seg_size - stride, 0, -stride):
        #             if j + i < configs.seq_len // self.seg_size:
        #                 new_weights[i, j+i] = period / configs.seq_len
            
        #     self.mappings[0].weight.data = new_weights
        #     self.mappings[0].bias.data = torch.zeros(configs.pred_len // self.seg_size)
        # self.head2 = nn.Linear(self.pred_len, self.pred_len)
        
        
        # self.aware = channel_id_aware(configs)
        self.SE = MySpectralEntropy(self.seq_len, normalized=True)
        # Without RevIN
        self.use_RevIN = configs.use_revin 
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)
        self.use_drop = configs.use_dropout if configs.use_dropout else 0.0
        self.dropout = nn.Dropout(self.use_drop)
    
    def forward(self, x, x_masks, is_training=False, target=None):
        # x: [B, T, C]
        B, T, C = x.shape
        # Normalization
        x = self.norm(x, 'norm')
        # x: [B, C, T]
        x = x.permute(0, 2, 1)
        # se = compute_spectral_entropy(x) # [B, C]
        # ss = self.SE(x, False)
        # print('==========================\n',ss[:,:10])
        se = self.SE(x)
        # print(se)
        x = x.reshape(-1, C*T) # [B, C*T]
        
        x = self.patch_embed(x) # [B, N, D]  N = [C*T / P]

        
        # x = self.aware(x) + x
        # x, moe_loss = self.backbone(x, masks, self.alpha, is_training)
        x, moe_loss = self.backbone(x, se)
        moe_loss = 0
        # print(moe_loss)
        # [B, C, T/P, D]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)) # [B, C, T]
        # x = self.head2(x)
        # if self.use_drop:
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        
        # De-Normalization
        x = self.norm(x, 'denorm')

        return x, moe_loss
        