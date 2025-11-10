import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from .vis import visualize_features_tsne, visualize_cross_similarity,visualize_weight_distribution, visualize_adj
# from layers.Transformer_EncDec import Encoder, EncoderLayer

def mask_topk(x, alpha=0.5, largest=False):
    # B, L = x.shape[0], x.shape[-1]
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]

def mask_abs_topk(x, alpha=0.5, largest=False):
    # B, L = x.shape[0], x.shape[-1]
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]

class GCN(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj, x):
        # adj [B, H, L, L]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        x = x.view(B, L, -1)
        return x


class time_GCN(nn.Module):
    def __init__(self, dim, n_heads, n_vars): #BLD->
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.n_vars = n_vars

    def forward(self, adj, x):
        # adj [B, H, N, P, P]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        B, L, H, D_ = x.shape
        x = x.reshape(B, self.n_vars, -1, self.n_heads, D_) # BNPHD
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhixy,bixhd->biyhd", adj, x).contiguous()  # [B, H, H, D_]
        x = x.reshape(B, L, -1)
        return x
###############################
# Ablation
###############################

# class ConvAttention(nn.Module):
#     def __init__(self, in_channels, heads=4, kernel_size=3):
#         super().__init__()
#         self.heads = heads
#         self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=heads)
#         self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=heads)
#         self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=heads)
#         self.scale = (in_channels // heads) ** -0.5

#     def forward(self, x):  # x: [B, C, H, W]
#         B, C, H, W = x.shape
#         Q = self.q_conv(x).view(B, self.heads, C // self.heads, H * W)  # [B, h, d, N]
#         K = self.k_conv(x).view(B, self.heads, C // self.heads, H * W)
#         V = self.v_conv(x).view(B, self.heads, C // self.heads, H * W)

#         attn = torch.softmax(torch.matmul(Q.transpose(-2, -1), K) * self.scale, dim=-1)  # [B, h, N, N]
#         out = torch.matmul(attn, V.transpose(-2, -1)).transpose(-2, -1)  # [B, h, d, N] → [B, C, H, W]
#         out = out.reshape(B, C, H, W)
#         return out


class ConvQKAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super().__init__()
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.scale = out_channels ** -0.5

    def forward(self, x):  # x: [B, H, N, D, D]
        B, H, N, D1, D2 = x.shape
        x_reshaped = x.view(B * H * N, 1, D1, D2)  # → [BHN, 1, D, D]

        Q = self.q_conv(x_reshaped)  # → [BHN, C, D, D]
        K = self.k_conv(x_reshaped)

        # flatten spatial dims
        Q = Q.flatten(2).squeeze(-2).reshape(B*H, N, -1)  # [BH, N, D*D]
        K = K.flatten(2).squeeze(-2).reshape(B*H, N, -1).transpose(1, 2)  # [BH, D*D, N]

        attn_logits = torch.bmm(Q, K) * self.scale  # [BH, N, N]
        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn.view(B, H, N, N)
        # out = torch.einsum('bhij,bhixy->bhjxy', attn, x)

        return attn, #out



class LearnablePatchSelector(nn.Module):
    def __init__(self, D, hidden_dim=64):
        super().__init__()
        # self.use_gumbel = use_gumbel
        self.score_net = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个打分
        )
    
    def forward(self, x, use_gumbel=False):
        """
        输入 x: shape [B, H, N, P, D]
        输出: 选择后的 patch: shape [B, H, N, D]
        """
        B, H, N, P, D = x.shape
        x_flat = x.reshape(-1, P, D)  # [B*H*N, P, D]
        
        scores = self.score_net(x_flat)  # [B*H*N, P, 1]
        scores = scores.squeeze(-1)  # [B*H*N, P]
        
        if use_gumbel:
            probs = F.gumbel_softmax(scores, tau=1.0, hard=True)  # [B*H*N, P]
        else:
            probs = F.softmax(scores, dim=-1)  # [B*H*N, P]

        # 加权和表示选择
        selected_patch = torch.einsum('bp, bpd -> bd', probs, x_flat) # - x_flat[:, 0, :].squeeze(1) #*0.05  # [B*H*N, D]
        selected_patch = selected_patch.view(B, H, N, 1, D)  # [B, H, N, D]
        
        return selected_patch, probs.view(B, H, N, P)



# def fold_windows(x_win, win, stride):
#     """
#     x_win: Tensor of shape (B, H, N, win_num, win, D)
#     win: window size
#     stride: stride of the sliding window

#     Returns:
#         Tensor of shape (B, H, N, L, D), L = win_num * stride + win - stride
#     """
#     B, H, N, win_num, win, D = x_win.shape
#     # 1. 先把win维度和D维度合并，方便fold操作
#     x = x_win.permute(0,1,2,5,3,4)  # B,H,N,D,win_num,win
#     x = x.reshape(B*H*N, D, win_num, win)  # 把B,H,N合并为batch，win_num视作宽度，win视作高度

#     # 2. fold的kernel_size 和 stride，对应win的空间是height=win，宽度=1吗？
#     # 这里是时间维度，我们假设时间维度在宽度方向处理，因此做个转置:
#     x = x.permute(0,1,3,2)  # B*H*N, D, win (height), win_num (width)

#     fold = nn.Fold(output_size=(1, (win_num -1)*stride + win), kernel_size=(win,1), stride=(stride,1))
#     x_folded = fold(x)

#     # 3. 构造归一化的除数
#     ones = torch.ones_like(x)
#     divisor = fold(ones)

#     x_folded /= divisor
#     # x_folded形状是 (B*H*N, D, 1, L) -> squeeze height维度
#     x_folded = x_folded.squeeze(2)  # (B*H*N, D, L)

#     # 4. reshape回 (B,H,N,L,D)
#     x_folded = x_folded.permute(0,2,1).reshape(B, H, N, -1, D)

#     return x_folded

class Context_Win_Graph(nn.Module):
    def __init__(self, dim, n_vars, dropout=0.1, n_heads=4, patch_num=2):
        super().__init__()
        self.dim = dim
        # self.proj_1 = nn.Linear(dim, dim)
        # self.proj_2 = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.proj_v = nn.Linear(dim*self.n_heads, dim*self.n_heads)
        self.n_vars = n_vars
        self.Q = nn.Parameter(
            0.05*torch.randn(dim, dim)
        )
        # 根据From Similarity to Superiority: Channel Clustering for Time Series Forecasting
        # self.scale = 0.05
        # self.n_cluster = 4
        # self.cluster_aware_feed_forward = nn.Parameter(
        #     self.scale * torch.randn(self.n_cluster, self.dim, self.dim)
        # )
        # self.fuse = nn.Parameter(self.scale * torch.randn(self.n_vars, self.n_cluster)) # 变量聚类
        # self.proj_3 = nn.Linear(dim, dim)
        # self.proj_4 = nn.Linear(dim, dim)
        
        self.win = 2 # win
        self.stride = 1
        
        # self.win_num = patch_num
        # self.win_linear = nn.Linear(dim, dim)
        # self.patch_selector = LearnablePatchSelector(dim)
        self.dropout = nn.Dropout(dropout)
        
        # self.golbal_proj_1 = nn.Linear(dim, dim)
        # self.golbal_proj_2 = nn.Linear(dim, dim)
        # self.golbal_proj_3 = nn.Linear(dim, dim)
        # self.fuse = nn.Linear(dim+32, dim)
        # self.convQK = ConvQKAttention()
        # 状态分布聚类 DUET 或者使用MoE
        
        # self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, alpha):
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        x = x.reshape(B, H, self.n_vars, -1, D)
        B, H, N, P, D = x.shape # P为patch数
        
        if self.win <= P:
            win_x = x.unfold(dimension=-2, size=self.win, step=self.stride) # B,H,N, win_num, win, D
        else:
            win_x = x.unfold(dimension=-2, size=P, step=self.stride) # B,H,N, 1, 1, D P==1
            self.win = 1
        
        win_num = win_x.shape[-3]
        win_x = win_x.permute(0, 1, 2, 4, 3, 5).reshape(B, H, -1, win_num, D) # B,H,N*win, win_num, D
        
        # win_adj = torch.einsum('bhiwd,bhjwd->bhwij', self.proj_1(win_x), self.proj_2(win_x)) # ij为窗口内patch序号，w为窗口号win_num
        XQ = torch.einsum('bhiwd,dq->bhiwq', win_x, self.Q)
        win_adj = torch.einsum('bhiwq,bhjwq->bhwij', XQ, win_x)
        # visualize_adj(win_adj, 'src')
        win_adj = win_adj * mask_topk(torch.abs(win_adj), alpha)
        # win_adj = F.softplus(win_adj)
        win_adj = torch.tanh(win_adj)
        # torch.sign(win_adj) * torch.softmax(torch.abs(win_adj), dim=-1)
        # 
        win_adj = self.dropout(win_adj)
        adj = F.normalize(win_adj, p=1, dim=-1) #adj = F.normalize(adj, p=1, dim=-1)
        # visualize_adj(adj,'tanh')
        # import time
        # print('finished')
        # time.sleep(10)
        
        # adj = self.dropout(adj)
        
        # GCN
        # B,H,N*win, win_num, D
        win_x = self.proj_v(win_x.permute(0, 2, 3, 1, 4).reshape(B, N*self.win, win_num , -1)).reshape(B, N * self.win, win_num, self.n_heads, -1)
        # print(adj.shape,win_x.shape)
        graph_stalbe_loss = torch.var(adj, dim=2, unbiased=False).mean()
        out = torch.einsum('bhwij,bjwhd->bhiwd', adj, win_x) # B,H,N*win, win_num, D
        out = out.reshape(B, H, N, -1, win_num, D).permute(0, 1, 2, 4, 3, 5) # B,H,N, win_num, win, D
        # out = fold_windows(out, win=self.win, stride=self.stride) # B, H, N, L, D
        out = out.reshape(B, H, N, -1, D)
        if self.win < P: # patch数不是1
            out_h = out[:, :, :, 0:1, :]
            out_t = out[:, :, :, -1:, :]
            out = torch.cat([out_h, out, out_t], dim=-2).reshape(B, H, N, -1, self.win, D).mean(-2) # BHNPD
        
        # global_channel_node = out.mean(-2) # B,H,N, D
        # global_channel_Q = self.golbal_proj_1(global_channel_node) # B,H,N, D
        # global_channel_K = self.golbal_proj_2(global_channel_node) # B,H,N, D
        # global_channel_V = self.golbal_proj_3(global_channel_node) # B,H,N, D
        # global_channel_adj = torch.einsum('bhid,bhjd->bhij', global_channel_Q, global_channel_K) # B,H,N
        # global_channel_adj = F.softplus(global_channel_adj)
        # global_channel_adj = self.dropout(global_channel_adj)
        # global_channel_out = torch.einsum('bhij,bhjd->bhid', global_channel_adj, global_channel_V) # B,H,N,D
        # out = out + global_channel_out.unsqueeze(-2)
        
        
        # else:
        # print(out.shape)
        #     out = out.mean(-2)
        # out = out.mean(-2) # B,H,N, win_num, D
        # 自适应patch选择关注,在前面cat上上一轮演化的patch表示时间推移,此外利用浅层信息
        # if self.win <= P:
        #     select_x, prob = self.patch_selector(x)
        #     out = torch.cat([select_x, out], dim=-2) # B,H,N, P, D
        out = out.permute(0, 2, 3, 1, 4).reshape(B, N*P, -1) # B*NP*HD
        
        return out, graph_stalbe_loss

    def __init__(self, dim, n_vars, dropout=0.1, n_heads=4, patch_num=2):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.proj_v = nn.Linear(dim*self.n_heads, dim*self.n_heads)
        self.n_vars = n_vars
        # 根据From Similarity to Superiority: Channel Clustering for Time Series Forecasting
        # self.scale = 0.05
        # self.n_cluster = 4
        # self.cluster_aware_feed_forward = nn.Parameter(
        #     self.scale * torch.randn(self.n_cluster, self.dim, self.dim)
        # )
        # self.fuse = nn.Parameter(self.scale * torch.randn(self.n_vars, self.n_cluster)) # 变量聚类
        # self.proj_3 = nn.Linear(dim, dim)
        # self.proj_4 = nn.Linear(dim, dim)
        
        self.win = 2 # win
        self.stride = 1
        
        # self.win_num = patch_num
        # self.win_linear = nn.Linear(dim, dim)
        # self.patch_selector = LearnablePatchSelector(dim)
        self.dropout = nn.Dropout(dropout)
        
        # self.golbal_proj_1 = nn.Linear(dim, dim)
        # self.golbal_proj_2 = nn.Linear(dim, dim)
        # self.golbal_proj_3 = nn.Linear(dim, dim)
        # self.fuse = nn.Linear(dim+32, dim)
        # self.convQK = ConvQKAttention()
        # 状态分布聚类 DUET 或者使用MoE
        
        # self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, alpha):
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        x = x.reshape(B, H, self.n_vars, -1, D)
        B, H, N, P, D = x.shape # P为patch数
        
        if self.win <= P:
            win_x = x.unfold(dimension=-2, size=self.win, step=self.stride) # B,H,N, win_num, win, D
        else:
            win_x = x.unfold(dimension=-2, size=P, step=self.stride) # B,H,N, 1, 1, D P==1
            self.win = 1
        
        win_num = win_x.shape[-3]
        win_x = win_x.permute(0, 1, 2, 4, 3, 5).reshape(B, H, -1, win_num, D) # B,H,N*win, win_num, D
        
        win_adj = torch.einsum('bhiwd,bhjwd->bhwij', self.proj_1(win_x), self.proj_2(win_x)) # ij为窗口内patch序号，w为窗口号win_num
        tmp_adj = win_adj * mask_topk(torch.abs(win_adj), alpha)
        # src_adj =tmp_adj
        # visualize_adj(win_adj,'src')
        
        
        # print(alpha)
        # win_adj = F.softplus(win_adj)
        # adj_pos = torch.clamp(win_adj, min=0.0)
        # adj_neg = torch.clamp(-win_adj, min=0.0)
        
        # weight = torch.softmax(torch.abs(win_adj), dim=-1)*torch.sum(win_adj, dim=-1, keepdim=True)
        # win_adj = torch.sign(tmp_adj) * torch.softmax(torch.abs(tmp_adj), dim=-1)
        
        # win_adj = self.dropout(win_adj)
        
        # adj = F.normalize(win_adj, p=1, dim=-1) #adj = F.normalize(adj, p=1, dim=-1)
        # visualize_adj(adj,'sign_softmax')
        
        win_adj = torch.tanh(tmp_adj)
        
        win_adj = self.dropout(win_adj)
        
        adj = F.normalize(win_adj, p=1, dim=-1) #adj = F.normalize(adj, p=1, dim=-1)
        # visualize_adj(adj,'tanh')
        
        # src_adj = src_adj * mask_topk(src_adj, alpha)
        # src_adj = F.softmax(src_adj, dim=-1)
        # visualize_adj(src_adj,'softmax')
        # time.sleep(20)
        # adj = self.dropout(adj)
        
        # GCN
        # B,H,N*win, win_num, D
        win_x = self.proj_v(win_x.permute(0, 2, 3, 1, 4).reshape(B, N*self.win, win_num , -1)).reshape(B, N * self.win, win_num, self.n_heads, -1)
        # print(adj.shape,win_x.shape)
        graph_stalbe_loss = torch.var(adj, dim=2, unbiased=False).mean()
        out = torch.einsum('bhwij,bjwhd->bhiwd', adj, win_x) # B,H,N*win, win_num, D
        out = out.reshape(B, H, N, -1, win_num, D).permute(0, 1, 2, 4, 3, 5) # B,H,N, win_num, win, D
        # out = fold_windows(out, win=self.win, stride=self.stride) # B, H, N, L, D
        out = out.reshape(B, H, N, -1, D)
        if self.win < P: # patch数不是1
            out_h = out[:, :, :, 0:1, :]
            out_t = out[:, :, :, -1:, :]
            out = torch.cat([out_h, out, out_t], dim=-2).reshape(B, H, N, -1, self.win, D).mean(-2) # BHNPD
        
        # global_channel_node = out.mean(-2) # B,H,N, D
        # global_channel_Q = self.golbal_proj_1(global_channel_node) # B,H,N, D
        # global_channel_K = self.golbal_proj_2(global_channel_node) # B,H,N, D
        # global_channel_V = self.golbal_proj_3(global_channel_node) # B,H,N, D
        # global_channel_adj = torch.einsum('bhid,bhjd->bhij', global_channel_Q, global_channel_K) # B,H,N
        # global_channel_adj = F.softplus(global_channel_adj)
        # global_channel_adj = self.dropout(global_channel_adj)
        # global_channel_out = torch.einsum('bhij,bhjd->bhid', global_channel_adj, global_channel_V) # B,H,N,D
        # out = out + global_channel_out.unsqueeze(-2)
        
        
        # else:
        # print(out.shape)
        #     out = out.mean(-2)
        # out = out.mean(-2) # B,H,N, win_num, D
        # 自适应patch选择关注,在前面cat上上一轮演化的patch表示时间推移,此外利用浅层信息
        # if self.win <= P:
        #     select_x, prob = self.patch_selector(x)
        #     out = torch.cat([select_x, out], dim=-2) # B,H,N, P, D
        out = out.permute(0, 2, 3, 1, 4).reshape(B, N*P, -1) # B*NP*HD
        
        return out, graph_stalbe_loss

class CSE(nn.Module):
    def __init__(self, dim, n_vars, n_heads=4, scale=None, dropout=0., patch_num=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = Context_Win_Graph(self.dim // self.n_heads, n_vars, dropout, n_heads, patch_num)
        # self.graph_conv = GCN(self.dim, self.n_heads)
        # self.graph_conv = time_GCN(self.dim, self.n_heads, n_vars)
        
    def forward(self, x, alpha):
        # x: [B, L, D]
        B, L, D = x.shape

        out, loss = self.graph_learner(x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), alpha)  
        return out, loss  # [B, L, D]


class timeGraph(nn.Module):
    def __init__(self, dim, n_vars, dropout=0.1):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, n_heads=4):
        B, L, D = x.shape
        x = x.reshape(B, L, n_heads, -1).permute(0, 2, 1, 3)
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        x = x.reshape(B, H, self.n_vars, -1, D)
        B, H, N, P, D = x.shape # P为patch数量
        
        time_adj = torch.einsum('bhixd,bhiyd->bhixy', self.proj_1(x), self.proj_2(x))  # i为变量，xy为patch序号(即对应时间)
        time_adj = torch.softmax(time_adj, dim=-1)
        time_adj = self.dropout(time_adj)
        
        time = torch.einsum('bhixy,bhiyd->bixhd', time_adj, x).contiguous() # BNPHD
        time = time.reshape(B, N, P, -1)
        return time

class spaceGraph(nn.Module):     
    def __init__(self, dim, n_heads, n_vars, dropout=0.8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, D = x.shape
        # print(x.shape)
        x = x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        # print(x.shape)
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        # adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))

        # adj = torch.softmax(adj, dim=-1)
        # adj = self.dropout(adj)
        # adj = F.normalize(adj, p=1, dim=-1)
        # space = torch.einsum('bhij,bhjd->bihd', adj, x).contiguous()
        
        x = x.reshape(B, H, self.n_vars, -1, D)
        B, H, N, P, D = x.shape # P为patch数量
        # 待看 todo
        new_x = x
        # new_x = x.detach()
        # print(x.shape)
        space_adj = F.gelu(torch.einsum('bhixd,bhjxd->bhxij', self.proj_1(new_x), self.proj_2(new_x)))  # i为变量，xy为patch序号(即对应时间)
        space_adj = torch.softmax(space_adj, dim=-1) #*0.1后不知道为什么效果不错
        space_adj = self.dropout(space_adj)
        
        space_adj = F.normalize(space_adj, p=1, dim=-1)
        
        space = torch.einsum('bhxij,bhjxd->bixhd', space_adj, x).contiguous() # BNPHD
        space = space.reshape(B, L, -1)
        # space = space.reshape(B, L, -1)
        
        return space

    
# class myatt(nn.Module):
#     def __init__(self, configs, win_size, activation="relu"):
#         super(myatt, self).__init__()
       
#         self.encoder = EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     configs.c_out,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
                
#     def forward(self, x):
#         B, L, D = x.size()
#         # x = x.view(x.size(0), N, -1, self.patch_size).reshape(B, -1, self.patch_size)
#         # B, N*num,P
        
#         # enc_out = enc_out.view(B, N, -1, self.patch_size).reshape(B, N, -1)
#         enc_out, attns = self.encoder(x, attn_mask=None)
        
#         return  enc_out# self.norm2(x + y)

class STBlock(nn.Module):
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, dropout=0., patch_num=2):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff

        self.n_vars = n_vars
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm_self = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)
        
        self.n_heads = n_heads
        # self.space_graph = spaceGraph(self.dim//n_heads, n_heads, n_vars, dropout)
        self.gnn = CSE(self.dim, n_vars, n_heads, dropout=dropout, patch_num=patch_num)
        



        self.att = nn.MultiheadAttention(embed_dim=self.dim, num_heads=8, batch_first=True, dropout=0.5)
        # self.mlp replace mha
        # self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim*4), nn.GELU(), nn.Linear(self.dim*4, self.dim))
        
    def forward(self, x, se=None, alpha=0.5, enable_env=0):
        # x: [B, L, D], time_embed: [B, time_embed_dim]
        # out, loss = self.MoE(self.norm1(x), self.router)
        # out = self.space_graph(x)
        B, L, D = x.shape
        out1, loss = self.gnn(self.norm1(x), alpha)
        loss = 0.0
        # print(x.shape, out.shape)
        
        
        # x = x + out1
        
        # x = self.att(x, x, x)[0]
        autocor = 1.0 - se
        mutcor = se
        
        # out2 = torch.einsum('bnpd,bn->bnpd', x.reshape(B, self.n_vars, -1, D), autocor).reshape(B, -1, D)
        # x = out1 + x
        out2 = 0
        # if L != self.n_vars:
        qkv = x.reshape(B * self.n_vars, -1, D)
        qkv = self.norm_self(qkv)
        out2 = self.att(qkv, qkv, qkv)[0]
        out2 = out2.reshape(B, -1, D)

        if not enable_env:
            # print("Warning: enable_env is False")
            out1 = torch.einsum('bnpd,bn->bnpd', out1.reshape(B, self.n_vars, -1, D), mutcor).reshape(B, -1, D)
            out2 = torch.einsum('bnpd,bn->bnpd', out2.reshape(B, self.n_vars, -1, D), autocor).reshape(B, -1, D)
        else:
            sim = torch.einsum('bnpd,bnpd->bnp', out1.reshape(B, self.n_vars, -1, D), out2.reshape(B, self.n_vars, -1, D))
            autocor = torch.einsum('bnp,bn->bnp', sim, autocor)
            mutcor = 1.0 - autocor
            out1 = torch.einsum('bnpd,bnp->bnpd', out1.reshape(B, self.n_vars, -1, D), mutcor).reshape(B, -1, D)
            out2 = torch.einsum('bnpd,bnp->bnpd', out2.reshape(B, self.n_vars, -1, D), autocor).reshape(B, -1, D)
        # assert torch.isnan(se).sum() == 0, print(se)
        # assert torch.isnan(out2).sum() == 0, print(out2)
        # assert torch.isnan(out1).sum() == 0, print(out1)
    
        x = x + out2 + out1
        
        x = x + self.ffn(self.norm2(x))
        return x, loss

class myGraphExpert(nn.Module):
    def __init__(self, dim, n_head, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.proj_1  = nn.Linear(dim, dim)
        self.proj_2  = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        B, H, L, D = x.shape # L为patch数，不一定是满patch
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        # 正负图分解 todo
        # adj = F.normalize(adj, p=1, dim=-1)
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        out = torch.einsum('bhij,bhjd->bhid', adj, x)
        
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        
        return out
        

class mySparseMoE(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.2, num_experts = 8):
        super().__init__()
        
        # self.A_num = A_num
        # self.B_num = B_num
        self.num_experts = num_experts # self.A_num * self.B_num
        self.n_heads = n_heads
        # self.n_embed = n_embed
        # self.r = r
        # self.scale = 0.02
        # self.A = nn.Parameter(self.scale * torch.randn(A_num, n_embed, r), requires_grad=True)
        # self.B = nn.Parameter(self.scale * torch.randn(B_num, n_embed, r), requires_grad=True)
        self.experts = nn.ModuleList([myGraphExpert(dim//n_heads, n_heads, dropout) for _ in range(self.num_experts)])
        
        self.expert_fix = myGraphExpert(dim//n_heads, n_heads, dropout)
        # self.top_k = top_k

    def forward(self, x, router): # B*L*D
        B, L, D = x.shape
        # 1. 输入进入router得到两个输出
        gating_output, indices = router(x.detach()) # B*L*e  , B*L*k
        # todo
        loss = 0
        # print(gating_output.shape, indices.shape)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            expert = self.expert_fix
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1) # B*L # 同一批次同一变量间共享topk选择
            # 5. 展平操作
            # flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if expert_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                # if not self.training:
                #     print(flat_mask.shape, flat_x.shape)
                expert_input = x[expert_mask]
                # 7. 将token输入expert得到输出
                # expert_input = expert_input.reshape(B, -1, self.n_heads, D//self.n_heads).permute(0, 2, 1, 3)
                expert_input = expert_input.reshape(B, -1, self.n_heads, D//self.n_heads).permute(0, 2, 1, 3)
                expert_output = expert(expert_input) #output B L D

                # 8. 计算当前专家对于有作用的token的权重分数
                # print(gating_output.shape, expert_mask.shape, expert_output.shape)
                gating_scores = gating_output[expert_mask][:, i].unsqueeze(1) #BL*1
                # print(gating_scores)
                # print(gating_scores.shape, expert_output.shape)
                # 9. 将expert输出乘上权重分数
                expert_output = expert_output.reshape(-1, D)
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output

        return final_output, loss


class myRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(myRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.noise_linear =nn.Linear(dim, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        B, L, E = mh_output.shape
        logits = self.topkroute_linear(mh_output) # B*L*E

        #Noise logits
        noise_logits = self.noise_linear(mh_output) # B*L*E

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        noisy_logits_batch_mean = noisy_logits.mean(dim=0) # L*E
        # noisy_logits_mean = noisy_logits.mean(dim=0).mean(dim=0)
        top_k_mean, indices_mean = noisy_logits_batch_mean.topk(self.top_k, dim=-1)
        indices = indices_mean.unsqueeze(0).repeat(B, 1, 1)
        # top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        top_k_logits = noisy_logits.gather(dim=-1, index=indices)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        
        
        
        # 测试，删去
        # router_output = torch.ones_like(router_output)
        # indices =torch.zeros_like(indices)
        
        
        return router_output, indices




class SEED_Backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dim = configs.dim
        self.n_heads = configs.n_heads
        self.n_blocks = configs.n_blocks
        self.n_vars = configs.n_vars
        self.num_patches = configs.num_patches
        self.dropout = configs.dropout
        self.enable_env = configs.enable_env
        self.alpha = configs.alpha
        
        self.d_ff = self.dim * 2 if configs.d_ff is None else configs.d_ff
        # graph blocks
        self.blocks = nn.ModuleList([
            STBlock(self.dim, self.n_vars, self.d_ff, self.n_heads, self.dropout, self.num_patches)
            for _ in range(self.n_blocks)
        ])


    def forward(self, x, se=None):
        # x: [B, N, T]
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, se, self.alpha, self.enable_env)
            moe_loss += loss
        moe_loss /= self.n_blocks
        return x, moe_loss  # [B, N, T]
