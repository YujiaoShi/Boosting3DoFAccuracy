from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from VGG import L2_norm
from swin_transformer_cross import SwinTransformerSelf


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super(CrossAttention, self).__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, x, y):
        """
        x: (B, M, C)
        y: (B, M, N, C)
        """
        
        B, M, C = x.shape
        _, M, N, C = y.shape

        # Project with multiple heads
        q = self.to_q(x).reshape(B, M, 1, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, 1, dim_head]
        k = self.to_k(y).reshape(B, M, N, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, N, dim_head]
        v = self.to_v(y).reshape(B, M, N, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, N, dim_head]

        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2)).reshape(B, self.heads, M, N, 1)  
        dot = dot.softmax(dim=-2)

        # Combine values (image level features).
        a = torch.sum(dot * v, dim=-2) # [B, self.heads, M, dim_heads]
        a = a.permute(0, 2, 1, 3).reshape(B, M, self.heads * self.dim_head)
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)  # [B, M, C]
    
        return z


def generate_xy_for_attn(x, y, u):
    '''
    x.shape = [B, C, S, S]
    y.shape = [B, C, H, W]
    uv.shape = [B, S, S]
    
    return:
    x.shape = [B, S^2, C]
    ys.shape = [B, S^2, 2H, C]
    '''
    B, C, S, _ = x.shape
    x = x.reshape(B, C, S*S).permute(0, 2, 1)
    
    _, C, H, W = y.shape
    
    with torch.no_grad():
        u_left = torch.floor(u)
        u_right = u_left + 1
        
        torch.clamp(u_left, 0, W -1, out=u_left)
        torch.clamp(u_right, 0, W -1, out=u_right)
    
    y = y.reshape(B, C*H, W)
    y_left = torch.gather(y, 2, u_left.long().view(B, 1, S*S).repeat(1, C*H, 1)).view(B, C, H, S*S)
    y_right = torch.gather(y, 2, u_right.long().view(B, 1, S*S).repeat(1, C*H, 1)).view(B, C, H, S*S)
    ys = torch.cat([y_left, y_right], dim=2).permute(0, 3, 2, 1)  # [B, S^2, 2H, C]
    
    return x, ys


def generate_y_for_attn(S, y, u):
    '''

    y.shape = [B, C, H, W]
    uv.shape = [B, S, S]

    return:

    ys.shape = [B, S^2, 2H, C]
    '''
    # B, C, S, _ = x.shape
    # x = x.reshape(B, C, S * S).permute(0, 2, 1)

    B, C, H, W = y.shape

    with torch.no_grad():
        u_left = torch.floor(u)
        u_right = u_left + 1

        torch.clamp(u_left, 0, W - 1, out=u_left)
        torch.clamp(u_right, 0, W - 1, out=u_right)

    y = y.reshape(B, C * H, W)
    y_left = torch.gather(y, 2, u_left.long().view(B, 1, S * S).repeat(1, C * H, 1)).view(B, C, H, S * S)
    y_right = torch.gather(y, 2, u_right.long().view(B, 1, S * S).repeat(1, C * H, 1)).view(B, C, H, S * S)
    ys = torch.cat([y_left, y_right], dim=2).permute(0, 3, 2, 1)  # [B, S^2, 2H, C]

    return ys
    

class CrossViewAttention(nn.Module):
    
    def __init__(self, blocks, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm) -> None:
        super(CrossViewAttention, self).__init__()

        self.blocks = blocks

        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        for _ in range(blocks):
            self.self_attention_layers.append(
                SwinTransformerSelf(img_size=[64, 64], patch_size=1, in_chans=256, num_classes=3,
                                embed_dim=48, depths=[1], num_heads=[3],
                                window_size=8, mlp_ratio=4., qkv_bias=False, proj_bias=False, qk_scale=None,
                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                use_checkpoint=False)
            )
            self.cross_attention_layers.append(
                CrossAttention(dim, heads, dim_head, qkv_bias, norm)
            )

    
    
    def forward(self, grd2sat, grd_x, u, geo_mask):
        '''
         grd2sat.shape = [B, C, S, S]
         grd_x.shape = [B, C, H, W]
         u.shape = [B, S, S]
        '''
        B, C, S, _ = grd2sat.shape

        x_attn = grd2sat
        y_attn = generate_y_for_attn(S, grd_x, u)

        for i in range(self.blocks):
            x_attn = self.self_attention_layers[i](x_attn, geo_mask)
            # x_attn = x_attn.reshape(B, C, S * S).permute(0, 2, 1)
            x_attn = self.cross_attention_layers[i](x_attn, y_attn)
            x_attn = x_attn.permute(0, 2, 1).reshape(B, C, S, S)

        # for layer in self.cross_attention_layers:
        #     x_attn = layer(x_attn, y_attn)
        
        # x_attn = x_attn.permute(0, 2, 1).reshape(B, C, S, S)
        
        return L2_norm(x_attn)
        
            
        