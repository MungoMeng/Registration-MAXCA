import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal


class MultiAxisXCA(nn.Module):   #input shape: n, h, w, d, c
    """The multi-axis XCA block."""
    
    def __init__(self, region_size, num_heads, num_channels, 
                 input_proj_factor=2, use_bias=True, dropout_rate=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels*input_proj_factor, bias=use_bias)
        self.gelu = nn.GELU()
        self.GlobalXCALayer = GlobalXCALayer(region_size=region_size, num_channels=num_channels*input_proj_factor//2, 
                                         num_heads=num_heads, use_bias=use_bias)
        self.LocalXCALayer = LocalXCALayer(region_size=region_size, num_channels=num_channels*input_proj_factor//2, 
                                           num_heads=num_heads, use_bias=use_bias)
        self.out_project = nn.Linear(num_channels*input_proj_factor, num_channels, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward_run(self, x_in):
        
        x = self.LayerNorm(x_in)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)
        
        #grid gMLP
        u = self.GlobalXCALayer(u)
        #block gMLP
        v = self.LocalXCALayer(v)
        
        #out projection
        x = torch.cat([u,v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        
        x_out = x + x_in
        return x_out

    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class GlobalXCALayer(nn.Module):  #input shape: n, h, w, d, c
    """Global XCA layer that performs global mixing of tokens."""
    
    def __init__(self, region_size, num_channels, num_heads, use_bias=True, dropout_rate=0):
        super().__init__()
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.fh = region_size[0]
        self.fw = region_size[1]
        self.fd = region_size[2]
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.qkv = nn.Conv3d(num_channels, num_channels*3, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.out_project = nn.Linear(num_channels, num_channels, use_bias)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        
        _, h, w, d, _ = x.shape
        
        # padding
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.fh - h % self.fh) % self.fh
        pad_b = (self.fw - w % self.fw) % self.fw
        pad_r = (self.fd - d % self.fd) % self.fd
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        shortcut = x
        x = self.LayerNorm(x)

        x = x.permute(0,4,1,2,3)  #n,c,h,w,d
        x = self.qkv(x)
        x = x.permute(0,2,3,4,1)  #n,h,w,d,c
        
        gh, gw, gd = x.shape[1] // self.fh, x.shape[2] // self.fw, x.shape[3] // self.fd
        x = split_images_einops(x, patch_size=(self.fh, self.fw, self.fd))  #n (gh gw gd) (fh fw fd) c
        
        # Global XCA part
        q,k,v = x.chunk(3, dim=-1)

        q = einops.rearrange(q, 'b g f (head c) -> b f head c g', head=self.num_heads)
        k = einops.rearrange(k, 'b g f (head c) -> b f head c g', head=self.num_heads)
        v = einops.rearrange(v, 'b g f (head c) -> b f head c g', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        x = einops.rearrange(out, 'b f head c g -> b g f (head c)')
        x = self.out_project(x)
        x = self.dropout(x)
        
        x = unsplit_images_einops(x, grid_size=(gh, gw, gd), patch_size=(self.fh, self.fw, self.fd))

        x = x + shortcut
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        
        return x


class LocalXCALayer(nn.Module):  #input shape: n, h, w, d, c
    """Local XCA layer that performs local mixing of tokens."""
    
    def __init__(self, region_size, num_channels, num_heads, use_bias=True, dropout_rate=0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.fh = region_size[0]
        self.fw = region_size[1]
        self.fd = region_size[2]
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.qkv = nn.Conv3d(num_channels, num_channels*3, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.out_project = nn.Linear(num_channels, num_channels, use_bias)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        
        _, h, w, d, _ = x.shape

        # padding
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.fh - h % self.fh) % self.fh
        pad_b = (self.fw - w % self.fw) % self.fw
        pad_r = (self.fd - d % self.fd) % self.fd
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        shortcut = x
        x = self.LayerNorm(x)

        x = x.permute(0,4,1,2,3)  #n,c,h,w,d
        x = self.qkv(x)
        x = x.permute(0,2,3,4,1)  #n,h,w,d,c
        
        gh, gw, gd = x.shape[1] // self.fh, x.shape[2] // self.fw, x.shape[3] // self.fd
        x = split_images_einops(x, patch_size=(self.fh, self.fw, self.fd))  #n (gh gw gd) (fh fw fd) c
        
        # Local XCA part
        q,k,v = x.chunk(3, dim=-1)

        q = einops.rearrange(q, 'b g f (head c) -> b g head c f', head=self.num_heads)
        k = einops.rearrange(k, 'b g f (head c) -> b g head c f', head=self.num_heads)
        v = einops.rearrange(v, 'b g f (head c) -> b g head c f', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        x = einops.rearrange(out, 'b g head c f -> b g f (head c)')
        x = self.out_project(x)
        x = self.dropout(x)
        
        x = unsplit_images_einops(x, grid_size=(gh, gw, gd), patch_size=(self.fh, self.fw, self.fd))

        x = x + shortcut
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        
        return x

########################################################
# Functions
########################################################

def split_images_einops(x, patch_size):  #n, h, w, d, c
    """Image to patches."""
    
    batch, height, width, depth, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    grid_depth = depth // patch_size[2]
    
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c",
        gh=grid_height, gw=grid_width, gd=grid_depth, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def unsplit_images_einops(x, grid_size, patch_size):
    """patches to images."""
    
    x = einops.rearrange(
        x, "n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c",
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x
    
