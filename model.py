import torch
from torch import nn
import torch.nn.functional as F

class PositionEncoder():
    def __init__(self, L):
        super().__init__()
        self.L = L

    def __call__(self, input):
        output = []
        for i in range(0, self.L):
            s = torch.sin((2.**i) * input)
            c = torch.cos((2.**i) * input)
            output.append(s)
            output.append(c)
        return torch.cat(output, dim=-1)

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, in_L=10, in_v_L=4, skips=[4]):
        super(NeRF, self).__init__()
        self.pe_x = PositionEncoder(in_L)
        self.pe_d = PositionEncoder(in_v_L)
        self.D = D
        self.W = W
        self.input_ch_net = input_ch * (in_L * 2 + 1)
        self.input_ch_views_net = input_ch_views * (in_v_L * 2 + 1)
        self.skips = skips
        
        self.nerf_net = nn.ModuleList([nn.Linear(self.input_ch_net, W)])
        
        for i in range(D-1):
            if i not in self.skips:
                self.nerf_net.append(nn.Linear(W, W))
            else:
                self.nerf_net.append(nn.Linear(self.input_ch_net + W, W))
        
        self.feature = nn.Linear(W, W)
        self.sigma = nn.Linear(W, 1)
        self.view = nn.Linear(self.input_ch_views_net + W, W // 2)
        self.rgb = nn.Linear(W // 2, 3)
        
    def forward(self, rays_samples, view_dirs):
        
        input_x = torch.cat([rays_samples ,self.pe_x(rays_samples)], dim=-1)  
        input_view = torch.cat([view_dirs, self.pe_d(view_dirs)], dim=-1)
        
        h = input_x
        for i in range(self.D):
            h = self.nerf_net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_x, h], -1)
        
        sigma = self.sigma(h)
        sigma = F.relu(sigma)
        
        feature = self.feature(h)
        h = torch.cat([feature, input_view], -1)
        
        h = self.view(h)
        h = F.relu(h)
        rgb = self.rgb(h)
        # rgb = torch.sigmoid(rgb)
        
        # output = torch.cat([rgb, sigma], -1)
        return rgb, sigma