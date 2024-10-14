import torch

def norm_minmax(x):
    C, H, W = x.shape
    max_vals = torch.max(x.view(C, -1), dim=1)[0][:, None, None]
    min_vals = torch.min(x.view(C, -1), dim=1)[0][:, None, None]  
    return (x-min_vals) / (max_vals-min_vals)

def batch_minmax(x):
    b, c, h, w = x.shape
    x = torch.cat([norm_minmax(i).unsqueeze(0) for i in x]) ## 三通道
    return x

        