# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model.transformer import LocalFeatureTransformer
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
class densemap_guided(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1))
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    def forward(self, x):
        return self.softmax(torch.relu(self.alpha) * x)

class SDME(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.curchan = config['conv']['inchan']
        self.mchan = config['conv']['mchan']
        self.ochan = config['conv']['ochan']
        self.dilated = config['conv']['dilated']
        self.dilation = config['conv']['dilation']
        
        ## repeatability
        self.sal = nn.Conv2d(self.ochan, 1, kernel_size=1)
        self.densemap_guided = densemap_guided()
        
        self.FeatureTransformer = LocalFeatureTransformer(config['transformer'])    
        self.agents = nn.Parameter(torch.randn(config['agents']['num'], config['agents']['dim']), requires_grad=True)
        
        self.ops = nn.ModuleList([])
        self._add_conv(1,  8*self.mchan) ## 3->32
        self._add_conv(2,  8*self.mchan) ## 32->32
        self._add_conv(3, 16*self.mchan, stride=2) ## 32->64
        self._add_conv(4, 16*self.mchan) ## 64->64
        self._add_conv(5, 32*self.mchan, stride=2) ## 64->128
        self._add_conv(6, 32*self.mchan) ## 128->128
        self._add_conv(7, 32*self.mchan, k=2, stride=2, relu=False) ## 128->128
        self._add_conv(8, 32*self.mchan, k=2, stride=2, relu=False) ## 128->128
        self._add_conv(9, self.ochan, k=2, stride=2, bn=False, relu=False) ## 128->128
        
    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=False)        
        
    def _add_conv(self, i, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        if i <= 4:
            d = self.dilation * dilation
            if self.dilated: 
                conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
                self.dilation *= stride
            else:
                conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
            self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
            if bn: self.ops.append(self._make_bn(outd))
            if relu: self.ops.append(nn.ReLU(inplace=True))
            self.curchan = outd
        elif i > 4:
            seperate_modulelist = nn.ModuleList([])
            d = self.dilation * dilation
            if self.dilated: 
                conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
                self.dilation *= stride
            else:
                conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
            seperate_modulelist.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
            if bn: seperate_modulelist.append(self._make_bn(outd))
            if relu: seperate_modulelist.append(nn.ReLU(inplace=True))
            self.curchan = outd
            if i == 9:
                seperate_modulelist_dense = nn.ModuleList([])
                seperate_modulelist_dense.append(nn.Conv2d(self.curchan, 128, kernel_size=k, **conv_params))
            else:
                seperate_modulelist_dense = copy.deepcopy(seperate_modulelist)            
            self.ops.append(seperate_modulelist_dense)  ## dense branch
            self.ops.append(seperate_modulelist) ## sparse branch
            
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x_for_dense, x_for_sparse, urepeatability, updated_agents=None):
        return dict(dense_descriptors = F.normalize(x_for_dense, p=2, dim=1),
                    descriptors = F.normalize(x_for_sparse, p=2, dim=1),
                    repeatability = self.softmax(urepeatability),
                    updated_agents = updated_agents,
                    )
                    
    def forward_one(self, x, phase, encoder_result, if_transformer):
        b, _, h, w = x.shape
        if phase == 'encoder':
            for n, op in enumerate(self.ops):
                if n < 12:
                    x = op(x)
                else:
                    break
            return dict(encoder_result = x)
        if phase == 'dense':
            x_for_dense = encoder_result
            for n_dense, op_list in enumerate(self.ops[12::2]):   
                for op in op_list:
                    x_for_dense = op(x_for_dense)     
            return dict(dense_descriptors = F.normalize(x_for_dense, p=2, dim=1))      
        if phase == 'sparse':
            x_for_sparse = encoder_result
            for n_sparse, op_list in enumerate(self.ops[13::2]):
                for op in op_list:
                    x_for_sparse = op(x_for_sparse) 
            if if_transformer:
                x_for_sparse, updated_agents = self.FeatureTransformer(self.agents.unsqueeze(0),
                                                                       x_for_sparse.flatten(-2).permute(0, 2, 1),
                                                                       None)
                x_for_sparse = x_for_sparse.permute(0, 2, 1).reshape(b, -1, h, w)
            else:
                x_for_sparse = x_for_sparse
                updated_agents = None
            urepeatability = self.sal(x_for_sparse**2) 
            return dict(descriptors=F.normalize(x_for_sparse, p=2, dim=1),
                        repeatability=self.softmax(urepeatability),
                        updated_agents=updated_agents)  
        if phase == 'all':    
            for n, op in enumerate(self.ops):
                if n < 12:
                    x = op(x)
                else:
                    break  
            x_for_dense = x
            x_for_sparse = x
            for n_dense, op_list in enumerate(self.ops[n::2]):   
                for op in op_list:
                    x_for_dense = op(x_for_dense) 
            for n_sparse, op_list in enumerate(self.ops[n+1::2]):
                for op in op_list:
                    x_for_sparse = op(x_for_sparse)
            if if_transformer:
                x_for_sparse, updated_agents = self.FeatureTransformer(self.agents, x_for_sparse.flatten(-2).permute(0, 2, 1), None)
                x_for_sparse = x_for_sparse.permute(0, 2, 1).reshape(b, -1, h, w)
            else:
                x_for_sparse = x_for_sparse
                updated_agents = None            
            urepeatability = self.sal(x_for_sparse**2)
        return self.normalize(x_for_dense, x_for_sparse, urepeatability, updated_agents)
    
    def forward(self, imgs, phase, encoder_result=[None,None], if_transformer=True, **kw):
        res = [self.forward_one(img, phase, e, if_transformer) for img, e in zip(imgs, encoder_result)]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)    
