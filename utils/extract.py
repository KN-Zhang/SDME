# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


import torch
import torch.nn as nn
import torch.nn.functional as F

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr
    
    def forward(self, repeatability, **kw):
        assert len(repeatability) == 1
        repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1,  
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    min_size = H
    max_size = H
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,Q,D = [],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            res = net(imgs=[img], phase='all', if_transformer=True)
    
            # get output and repeatability map
            descriptors = res['descriptors'][0]
            repeatability = res['repeatability'][0]

            # extract maxima and descs
            y,x = detector(**res) # nms
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(Q) # scores = repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores, res['repeatability'][0], res['dense_descriptors'][0]

@torch.no_grad()
def extract_keypoints(conf, net, img, return_numpy=True):

    args = conf
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rep_thr = args['repeatability_thr'])
    
    # extract keypoints/descriptors for a single image
    xys, desc, scores, repeatability_map, dense_descriptors = extract_multiscale(net, img, detector,
        scale_f   = args['scale_f'], 
        min_scale = args['min_scale'], 
        max_scale = args['max_scale'], 
        verbose = False)
    if return_numpy:
        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-args['top_k'] or None:]
    else:
        idxs = scores.argsort()[-args['top_k'] or None:]        
    
    return xys[idxs], desc[idxs], repeatability_map, dense_descriptors

  




