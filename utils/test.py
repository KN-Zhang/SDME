import kornia
import torch

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy(), mask.cpu().numpy()

def initial_H_vanilla():
    src_points = torch.tensor([[0, 0], [127, 0], [127, 127], [0, 127]]).float()
    tgt_points = torch.tensor([[32, 32], [160, 32], [160, 160], [32, 160]]).float()
    H = kornia.geometry.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0)).float()

    return H

def average_cornner_error(matrix, points_gt):
    B = points_gt.shape[0]
    src_points = torch.tensor([[0, 0], [127, 0], [127, 127], [0, 127]]).float().to(points_gt.device)
    predicted = kornia.geometry.linalg.transform_points(matrix, src_points.unsqueeze(0).repeat(B, 1, 1))
    return ((points_gt - predicted)**2).sum(-1).sqrt().mean()
