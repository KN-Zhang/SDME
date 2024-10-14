import torch
import torch.nn as nn
import kornia
import math
from torch.nn import functional as nnF
from packaging import version

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky

class Lucas_Kanade_layer(nn.Module):
    def __init__(self, batch_size, height_template, width_template, interpolator):
        nn.Module.__init__(self)
        self.batch_size = batch_size
        self.interpolator = interpolator
        
        template_grid_x, template_grid_y = self.meshgrid(self.batch_size, height_template, width_template) ## B N 1
        self.register_buffer('template_grid_x', template_grid_x, persistent=False)
        self.register_buffer('template_grid_y', template_grid_y, persistent=False)
        p_W_p_2D_once = self.p_W_p_2D()
        self.register_buffer('p_W_p_2D_once', p_W_p_2D_once, persistent=False)
        
    def meshgrid(self, batch_size, height, width):

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        xy = torch.stack([x, y], dim=-1).view(-1, 2).float()
        x = xy[:, 0].unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) ## B N 1
        y = xy[:, 1].unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) ## B N 1
   
        return x, y         
    
        
    def p_F_p_2D(self, template_feature):

        b, c, h, w = template_feature.shape
        padding = torch.nn.ReflectionPad2d(1)
        template_feature_padding = padding(template_feature)
        template_feature_partial_u = template_feature_padding[:, :, 1:h+1, 2:] - template_feature_padding[:, :, 1:h+1, :w]
        template_feature_partial_v = template_feature_padding[:, :, 2:, 1:w+1] - template_feature_padding[:, :, :h, 1:w+1]
        
        template_feature_partial_u = template_feature_partial_u.reshape(b, -1, c).unsqueeze(-1) ## B N C 1
        template_feature_partial_v = template_feature_partial_v.reshape(b, -1, c).unsqueeze(-1) ## B N C 1
        
        p_F_p_2D = torch.cat((template_feature_partial_u, template_feature_partial_v), dim=-1) ## B N C 2
        
        return p_F_p_2D
     
    def p_W_p_2D(self):

        ## only needs once if the resolution of the image is fixed
        x = self.template_grid_x ## B N 1
        y = self.template_grid_y

        ones = torch.ones_like(x)
        zeros = torch.ones_like(x)
        
        first_row = torch.cat([x, y, ones, zeros, zeros, zeros, -x * x, -x * y], dim=-1).unsqueeze(-1) ## B N 8 1
        second_row = torch.cat([zeros, zeros, zeros, x, y, ones, -y * x, -y * y], dim=-1).unsqueeze(-1)
        
        p_W_p_2D = torch.cat([first_row, second_row], dim=-1) ## B N 8 2
        p_W_p_2D = p_W_p_2D.permute(0, 1, 3, 2) ## B N 2 8
        
        return p_W_p_2D
    
    def calculate_J(self, template_feature):

        assert len(template_feature.shape) == 4
        b, c, h, w = template_feature.shape
        p_F_p_2D = self.p_F_p_2D(template_feature) ## B N C 2
        J = p_F_p_2D @ self.p_W_p_2D_once ## B N C 8
        
        return J
    
    def calculate_r(self, template_feature, input_feature, H):

        b, c, h, w = template_feature.shape
        ## warp the template to input
        warped_template_grid = self.calculate_warped_template_grid(H)
        
        interpolater_feature, mask, _ = self.interpolator(input_feature, warped_template_grid) ## B N C
        photoness_loss = interpolater_feature - template_feature.reshape(b, c, -1).permute(0, 2, 1) ## B N C

        return photoness_loss 
   
    
    def calculate_warped_feature_map(self, template_feature, input_feature, H_gt):
        b, c, h, w = template_feature.shape
        ## warp the template to input
        warped_template_grid = self.calculate_warped_template_grid(H_gt)
        interpolater_feature, mask, _ = self.interpolator(input_feature, warped_template_grid) ## B N C

        return interpolater_feature.reshape(b, h, w, c).permute(0, 3, 1, 2)       
    
    def calculate_warped_template_grid(self, matrix):
        warped_template_grid = \
        kornia.geometry.linalg.transform_points(matrix, torch.cat((self.template_grid_x, self.template_grid_y), dim=-1))
        
        return warped_template_grid
    
    def build_system(self, J, res, weights):
        '''
        Input:
        J: B N C 8
        res: B N C
        weights: B N 

        Output:
        grad: B 8
        Hess:  B 8 8
        '''
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weights[..., None] * grad
        grad = grad.sum(-2) / grad.shape[1]  # ... x 6

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3) / Hess.shape[1]  # ... x 6 x6

        return grad, Hess        
    def optimizer_step(self, g, H, lambda_=0, mute=True, mask=None, eps=1e-6):
        """One optimization step with Gauss-Newton or Levenberg-Marquardt.
        Args:
            g: batched gradient tensor of size (..., N).
            H: batched hessian tensor of size (..., N, N).
            lambda_: damping factor for LM (use GN if lambda_=0).
            mask: denotes valid elements of the batch (optional).
        """
        if lambda_ == 0:  # noqa
            diag = torch.zeros_like(g)
        else:
            diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
        H = H + diag.clamp(min=eps).diag_embed()

        if mask is not None:
            # make sure that masked elements are not singular
            H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
            # set g to 0 to delta is 0 for masked elements
            g = g.masked_fill(~mask[..., None], 0.)

        H_, g_ = H.cpu(), g.cpu()
        # delta = -torch.solve(g_[..., None], H_)[0][..., 0]
        try:
            U = cholesky(H_)
        except RuntimeError as e:
            if 'singular U' in str(e):
                if not mute:
                    logger.debug(
                        'Cholesky decomposition failed, fallback to LU.')
                delta = -torch.solve(g_[..., None], H_)[0][..., 0]
            else:
                raise
        else:
            delta = -torch.cholesky_solve(g_[..., None], U)[..., 0]

        return delta.to(H.device)

    def calculate_delta_p(self, template_feature, input_feature, template_feature_weight, input_feature_weight, matrix, mode, damping=0):
        
        warped_template_grid = self.calculate_warped_template_grid(matrix)
        weight2, mask, _ = self.interpolator(input_feature_weight, warped_template_grid)
        weight1 = template_feature_weight.squeeze().reshape(self.batch_size, -1) ## B N
        r = self.calculate_r(template_feature, input_feature, matrix) ## r: B N C
        if mode == 'weight':
            weight = weight1 * weight2.squeeze(-1)
        elif mode == 'no_weight':
            weight = torch.ones_like(mask)
        J = self.calculate_J(template_feature) ## B N C 8
        g, H = self.build_system(J, -r, weight)
        delta_p = self.optimizer_step(g, H, lambda_=damping, eps=1e-4) ## B 8 

        return delta_p[..., None]
        
    def construct_matrix(self, initial_matrix, scale_factor):
        if not isinstance(initial_matrix, torch.Tensor):
            return 0
        else:
            scale_matrix = scale_factor * torch.eye(3, 3, dtype=torch.float32, device=initial_matrix.device).unsqueeze(0).repeat(self.batch_size, 1, 1)   
            scale_matrix[:, 2, 2] = 1.0  
        
            scale_matrix_inverse = torch.linalg.inv(scale_matrix)
        
            final_matrix = scale_matrix @ initial_matrix @ scale_matrix_inverse
        
            return final_matrix
        
    def update_matrix(self, template_feature_real, input_feature_real, template_feature_weight, input_feature_weight, H0, scale_factor, mode, damping=0):
        H_pre = self.construct_matrix(H0, scale_factor)

        for _ in range(15):    
            delta_p_1 = self.calculate_delta_p(template_feature_real, input_feature_real, template_feature_weight, input_feature_weight, H_pre, mode, damping) ## B 8 1
            zeros = torch.zeros_like(delta_p_1)[:, 0, :].unsqueeze(1) ## B 1 1
            delta_p_2 = torch.cat((delta_p_1, zeros), dim=1).reshape(self.batch_size, 3, 3) ## B 3 3
            
            delta_p_3 = delta_p_2 + torch.eye(3, device=delta_p_2.device)
            
            H = H_pre @ torch.inverse(delta_p_3)
            normaliza_value = H[:, -1, -1].unsqueeze(-1).unsqueeze(-1) ## B 1 1
            H = H / normaliza_value.repeat(1, 3, 3) ## B 3 3      
            
            H_pre = H 
        
        return H
            
def calculate_feature_map(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = nnF.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    patches = patches.view(b, -1, c, h, w).permute(0, 3, 4, 2, 1) ## b h w c 9
    patch_extract_mean = torch.mean(patches, dim=3, keepdim=True)
    
    patches = patches - patch_extract_mean
    patch_transpose = patches.permute(0, 1, 2, 4, 3)
    variance_matrix = patch_transpose @ patches
    
    eye_matrix = torch.eye(kernel*kernel, device=x.device)[None, None, None, :, :]
    trace_value = torch.sum(torch.sum(variance_matrix * eye_matrix, dim=-1), dim=-1)
    row_sum = torch.sum(variance_matrix, dim=-1)
    max_row_sum = torch.max(row_sum, dim=-1)[0]
    min_row_sum = torch.min(row_sum, dim=-1)[0]
    mimic_ratio=(max_row_sum+min_row_sum)/2.0/trace_value

    return mimic_ratio.unsqueeze(1)  ## b 1 h w
    
