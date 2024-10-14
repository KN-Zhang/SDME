import torch
import torch.nn as nn
import math
from torch.nn import functional as nnF

from loss.utils.Lucas_Kanade_optimization import Lucas_Kanade_layer
from loss.utils.interpolation import Interpolator


def generate_lambda_noise():
    lambda_one = (torch.rand(9) - 0.5) / 6
    for mm in range(len(lambda_one)):
        if lambda_one[mm]>0 and lambda_one[mm]<0.02:
            lambda_one[mm]=0.02
        if lambda_one[mm]<0 and lambda_one[mm]>-0.02:
            lambda_one[mm]=-0.02    
    lambda_one[-1] = 0.
    return lambda_one

def gt_motion_rs(H, lambda_noisy):
    b, _, _ = H.shape
    lambda_noisy = lambda_noisy.reshape(3, 3).unsqueeze(0).repeat(b, 1, 1)
    noisy_matrix = lambda_noisy * H
    return noisy_matrix

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

def calculate_ssim(Lucas_Kanade_layer_fine, template_calculated_feature_map, input_calculated_feature_map, template_weight, input_weight, H_gt, H_noisy):
    input_warped_to_template_calculated_feature_map_fine = \
        Lucas_Kanade_layer_fine.calculate_warped_feature_map(template_calculated_feature_map,
                                                            input_calculated_feature_map,
                                                            H_gt + H_noisy)
    input_warped_to_template_weight = \
        Lucas_Kanade_layer_fine.calculate_warped_feature_map(template_weight,
                                                            input_weight,
                                                            H_gt + H_noisy)    
    B, C, _, _, = template_calculated_feature_map.shape
    template_calculated_feature_map = template_calculated_feature_map.reshape(B, C, -1) ## B C N
    input_warped_to_template_calculated_feature_map_fine = input_warped_to_template_calculated_feature_map_fine.reshape(B, C, -1)
    
    template_weight = template_weight.reshape(B, C, -1) ## B C N
    input_warped_to_template_weight = input_warped_to_template_weight.reshape(B, C, -1) ## B C N

    
    ssim_middle_fine = torch.mean(template_weight * input_warped_to_template_weight * torch.pow(template_calculated_feature_map - input_warped_to_template_calculated_feature_map_fine, 2))
    return ssim_middle_fine

class ConvergeLoss(nn.Module):
    def __init__(self, batch_size):
        nn.Module.__init__(self)
        interpolater = Interpolator()        
        self.Lucas_Kanade_layer_fine = Lucas_Kanade_layer(batch_size=batch_size,
                                                          height_template=128, 
                                                          width_template=128, 
                                                          interpolator=interpolater)
    def calculate_ConvergeLoss_weight(self, template_features, input_features, template_weights, input_weights, H_gt, lambda_=2., damping=1.):
        
        DEV = H_gt.device
        template_calculated_feature_map = calculate_feature_map(template_features, 3)
        input_calculated_feature_map = calculate_feature_map(input_features, 3)
                
        ssim_middle_fine = calculate_ssim(self.Lucas_Kanade_layer_fine, 
                                          template_calculated_feature_map, 
                                          input_calculated_feature_map,
                                          template_weights,
                                          input_weights,
                                          H_gt, 
                                          0)
        convex_loss = 0.
        for nn in range(4):
            lambda_noisy = generate_lambda_noise().to(H_gt.device)
            noisy_gt_matrix = gt_motion_rs(H_gt, lambda_noisy)
            ssim_shift_left_fine = calculate_ssim(self.Lucas_Kanade_layer_fine,
                                                  template_calculated_feature_map,
                                                  input_calculated_feature_map,
                                                  template_weights,
                                                  input_weights,
                                                  H_gt, 
                                                  noisy_gt_matrix)            
            ssim_shift_left_left_fine = calculate_ssim(self.Lucas_Kanade_layer_fine,
                                                  template_calculated_feature_map,
                                                  input_calculated_feature_map,
                                                  template_weights,
                                                  input_weights,
                                                  H_gt, 
                                                  2. * noisy_gt_matrix) 
            ssim_shift_right_fine = calculate_ssim(self.Lucas_Kanade_layer_fine,
                                                  template_calculated_feature_map,
                                                  input_calculated_feature_map,
                                                  template_weights,
                                                  input_weights,
                                                  H_gt, 
                                                  -noisy_gt_matrix)            
            ssim_shift_right_right_fine = calculate_ssim(self.Lucas_Kanade_layer_fine,
                                                  template_calculated_feature_map,
                                                  input_calculated_feature_map,
                                                  template_weights,
                                                  input_weights,
                                                  H_gt, 
                                                  -2. * noisy_gt_matrix)

            if not isinstance(damping, torch.Tensor):
                ssim_convex_fine_left = -torch.minimum((ssim_shift_left_fine - ssim_middle_fine)-
                                                        torch.sum(torch.pow(lambda_noisy, 2)), torch.tensor(0., device=DEV)) 
                ssim_convex_fine_left_left = -torch.minimum((ssim_shift_left_fine - ssim_shift_left_left_fine)+
                                                            (torch.sum(torch.pow(2 * lambda_noisy, 2))-
                                                            torch.sum(torch.pow(lambda_noisy, 2))), torch.tensor(0., device=DEV)) 
                ssim_convex_fine_right = -torch.minimum((ssim_shift_right_fine - ssim_middle_fine)-
                                                        torch.sum(torch.pow(lambda_noisy, 2)), torch.tensor(0., device=DEV))    
                ssim_convex_fine_right_right = -torch.minimum((ssim_shift_right_fine - ssim_shift_right_right_fine)+
                                                            (torch.sum(torch.pow(2 * lambda_noisy, 2))-
                                                            torch.sum(torch.pow(lambda_noisy, 2))), torch.tensor(0., device=DEV)) 
            else:
                ssim_convex_fine_left = -torch.minimum((ssim_shift_left_fine - ssim_middle_fine)-
                                                        torch.sum(damping * torch.pow(lambda_noisy[:-1], 2)), torch.tensor(0., device=DEV)) 
                ssim_convex_fine_left_left = -torch.minimum((ssim_shift_left_fine - ssim_shift_left_left_fine)+
                                                            torch.sum(damping * (torch.pow(lambda_ * lambda_noisy, 2)-
                                                            torch.pow(lambda_noisy, 2))[:-1]), torch.tensor(0., device=DEV)) 
                ssim_convex_fine_right = -torch.minimum((ssim_shift_right_fine - ssim_middle_fine)-
                                                        torch.sum(damping * torch.pow(lambda_noisy[:-1], 2)), torch.tensor(0., device=DEV))    
                ssim_convex_fine_right_right = -torch.minimum((ssim_shift_right_fine - ssim_shift_right_right_fine)+
                                                            torch.sum(damping * (torch.pow(lambda_ * lambda_noisy, 2)-
                                                            torch.pow(lambda_noisy, 2))[:-1]), torch.tensor(0., device=DEV))                 
                   
            convex_loss = convex_loss + ssim_convex_fine_left + ssim_convex_fine_left_left \
                                      + ssim_convex_fine_right + ssim_convex_fine_right_right           
        
        
        return convex_loss, ssim_middle_fine, template_calculated_feature_map, input_calculated_feature_map    
    