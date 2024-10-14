import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm

from datasets.pair_dataset import PairDataset
from loss.converge_loss import calculate_feature_map
from loss.utils.Lucas_Kanade_optimization import Lucas_Kanade_layer
from loss.utils.interpolation import Interpolator
from utils.test import initial_H_vanilla, mnn_matcher, average_cornner_error
from utils.extract import extract_keypoints
class HomogBenchmark(nn.Module):
    def __init__(self, dataset, data_root_path, conf, device) -> None:
        super().__init__()
        test_dataset = PairDataset(name=dataset, mode='val', data_path=data_root_path)
        self.dataset = test_dataset
        self.conf = conf
        self.device = device
        self.register_buffer('H0', initial_H_vanilla(), persistent=False)
        
        self.Lucas_Kanade_layer_fine = Lucas_Kanade_layer(batch_size=1,
                                                    height_template=128, 
                                                    width_template=128, 
                                                    interpolator=Interpolator())
    def todevice(self, x):
        for key, value in x.items(): 
            if type(x[key]) == torch.Tensor: x[key] = x[key].to(self.device)
    
    @torch.no_grad()                
    def run(self, model, if_print=True):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=1, shuffle=False
        )        
        model.eval()
        sr = 0
        error0_list = []
        error1_list = []
        error2_list = []
        results = {}
        for batch in tqdm(dataloader):
            self.todevice(batch)
            B = batch['img1'].shape[0]
            self.H0 = self.H0.repeat(B, 1, 1) ## B 3 3
            
            keypoints_a, descriptors_a, repeatability_a, dense_descriptors_a =\
            extract_keypoints(
                self.conf['Extractor'],
                model,
                batch['img1'],
                return_numpy=False
                )
            keypoints_b, descriptors_b, repeatability_b, dense_descriptors_b =\
            extract_keypoints(
                self.conf['Extractor'],
                model,
                batch['img2'],
                return_numpy=False
                )
            
            try:
                matches, mask_mnn = mnn_matcher(descriptors_a, descriptors_b)
                H1, inliers = cv2.findHomography(keypoints_a[matches[:, 0], :2].cpu().numpy(),
                                                keypoints_b[matches[:, 1], :2].cpu().numpy(),
                                                cv2.USAC_MAGSAC, 1, 0.999, 100000)
                H1 = H1 / H1 [-1, -1]
                H1 = torch.from_numpy(H1).unsqueeze(0).repeat(B, 1, 1).float().to(self.device)
            except:
                H1 = self.H0

            error0 = average_cornner_error(self.H0.repeat(B, 1, 1), batch['points_gt'])
            error1 = average_cornner_error(H1, batch['points_gt'])
                      

            descriptors_a = calculate_feature_map(dense_descriptors_a, 3)
            descriptors_b = calculate_feature_map(dense_descriptors_b, 3)
                        
            H2 = self.Lucas_Kanade_layer_fine.update_matrix(descriptors_a,
                                                            descriptors_b,
                                                            repeatability_a,
                                                            repeatability_b,
                                                            H1,
                                                            1.,
                                                            mode = 'weight',
                                                            )
            error2 = average_cornner_error(H2, batch['points_gt'])
            
            if error2 < error0:
                sr += 1
                error0_list.append(error0.item())
                error1_list.append(error1.item())
                error2_list.append(error2.item())
                
        if  if_print:
            print(f'{np.mean(error0_list)}---->{np.mean(error1_list)}')
            print(f'{np.mean(error1_list)}---->{np.mean(error2_list)}')
            print(f'SR: {sr/len(dataloader)}')

            print(f'PE<0.5: {np.sum(np.array(error1_list) < 0.5) / len(error1_list)}')
            print(f'PE<1: {np.sum(np.array(error1_list) < 1) / len(error1_list)}')
            print(f'PE<3: {np.sum(np.array(error1_list) < 3) / len(error1_list)}')
            print(f'PE<5: {np.sum(np.array(error1_list) < 5) / len(error1_list)}')
            print(f'PE<10: {np.sum(np.array(error1_list) < 10) / len(error1_list)}')
            print(f'PE<20: {np.sum(np.array(error1_list) < 20) / len(error1_list)}')

            print('\n')

            print(f'PE<0.5: {np.sum(np.array(error2_list) < 0.5) / len(error2_list)}')
            print(f'PE<1: {np.sum(np.array(error2_list) < 1) / len(error2_list)}')
            print(f'PE<3: {np.sum(np.array(error2_list) < 3) / len(error2_list)}')
            print(f'PE<5: {np.sum(np.array(error2_list) < 5) / len(error2_list)}')
            print(f'PE<10: {np.sum(np.array(error2_list) < 10) / len(error2_list)}')
            print(f'PE<20: {np.sum(np.array(error2_list) < 20) / len(error2_list)}')
        
        thresholds = [0.5, 1, 3, 5, 10, 20]
        ratio = {i: np.sum(np.array(error2_list)<i) / len(error2_list) for i in thresholds}
        results.update({
                    f"ACE_stage1": np.mean(error1_list),
                    f"ACE_stage2": np.mean(error2_list),
                })
        results.update({f'homog@{t}': v for t, v in ratio.items()})
        return results