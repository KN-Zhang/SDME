import numpy as np
import os, json, cv2
import kornia
import torch
import torchvision.transforms as tvf

from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, name, mode='train', data_path='/home/kz23d522/data/SDME/Dataset') -> None:
        super().__init__()
        self.name = name
        self.mode = mode
        self.data_path = data_path
        
        self.load_path()
        self.norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.norm_RGB = tvf.Compose([tvf.ToTensor()])
        
    def load_path(self):
        if self.name == 'MSCOCO' or self.name == 'GoogleEarth':
            self.img_name = os.listdir(os.path.join(self.data_path, self.name, f'{self.mode}2014_input/'))
            self.input_path = os.path.join(self.data_path, self.name, f'{self.mode}2014_input/')
            self.label_path = os.path.join(self.data_path, self.name, f'{self.mode}2014_label/')
            self.template_path = os.path.join(self.data_path, self.name, f'{self.mode}2014_template/')

        elif self.name == 'VIS_NIR':
            if self.mode == 'train':
                self.img_name = os.listdir(os.path.join(self.data_path, self.name, 'train_small_size/NIR/'))
                self.input_path = os.path.join(self.data_path, self.name, 'train_small_size/VIS')
                self.label_path = os.path.join(self.data_path, self.name, 'train_small_size/label')
                self.template_path = os.path.join(self.data_path, self.name, 'train_small_size/NIR')
            else:
                self.img_name = open(os.path.join(self.data_path, self.name, 'test_small_size/test_list.txt')).read().split('\n')
                self.input_path = os.path.join(self.data_path, self.name, 'test_small_size/VIS')
                self.label_path = os.path.join(self.data_path, self.name, 'test_small_size/label')
                self.template_path = os.path.join(self.data_path, self.name, 'test_small_size/NIR')                
        elif self.name == 'VIS_IR_drone':
            if self.mode == 'train':
                self.img_name = os.listdir(os.path.join(self.data_path, self.name, 'train_small_size/IR/'))
                self.input_path = os.path.join(self.data_path, self.name, 'train_small_size/VIS')
                self.label_path = os.path.join(self.data_path, self.name, 'train_small_size/label')
                self.template_path = os.path.join(self.data_path, self.name, 'train_small_size/IR')
            else:
                self.img_name = open(os.path.join(self.data_path, self.name, 'test_small_size/test_list.txt')).read().split('\n')
                self.input_path = os.path.join(self.data_path, self.name, 'test_small_size/VIS')
                self.label_path = os.path.join(self.data_path, self.name, 'test_small_size/label')
                self.template_path = os.path.join(self.data_path, self.name, 'test_small_size/IR')
        self.nimg = len(self.img_name)    
        self.input_image_path = [os.path.join(self.input_path, self.img_name[idx].split('/')[-1]) for idx in range(self.nimg)]
        self.template_image_path = [os.path.join(self.template_path, self.img_name[idx].split('/')[-1]) for idx in range(self.nimg)]            

    def H_gt(self, u, v):
        src_points = torch.tensor([[0,0], [127,0], [127,127], [0,127]]).float()
        tgt_points = torch.stack([torch.tensor(u), torch.tensor(v)], dim=-1).float()

        H = kornia.geometry.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0))
        
        return H.squeeze(0), tgt_points
    
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, index):
        sample_name = self.img_name[index]
        input_image = Image.open(self.input_image_path[index]).convert('RGB')
        template_image = Image.open(self.template_image_path[index]).convert('RGB')

        label_path = os.path.join(self.label_path, sample_name.split('.')[0] + '_label.txt')
        with open(label_path, 'r') as outfile:
            data = json.load(outfile)
        u_list = [data['location'][0]['top_left_u'], data['location'][1]['top_right_u'], data['location'][3]['bottom_right_u'], data['location'][2]['bottom_left_u']]
        v_list = [data['location'][0]['top_left_v'], data['location'][1]['top_right_v'], data['location'][3]['bottom_right_v'], data['location'][2]['bottom_left_v']]
        
        H_t2i, points_gt = self.H_gt(u_list, v_list)
        # trf = tuple(np.reshape(H_t2i, -1)[:-1])
        
        W, H = template_image.size 
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        xy = torch.stack([x, y], dim=-1).view(-1, 2).float()
        aflow = kornia.geometry.linalg.transform_points(H_t2i.unsqueeze(0), xy.unsqueeze(0)).reshape(H, W, 2).permute(2, 0, 1).float()
        mask = torch.ones_like(aflow[0, :, :])
        
        return {
            'img1': self.norm_RGB(template_image),
            'img2': self.norm_RGB(input_image),
            'H_gt': H_t2i,
            'aflow': aflow,
            'mask': mask,
            'points_gt': points_gt,
        }
        
           
