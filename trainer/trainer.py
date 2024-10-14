import torch
import torch.nn as nn
from torch.autograd import Variable

from trainer.min_norm_solvers import gradient_normalizers, MinNormSolver
from utils.common import batch_minmax
class MTL_Trainer(nn.Module):
    def __init__(self, function_sparse_loss, function_dense_loss, optimizer, device) -> None:
        super().__init__()
        self.function_sparse_loss = function_sparse_loss
        self.function_dense_loss = function_dense_loss
        self.optimizer = optimizer
        self.device = device
    def todevice(self, x):
        for key, value in x.items(): 
            if type(x[key]) == torch.Tensor: x[key] = x[key].to(self.device)        
    
    def forward(self, model, batch):

        self.todevice(batch)
        loss_data = {}
        grads = {}
        scale = {}

        output1 = model(imgs=[batch['img1'], batch['img2']], phase='encoder')
        encoder_result_variable = [Variable(output1['encoder_result'][i].data.clone(), requires_grad=True) for i in range(len(output1['encoder_result']))]
        
        self.optimizer.zero_grad()
        output2 = model(imgs=[batch['img1'], batch['img2']], phase='dense', encoder_result=encoder_result_variable)
        
        B, C, H1, W1 = output2['dense_descriptors'][0].shape
        B, C, H2, W2 = output2['dense_descriptors'][1].shape

        weight1 = batch['mask'].unsqueeze(1).float()
        weight2 = torch.ones((B, 1, H2, W2), device=self.device)

        c_loss, ssim_loss, template_calculated_feature_map, input_calculated_feature_map =\
        self.function_dense_loss.calculate_ConvergeLoss_weight(
            output2['dense_descriptors'][0], 
            output2['dense_descriptors'][1],
            weight1,
            weight2,
            batch['H_gt']
        )
                
        loss_dense = c_loss + ssim_loss
        loss_data['dense'] = loss_dense.data
        loss_dense.backward()
        grads['dense'] = []
        grads['dense'].append(Variable(encoder_result_variable[0].grad.data.clone(), requires_grad=False))
        encoder_result_variable[0].grad.data.zero_()
        encoder_result_variable[1].grad.data.zero_()
        
        self.optimizer.zero_grad()
        output3 = model(imgs=[batch['img1'],batch['img2']], phase='sparse', encoder_result=encoder_result_variable)
        
        
        output3['image_gradient'] = [] 
        output3['image_gradient'].append(model.densemap_guided(1 - batch_minmax(template_calculated_feature_map.detach())))
        output3['image_gradient'].append(model.densemap_guided(1 - batch_minmax(input_calculated_feature_map.detach())))  

        allvars = dict(batch, **output3)
        loss, details = self.function_sparse_loss(**allvars)

        loss_data['sparse'] = loss.data
        loss.backward()
        grads['sparse'] = []
        grads['sparse'].append(Variable(encoder_result_variable[0].grad.data.clone(), requires_grad=False))
        encoder_result_variable[0].grad.data.zero_()
        encoder_result_variable[1].grad.data.zero_()
        
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        
        for t in ['dense', 'sparse']:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]
        try:
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in ['dense', 'sparse']])
            for i, t in enumerate(['dense', 'sparse']):
                scale[t] = float(sol[i])
        except:
            print("failed to get scale!!!!")
            scale['dense'] = 0.5
            scale['sparse'] = 0.5     
            
        self.optimizer.zero_grad()
        del output1, output2, output3, encoder_result_variable
        torch.cuda.empty_cache()

        output = model(imgs=[batch['img1'],batch['img2']], phase='all')      
        
        weight1 = batch['mask'].unsqueeze(1).float()*output['repeatability'][0].detach()
        weight2 = output['repeatability'][1].detach()

        c_loss, ssim_loss, template_calculated_feature_map, input_calculated_feature_map =\
        self.function_dense_loss.calculate_ConvergeLoss_weight(
            output['dense_descriptors'][0], 
            output['dense_descriptors'][1],
            weight1,
            weight2,
            batch['H_gt'],
        )        


        output['image_gradient'] = []   
        output['image_gradient'].append(model.densemap_guided(1 - batch_minmax(template_calculated_feature_map.detach())))
        output['image_gradient'].append(model.densemap_guided(1 - batch_minmax(input_calculated_feature_map.detach())))     

        allvars = dict(batch, **output)
        loss, details = self.function_sparse_loss(**allvars)    
        
        loss_total = scale['dense'] * (c_loss + ssim_loss) + scale['sparse'] * loss
        loss_total.backward()

        details['dense_ratio'] = scale['dense']
        details['converge_loss'] = c_loss.item()
        details['ssim_loss'] = ssim_loss.item()
        
             
        torch.cuda.empty_cache()

        self.optimizer.step()

        
        plot_for_show = dict(
            img1=batch['img1'],
            img2=batch['img2'],
            repeatability1=output['repeatability'][0],
            repeatability2=output['repeatability'][1],
            dense_featuremap1=template_calculated_feature_map,
            dense_featuremap2=input_calculated_feature_map,
        )
        
        return details, plot_for_show        