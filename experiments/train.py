import argparse, os, wandb, sys
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets.pair_dataset import PairDataset
from model.SDME import SDME
from loss.sparse_losses import *
from loss.sampler import *
from loss.converge_loss import ConvergeLoss
from trainer.trainer import MTL_Trainer
from utils.checkpoint import CheckPoint
from utils.common import batch_minmax

from benchmark.benchmark_homography import HomogBenchmark

def train_one_epoch(args, dataloader, model, trainer, lr_scheduler):
    model.train()
    for idx, batch in enumerate(tqdm(dataloader)):
        loss_details, plot_for_show = trainer(model, batch)
        if not idx % args.plot_freq == 0:
            log_dict = {**loss_details, 'lr': lr_scheduler.get_last_lr()[0]}
            wandb.log(log_dict)
        else:
            wandb_table = wandb.Table(columns=['image_name', 'image'])
            plot_for_show = {key: batch_minmax(value) for key, value in plot_for_show.items()}
            for key, value in plot_for_show.items():
                wandb_image = wandb.Image(value[0].permute(1, 2, 0).detach().cpu().numpy(), caption=key)
                wandb_table.add_data(key, wandb_image)

            log_dict = {**loss_details, 'visualization': wandb_table, 'lr': lr_scheduler.get_last_lr()[0]}
            wandb.log(log_dict)
    lr_scheduler.step()
      
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--dataset_root_path', type=str, default='/home/kz23d522/data/SDME/Dataset', help='dataset root path')    
    parser.add_argument('--dataset', type=str, default='VIS_IR_drone', help='dataset')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--plot_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10)
    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpuid))

    experiment_name = os.path.splitext(os.path.basename(__file__))[0] + '-' + args.dataset
    wandb_mode = "online" if not args.dont_log_wandb else "disabled"
    wandb.init(project="SDME", name=f'{experiment_name}', reinit=False, mode = wandb_mode)

    conf = OmegaConf.load(args.config)
    batchsz = conf['Trainer']['batch_size']
    global_step = 0
    checkpointer = CheckPoint(dir='./checkpoints/', name=experiment_name)

    ## create model, optimizer, lr_schedule
    model = SDME(conf['Model'])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=conf['Trainer']['lr'],
        weight_decay=conf['Trainer']['weight_decay'],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf['Trainer']['epoch'])
    
    if args.restore:
        model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    model = model.to(device)    
    
    ## create losses
    loss_sparse = conf['Loss']['sparse']['loss'].replace('`sampler`', conf['Loss']['sparse']['sampler']).replace('`N`', str(conf['Loss']['sparse']['patch_size']))
    loss_sparse = eval(loss_sparse.replace('\n', ''))
    loss_dense = ConvergeLoss(batchsz)    
    
    ## create trainer
    trainer = MTL_Trainer(loss_sparse, loss_dense, optimizer, device).to(device)
    
    ## create dataset
    train_dataset = PairDataset(name=args.dataset, mode='train', data_path=args.dataset_root_path)
    
    ## create benchmark
    benchmark = HomogBenchmark(dataset=args.dataset,
                               data_root_path=args.dataset_root_path,
                               conf=conf,
                               device=device).to(device)

    try:
        for n in range(args.epoch):
            dataloader = iter(
                torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size = batchsz,
                    num_workers = 8,
                    shuffle = True,
                    drop_last=True,
                )
            )
            train_one_epoch(args, dataloader, model, trainer, lr_scheduler)
            if n % args.save_freq == 0:
                if not args.dont_log_wandb:
                    checkpointer.save(model, optimizer, lr_scheduler, n)
                wandb.log(benchmark.run(model, if_print=False))
    except KeyboardInterrupt:
            print('Interrupted')
            if not args.dont_log_wandb:
                checkpointer.save(model, optimizer, lr_scheduler, n)
            sys.exit(0)                

if __name__ == "__main__":
    main()