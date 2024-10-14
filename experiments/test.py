import argparse
import torch
from omegaconf import OmegaConf
from model.SDME import SDME
from benchmark.benchmark_homography import HomogBenchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--dataset_root_path', type=str, default='/home/kz23d522/data/SDME/Dataset', help='dataset root path')
    parser.add_argument('--dataset', type=str, default='VIS_IR_drone', help='dataset')
    parser.add_argument('--config', type=str, default='configs/test.yaml', help='config')
    parser.add_argument('--ck_path', type=str, default='checkpoints/VIS_IR_drone.pth',)
    args = parser.parse_args()
    
    device = torch.device('cuda:'+ str(args.gpuid))
    conf = OmegaConf.load(args.config)
    
    model = SDME(conf['Model'])
    states = torch.load(args.ck_path)
    model.load_state_dict(states["model"])
    model = model.to(device)

    benchmark = HomogBenchmark(dataset=args.dataset,
                               data_root_path=args.dataset_root_path,
                               conf=conf,
                               device=device).to(device)
    benchmark.run(model, if_print=True)


if __name__ == "__main__":
    main()