import argparse
from argparse import Namespace
from collections import OrderedDict

import torch
from pathlib import Path

from modules.eval import Eval
from modules.model import LseRepFusNet, repvgg_model_convert, LseRepNet


def parse_opt() -> Namespace:
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--src', type=str, help='fusion data root path  DataSets include:[TNO/RoadScene/MSRS/M3FD]', default='data/test/TNO' )
    parser.add_argument('--dst', type=str, help='fusion images save path run save include:[TNO/RoadScene/MSRS/M3FD]', default='runs/test/TNO')

    parser.add_argument('--weights', type=str, default='cache/a1/300.pth', help='pretrained weights path')
    parser.add_argument('--deploy_weight', type=str, default='cache/a2/300.pth', help='pretrained weights path')
    parser.add_argument('--color', action='store_true', help='colorize fused images with visible color channels', default=True)

    # fusion opt
    parser.add_argument('--dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--depth', type=int, default=3, help='network dense depth')
    parser.add_argument('--cudnn', action='store_true', help='accelerate network forward with cudnn')
    parser.add_argument('--eval', action='store_true', help='use eval mode for new pytorch models')
    parser.add_argument('--half', action='store_true', help='use half mode for new pytorch models')
    # gpus
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='2',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    # deploy
    parser.add_argument('--mode', metavar='MODE', default='train', choices=['train', 'deploy'], help='train or deploy')


    return parser.parse_args()


def img_filter(x: Path) -> bool:
    return x.suffix in ['.png', '.bmp', '.jpg']



if __name__ == '__main__':
    config = parse_opt()
    cuda = (config.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(config.gpus[0]) if cuda else "cpu")

    # init model
    lseRepFusNet = LseRepFusNet(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
    lseRepNet = LseRepNet(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
    # load pretrained weights
    ck_pt = torch.load(config.weights, map_location=device)
    # Multi card parallelism, removing the module in the weight
    for i, j in ck_pt.items():
        state_dict = OrderedDict()
        for k, v in j.items():
            name = k[7:]  # remove `module.`
            state_dict[name] = v
        ck_pt[i] = state_dict

    lseRepFusNet.load_state_dict(ck_pt['lseRepFusNet'])
    lseRepNet.load_state_dict(ck_pt['lseRepNet'])
    save_path = config.dst
    if config.mode == 'deploy':
        lseRepFusNet = repvgg_model_convert(lseRepFusNet, save_path=config.deploy_weight)
        lseRepNet = repvgg_model_convert(lseRepNet, save_path=config.deploy_weight)
        save_path = config.dst + '/' + 'deploy'

    # images
    root = Path(config.src)
    ir_paths = [x for x in sorted((root / 'ir').glob('*')) if img_filter]
    vi_paths = [x for x in sorted((root / 'vi').glob('*')) if img_filter]
    # vi_paths = [x for x in sorted((root / 'lab').glob('*')) if img_filter]

    f = Eval(lseRepFusNet, device=device, cudnn=config.cudnn, half=config.half, eval=config.eval)
    f(ir_paths, vi_paths, Path(save_path), config.color)






