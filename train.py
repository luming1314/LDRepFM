import argparse
import logging
from argparse import Namespace

import torch

from tensorboardX import SummaryWriter

from modules.environment_probe import EnvironmentProbe
from modules.train import Train


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='a1', help='train process identifier')
    parser.add_argument('--folder', default='data/train', help='data root path')
    parser.add_argument('--size', default=120, help='resize image to the specified size')
    parser.add_argument('--cache', default='cache', help='weights cache folder')

    # LseRepFusNet opt
    parser.add_argument('--depth', default=3, type=int, help='network dense depth')
    parser.add_argument('--dim', default=32, type=int, help='network features dimension')
    parser.add_argument('--mask', default='m1', help='mark index')
    parser.add_argument('--weight', nargs='+', type=float, default=[1, 20, 0.1], help='loss weight')
    parser.add_argument('--adv_weight', nargs='+', type=float, default=[1, 1], help='discriminator balance')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=300, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=40, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', action='store_true', help='debug mode (default: off)')

    # log
    parser.add_argument('--log', type=str, default='logs/log.log', help='save log path')

    #tensorboardX
    parser.add_argument('--summary_name', type=str, default='DoubleAEcoder',
                        help='Name of the tensorboard summmary')
    parser.add_argument('--tensorboardX_path', type=str, default='logs/tensorboardX',
                        help='Path of the tensorboard summmary')

    # gpus
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1,3',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

    return parser.parse_args()

def log():
    logger = logging.getLogger('fusion')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(config.log)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':



    config = parse_args()
    writer = SummaryWriter(config.tensorboardX_path, comment=config.summary_name)

    logger = log()
    logger.info("Start train")
    logger.info("our use device is {}".format(config.gpus))

    cuda = (config.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(config.gpus[0]) if cuda else "cpu")

    environment_probe = EnvironmentProbe(device)

    train_process = Train(environment_probe, config, logger, writer)
    for epoch in range(1, config.epochs + 1):
        train_process.train(epoch)
        train_process.eval(epoch)
    logger.info("train over")
    writer.close()
