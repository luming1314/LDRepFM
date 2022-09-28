import logging
from argparse import Namespace

import torch
from kornia.filters import SpatialGradient
from kornia.losses import SSIMLoss
from torch import nn, Tensor
from torchvision.transforms import transforms


from torch.optim import RMSprop
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from modules.dataset import DataSets
from modules.environment_probe import EnvironmentProbe
from modules.model import LseRepFusNet


class Train:

    def __init__(self, environment_probe: EnvironmentProbe, config: Namespace, logging: logging, writer: SummaryWriter):
        self.logging = logging
        self.writer = writer
        logging.info(f'our Training | mask: {config.mask} | weight: {config.weight} | adv: {config.adv_weight}')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        logging.info(f'autoEncoder | dim: {config.dim} | depth: {config.depth}')

        self.lseRepFusNet = LseRepFusNet(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)


        # WGAN adam optim
        logging.info(f'RMSprop | learning rate: {config.learning_rate}')

        self.opt_lseRepFusNet = RMSprop(self.lseRepFusNet.parameters(), lr=config.learning_rate)


        # move to device
        logging.info(f'module device: {environment_probe.device}')


        self.lseRepFusNet.to(environment_probe.device)


        if len(config.gpus) > 1:
            # parallel
            self.lseRepFusNet = nn.DataParallel(self.lseRepFusNet, device_ids=config.gpus, output_device=config.gpus[0])

        # loss
        self.l1 = nn.L1Loss(reduction='none')
        self.ssim = SSIMLoss(window_size=11, reduction='none')

        self.spatial = SpatialGradient('diff')



        self.l1.cuda(environment_probe.device)
        self.ssim.cuda(environment_probe.device)

        # datasets
        folder = Path(config.folder)
        self.resize = transforms.Resize((config.size, config.size))
        dataset = DataSets(folder, transform=self.resize)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(
            dataset.train_ind)  # sampler will assign the whole data according to batchsize.
        valid_sampler = SubsetRandomSampler(dataset.val_ind)

        self.train_loader = DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size,
                                       sampler=train_sampler, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size,
                                     sampler=valid_sampler, pin_memory=True)


        logging.info(f'dataset | folder: {str(folder)} | size: {dataset.__len__()}')


    def train(self, epoch):
        process_train = tqdm(enumerate(self.train_loader), disable=not self.config.debug)
        loss_total = 0.
        for idx, sample in process_train:
            ir, vi = sample[0].to(self.environment_probe.device), sample[1].to(self.environment_probe.device)
            loss = self.train_fusion(ir, vi)
            loss_total += loss['loss']
        loss_avg = loss_total / len(self.train_loader)
        self.logging.info(f'train----------fuse: {loss_avg:03f}')
        self.writer.add_scalar('Train/loss_avg', loss_avg, epoch)

    def train_fusion(self, ir: Tensor, vi: Tensor):
        self.lseRepFusNet.train()
        fus = self.lseRepFusNet(ir, vi)
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        vi_grad = self.gradient(vi)
        ir_grad = self.gradient(ir)
        grad_max = torch.max(vi_grad, ir_grad)
        l_f = b1 * self.ssim(fus, torch.max(ir, vi)) + b2 * self.l1(fus, torch.max(ir, vi)) + b2 * self.l1(self.gradient(fus), grad_max)
        l_f = l_f.mean()
        loss = l_f
        # backward
        self.opt_lseRepFusNet.zero_grad()
        loss.backward()
        self.opt_lseRepFusNet.step()
        # loss state
        state = {
            'loss': loss.item(),
        }
        return state


    def eval(self, epoch):
        process_val = tqdm(enumerate(self.val_loader), disable=not self.config.debug)
        loss_total = 0.
        for idx, sample in process_val:
            ir, vi = sample[0].to(self.environment_probe.device), sample[1].to(self.environment_probe.device)
            loss = self.val_fusion(ir, vi)
            loss_total += loss['loss']
        loss_avg = loss_total / len(self.train_loader)
        self.logging.info(f'val----------fuse: {loss_avg:03f}')
        self.writer.add_scalar('Val/loss_avg', loss_avg, epoch)
        self.save(epoch)

    @torch.no_grad()
    def val_fusion(self, ir: Tensor, vi: Tensor):
        self.lseRepFusNet.eval()
        fus = self.lseRepFusNet(ir, vi)
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        vi_grad = self.gradient(vi)
        ir_grad = self.gradient(ir)
        grad_max = torch.max(vi_grad, ir_grad)
        l_f = b1 * self.ssim(fus, torch.max(ir, vi)) + b2 * self.l1(fus, torch.max(ir, vi)) + b2 * self.l1(self.gradient(fus), grad_max)
        l_f = l_f.mean()
        loss = l_f
        # loss state
        state = {
            'loss': loss.item(),
        }
        return state


    def save(self, epoch: int):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)
        cache = path / f'{epoch:03d}.pth'
        self.logging.info(f'save checkpoint to {str(cache)}')
        state = {
            'lseRepFusNet': self.lseRepFusNet.state_dict(),
        }
        torch.save(state, cache)

    def gradient(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
        return u





