import copy
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
from modules.model import LseRepFusNet, LseRepNet


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
        self.lseRepNet = LseRepNet(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)


        # WGAN adam optim
        logging.info(f'RMSprop | learning rate: {config.learning_rate}')

        self.opt_lseRepFusNet = RMSprop(self.lseRepFusNet.parameters(), lr=config.learning_rate)
        self.opt_lseRepNet = RMSprop(self.lseRepNet.parameters(), lr=config.learning_rate)


        # move to device
        logging.info(f'module device: {environment_probe.device}')


        self.lseRepFusNet.to(environment_probe.device)
        self.lseRepNet.to(environment_probe.device)


        if len(config.gpus) > 1:
            # parallel
            self.lseRepFusNet = nn.DataParallel(self.lseRepFusNet, device_ids=config.gpus, output_device=config.gpus[0])
            self.lseRepNet = nn.DataParallel(self.lseRepNet, device_ids=config.gpus, output_device=config.gpus[0])

        # loss
        self.l1 = nn.L1Loss(reduction='none')
        self.ssim = SSIMLoss(window_size=11, reduction='none')
        self.mse = nn.MSELoss()

        self.spatial = SpatialGradient('diff')



        self.l1.cuda(environment_probe.device)
        self.ssim.cuda(environment_probe.device)
        self.mse.cuda(environment_probe.device)


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
        loss_total, loss_mask = 0., 0.
        for idx, sample in process_train:
            ir, vi = sample[0].to(self.environment_probe.device), sample[1].to(self.environment_probe.device)
            people_mask = sample[2].to(self.environment_probe.device)
            m3fd_mask = sample[3].to(self.environment_probe.device)
            loss = self.train_fusion(ir, vi, people_mask, m3fd_mask)
            loss_total += loss['loss']
            loss_mask += loss['l_mask']
        loss_avg = loss_total / len(self.train_loader)
        loss_mask_avg = loss_mask / len(self.train_loader)
        self.logging.info(f'train----------fuse: {loss_avg:03f}-------{loss_mask_avg:03f}')
        self.writer.add_scalar('Train/loss_avg', loss_avg, epoch)
        self.writer.add_scalar('Train/loss_mask_avg', loss_mask_avg, epoch)

    def train_fusion(self, ir: Tensor, vi: Tensor, people_mask: Tensor, m3fd_mask: Tensor):
        self.lseRepNet.train()
        mask = self.lseRepNet(ir)
        zeros = torch.zeros_like(ir)
        ones = torch.ones_like(ir)
        mean = torch.mean(ir)
        w = torch.where(ir > mean, ir, mean)
        for i in range(1):
            mean = torch.mean(w)
            w = torch.where(w > mean, w, mean)
        mask_lab = torch.where(w > torch.mean(w), ones, zeros)
        people_mask = torch.where(people_mask > zeros, ones, zeros)
        m3fd_mask = torch.where(m3fd_mask > zeros, ones, zeros)
        unite_mask = mask_lab + m3fd_mask
        unite_mask = torch.where(unite_mask > zeros, ones, zeros)
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        l_mask = b1 * self.ssim(mask, unite_mask * ir) + b2 * self.l1(mask, unite_mask * ir)
        mask_grad = self.gradient(mask)
        ir_grad = self.gradient(ir)
        l_mask_grad = b1 * self.mse(mask_grad, ir_grad) + b2 * self.l1(mask_grad, ir_grad)

        # mask_grad_bin = torch.where(mask_grad > mask_grad.mean(), ones, zeros)
        # ir_grad_bin = torch.where(ir_grad > ir_grad.mean(), ones, zeros)
        # l_mask_grad = b1 * self.mse(mask_grad_bin, ir_grad_bin) + b2 * self.l1(mask_grad_bin, ir_grad_bin)
        l_mask = l_mask.mean() + l_mask_grad.mean()
        # backward
        self.opt_lseRepNet.zero_grad()
        l_mask.backward()
        self.opt_lseRepNet.step()

        self.lseRepNet.eval()
        self.lseRepFusNet.train()
        mask = self.lseRepNet(ir)
        mask = torch.where(mask > torch.mean(mask), ones, zeros)
        # people_mask = torch.where(people_mask > zeros, ones, zeros)
        fus = self.lseRepFusNet(ir , vi )
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        vi_grad = self.gradient(vi)
        ir_grad = self.gradient(ir)
        grad_max = torch.max(vi_grad, ir_grad)
        # l_vi = b1 * self.ssim(fus , vi * (1 - people_mask) + ir * people_mask  ) + b2 * self.l1(fus , vi * (1 - people_mask) + ir * people_mask )
        l_vi = b1 * self.ssim(fus , vi ) + b2 * self.l1(fus , vi  )
        # l_ir = b1 * self.ssim(fus * mask, torch.max(ir, vi) * mask) + b2 * self.l1(fus * mask, torch.max(ir, vi) * mask)
        l_ir = b1 * self.ssim(fus * mask, ir * mask) + b2 * self.l1(fus * mask, ir * mask)
        l_bright = b1 * self.ssim(fus * people_mask, ir * people_mask) + b2 * self.l1(fus * people_mask, ir * people_mask)
        l_grad = b1 * self.ssim(self.gradient(fus), grad_max) + b2 * self.l1(self.gradient(fus), grad_max)
        # light_model = self.repvgg_model_convert(self.lseRepFusNet)
        # fus_light = light_model(ir,vi)
        # l_gap = b1 * self.ssim(fus, fus_light) + b2 * self.l1(fus, fus_light)

        l_f = l_vi.mean() + l_ir.mean() + l_grad.mean() + l_bright.mean()
        # l_f = l_vi.mean() + l_ir.mean() + l_grad.mean()

        loss = l_f
        # backward
        self.opt_lseRepFusNet.zero_grad()
        loss.backward()
        self.opt_lseRepFusNet.step()
        # loss state
        state = {
            'loss': loss.item(),
            'l_mask':l_mask.item()
        }
        return state


    def eval(self, epoch):
        process_val = tqdm(enumerate(self.val_loader), disable=not self.config.debug)
        loss_total, loss_mask = 0., 0.
        for idx, sample in process_val:
            ir, vi = sample[0].to(self.environment_probe.device), sample[1].to(self.environment_probe.device)
            people_mask = sample[2].to(self.environment_probe.device)
            m3fd_mask = sample[3].to(self.environment_probe.device)
            loss = self.val_fusion(ir, vi, people_mask, m3fd_mask)
            loss_total += loss['loss']
            loss_mask += loss['l_mask']
        loss_avg = loss_total / len(self.val_loader)
        loss_mask_avg = loss_mask / len(self.val_loader)
        self.logging.info(f'val----------fuse: {loss_avg:03f}-----{loss_mask_avg:03f}')
        self.writer.add_scalar('Val/loss_avg', loss_avg, epoch)
        self.writer.add_scalar('Val/loss_mask_avg', loss_mask_avg, epoch)
        self.save(epoch)

    @torch.no_grad()
    def val_fusion(self, ir: Tensor, vi: Tensor, people_mask: Tensor, m3fd_mask: Tensor):
        self.lseRepNet.eval()
        mask = self.lseRepNet(ir)
        zeros = torch.zeros_like(ir)
        ones = torch.ones_like(ir)
        mean = torch.mean(ir)
        w = torch.where(ir > mean, ir, mean)
        for i in range(1):
            mean = torch.mean(w)
            w = torch.where(w > mean, w, mean)
        mask_lab = torch.where(w > torch.mean(w), ones, zeros)
        people_mask = torch.where(people_mask > zeros, ones, zeros)
        m3fd_mask = torch.where(m3fd_mask > zeros, ones, zeros)
        unite_mask = mask_lab + m3fd_mask
        unite_mask = torch.where(unite_mask > zeros, ones, zeros)
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        l_mask = b1 * self.ssim(mask, unite_mask * ir) + b2 * self.l1(mask, unite_mask * ir)
        mask_grad = self.gradient(mask)
        ir_grad = self.gradient(ir)
        l_mask_grad = b1 * self.mse(mask_grad, ir_grad) + b2 * self.l1(mask_grad, ir_grad)
        # mask_grad_bin = torch.where(mask_grad > mask_grad.mean(), ones, zeros)
        # ir_grad_bin = torch.where(ir_grad > ir_grad.mean(), ones, zeros)
        # l_mask_grad = b1 * self.mse(mask_grad_bin, ir_grad_bin) + b2 * self.l1(mask_grad_bin, ir_grad_bin)
        l_mask = l_mask.mean() + l_mask_grad.mean()


        self.lseRepNet.eval()
        self.lseRepFusNet.eval()
        mask = self.lseRepNet(ir)
        mask = torch.where(mask > torch.mean(mask), ones, zeros)
        # people_mask = torch.where(people_mask > zeros, ones, zeros)
        fus = self.lseRepFusNet(ir , vi )
        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1
        vi_grad = self.gradient(vi)
        ir_grad = self.gradient(ir)
        grad_max = torch.max(vi_grad, ir_grad)
        # l_vi = b1 * self.ssim(fus , vi * (1 - people_mask) + ir * people_mask) + b2 * self.l1(fus , vi * (1 - people_mask) + ir * people_mask )
        l_vi = b1 * self.ssim(fus , vi ) + b2 * self.l1(fus , vi )
        # l_ir = b1 * self.ssim(fus * mask, torch.max(ir, vi) * mask) + b2 * self.l1(fus * mask, torch.max(ir, vi) * mask)
        l_ir = b1 * self.ssim(fus * mask, ir * mask) + b2 * self.l1(fus * mask, ir * mask)
        l_bright = b1 * self.ssim(fus * people_mask, ir * people_mask) + b2 * self.l1(fus * people_mask, ir * people_mask)
        l_grad = b1 * self.ssim(self.gradient(fus), grad_max) + b2 * self.l1(self.gradient(fus), grad_max)

        # light_model = self.repvgg_model_convert(self.lseRepFusNet)
        # fus_light = light_model(ir,vi)
        # l_gap = b1 * self.ssim(fus, fus_light) + b2 * self.l1(fus, fus_light)

        l_f = l_vi.mean() + l_ir.mean() + l_grad.mean() + l_bright.mean()
        # l_f = l_vi.mean() + l_ir.mean() + l_grad.mean()

        loss = l_f
        # loss state
        state = {
            'loss': loss.item(),
            'l_mask': l_mask.item()

        }
        return state


    def save(self, epoch: int):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)
        cache = path / f'{epoch:03d}.pth'
        self.logging.info(f'save checkpoint to {str(cache)}')
        state = {
            'lseRepFusNet': self.lseRepFusNet.state_dict(),
            'lseRepNet': self.lseRepNet.state_dict(),
        }
        torch.save(state, cache)

    def gradient(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
        return u

    def gaussian(self, img, mean, std):
        b, c, h, w = img.size()
        noise = torch.randn([b, c, h, w]).to(img.device) * std + mean
        return noise

    def repvgg_model_convert(self, model:torch.nn.Module, save_path=None, do_copy=True):
        if do_copy:
            model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        return model





