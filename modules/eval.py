import time
from pathlib import Path
from typing import List

import torch
import torch.backends.cudnn
import torchvision.transforms
from kornia.filters import SpatialGradient
from torch import nn, Tensor
from torchvision.transforms import transforms

from tqdm import tqdm

from modules.image_pair import ImagePair
from thop import profile


class Eval:
    def __init__(self, net, device, cudnn: bool = True, half: bool = False, eval: bool = False):
        torch.backends.cudnn.benchmark = cudnn
        self.device = device
        self.half = half
        _ = net.half() if half else None
        _ = net.to(self.device)
        _ = net.eval() if eval else None
        self.net = net
        self.spatial = SpatialGradient('diff')
    @torch.no_grad()
    def __call__(self, ir_paths: List[Path], vi_paths: List[Path], dst: Path, color: bool = False):
        img_len= len(ir_paths)
        time_used_avg, flops_avg, params_avg = 0., 0., 0.
        p_bar = tqdm(enumerate(zip(ir_paths, vi_paths)), total=len(ir_paths))
        for idx, (ir_path, vi_path) in p_bar:
            assert ir_path.stem == vi_path.stem
            p_bar.set_description(f'fusing {ir_path.stem} | device: {str(self.device)}')
            pair = ImagePair(ir_path, vi_path)
            ir, vi = pair.ir_t, pair.vi_t
            ir, vi = [ir.half(), vi.half()] if self.half else [ir, vi]
            ir, vi = ir.to(self.device), vi.to(self.device)
            start = time.time()
            fus = self.net(ir.unsqueeze(0), vi.unsqueeze(0))[0].clip(0., 1.)
            time_used = time.time() - start
            if idx != 0:
                time_used_avg += (time_used / (img_len - 1))
                flops, params = profile(self.net, (ir.unsqueeze(0), vi.unsqueeze(0)))
                flops_avg += flops / (img_len - 1)
                params_avg += params / (img_len - 1)
            pair.save_fus(dst / ir_path.name, fus, color)

        print('FLOPs = ' + str(flops_avg / 1000 ** 3) + 'G')
        print('Params = ' + str(params_avg / 1000 ** 2) + 'M')
        print('Time = ' + str(time_used_avg) + 'S')

    @torch.no_grad()
    def getGradMap(self, ir_paths: List[Path], vi_paths: List[Path], dst: Path, color: bool = False):
        p_bar = tqdm(enumerate(zip(ir_paths, vi_paths)), total=len(ir_paths))
        for idx, (ir_path, vi_path) in p_bar:
            # assert ir_path.stem == vi_path.stem
            p_bar.set_description(f'fusing {ir_path.stem} | device: {str(self.device)}')
            pair = ImagePair(ir_path, vi_path)
            ir, vi = pair.ir_t, pair.vi_t
            ir, vi = [ir.half(), vi.half()] if self.half else [ir, vi]
            ir, vi = ir.to(self.device), vi.to(self.device)

            max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 可调整 kernel_size 和 padding
            max_pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # 可调整 kernel_size 和 padding
            fus_f = self.gradient(ir.unsqueeze(0))
            zeros = torch.zeros_like(ir)
            ones = torch.ones_like(ir)
            fus_f = torch.where(fus_f > fus_f.mean(), ones, zeros)
            for i in range(2):
                fus_f = -max_pool(-fus_f)
            for i in range(10):
                fus_f = max_pool2(fus_f)
            from scipy import ndimage as ndi
            fill_coins = ndi.binary_fill_holes(fus_f.squeeze(0).squeeze(0).cpu().numpy())
            toTensor = torchvision.transforms.ToTensor()
            fus = toTensor(fill_coins)
            fus = torch.where(fus == True, ones.cpu(), zeros.cpu())
            pair.save_fus(dst / ir_path.name, fus, color=False)

    def gradient(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
        return u