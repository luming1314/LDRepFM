from pathlib import Path
from typing import List

import torch
import torch.backends.cudnn
import torchvision.transforms
from torchvision.transforms import transforms

from tqdm import tqdm

from modules.image_pair import ImagePair


class Eval:
    def __init__(self, net, device, cudnn: bool = True, half: bool = False, eval: bool = False):
        torch.backends.cudnn.benchmark = cudnn
        self.device = device
        self.half = half
        _ = net.half() if half else None
        _ = net.to(self.device)
        _ = net.eval() if eval else None
        self.net = net

    @torch.no_grad()
    def __call__(self, ir_paths: List[Path], vi_paths: List[Path], dst: Path, color: bool = False):
        p_bar = tqdm(enumerate(zip(ir_paths, vi_paths)), total=len(ir_paths))
        for idx, (ir_path, vi_path) in p_bar:
            # assert ir_path.stem == vi_path.stem
            p_bar.set_description(f'fusing {ir_path.stem} | device: {str(self.device)}')
            pair = ImagePair(ir_path, vi_path)
            ir, vi = pair.ir_t, pair.vi_t
            ir, vi = [ir.half(), vi.half()] if self.half else [ir, vi]
            ir, vi = ir.to(self.device), vi.to(self.device)
            fus = self.net(ir.unsqueeze(0), vi.unsqueeze(0))[0].clip(0., 1.)

            # fus = torch.max(ir,vi)



            pair.save_fus(dst / ir_path.name, fus, color)