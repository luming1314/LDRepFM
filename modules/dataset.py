from __future__ import print_function
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
import os
from PIL import Image
import numpy as np



class DataSets(Data.Dataset):
    def __init__(self, root, transform=None, gray=True, partition='train'):

        self.files_VIS = glob(os.path.join(root, "M3FD", "VIS", '*.*'))
        self.files_IR = glob(os.path.join(root, "M3FD", "IR", '*.*'))
        self.files_Lab = glob(os.path.join(root, "M3FD", "Lab", '*.*'))

        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transform
        self.num_examples = len(self.files_VIS)

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

    def __len__(self):
        return len(self.files_VIS)

    def __getitem__(self, index):
        img_VIS = Image.open(self.files_VIS[index])
        img_IR = Image.open(self.files_IR[index])
        img_Lab = Image.open(self.files_Lab[index])

        if self.transform is not None:
            img_VIS = self.transform(img_VIS)
            img_IR = self.transform(img_IR)
            img_Lab = self.transform(img_Lab)

        if self.gray:
            img_VIS = img_VIS.convert('L')
            img_IR = img_IR.convert('L')
            img_Lab = img_Lab.convert('L')

        img_VIS = self._tensor(img_VIS)
        img_IR = self._tensor(img_IR)
        img_Lab = self._tensor(img_Lab)
        return img_IR, img_VIS, img_Lab