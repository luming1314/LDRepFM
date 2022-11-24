import numpy as np
from skimage.transform import (hough_line, hough_line_peaks, hough_circle,
hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.data import astronaut
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage import img_as_float, morphology
from skimage.morphology import skeletonize
from skimage import data, img_as_float
import matplotlib.pyplot as pylab
from matplotlib import cm, pyplot as plt
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.segmentation import watershed
from skimage import io
from scipy import ndimage as ndi
from pathlib import Path
# coins = data.coins()
from tqdm import tqdm


def img_filter(x: Path) -> bool:
 return x.suffix in ['.png', '.bmp', '.jpg']


root = Path('./data/train/M3FD')
# 转灰度
# ir_paths = [x for x in sorted((root / 'IR').glob('*')) if img_filter]
# vi_paths = [x for x in sorted((root / 'IR').glob('*')) if img_filter]
# p_bar = tqdm(enumerate(zip(ir_paths, vi_paths)), total=len(ir_paths))
# for idx, (ir_path, vi_path) in p_bar:
#  coins = io.imread(str(ir_path), as_gray=True)
#  io.imsave("./data/train/M3FD/ir_gray/" + ir_path.name, coins)



ir_paths = [x for x in sorted((root / 'ir_gray').glob('*')) if img_filter]
vi_paths = [x for x in sorted((root / 'ir_gray').glob('*')) if img_filter]
p_bar = tqdm(enumerate(zip(ir_paths, vi_paths)), total=len(ir_paths))
for idx, (ir_path, vi_path) in p_bar:
 coins = io.imread(str(ir_path))
 elevation_map = sobel(coins)
 markers = np.zeros_like(coins)
 markers[coins < 50] = 1
 markers[coins > 150] = 2
 segmentation = watershed(elevation_map, markers)
 segmentation = ndi.binary_fill_holes(segmentation - 1)
 io.imsave("./data/train/M3FD/SLab/" + ir_path.name, segmentation)









