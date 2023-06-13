# LDRepFM
**LDRepFM: A Real-time End-to-End Visible and Infrared Image Fusion Model Based on Layer Decomposition and Re-parameterization**


- [*[Paper]*](https://doi.org/10.1109/TIM.2023.3280496)

<img src="/assets/figure_1.png"/>

## Recommended Environment
 - [ ] python  3.8
 - [ ] torch  1.11.0
 - [ ] torchvision 0.12.0
## To Test
**Pre-training weights have been uploaded**

**Ours-training** `python fuse.py --mode train`

**Ours-inference** `python fuse.py --mode deploy`
## To Train
Firstly, you need to download the [M3FD](https://github.com/JinyuanLiu-CV/TarDAL) dataset.

Secondly, clone the folders into our repository as
```
  data
  ├── train
  |   ├── M3FD
  |   |   ├──Annotations
  |   |   |  ├── 01863.xml
  |   |   |  └── ...
  |   |   ├──ir
  |   |   |  ├── 01863.png
  |   |   |  └── ...
  |   |   ├──vi
  |   |   |  ├── 01863.png
  |   |   |  └── ...
  |   |   ├──la
  |   |   |  ├── 01863.png
  |   |   |  └── ...
  |   |   ├──lb
  |   |   |  ├── 01863.png
  |   |   |  └── ...
      └── ...
```
Thirdly, run `python lab_xml2png.py --mode pm` to generate mask images in the **la**  folders.

Fourthly, run `python lab_xml2png.py --mode mm` to generate mask images in the **lb** folders.

Moreover, if you want to view the mask images under the **la** or **lb** folders. We provide a method using MATLAB. Of course, you can also use other methods.

```
I = imread('./data/train/M3FD/la/00000.png');
imagesc(I);
```
Finally, run `python train.py --gpus -1` to train LDRepFM.
## Any Question
If you have any other questions about the code, please email `luming@stu.jiangnan.edu.cn`
## Citation
If this work has been helpful to you, please feel free to cite our paper!

```
@ARTICLE{10138238,
  author={Lu, Ming and Jiang, Min and Kong, Jun and Tao, Xuefeng},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={LDRepFM: A Real-Time End-to-End Visible and Infrared Image Fusion Model Based on Layer Decomposition and Re-Parameterization}, 
  year={2023},
  volume={72},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2023.3280496}}
```
