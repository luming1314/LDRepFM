import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', metavar='MODE', default='pm', choices=['pm', 'mm'], help='pm is only people mask; mm is all m3fd mask')
config = parser.parse_args()
ONLY_PEOPLE = True if config.mode == 'pm' else False
root_path = r'./data/train/M3FD'
xml_name = 'Annotations'
xml_file = os.path.join(root_path, xml_name)
l = ['People', 'Bus', 'Car', 'Motorcycle', 'Lamp', 'Truck']
classes_id = {'People': '0', 'Bus': '5', 'Car': '2', 'Motorcycle': '3', 'Lamp': '9', 'Truck': '7'}


def convert(box,dw,dh):
    x=(box[0]+box[2])/2.0
    y=(box[1]+box[3])/2.0
    w=box[2]-box[0]
    h=box[3]-box[1]

    x=x/dw
    y=y/dh
    w=w/dw
    h=h/dh

    return x,y,w,h

def f(name_id):
    path = os.path.join(xml_file, '%s.xml' % name_id)
    xml_o = open(path)
    # xml_o=open(r'./test/Annotations\%s.xml'%name_id)
    # txt_o=open(r'./test/labels\%s.txt'%name_id,'w')

    pares=ET.parse(xml_o)
    root=pares.getroot()
    objects=root.findall('object')
    size=root.find('size')
    dw=int(size.find('width').text)
    dh=int(size.find('height').text)

    mask = torch.rand(dh, dw)
    mask = torch.zeros_like(mask)
    for obj in objects :
        c = l.index(obj.find('name').text)
        if ONLY_PEOPLE and c != 0:
            continue
        # c=classes_id[obj.find('name').text]
        bnd=obj.find('bndbox')

        b=(float(bnd.find('xmin').text),float(bnd.find('ymin').text),
           float(bnd.find('xmax').text),float(bnd.find('ymax').text))



        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if j >= b[0] and j<= b[2] and i >= b[1] and i<= b[3]:
                    mask[i][j] = 1.
        # x,y,w,h=convert(b,dw,dh)
        #
        # write_t="{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(c,x,y,w,h)
        # txt_o.write(write_t)
    image = Image.fromarray(np.uint8(mask))
    path = os.path.join(root_path, 'la' if ONLY_PEOPLE else 'lb', name_id + '.png')
    image.save(path)
    xml_o.close()
    # txt_o.close()

name = glob.glob(os.path.join(xml_file,"*.xml"))
for i in name :
    name_id = os.path.basename(i)[:-4]
    f(name_id)
