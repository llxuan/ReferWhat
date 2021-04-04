from __future__ import print_function

import numpy as np
import argparse
import os.path as osp
import h5py
import json
import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import vgg16, resnet152
import torchvision.transforms as transforms

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

__normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

torch.set_grad_enabled(False)

# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    #return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))
    [x,y,w,h] = boxes
    return [x,y,x+w,y+h]

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    [x1,y1,x2,y2] = boxes
    #return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))
    return [x1,y1, x2-x1,y2-y1]

def _clip_boxes(boxes, image_size):
    """Clip boxes to image boundaries."""
    boxes[0] = np.maximum(boxes[0],0)
    boxes[1] = np.maximum(boxes[1], 0)
    boxes[2] = np.minimum(boxes[2], image_size[0])
    boxes[3] = np.minimum(boxes[3], image_size[1])

    return boxes

def extract_feats(image, cnn):
    scale_size = 224
    normalize = __normalize
    img_transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
    if image.size[1] == 0:
        print(image.size)
    img = img_transform(image).unsqueeze(0)
    img_feat = cnn(img.cuda()).squeeze()
    return img_feat

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name:  refcoco, refcoco+')
    parser.add_argument('--cnn', type=str, default='resnet152', help='Type of CNN')
    parser.add_argument('--image_root', type=str, default='/home/ntr/dataset/train2014', help='path of coco images')
    parser.add_argument('--data_root', type=str, default='../../data/', help='path of dataset root')

    args = parser.parse_args()

    IMAGE_DIR = args.image_root
    write_path = args.data_root + args.dataset
    data_json = osp.join(write_path, 'instances.json')
    # construct  imange_id , region_id to image_h5_id , region_h5_id map dict
    img_h5_id = {}
    region_h5_id = {}

    data = json.load(open(data_json,'r'))
    images = data['images']
    regions = data['annotations']
    len_imgs = len(images)
    len_regions = len(regions)

    if args.cnn == 'vgg16':
        vgg = vgg16(pretrained=True).cuda().eval()
        vgg_feats = nn.Sequential(*list(vgg.features.children()))
        vgg_classifier= nn.Sequential(*list(vgg.classifier.children())[:4])
        vgg_fc7 = nn.Sequential(*[vgg_feats, vgg.avgpool, Flatten(), vgg_classifier])
        cnn = vgg_fc7
        feat_file_name = 'feats_vgg16_fc7.h5'
        id_file_name = 'id_to_h5_vgg16_fc7.json'
        feats_dim = 4096
    elif args.cnn == 'resnet152':
        # For CNN is ResNet152
        resnet = resnet152(pretrained=True).cuda().eval()
        cnn = nn.Sequential(*(list(resnet.children())[:-1]))
        feat_file_name = 'feats_resnet152.h5'
        id_file_name = 'id_to_h5_resnet152.json'
        feats_dim = 2048
    else:
        raise TypeError ('CNN type is not valid')

    feat_file = osp.join(write_path, 'feats', feat_file_name)
    # write
    f = h5py.File(feat_file, 'w')
    image_feats_set = f.create_dataset('image_feats', (len_imgs, feats_dim), dtype=np.float32)
    region_feats_set = f.create_dataset('region_feats', (len_regions, feats_dim), dtype=np.float32)

    image_file_dict = {}

    # extract image feats
    for i, image_info in enumerate(images):
        image_file = image_info['file_name']
        image = Image.open(osp.join(IMAGE_DIR, image_file)).convert('RGB')
        image_feat = extract_feats(image, cnn)

        img_h5_id[image_info['id']] = i
        image_file_dict[image_info['id']] = image_file
        image_feats_set[i] = image_feat.detach().cpu().numpy()
        if i % 100 == 0:
            print(args.dataset, ':', i, 'images finish feature extract')

    # extract region feats
    for j, region_info in enumerate(regions):
        image_file = image_file_dict[region_info['image_id']]
        image = Image.open(osp.join(IMAGE_DIR, image_file)).convert('RGB')
        image_size = image.size
        region_box = region_info['bbox']
        ann_id = region_info['id']
        ann_box = xywh_to_xyxy(region_box)  # [x1,y1,x2,y2]
        ann_box = _clip_boxes(ann_box, image_size)
        crop_img = image.crop([ann_box[0], ann_box[1], ann_box[2], ann_box[3]])  # [x1,y1,x2,y2]
        if crop_img.size[1] <= 0 or crop_img.size[0] <= 0:
            print(args.dataset, ':', 'candidate:', ann_id, ' region size:', crop_img.size, ' is throw away')
            continue
        region_feat = extract_feats(crop_img, cnn)
        region_h5_id[ann_id] = j

        region_feats_set[j] = region_feat.detach().cpu().numpy()
        if j % 1000 == 0:
            print(args.dataset, ':', j, 'region finish feature extract')

    print(args.dataset, ':', i, 'images finish feature extract')
    print(args.dataset, ':', j, 'regions finish feature extract')
    f.close()

    f.close()
    print('%s writtern.' % feat_file)

    id_file = osp.join(write_path, 'feats', id_file_name)

    hf = open(id_file, 'w')
    json.dump({'image_h5_id': img_h5_id,
               'region_h5_id': region_h5_id}, hf)

    hf.close()
    print('%s writtern.' % id_file)








