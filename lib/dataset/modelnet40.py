from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


class ModelNet40(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_category = 40
        self.cfg = cfg
        self.root = root
        self.subroot = cfg.DATASET.SUBROOT
        self.subroot_list = []
        self.is_train = is_train
        self.image_set = image_set
        self.transform = transform
        self.image_path = os.path.join(self.root, self.subroot, self.image_set, 'images')
        if self.subroot == 'mix':
            if self.is_train == True:
                self.image_path = []
                self.subroot_list = ['rgb', 'gbr', 'brg']
                for i in range(3):
                    self.image_path.append(os.path.join(self.root, self.subroot_list[i], self.image_set, 'images'))
            else:
                self.subroot = 'rgb'
                self.image_path = os.path.join(self.root, self.subroot, self.image_set, 'images')
        self.db = self._get_db()
        logger.info('=> Loading {} images for {}'.format(self.image_set, self.subroot))
        logger.info('=> num_images: {}'.format(len(self.db)))
    
    def _get_db(self):
        db = []
        if not isinstance(self.image_path, list):
            labels_path = os.path.join(self.root, self.subroot, self.image_set, 'labels.txt')
            fp = open(labels_path)
            labels = fp.readlines()
            for i in range(len(labels)):
                img = str(i) + '.npy'
                label = int(labels[i])
                db.append((img, label))

        else:
            for j in range(3):
                labels_path = os.path.join(self.root, self.subroot_list[j], self.image_set, 'labels.txt')
                fp = open(labels_path)
                labels = fp.readlines()
                for i in range(len(labels)):
                    img = str(i) + '.npy'
                    label = int(labels[i])
                    db.append((img, label, j))

        return db
    
    def __len__(self,):
        return len(self.db)
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        if not isinstance(self.image_path, list):
            image_file = os.path.join(self.image_path, db_rec[0])
        else:
            image_file = os.path.join(self.image_path[db_rec[2]], db_rec[0])
        
        label = db_rec[1]
        init_image = np.load(image_file)
        

        if self.cfg.DATASET.SUBROOT in ['rgb_32', 'rgb_1024', 'random_1024', 'random_1024_2']:
            padding = False
        elif not self.cfg.DATASET.PADDING:
            padding = False
        else:
            padding = True
        
        if self.cfg.DATASET.SUBROOT in ['brg', 'gbr', 'rgb_noise_50'] or (isinstance(self.image_path, list) and db_rec[2] > 0):
            init_image = init_image + 1
            init_image = init_image / 2
        
        if padding:
            image = np.zeros((64, 64, 3))
            for i in range(init_image.shape[0]):
                for j in range(init_image.shape[1]):
                    image[10 + i][10 + j] = init_image[i][j]
        else:
            #print('No padding!')
            image = init_image

        if self.transform and not isinstance(self.image_path, list):
            raw_image = np.swapaxes(image, 1, 2)
            raw_image = np.swapaxes(raw_image, 0, 1)
            non_norm_image = torch.from_numpy(raw_image).float()
            #print(type(non_norm_image))
            norm_image = self.transform(non_norm_image)
        else:
            raw_image = np.swapaxes(image, 1, 2)
            raw_image = np.swapaxes(raw_image, 0, 1)
            non_norm_image = torch.from_numpy(raw_image).float()
            #print(type(non_norm_image))
            if db_rec[2] == 0:
                norm_image = self.transform(non_norm_image)
            else:
                if db_rec[2] == 1:
                    mean = [0.247, 0.247, 0.247]
                    std = [0.276, 0.281, 0.275]
                else:
                    mean = [0.247, 0.247, 0.247]
                    std = [0.281, 0.275, 0.276]
                normalize = transforms.Normalize(mean=mean, std=std)
                transform = transforms.Compose([normalize,])
                norm_image = transform(non_norm_image)
        image = torch.from_numpy(image)

        meta = {
            'image': db_rec[0],
            'label': label
        }

        return norm_image, label, meta, image
