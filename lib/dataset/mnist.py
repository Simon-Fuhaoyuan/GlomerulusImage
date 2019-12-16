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


class Mnist(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_category = 10
        self.cfg = cfg
        self.root = root
        self.subroot = cfg.DATASET.SUBROOT
        self.is_train = is_train
        self.image_set = image_set
        self.transform = transform
        self.image_path = os.path.join(self.root, self.subroot, self.image_set, 'images')
        self.db = self._get_db()
        logger.info('=> Loading {} images for {}'.format(self.image_set, self.subroot))
        logger.info('=> num_images: {}'.format(len(self.db)))
    
    def _get_db(self):
        db = []

        labels_path = os.path.join(self.root, self.subroot, self.image_set, 'labels.txt')
        fp = open(labels_path)
        labels = fp.readlines()
        for i in range(len(labels)):
            img = str(i) + '.npy'
            label = int(labels[i])
            db.append((img, label))

        return db
    
    def __len__(self,):
        return len(self.db)
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = os.path.join(self.image_path, db_rec[0])
        
        label = db_rec[1]
        image = np.load(image_file)

        if self.transform:
            raw_image = np.swapaxes(image, 1, 2)
            raw_image = np.swapaxes(raw_image, 0, 1)
            non_norm_image = torch.from_numpy(raw_image).float()
            norm_image = self.transform(non_norm_image)

        image = torch.from_numpy(image)

        meta = {
            'image': db_rec[0],
            'label': label
        }

        return norm_image, label, meta, image
