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


class Glomerulus(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_category = 7
        self.cfg = cfg
        self.root = root
        self.is_train = is_train
        self.image_set = image_set
        self.transform = transform
        self.image_path = os.path.join(self.root, self.image_set, 'images')
        self.db = self._get_db()
        logger.info('=> Loading {} images'.format(self.image_set))
        logger.info('=> num_images: {}'.format(len(self.db)))

    def _get_db(self):
        db = []
        labels_path = os.path.join(self.root, self.image_set, 'labels.txt')
        fp = open(labels_path)
        labels = fp.readlines()
        for i in range(len(labels)):
            img = str(i) + '.png'
            label = int(labels[i])
            db.append((img, label))

        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        label = db_rec[1]
        image_file = os.path.join(self.image_path, db_rec[0])
        init_image = cv2.imread(image_file)

        image_size = (self.cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[1])
        resized_image = cv2.resize(init_image, image_size)

        if self.transform is not None:
            norm_image = self.transform(resized_image)
        else:
            norm_image = resized_image
        resized_image = torch.from_numpy(resized_image)

        meta = {
            'image': db_rec[0],
            'label': label
        }

        return norm_image, label, meta, resized_image
