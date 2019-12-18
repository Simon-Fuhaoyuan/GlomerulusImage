from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.nn import functional as F

class GradCAM(object):
    def __init__(self, i, cfg, model_dict, input, output, fmaps, target):
        self.init_image_size = cfg.MODEL.IMAGE_SIZE
        self.output_dir = cfg.OUTPUT_DIR
        self.subroot = cfg.DATASET.SUBROOT
        self.fmap_size = fmaps.size()[2:]
        self.fmaps = fmaps.cpu().numpy()
        self.input = input.cpu().numpy()
        #self.input = np.swapaxes(self.input, 1, 2)
        #self.input = np.swapaxes(self.input, 2, 3)
        self.output = output.cpu().numpy()
        self.grads = self._get_grads(model_dict).cpu().numpy()
        self.target = target.cpu().numpy()
        self.group = i
    
    def _get_grads(self, model_dict):
        model = torch.load(model_dict)
        grads = model['fullyc.weight']
        
        return grads
    
    def save(self, gcam, raw_image, id, category_id):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        if(gcam.max() != 0):
           gcam = gcam / gcam.max() * 255.0
        output_dir = os.path.join(self.output_dir, self.subroot, 'gradcam')
        filename = '/group%d_num_%d_cat%d.png' %(self.group, id, category_id)
        filename_init = '/group%d_num_%d.png' %(self.group, id)
        cv2.imwrite(output_dir + filename, np.uint8(gcam))
        cv2.imwrite(output_dir + filename_init, np.uint8(raw_image))
        # np.save(output_dir + filename, gcam)
        # np.save(output_dir + filename_init, raw_image)

    def generate(self):
        for i in range(self.fmaps.shape[0]): # batch size
            gcam = np.zeros((self.fmap_size[0], self.fmap_size[1]))
            category_id = np.argmax(self.output[i])
            target_id = self.target[i]
            if not category_id == target_id:
                continue
            raw_image = self.input[i]
            for j in range(self.fmaps.shape[1]):
                fmap = self.fmaps[i][j]
                grad = self.grads[category_id][j]
                gcam += fmap * grad

            gcam -= gcam.min()
            if(gcam.max() != 0):
                gcam /= gcam.max()
            gcam = cv2.resize(gcam, (self.init_image_size[0], self.init_image_size[1]))
            self.save(gcam, raw_image, i, category_id)
        print('Group %d finish!' % self.group)
