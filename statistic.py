import numpy as np
import os

root = '../../../disk4/data/mnist/'
subroot = ['random_3']
image_set = 'test/images/'

for i in range(len(subroot)):
    path = os.path.join(root, subroot[i], image_set)
    images = os.listdir(path)
    mean = np.array([0, 0, 0], dtype='float64')
    std = np.array([0, 0, 0], dtype='float64')
    for image in images:
        image = os.path.join(path, image)
        initImage = np.load(image)
    #    image_padding = np.zeros((64, 64, 3))
    #    for h in range(initImage.shape[0]):
    #        for w in range(initImage.shape[1]):
    #            image_padding[h + 10][w + 10] = initImage[h][w]
        
        mean += initImage.mean(axis=(0,1))
        std += initImage.var(axis=(0,1))
    mean = mean / len(images)
    std = std / len(images)
    for c in range(3):
        std[c] = std[c] ** 0.5
    print(subroot[i], 'mean:', mean)
    print(subroot[i], 'std:', std)
