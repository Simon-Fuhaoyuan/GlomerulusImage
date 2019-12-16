import cv2
import os
from tqdm import tqdm

label_file = 'labels.txt'
new_label_file = 'labels_new.txt'
image_prefix = 'images/'
new_image_prefix = 'images_new/'

times = {
    0: 3,
    1: 1,
    2: 3,
    3: 3, 
    4: 1,
    5: 25,
    6: 1
}

idx = 0

with open(label_file) as fp:
    labels_init = fp.readlines()
fw = open(new_label_file, 'w')

if not os.path.exists(new_image_prefix[:-1]):
    os.makedirs(new_image_prefix[:-1])

for init_idx, label in tqdm(enumerate(labels_init)):
    image = image_prefix + str(init_idx) + '.png'
    label = int(label)
    img = cv2.imread(image)
    img_flip = cv2.flip(img, 1)
    for i in range(times[label]):
        cv2.imwrite(new_image_prefix + str(idx) + '.png', img)
        fw.write(str(label) + '\n')
        idx += 1
        cv2.imwrite(new_image_prefix + str(idx) + '.png', img_flip)
        fw.write(str(label) + '\n')
        idx += 1

fp.close()
fw.close()
