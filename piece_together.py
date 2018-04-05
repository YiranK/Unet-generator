import numpy as np
import scipy
import skimage
from skimage.io import imread
from PIL import Image
from skimage import exposure
import random
import os
import cv2


def get_piecename_from_line(line,img_dir):
    img_name = line.strip().split(' ')[0]+'_fakeB.jpg'
    img_name = os.path.join(img_dir, img_name)
    return img_name

def get_imgname_from_line(line,img_dir):
    img_name = line.strip().split(' ')[0]
    img_name = img_name[:-5]+'.jpg'
    img_name = os.path.join(img_dir, img_name)
    return img_name


dataset_path = '/data/unet/synv19box.txt'

patch_version = 'Patch_shrink_bbox'
patch_version = 'patch_big_0329'
patch_version = 'patchSYN_v13_0329'
patch_version = 'patchSYN_v19_0329'


# dataset_path = './localGMU_test.txt'
# dataset_path = './localGMU_train.txt'

phase = 'GMUtrain'
phase = 'SYNv19'

GMUfore_dir='/data/GMU/{0}/{1}_fore_patch/'.format(patch_version, phase)
GMUback_dir='/data/GMU/{0}/{1}_back_patch/'.format(patch_version, phase)
GMUmask_dir='/data/GMU/{0}/{1}_mask_patch/'.format(patch_version, phase)
GMUgt_dir='/data/GMU/{0}/{1}_gt_patch/'.format(patch_version, phase)

# GMUfore_dir='/data/unet/synv13big/'
# GMUback_dir='/data/unet/synv13big/'
# GMUmask_dir='/data/unet/synv13big/'
# GMUgt_dir='/data/unet/synv13big/'

dirs = [GMUfore_dir, GMUback_dir, GMUmask_dir, GMUgt_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# height = 1080
# width = 1920
height = 480
width = 640

#v1 v13noise
img_dir = '/data/GMU/syn_v19'
piece_dir = '/data/GMU/unet0330_vgg2_synv13noise__patchSYN_v19_0329'
save_dir = '/data/GMU/join_synv19_unet0330_vgg2_synv13noise'

#v2 gmubig
# img_dir = '/data/GMU/syn_v19'
# piece_dir = '/data/GMU/unet0330_vgg2_big__patchSYN_v19_0329'
# save_dir = '/data/GMU/join_synv19_unet0330_vgg2_big'

#v3 gmusmall
# img_dir = '/data/GMU/syn_v19'
# piece_dir = '/data/GMU/unet0327_vgg5__patchSYN_v19_0329'
# save_dir = '/data/GMU/join_synv19_unet0327_vgg5'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with open(dataset_path,'r') as f:
    lines= f.readlines()
    i = 79197
    print len(lines)
    boxs = []
    while i < len(lines):
        # img_name = lines[i].strip().split(' ')[0]
        # img_name = img_name[:-5]+'.png'
        # img_name = os.path.join(img_dir, img_name)
        img_name = get_imgname_from_line(lines[i],img_dir)
        img = imread(img_name)

        while i < len(lines) and get_imgname_from_line(lines[i],img_dir) == img_name:
            print lines[i]
            piece_name = get_piecename_from_line(lines[i], piece_dir)
            piece = imread(piece_name)

            boxs = lines[i].strip().split(' ')[1:]
            boxs = [int(x) for x in boxs]
            ymin, ymax, xmin, xmax = boxs
            h = ymax - ymin
            w = xmax - xmin
            piece = scipy.misc.imresize(piece, [w,h])
            img[ymin:ymax, xmin:xmax] = cv2.addWeighted(piece, 0.7, img[ymin:ymax, xmin:xmax], 0.3, 0)
            i = i + 1
        # print lines[i],i
        print os.path.join(save_dir,img_name.split('/')[-1])
        scipy.misc.imsave(os.path.join(save_dir,img_name.split('/')[-1]),img)
