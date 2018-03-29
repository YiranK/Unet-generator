import numpy as np
import scipy
import skimage
from skimage.io import imread
from skimage import exposure
import random
import os

patch_version = 'Patch_shrink_bbox'
patch_version = 'patch_big_0329'

dataset_path = './synv13/local_synv13.txt'
dataset_path = './localGMU_train.txt'

GMUfore_dir='/data/GMU/{}/GMUtest_fore_patch/'.format(patch_version)
GMUback_dir='/data/GMU/{}/GMUtest_back_patch/'.format(patch_version)
GMUmask_dir='/data/GMU/{}/GMUtest_mask_patch/'.format(patch_version)
GMUgt_dir='/data/GMU/{}/GMUtest_gt_patch/'.format(patch_version)

# GMUfore_dir='/data/unet/synv13big/'
# GMUback_dir='/data/unet/synv13big/'
# GMUmask_dir='/data/unet/synv13big/'
# GMUgt_dir='/data/unet/synv13big/'

dirs = [GMUfore_dir, GMUback_dir, GMUmask_dir, GMUgt_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

height = 1080
width = 1920
# height = 480
# width = 640

count = 0
with open(dataset_path) as f:
    while True:
        count += 1
        line = f.readline()
        if line == '':
            break
        # if count > 20472:
        #     break
        # if count > 20227:
        #     break
        if line.find('.png')>-1:
            #foreground = np.zeros((1080,1920,3))
            img = imread(line.strip())
            #background = img.copy()
            img_name = line.strip().split('/')[-3]+'_'+line.strip().split('/')[-1]
            for i in range(5):
                t = f.readline()
            #t = int(f.readlines(4)[-1].strip())
            t = int(t)
            print img_name, t
            for i in range(t):
                gt_tmp = img.copy()
                fore_tmp = img.copy()
                bbox = f.readline().strip().split(' ')
                bbox = bbox[-4:]
                int_bbox = [int(x) for x in bbox]

                ymin = max(0, int_bbox[1]-4)
                ymax = min(height, int_bbox[3]+4)
                xmin = max(0, int_bbox[0]-4)
                xmax = min(width, int_bbox[2]+4)


                w = xmax - xmin
                h = ymax - ymin

                patch_size = 256
                while w > patch_size or h > patch_size or w * h > patch_size * patch_size * 2/3:
                    patch_size += 32

                # while (w < patch_size/3 or h < patch_size/3) and patch_size > 20 and w < patch_size-8 and h < patch_size-8:
                #     patch_size -= 8

                # 0329
                while (w * h < patch_size * patch_size /2) and patch_size > 20 and w < patch_size-8 and h < patch_size-8:
                    patch_size -= 8

                gt = np.zeros((patch_size, patch_size, 3))
                fore = np.zeros((patch_size, patch_size, 3))
                back = np.zeros((patch_size, patch_size, 3))
                mask = np.zeros((patch_size, patch_size))
                pad_w = patch_size - w
                pad_h = patch_size - h


                if xmin < width-xmax:
                    pad_left = random.randint(0, min(pad_w, xmin))
                    pad_right = pad_w - pad_left
                else:
                    pad_right = random.randint(0, min(pad_w, width - xmax))
                    pad_left = pad_w - pad_right

                if ymin < height-ymax:
                    pad_up = random.randint(0, min(pad_h, ymin))
                    pad_bottom = pad_h - pad_up
                else:
                    pad_bottom = random.randint(0, min(pad_h, height - ymax))
                    pad_up = pad_h - pad_bottom

                pad_xmin = xmin - pad_left
                pad_xmax = xmax + pad_right
                pad_ymin = ymin - pad_up
                pad_ymax = ymax + pad_bottom


                gt = gt_tmp[pad_ymin:pad_ymax,pad_xmin:pad_xmax]
                fore[pad_up:pad_up+h,pad_left:pad_left+w] = fore_tmp[ymin:ymax,xmin:xmax] # only for bounding box
                mask[pad_up:pad_up+h,pad_left:pad_left+w] = 1
                back = gt.copy()
                back[pad_up:patch_size-pad_bottom,pad_left:patch_size-pad_right] = 0

                if patch_size != 256 or patch_size != 256:
                    print gt.shape, pad_ymin, pad_ymax, pad_up, pad_bottom
                    gt = scipy.misc.imresize(gt, [256,256])
                    fore = scipy.misc.imresize(fore,[256,256])
                    back = scipy.misc.imresize(back,[256,256])
                    mask = scipy.misc.imresize(mask,[256,256])


                nm = random.randint(0, 2)
                if nm > 2:
                    noise_mode = ['gaussian', 'poisson']
                    fore = skimage.util.random_noise(fore, mode=noise_mode[nm])
                degree = round(random.random()*3+0.2, 2)
                fore = exposure.adjust_gamma(fore, degree)
                scipy.misc.imsave(GMUfore_dir + img_name[:-4] + '_box' + str(i) + '_fore.png', fore)
                scipy.misc.imsave(GMUback_dir + img_name[:-4] + '_box' + str(i) + '_back.png', back)
                scipy.misc.imsave(GMUmask_dir + img_name[:-4] + '_box' + str(i) + '_mask.png', mask)
                scipy.misc.imsave(GMUgt_dir + img_name[:-4] + '_box' + str(i) + '_gt.png', gt)

                # foreground[ymin:ymax,xmin:xmax] = temp[ymin:ymax,xmin:xmax]
                # background[ymin:ymax,xmin:xmax] = 0

            #scipy.misc.imsave(GMUfore_dir+img_name[:-4]+'_fore.png', foreground)
            #scipy.misc.imsave(GMUback_dir+ img_name[:-4] + '_back.png', background)