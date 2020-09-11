import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from torch import optim
from PIL import Image
from torch.autograd import Function, Variable
#import matplotlib.pyplot as plt
#import matplotlib
from torchvision import transforms
import glob
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import cv2
class Flip(object):
    """
    Flip the image left or right for data augmentation, but prefer original image.
    """

    def __init__(self, ori_probability=0.60):
        self.ori_probability = ori_probability

    def __call__(self, sample):
        if random.uniform(0, 1) < self.ori_probability:
            return sample
        else:
            img, label = sample['img'], sample['label']
            img_flip = img[:, :, ::-1]
            label_flip = label[:, ::-1]

            return {'img': img_flip, 'label': label_flip}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label ,index,d,w,h= sample['img'], sample['label'],sample['index'],sample['d'],sample['w'],sample['h']

        return {'img': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                'label': torch.from_numpy(label.copy()).type(torch.FloatTensor),
                'index':index,
                'd':d,
                'w':w,
                'h':h}


# the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_masks, transforms=None):
        self.image_masks = image_masks
        self.transforms = transforms

    def __len__(self):  # return count of sample we have
        return len(self.image_masks)

    def __getitem__(self, index):
        image = self.image_masks[index][0]  # H, W, C
        mask = self.image_masks[index][1]
        ii = self.image_masks[index][2]
        d = self.image_masks[index][3]
        w = self.image_masks[index][4]
        h = self.image_masks[index][5]


        #image = np.transpose(image, axes=[2, 0, 1])  # C, H, W

        sample = {'img': image, 'label': mask, 'index': ii, 'd': d, 'w': w, 'h': h}
        if transforms:
            sample = self.transforms(sample)

        return sample
# def collate_pool(dataset_list):
#     batch_img=[]
#     batch_mask=[]
#     for i, sample in enumerate(dataset_list):
#         print()
#         batch_img.append(sample['img'])
#         batch_mask.append(sample['label'])
#
#     batch_sample={'img':torch.cat(batch_img,dim=0),'label':torch.cat(batch_mask,dim=0)}
#
#     return batch_sample
# try to split the whole train dataset into train and validation, and match the train image path with the
# corresponding label path
def split_train_val(image_paths, mask_paths, train_size):
    img_paths_dic = {}
    mask_paths_dic = {}
    len_data = len(image_paths)
    print('total len:', len_data)
    for i in range(len(image_paths)):
        #去除.png
        img_paths_dic[os.path.basename(image_paths[i])[:-4]] = image_paths[i]

    for i in range(len(mask_paths)):
        mask_paths_dic[os.path.basename(mask_paths[i])[:-9]] = mask_paths[i]

    img_mask_list = []
    for key in img_paths_dic:
        img_mask_list.append((img_paths_dic[key], mask_paths_dic[key]))

    train_img_mask_paths = img_mask_list[:int(len_data * train_size)]
    val_img_mask_paths = img_mask_list[int(len_data * train_size):]
    return train_img_mask_paths, val_img_mask_paths


# read in the image and label pair, and then resize them from 1280*1918 to 80*100 by consideration of
# your computer memory limitation
def preprocess_image(image_mask_paths):
    img_mask_list = []
    new_h, new_w = 80, 100#80,100 288,384 96,72

    for i in tqdm(range(len(image_mask_paths))):
        # cv2 cannot read .gif files
        # Use Image.open() to open image and mask, then convert them to np arrays
        # For images use float32, mask use uint8
        # Normalize img to range (0,1)
        img = np.array(Image.open(image_mask_paths[i][0]), np.float32) / 255.0
        #mask = np.array(Image.open(image_mask_paths[i][1]), np.uint8)
        mask=cv2.imread(image_mask_paths[i][1],0)

        # Use cv2 to resize images to 80x100, use INTER_CUBIC interpolation
        img_resize = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        mask_resize = np.uint8(cv2.resize(mask, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC))
        mask_resize[mask_resize>0]=1
        img_mask_list.append((img_resize, mask_resize))
    return img_mask_list
# save the data into pickle file and you can just reload this file, which can help you avoid reading the image
# file again in the future, since reading in image file from hard drive would take quite a long time
def pickle_store(file_name, save_data):
    fileObj = open(file_name, 'wb')
    pickle.dump(save_data, fileObj)
    fileObj.close()
if __name__=="__main__":

    # get all the image and mask path and number of images
    image_paths = glob.glob("image/*.tif")
    mask_paths = glob.glob("mask/*.tif")

    # split these path using a certain percentage
    train_size = 0.80
    print('original image shape: {}'.format(np.array(Image.open(image_paths[0])).shape))
    print('orginal mask shape: {}'.format(np.array(Image.open(mask_paths[0])).shape))

    train_img_mask_paths, val_img_mask_paths = split_train_val(image_paths, mask_paths, train_size)

    print('train len: {}, val len: {}'.format(len(train_img_mask_paths),len(val_img_mask_paths)))

    ## This part preprocess the images and save them to pickle files.
    ## If the pickle file already exists, this part will not run
    ## IMPORTANT: If you made any change to the prepreprocess function,
    ## you need to delete the pickle files before you rerun this part
    train_img_masks_save_path = './train_img_masks.pickle'
    if os.path.exists(train_img_masks_save_path):
        with open(train_img_masks_save_path,'rb') as f:
            train_img_masks = pickle.load(f)
        f.close()
    else:
        train_img_masks = preprocess_image(train_img_mask_paths)
        pickle_store(train_img_masks_save_path,train_img_masks)
    print('train len: {}'.format(len(train_img_masks)))

    val_img_masks_save_path = './val_img_masks.pickle'
    if os.path.exists(val_img_masks_save_path):
        with open(val_img_masks_save_path,'rb') as f:
            val_img_masks = pickle.load(f)
        f.close()
    else:
        val_img_masks = preprocess_image(val_img_mask_paths)
        pickle_store(val_img_masks_save_path,val_img_masks)
    print('val len: {}'.format(len(val_img_masks)))


    #### Let us display some of the images to make sure the data loading and processing is correct.
    ### the original size of the image is: 1280*1918, but we resize the image to 80*100 for
    ### training the segmentation network
    img_num = 60
    plt.imshow(train_img_masks[img_num][0])
    plt.title("sample image")
    plt.show()

    plt.imshow(train_img_masks[img_num][1], cmap='gray')
    plt.title("ground true segmentation")
    plt.show()
    print(train_img_masks[img_num][1].tolist())
