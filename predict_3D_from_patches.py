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
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
import glob
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import cv2
from utils.util import *
from Model.HRNet_3D import  *
import config
import nibabel as nib

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# define dice coefficient
def dice_coeff_cpu(prediction, target):
    s=[]
    eps = 1.0
    for i, (a, b) in enumerate(zip(prediction, target)):
        A = a.flatten()
        B = b.flatten()
        inter = np.dot(A, B)
        union = np.sum(A) + np.sum(B) + eps
        # Calculate DICE
        d = (2 * inter+eps) / union
        s.append(d)
    return s

################################################ [TODO] ###################################################
# This function is used to evaluate the network after each epoch of training
# Input: network and validation dataset
# Output: average dice_coeff
def eval_net(net, dataset,mask_list):
    # set net mode to evaluation
    net.eval()
    zero_index=0
    reconstucted_mask=[]
    original_mask=[]
    mask_blank_list=[]
    for i in range(len(mask_list)):
        if i==0:
            zero_index=mask_list[i][1]
        mask=mask_list[i][0]
        mask_blank_list.append((np.zeros((mask.shape)),np.zeros((mask.shape))))
        original_mask.append(mask)
    #这里需要考虑batch_size,不过根本不会太大
    for i, b in enumerate(dataset):
        img = b['img'].to(device)
        index = b['index']
        d = b['d']
        w = b['w']
        h = b['h']
        ################################################ [TODO] ###################################################
        # Feed the image to network to get predicted mask
        mask_pred =net(img)
        mask_blank_list[index - zero_index][0][0:1,d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(0).cpu().numpy()
        mask_blank_list[index - zero_index][1][0:1,d:d + 64, w:w + 64, h:h + 64] += 1

    for i in range(len(mask_blank_list)):
        mask_blank_list[i][1][ mask_blank_list[i][1]==0 ]=1
        reconstucted_mask.append(mask_blank_list[i][0]/mask_blank_list[i][1])

    return dice_coeff_cpu(reconstucted_mask,original_mask)


if __name__=="__main__":
    # get all the image and mask path and number of images
    mask_list = []
    patch_img_mask_list = []
    train_img_masks=[]
    val_img_masks=[]
    new_h, new_w, new_d = 64, 64, 64  # 80,100 288,384 96,72
    voxel=64*64*64*0.0001
    # 遍历根目录
    index = 0
    for root, dirs, files in os.walk("../TrainingDataset_MSSEG/"):
        if dirs == []:
            # 对每一个文件夹
            print(root)
            img = np.array(nib.load(os.path.join(root, "FLAIR_preprocessed.nii.gz")).get_fdata())
            mask = np.array(nib.load(os.path.join(root, "Consensus.nii.gz")).get_fdata())
            max_img = np.max(img)
            img = img / max_img
            depth = mask.shape[0]
            width = mask.shape[1]
            height = mask.shape[2]
            #pad zero
            pad_size = [0, 0, 0]
            div=[64,64,64]
            pad = False
            for i in range(len(img.shape)):
                remain = img.shape[i] % div[i]
                if remain != 0:
                    pad = True
                    pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
            if pad:
                # deal with odd number of padding
                pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
                pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
                pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
                img=np.pad(img, (pad0, pad1, pad2), 'constant')
                mask=np.pad(mask, (pad0, pad1, pad2), 'constant')
            depth = mask.shape[0]
            width = mask.shape[1]
            height = mask.shape[2]
            for d in range(0, depth - 64+16, 16):
                for w in range(0, width - 64+16, 16):
                    for h in range(0, height -64+16, 16):
                        patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                        patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                        if np.sum(patch_mask)>voxel and index<13:
                            patch_img = np.expand_dims(patch_img, 0)
                            patch_mask = np.expand_dims(patch_mask, 0)
                            train_img_masks.append((patch_img, patch_mask, index, d, w, h))
                        if index>=13:
                            patch_img = np.expand_dims(patch_img, 0)
                            patch_mask = np.expand_dims(patch_mask, 0)
                            val_img_masks.append((patch_img, patch_mask, index, d, w, h))

                        #patch_img_mask_list.append((patch_img, patch_mask, index, d, w, h))
            mask = np.expand_dims(mask, 0)
            mask_list.append((mask, index))

            index += 1
    # train_img_masks = patch_img_mask_list[:int(len(patch_img_mask_list) * 0.9)]
    # val_img_masks = patch_img_mask_list[int(len(patch_img_mask_list) * 0.9):]
    real_train_img_mask_list = mask_list[:int(len(mask_list) * 0.9)]
    real_test_img_mask_list = mask_list[int(len(mask_list) * 0.9):]

    train_dataset = CustomDataset(train_img_masks, transforms=transforms.Compose([ToTensor()]))
    val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))

    ################################################ [TODO] ###################################################
    # Create a UNET object from the class defined above. Input channels = 3, output channels = 1
    #net= UNet3D(1, 1)
    net=get_pose_net(config.cfg, 0)
    net.to(device)
    # run net.to(device) if using GPU
    # If continuing from previously saved model, use
    net.load_state_dict(torch.load('HRNet_model_MS/5.pth',map_location='cuda:2'))
    #print(net)

    # This shows the number of parameters in the network
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters in network: ', n_params)



    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dice = eval_net(net, train_loader, real_train_img_mask_list)
    print('Validation Dice Coeff: {}'.format(val_dice))

    val_dice = eval_net(net, val_loader, real_test_img_mask_list)
    print('Validation Dice Coeff: {}'.format(val_dice))
