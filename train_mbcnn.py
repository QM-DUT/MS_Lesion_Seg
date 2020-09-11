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
from utils.util_for_mbcnn import *
import nibabel as nib
from Model.mbcnn import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        x = b['img'][0].to(device)
        y = b['img'][1].to(device)
        z = b['img'][2].to(device)
        index = b['index']
        d = b['d']
        w = b['w']
        h = b['h']
        ################################################ [TODO] ###################################################
        # Feed the image to network to get predicted mask
        mask_pred =net(x,y,z)
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
    for root, dirs, files in os.walk("../Pre-processed_training_dataset/"):
        if dirs == []:
            # 对每一个文件夹
            print(root)
            img = np.array(nib.load(os.path.join(root, "T1_preprocessed.nii.gz")).get_fdata())
            img2 = np.array(nib.load(os.path.join(root, "T2_preprocessed.nii.gz")).get_fdata())
            img3 = np.array(nib.load(os.path.join(root, "FLAIR_preprocessed.nii.gz")).get_fdata())
            mask = np.array(nib.load(os.path.join(root, "Consensus.nii.gz")).get_fdata())
            max_img = np.max(img)
            max_img2 = np.max(img2)
            max_img3 = np.max(img3)
            img = img / max_img
            img2 = img2 / max_img2
            img3 = img3 / max_img3

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
                img2 = np.pad(img2, (pad0, pad1, pad2), 'constant')
                img3 = np.pad(img3, (pad0, pad1, pad2), 'constant')
                mask=np.pad(mask, (pad0, pad1, pad2), 'constant')
            depth = mask.shape[0]
            width = mask.shape[1]
            height = mask.shape[2]
            for d in range(0, depth - 64 +16, 16):
                for w in range(0, width - 64 +16, 16):
                    for h in range(0, height -64 +16, 16):
                        patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                        patch_img2 = img2[d:d + 64, w:w + 64, h:h + 64]
                        patch_img3 = img3[d:d + 64, w:w + 64, h:h + 64]
                        patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                        if np.sum(patch_mask)>voxel and index<13:
                            patch_img = np.expand_dims(patch_img, 0)
                            patch_img2 = np.expand_dims(patch_img2, 0)
                            patch_img3 = np.expand_dims(patch_img3, 0)
                            patch_mask = np.expand_dims(patch_mask, 0)
                            train_img_masks.append(([patch_img,patch_img2,patch_img3], patch_mask, index, d, w, h))
                        if index>=13:
                            patch_img = np.expand_dims(patch_img, 0)
                            patch_img2 = np.expand_dims(patch_img2, 0)
                            patch_img3 = np.expand_dims(patch_img3, 0)
                            patch_mask = np.expand_dims(patch_mask, 0)
                            val_img_masks.append(([patch_img,patch_img2,patch_img3], patch_mask, index, d, w, h))

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
    #net= Mul(1, 1)
    net=torch.nn.DataParallel(Mul(1,1), device_ids=[0, 1, 2,3])
    # net=get_pose_net(config.cfg, 0)
    net.to(device)
    # run net.to(device) if using GPU
    # If continuing from previously saved model, use
    #net.load_state_dict(torch.load('unet_model/50.pth'))
    #print(net)

    # This shows the number of parameters in the network
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters in network: ', n_params)

    ################################################ [TODO] ###################################################
    # Specify number of epochs, image scale factor, batch size and learning rate
    epochs = 50  # e.g. 10, or more until dice converge
    batch_size = 16  # e.g. 16
    lr = 0.001  # e.g. 0.01
    N_train = len(train_img_masks)
    model_save_path = './results_mbcnn/'  # directory to same the model after each epoch.
    if not os.path.isdir(model_save_path):
        os.mkdir('results_mbcnn')
    ################################################ [TODO] ###################################################
    # Define an optimizer for your model.
    # Pytorch has built-in package called optim. Most commonly used methods are already supported.
    # Here we use stochastic gradient descent to optimize
    # For usage of SGD, you can read https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html
    # Also you can use ADAM as the optimizer
    # For usage of ADAM, you can read https://www.programcreek.com/python/example/92667/torch.optim.Adam

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # suggested parameter settings: momentum=0.9, weight_decay=0.0005

    # The loss function we use is binary cross entropy: nn.BCELoss()
    criterion = nn.BCELoss()
    # note that although we want to use DICE for evaluation, we use BCELoss for training in this example

    ################################################ [TODO] ###################################################
    # Start training  #This part takes very long time to run if using CPU
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # Reload images and masks for training and validation and perform random shuffling at the begining of each epoch
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

        epoch_loss = 0
        count = 0

        for i, b in enumerate(train_loader):
            ################################################ [TODO] ###################################################
            # Get images and masks from each batch


            x = b['img'][0].to(device)
            y = b['img'][1].to(device)
            z = b['img'][2].to(device)
            true_masks = b['label'].to(device)
            ################################################ [TODO] ###################################################
            # Feed your images into the network
            masks_pred = net(x,y,z).squeeze(0)
            # masks_pred = nn.functional.interpolate(masks_pred, size=(80,100), mode='bilinear')
            # Flatten the predicted masks and true masks. For example, A_flat = A.view(-1)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)
            ################################################ [TODO] ###################################################
            # Calculate the loss by comparing the predicted masks vector and true masks vector
            # And sum the losses together
            loss = criterion(masks_probs_flat, true_masks_flat)
            dice_score=(2*torch.dot(masks_probs_flat,true_masks_flat)+1)/(torch.sum(masks_probs_flat)
                                                                            +torch.sum(true_masks_flat)+1)
            # if loss.item()<0:
            #     mm=true_masks_flat.cpu().detach().numpy().tolist()
            #     print(mm)

            epoch_loss += loss.item()
            if count % 20 == 0:  # Print status every 20 batch
                print('{0:.4f} --- loss: {1:.6f},dice_score:{2:.6f}'.format(i * batch_size / N_train, loss.item(),dice_score))
            count = count + 1
            # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
            # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
            optimizer.zero_grad()
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # These are accumulated into x.grad for every parameter x
            loss.backward()
            # optimizer.step updates the value of x using the gradient x.grad.
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (len(train_loader)*batch_size)))
        ################################################ [TODO] ###################################################
        # Perform validation with eval_net() on the validation data
        val_dice = eval_net(net, val_loader,real_test_img_mask_list)
        print('Validation Dice Coeff: {}'.format(val_dice))

        if np.mean(val_dice) < min_dice:
            min_dice = np.mean(val_dice)
            print('New best performance! saving')
            save_name = os.path.join(model_save_path, 'model_best.pth')
            torch.save(net.state_dict(), save_name)
            print('model saved to {}'.format(save_name))

        if (epoch + 1) % log_interval == 0:
            save_name = os.path.join(model_save_path, 'model_routine.pth')
            torch.save(net.state_dict(), save_name)
            print('model saved to {}'.format(save_name))