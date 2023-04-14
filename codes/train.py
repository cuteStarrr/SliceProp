import os
import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from torch.cuda.amp import autocast as autocast

from UNet_COPY import *
from interact_dataset import interact_dataset_image_all, get_multiclass_labels, interact_dataset_image

import torch
from torch import Tensor

def accuracy_all_numpy(label: np.ndarray, prediction: np.ndarray):
    """
    output: dimension - 4
    """
    epsilon = 1
    #output = torch.softmax(output, dim=1)
    #prediction = torch.argmax(output, dim=1)
    label = np.uint8(label)
    prediction = np.uint8(prediction)
    # print(label.shape)
    # print(prediction.shape)
    
    total_num = np.sum(label > 0) + np.sum(prediction > 0)
    dist = label - prediction
    add_dist = label + prediction
    zero_num = np.sum(add_dist == 0)
    right_num = np.sum(dist == 0) - zero_num
    wrong_num = np.sum(dist != 0)

    return 2 * right_num / total_num



def accuracy_all(label: Tensor, prediction: Tensor):
    """
    output: dimension - 4
    """
    epsilon = 1
    #output = torch.softmax(output, dim=1)
    #prediction = torch.argmax(output, dim=1)
    
    total_num = torch.sum(label.int() > 0) + torch.sum(prediction.int() > 0)
    dist = label.int() - prediction.int()
    add_dist = label.int() + prediction.int()
    zero_num = torch.sum(add_dist == 0)
    right_num = torch.sum(dist == 0) - zero_num
    wrong_num = torch.sum(dist != 0)

    return 2 * right_num / total_num


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
    
def dice_accuracy(input: Tensor, target: Tensor, multiclass: bool = True):
    return 1 - dice_loss(input, target, multiclass)


def scribble_loss(scribbles, out_masks):
    dist = scribbles * 0.5 - torch.sigmoid(out_masks)
    F = nn.ReLU() # 先不平方来看看
    dist = F(dist)

    return torch.mean(dist)


def scribble_loss_all(scribbles, out_masks, device):
    #scribbles = in_images[:,2,:,:]
    #print(scribbles.shape)
    dist = torch.zeros(out_masks.shape).to(device)
    #print(dist.shape)
    #print(out_masks.shape)
    for i in range(scribbles.shape[0]):
        dist[i, :, :, :] = torch.tensor(get_multiclass_labels(scribbles[i,:,:].cpu().detach().numpy(), out_masks.shape[1])).to(device)
        #print(tmp.shape)
        #dist[i, :, :, :] = tmp
    predict =  torch.max(torch.softmax(out_masks[:,:,:,:], dim=1), dim=1)[1]
    predict_multi = torch.zeros(out_masks.shape).to(device)
    for i in range(predict.shape[0]):
        predict_multi[i,:,:,:] = torch.tensor(get_multiclass_labels(predict[i,:,:].cpu().detach().numpy(), out_masks.shape[1])).to(device)
    dist = dist[:,1:,:,:] - predict_multi[:,1:,:,:]
    # dist = dist[:,1:,:,:] - torch.softmax(out_masks[:,:,:,:], dim=1)[:,1:,:,:]
    F = nn.ReLU() # 先不平方来看看
    dist = F(dist)

    return torch.mean(dist)
    


def train(epochs: int = 80,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        window_transform_flag: bool = True,
        FLT_flag: bool = False,
        sobel_flag: bool = True,
        feature_flag: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,
        ):
    
    """define training paras"""
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    """prepare dataset"""
    # 6 images for training, 3 images for testing
    train_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 161, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    #train_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 140, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    validate_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 2, end_file2 = 8, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    #validate_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 2, end_file2 = 3, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    
    #train_dataset = interact_dataset_image_all(three_class_path = r'/data/xuxin/ImageTBAD_processed/three_class/', start_file3 = 180, end_file3 = 193, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, model_flag = model_flag)
    #validate_dataset = interact_dataset_image_all(three_class_path = r'/data/xuxin/ImageTBAD_processed/three_class/', start_file3 = 3, end_file3 = 6, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, model_flag = model_flag)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    n_train = len(train_dataset)
    n_val = len(validate_dataset)

    print(f'using {n_train} images for training, {n_val} images for validation.')

    """prepare network"""
    model = U_Net(in_channels, out_channels) 
    #model.load_state_dict(torch.load(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/notransform_sobel_scribble/U_Net_notransform_sobel_scribble_loss_2.pth', map_location = device))
    model.to(device)

    """set loss function, optimazier"""
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    binary_flag = False if out_channels > 1 else True
    criterion = nn.BCEWithLogitsLoss() if binary_flag else nn.CrossEntropyLoss()


    """prepare for saving and log"""
    save_path_loss = r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_11.pth'
    save_path_acc = r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_acc_11.pth'
    log = open(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/train_log_transform_sobel_scribble_11.txt', "a+", buffering=1)
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    least_loss = 999999999
    accuracy =  -1

    # begin training
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        step = 0
        with tqdm(iterable=train_loader, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for data in pbar:
                images, true_masks = data
                images = images.to(device=device, dtype=torch.float32) # , dtype=torch.float32
                true_masks = true_masks.to(device=device) # , dtype=torch.long
                
                optimizer.zero_grad()

                masks_pred = model(images)
                # print(masks_pred.shape)
                # print(true_masks.shape)
                loss = criterion(masks_pred.squeeze(1), true_masks.float()) if binary_flag else criterion(masks_pred, true_masks.long())
                loss += scribble_loss(images[:,2,:,:], masks_pred.squeeze(1)) if binary_flag else scribble_loss_all(images[:,2,:,:] if feature_flag else images[:,1,:,:], masks_pred, device)
                # loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                step += 1
                train_loss += loss.item()
                acc_tmp = accuracy_all(true_masks.int(), torch.round(torch.sigmoid(masks_pred.squeeze(1)))) if binary_flag else accuracy_all(true_masks.int(), torch.argmax(torch.softmax(masks_pred, dim=1), dim=1))
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_postfix(**{'acc (batch)': acc_tmp})
                train_acc += acc_tmp
                # print('[epoch %d] [step %d / %d] loss: %.3f' %
                #         (epoch, step, train_steps, loss.item()))
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        step = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device=device, dtype=torch.float32)
                val_labels = val_labels.to(device=device)
                outputs = model(val_images)
                loss = criterion(outputs.squeeze(1), val_labels.float()) if binary_flag else criterion(outputs, val_labels.long())
                loss += scribble_loss(val_images[:,2,:,:], outputs.squeeze(1)) if binary_flag else scribble_loss_all(val_images[:,2,:,:] if feature_flag else val_images[:,1,:,:], outputs, device)
                # loss += dice_loss(torch.sigmoid(outputs.squeeze(1)), val_labels.float(), multiclass=False)
                val_loss += loss.item()
                step += 1
                acc_tmp = accuracy_all(val_labels.int(), torch.round(torch.sigmoid(outputs.squeeze(1)))) if binary_flag else accuracy_all(val_labels.int(), torch.argmax(torch.softmax(outputs, dim=1), dim=1))
                #acc_tmp = accuracy_all(val_labels, outputs)
                val_bar.set_postfix(**{'loss (batch)': loss.item()})
                val_bar.set_postfix(**{'acc (batch)': acc_tmp})
                val_acc += acc_tmp
                # print('[epoch %d] [step %d / %d] loss: %.3f' %
                #         (epoch, step, val_steps, loss.item()))

        print('[epoch %d] train_loss: %.5f  val_loss: %.5f  train_acc: %.5f val_acc: %.5f' %
            (epoch, train_loss / train_steps, val_loss / val_steps, train_acc / train_steps, val_acc / val_steps))
        log.write('[epoch %d] train_loss: %.5f  val_loss: %.5f  train_acc: %.5f val_acc: %.5f\n' %
            (epoch, train_loss / train_steps, val_loss / val_steps, train_acc / train_steps, val_acc / val_steps))

        if val_loss / val_steps < least_loss:
            least_loss = val_loss / val_steps
            torch.save(model.state_dict(), save_path_loss)
            
        if val_acc / val_steps > accuracy:
            accuracy = val_acc / val_steps
            torch.save(model.state_dict(), save_path_acc)

    log.close()


def train_region(epochs: int = 80,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        window_transform_flag: bool = True,
        FLT_flag: bool = False,
        sobel_flag: bool = True,
        seeds_flag: bool = True,
        in_channels: int = 3,
        out_channels: int = 1,
        ):
    
    """define training paras"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    """prepare dataset"""
    # 6 images for training, 3 images for testing
    train_dataset = interact_dataset_image(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 161, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, seeds_flag = seeds_flag)
    #train_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 140, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    validate_dataset = interact_dataset_image(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 2, end_file2 = 8, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, seeds_flag = seeds_flag)
    #validate_dataset = interact_dataset_image_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 2, end_file2 = 3, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, feature_flag = feature_flag)
    
    #train_dataset = interact_dataset_image_all(three_class_path = r'/data/xuxin/ImageTBAD_processed/three_class/', start_file3 = 180, end_file3 = 193, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, model_flag = model_flag)
    #validate_dataset = interact_dataset_image_all(three_class_path = r'/data/xuxin/ImageTBAD_processed/three_class/', start_file3 = 3, end_file3 = 6, window_transform_flag = window_transform_flag, FLT_flag = FLT_flag, sobel_flag = sobel_flag, model_flag = model_flag)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    n_train = len(train_dataset)
    n_val = len(validate_dataset)

    print(f'using {n_train} images for training, {n_val} images for validation.')

    """prepare network"""
    model = U_Net(in_channels, out_channels) 
    # model.load_state_dict(torch.load(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/notransform_sobel_scribble/U_Net_region_notransform_sobel_scribble_loss_4.pth', map_location = device))
    model.to(device)

    """set loss function, optimazier"""
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    binary_flag = False if out_channels > 1 else True
    criterion = nn.BCEWithLogitsLoss() if binary_flag else nn.CrossEntropyLoss()


    """prepare for saving and log"""
    save_path_loss = r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_5.pth'
    save_path_acc = r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_acc_5.pth'
    log = open(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/train_log_region_transform_sobel_scribble_5.txt', "a+", buffering=1)
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    least_loss = 999999999
    accuracy =  -1

    # begin training
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        step = 0
        with tqdm(iterable=train_loader, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for data in pbar:
                images, true_masks = data
                images = images.to(device=device, dtype=torch.float32) # , dtype=torch.float32
                true_masks = true_masks.to(device=device) # , dtype=torch.long
                
                optimizer.zero_grad()

                masks_pred = model(images)
                # print(masks_pred.shape)
                # print(true_masks.shape)
                loss = criterion(masks_pred.squeeze(1), true_masks.float()) if binary_flag else criterion(masks_pred, true_masks.long())
                loss += scribble_loss(images[:,2,:,:], masks_pred.squeeze(1)) if binary_flag else scribble_loss_all(images[:,2,:,:], masks_pred, device)
                # loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                step += 1
                train_loss += loss.item()
                acc_tmp = accuracy_all(true_masks.int(), torch.round(torch.sigmoid(masks_pred.squeeze(1)))) if binary_flag else accuracy_all(true_masks.int(), torch.argmax(torch.softmax(masks_pred, dim=1), dim=1))
                # acc_tmp = accuracy_all(true_masks, torch.argmax(torch.softmax(masks_pred, dim=1), dim=1))
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_postfix(**{'acc (batch)': acc_tmp})
                train_acc += acc_tmp
                # print('[epoch %d] [step %d / %d] loss: %.3f' %
                #         (epoch, step, train_steps, loss.item()))
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        step = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device=device, dtype=torch.float32)
                val_labels = val_labels.to(device=device)
                outputs = model(val_images)
                loss = criterion(outputs.squeeze(1), val_labels.float()) if binary_flag else criterion(outputs, val_labels.long())
                loss += scribble_loss(val_images[:,2,:,:], outputs.squeeze(1)) if binary_flag else scribble_loss_all(val_images[:,2,:,:], outputs, device)
                # loss += dice_loss(torch.sigmoid(outputs.squeeze(1)), val_labels.float(), multiclass=False)
                val_loss += loss.item()
                step += 1
                acc_tmp = accuracy_all(val_labels, torch.round(torch.sigmoid(outputs.squeeze(1)))) if binary_flag else accuracy_all(val_labels, torch.argmax(torch.softmax(outputs, dim=1), dim=1))
                #acc_tmp = accuracy_all(val_labels, outputs)
                val_bar.set_postfix(**{'loss (batch)': loss.item()})
                val_bar.set_postfix(**{'acc (batch)': acc_tmp})
                val_acc += acc_tmp
                # print('[epoch %d] [step %d / %d] loss: %.3f' %
                #         (epoch, step, val_steps, loss.item()))

        print('[epoch %d] train_loss: %.5f  val_loss: %.5f  train_acc: %.5f val_acc: %.5f' %
            (epoch, train_loss / train_steps, val_loss / val_steps, train_acc / train_steps, val_acc / val_steps))
        log.write('[epoch %d] train_loss: %.5f  val_loss: %.5f  train_acc: %.5f val_acc: %.5f\n' %
            (epoch, train_loss / train_steps, val_loss / val_steps, train_acc / train_steps, val_acc / val_steps))

        if val_loss / val_steps < least_loss:
            least_loss = val_loss / val_steps
            torch.save(model.state_dict(), save_path_loss)
            
        if val_acc / val_steps > accuracy:
            accuracy = val_acc / val_steps
            torch.save(model.state_dict(), save_path_acc)

    log.close()


if __name__ == '__main__':
    # train_region() 
    train()  