import torch
import numpy as np
import os
import cv2
import h5py
import SimpleITK as sitk

from UNet_COPY import *
from interact_dataset import *
from train import accuracy_all_numpy
import matplotlib.pyplot as plt

def get_network_input(image, seeds, window_transform_flag):
    ele = []
    for i in range(seeds.shape[0]):
        ele.append(image[seeds[i,0], seeds[i,1]])
    ele = np.array(ele)

    image_processed = window_transform(image, max(ele.max() - ele.min() + 2 * np.sqrt(ele.var()), 255), (ele.max() + ele.min()) / 2) if window_transform_flag else image
    image_float = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.GetArrayFromImage(sobel_sitk)
    sobel_sitk = sobel_sitk - sobel_sitk.min()
    sobel_sitk = sobel_sitk / sobel_sitk.max()

    seeds_image = np.zeros(image.shape)
    for i in range(seeds.shape[0]):
        seeds_image[seeds[i,0], seeds[i,1]] = 1

    return np.stack((image_processed, sobel_sitk, seeds_image))


def get_network_input_all(image, seeds, seeds_image, window_transform_flag):
    ele = []
    for i in range(seeds.shape[0]):
        ele.append(image[seeds[i,0], seeds[i,1]])
    ele = np.array(ele)

    image_processed = window_transform(image, max(ele.max() - ele.min() + 2 * np.sqrt(ele.var()), 255), (ele.max() + ele.min()) / 2) if window_transform_flag else image
    # image_processed = window_transform(image, ele.max() - ele.min(), (ele.max() + ele.min()) / 2) if window_transform_flag else image

    image_float = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.GetArrayFromImage(sobel_sitk)
    sobel_sitk = sobel_sitk - sobel_sitk.min()
    sobel_sitk = sobel_sitk / sobel_sitk.max()

    # plt.imshow(sobel_sitk, cmap='gray')
    # plt.axis('off')
    # plt.show()


    return np.stack((image_processed, sobel_sitk, seeds_image))


    
def get_prediction(model, indata):
    prediction = model(indata).cpu().squeeze()
    prediction = torch.sigmoid(prediction)
    # prediction = torch.sigmoid(prediction).detach().numpy()
    prediction = prediction.detach().numpy()
    # prediction = prediction - prediction.min()
    # prediction = prediction / prediction.max()
    prediction = np.where(prediction > 0.5, 1, 0)

    return np.uint8(prediction)

def get_prediction_all(model, indata):
    prediction = model(indata).cpu().squeeze()
    prediction = torch.softmax(prediction, dim=0)
    uncertainty =  -torch.sum(prediction * torch.log(prediction   + 1e-16), dim=0).cpu().detach().numpy()
    # print(uncertainty.shape)
    # prediction = torch.sigmoid(prediction).detach().numpy()
    prediction = prediction.detach().numpy()
    # prediction = prediction - prediction.min()
    # prediction = prediction / prediction.max()
    prediction = np.uint8(np.argmax(prediction, axis=0))
    prediction_mask = prediction > 0


    return prediction, torch.sum(uncertainty[prediction_mask]) / torch.sum(prediction_mask)

def test_region(image_path, save_path, model_weight_path, window_transform_flag):
    file_image = h5py.File(image_path, 'r')

    image_data = (file_image['image'])[()]
    image_label = (file_image['label'])[()]

    image_data = image_data - image_data.min()
    image_label = np.uint8(image_label)
    height, width, depth = image_data.shape

    # array_ones = np.ones((height, width))
    # array_zeros = np.zeros((height, width))
    array_predict = np.zeros(image_data.shape)
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = U_Net()
    # model_weight_path = r'../training_files/two_class/train5_validate2/U_Net_1.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    for cur_piece in range(depth):
        if image_label[:,:,cur_piece].max() == 0:
            continue
        cur_image = image_data[:,:,cur_piece]
        cur_label = image_label[:,:,cur_piece]
        
        cur_kindregion = 0
        for cur_class in range(1, cur_label.max() + 1):
            cur_curclass_label = np.where(cur_label == cur_class, 1, 0)
            cur_connected_num, cur_connected_labels = cv2.connectedComponents(np.uint8(cur_curclass_label))
            cur_connected_labels = np.uint8(cur_connected_labels)

            for cur_region in range(1, cur_connected_num):
                cur_kindregion = cur_kindregion + 1
                cur_curkind_label = np.where(cur_connected_labels == cur_region, 1, 0)
                flag, seeds = get_right_seeds(cur_curkind_label, cur_image, cur_image, 6)
                if not flag:
                    continue
                indata = get_network_input(cur_image, seeds, window_transform_flag)
                indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                prediction = get_prediction(model, indata)
                array_predict[:,:,cur_piece] = np.where(prediction == 1, prediction * cur_kindregion, array_predict[:,:,cur_piece])
                print(f'cur piece: [{cur_piece}/{depth}], cur class: [{cur_class} / {cur_label.max()}] cur region: [{cur_region}/{cur_connected_num - 1}], ')

    save2h5(save_path, ['image', 'label', 'prediction'], [image_data, image_label, array_predict])


    # label_class = int(start_label.max())
    # for cur_class in range(label_class, 0, -1):
    #     last_curkind_label = np.where(start_label == cur_class, array_ones, array_zeros)
    #     last_image = start_image
    #     for i in range(start_piece, depth):
    #         cur_image = image_data[:,:,i]
    #         flag, seeds = get_right_seeds(last_curkind_label, cur_image, last_image)
    #         if not flag:
    #             break
    #         indata = get_network_input(cur_image, seeds, window_transform_flag)
    #         indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
    #         # prediction = (model(indata).cpu().squeeze()).detach().numpy()
    #         # prediction = prediction - prediction.min()
    #         # prediction = prediction / prediction.max()
    #         # prediction = np.where(prediction > 0.5, array_ones, array_zeros)
    #         prediction = get_prediction(model, indata)
    #         # print(np.unique(prediction, return_counts = True))
    #         # print(prediction.shape)
    #         array_predict[:,:,i] = np.where(prediction == 1, prediction * cur_class, array_predict[:,:,i])
    #         last_image = image_data[:,:,i]
    #         last_curkind_label = prediction
    #         print(f'cur label class: [{cur_class}/{label_class}], cur piece: [{i}/{depth}]')

    #     last_curkind_label = np.where(start_label == cur_class, array_ones, array_zeros)
    #     last_image = start_image
    #     for i in range(start_piece, -1, -1):
    #         cur_image = image_data[:,:,i]
    #         flag, seeds = get_right_seeds(last_curkind_label, cur_image, last_image)
    #         if not flag:
    #             break
    #         indata = get_network_input(cur_image, seeds, window_transform_flag)
    #         indata = torch.from_numpy(indata).unsqueeze(0).to(device=device, dtype=torch.float32)
    #         # prediction = (model(indata).cpu().squeeze()).detach().numpy()
    #         # prediction = prediction - prediction.min()
    #         # prediction = prediction / prediction.max()
    #         # prediction = np.where(prediction > 0.5, array_ones, array_zeros)
    #         prediction = get_prediction(model, indata)
    #         # print(np.unique(prediction, return_counts = True))
    #         array_predict[:,:,i] = np.where(prediction == 1, prediction * cur_class, array_predict[:,:,i])
    #         last_image = image_data[:,:,i]
    #         last_curkind_label = prediction
    #         print(f'cur label class: [{cur_class}/{label_class}], cur piece: [{i}/{depth}]')
    

def test_all(image_path, save_path, model_weight_path, window_transform_flag, FLT_flag, sobel_flag, feature_flag, in_channels, out_channels):
    file_image = h5py.File(image_path, 'r')

    image_data = (file_image['image'])[()]
    image_label = (file_image['label'])[()]

    image_data = image_data - image_data.min()
    
    if not FLT_flag:
        image_label = np.where(image_label == 3, 0, image_label)
    image_label = np.uint8(image_label)
    height, width, depth = image_data.shape

    array_predict = np.zeros(image_data.shape)
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    
    model = U_Net(in_channels, out_channels) 
    # model_weight_path = r'../training_files/two_class/train5_validate2/U_Net_1.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()
    


    start_piece = 103 # int(depth / 2)
    start_image = image_data[:,:,start_piece]
    start_label = image_label[:,:,start_piece]
    cur_image = image_data[:,:,start_piece]
    last_image = image_data[:,:,start_piece]
    last_label = start_label

    for i in range(start_piece, depth):
        cur_image = image_data[:,:,i]
        flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image)
        if not flag:
            break
        indata = get_network_input_all(cur_image, seeds, seeds_map, window_transform_flag, feature_flag)
        if not sobel_flag:
            indata[1,:,:] = array_predict[:,:,i - 1]
        indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        prediction,_ = get_prediction_all(model, indata)
        prediction = np.uint8(prediction)
        # print(np.unique(prediction, return_counts = True))
        # print(prediction.shape)
        array_predict[:,:,i] = prediction
        last_image = image_data[:,:,i]
        last_label = prediction
        
        print(f'cur piece: [{i}/{depth}]')

    
    last_image = start_image
    last_label = start_label
    for i in range(start_piece - 1, -1, -1):
        cur_image = image_data[:,:,i]
        flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image)
        if not flag:
            break
        indata = get_network_input_all(cur_image, seeds, seeds_map, window_transform_flag, feature_flag)
        if not sobel_flag:
            indata[1,:,:] = array_predict[:,:,i + 1]
        indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        
        prediction,_ = get_prediction_all(model, indata)
        prediction = np.uint8(prediction)
        # print(np.unique(prediction, return_counts = True))
        # print(prediction.shape)
        array_predict[:,:,i] = prediction
        last_image = image_data[:,:,i]
        last_label = prediction
        print(f'cur piece: [{i}/{depth}]')

    save2h5(save_path, ['image', 'label', 'prediction'], [image_data, image_label, array_predict])


def get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, nostart_flag, device, model, seeds_case):
    flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image, seeds_case=seeds_case)
    # plt.imshow(seeds_map, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # print("seeds")
    if not flag:
        return False, None, None
    indata = get_network_input_all(cur_image, seeds, seeds_map, window_transform_flag)
    # print("input")
    if not sobel_flag:
        if nostart_flag:
            indata[1,:,:] = array_predict[:,:,last_label]
        else:
            indata[1,:,:] = np.zeros(last_label.shape)
    indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
    prediction,_ = get_prediction_all(model, indata)
    # print("prediction")
    prediction = np.uint8(prediction)

    return True, prediction, seeds_map


def test_all_bidirectional(image_path, save_path, model_weight_path, window_transform_flag, FLT_flag, sobel_flag, feature_flag, in_channels, out_channels, dice_coeff_thred, seeds_case):
    """
    img_7 for test bidirectionally
    """
    file_image = h5py.File(image_path, 'r')

    image_data = (file_image['image'])[()]
    image_label = (file_image['label'])[()]

    image_data = image_data - image_data.min()
    
    image_label = np.uint8(image_label)
    height, width, depth = image_data.shape

    array_predict = np.zeros(image_data.shape)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model = U_Net(in_channels, out_channels) 
    # model_weight_path = r'../training_files/two_class/train5_validate2/U_Net_1.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()
    


    start_piece = 42 # int(depth / 2)
    start_image = image_data[:,:,start_piece]
    start_label = image_label[:,:,start_piece]
    cur_image = image_data[:,:,start_piece]
    last_image = image_data[:,:,start_piece]
    last_label = start_label

    for i in range(start_piece, depth):
        cur_image = image_data[:,:,i]
        flag, prediction,_ = get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, i - start_piece, device, model, seeds_case)
        if not flag:
            break
        # print(np.unique(prediction, return_counts = True))
        # print(prediction.shape)
        array_predict[:,:,i] = prediction
        if prediction.max() < 0.5:
            break
        cur_piece = i
        cur_coeff = accuracy_all_numpy(array_predict[:,:,cur_piece-1], array_predict[:,:,cur_piece])
        while cur_piece > 0 and cur_coeff  < dice_coeff_thred:
            roll_flag, roll_prediction,_ = get_prediction_all_bidirectional(array_predict[:,:,cur_piece], image_data[:,:,cur_piece-1], image_data[:,:,cur_piece], window_transform_flag, feature_flag, sobel_flag, array_predict, 1, device, model, seeds_case)
            if not roll_flag:
                break
            if accuracy_all_numpy(array_predict[:,:,cur_piece - 1], roll_prediction) < 0.98:
                array_predict[:,:,cur_piece - 1] = roll_prediction
            else:
                break
            if roll_prediction.max() < 0.5:
                break
            cur_piece = cur_piece - 1
            cur_coeff = accuracy_all_numpy(array_predict[:,:,cur_piece-1], array_predict[:,:,cur_piece])
        last_image = image_data[:,:,i]
        last_label = prediction
        
        print(f'cur piece: [{i}/{depth}]')

    
    # last_image = start_image
    # last_label = start_label
    # for i in range(start_piece - 1, -1, -1):
    #     cur_image = image_data[:,:,i]
    #     flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image)
    #     if not flag:
    #         break
    #     indata = get_network_input_all(cur_image, seeds, seeds_map, window_transform_flag, feature_flag)
    #     if not sobel_flag:
    #         indata[1,:,:] = array_predict[:,:,i + 1]
    #     indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        
    #     prediction = get_prediction_all(model, indata)
    #     prediction = np.uint8(prediction)
    #     # print(np.unique(prediction, return_counts = True))
    #     # print(prediction.shape)
    #     array_predict[:,:,i] = prediction
    #     last_image = image_data[:,:,i]
    #     last_label = prediction
    #     print(f'cur piece: [{i}/{depth}]')

    save2h5(save_path, ['image', 'label', 'prediction'], [image_data, image_label, array_predict])


if __name__ == '__main__':
    test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_loss_14_0.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_14.pth', True, False, True, True, 3, 3, 0.75, 0)
    test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_loss_14_6.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_14.pth', True, False, True, True, 3, 3, 0.75, 6)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_6.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_5.pth', True)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/notransform_sobel_scribble/validate_2_region_notransform_sobel_scribble_loss_5.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/notransform_sobel_scribble/U_Net_region_notransform_sobel_scribble_loss_5.pth', False)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_4.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_4.pth', True)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_3.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_3.pth', True)
