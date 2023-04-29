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


    return np.stack((image_processed, sobel_sitk, get_curclass_label(seeds_image, 0), get_curclass_label(seeds_image, 1), get_curclass_label(seeds_image, 2)))


    
def get_prediction(model, indata):
    prediction = model(indata).cpu().squeeze()
    prediction = torch.sigmoid(prediction)
    # prediction = torch.sigmoid(prediction).detach().numpy()
    prediction = prediction.detach().numpy()
    # prediction = prediction - prediction.min()
    # prediction = prediction / prediction.max()
    prediction = np.where(prediction > 0.5, 1, 0)

    return np.uint8(prediction)




def get_prediction_all(model, indata, uncertainty_flag = True):
    """
    除了不确定性 还需要考虑 
    scribble loss
    同一连通区域不应该有多种label
    """
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

    uncertainty_value = (np.sum(uncertainty[prediction_mask]) / np.sum(prediction_mask) if prediction_mask.any() else 0) if uncertainty_flag else 0

    return prediction, uncertainty_value

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


def get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, nostart_flag, device, model, seeds_case, clean_region_flag = False):
    flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image, seeds_case=seeds_case, clean_region_flag=clean_region_flag)
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


def get_prediction_all_bidirectional_mask(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, nostart_flag, device, model, seeds_case, clean_region_flag = False):
    flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image, seeds_case=seeds_case, clean_region_flag=clean_region_flag)
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
    seeds_map = np.uint8(seeds_map)
    # print("prediction")
    prediction = np.where(seeds_map == 1, 1, prediction)
    prediction = np.where(seeds_map == 2, 2, prediction)
    prediction = np.uint8(prediction)

    return True, prediction, seeds_map


def test_all_bidirectional(image_path, save_path, model_weight_path, window_transform_flag, FLT_flag, sobel_flag, feature_flag, in_channels, out_channels, dice_coeff_thred, seeds_case, clean_region_flag = False):
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
        flag, prediction,_ = get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, i - start_piece, device, model, seeds_case, clean_region_flag=clean_region_flag)
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
            roll_flag, roll_prediction,_ = get_prediction_all_bidirectional(array_predict[:,:,cur_piece], image_data[:,:,cur_piece-1], image_data[:,:,cur_piece], window_transform_flag, feature_flag, sobel_flag, array_predict, 1, device, model, seeds_case, clean_region_flag=clean_region_flag)
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



def test_all_bidirectional_mask(image_path, save_path, model_weight_path, window_transform_flag, FLT_flag, sobel_flag, feature_flag, in_channels, out_channels, dice_coeff_thred, seeds_case, clean_region_flag = False):
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
        flag, prediction,_ = get_prediction_all_bidirectional_mask(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, i - start_piece, device, model, seeds_case, clean_region_flag=clean_region_flag)
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
            roll_flag, roll_prediction,_ = get_prediction_all_bidirectional_mask(array_predict[:,:,cur_piece], image_data[:,:,cur_piece-1], image_data[:,:,cur_piece], window_transform_flag, feature_flag, sobel_flag, array_predict, 1, device, model, seeds_case, clean_region_flag=clean_region_flag)
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


    save2h5(save_path, ['image', 'label', 'prediction'], [image_data, image_label, array_predict])


def cal_image_acc_experiment(array_predict_ori, image_label_ori, log, file_name):
    height, width, depth = array_predict_ori.shape
    array_predict_tl = np.uint8(np.where(array_predict_ori == 1, 1, 0))
    image_label_tl = np.uint8(np.where(image_label_ori == 1, 1, 0))
    array_predict_fl = np.uint8(np.where(array_predict_ori == 2, 1, 0))
    image_label_fl = np.uint8(np.where(image_label_ori == 2, 1, 0))
    array_predict = np.uint8(np.where(array_predict_ori > 0, 1, 0))
    image_label = np.uint8(np.where(image_label_ori > 0, 1, 0))
    acc_tl = 0.0
    acc_fl = 0.0
    acc = 0.0
    acc_ori = 0.0

    for d in range(depth):
        tmp_acc_tl = accuracy_all_numpy(array_predict_tl[:,:,d], image_label_tl[:,:,d])
        # print(f'current file: {file_name}, current piece: {d}/{depth}, acc: {tmp_acc}')
        acc_tl += tmp_acc_tl


    for d in range(depth):
        tmp_acc_fl = accuracy_all_numpy(array_predict_fl[:,:,d], image_label_fl[:,:,d])
        # print(f'current file: {file_name}, current piece: {d}/{depth}, acc: {tmp_acc}')
        acc_fl += tmp_acc_fl

    for d in range(depth):
        tmp_acc = accuracy_all_numpy(array_predict[:,:,d], image_label[:,:,d])
        # print(f'current file: {file_name}, current piece: {d}/{depth}, acc: {tmp_acc}')
        acc += tmp_acc

    for d in range(depth):
        tmp_acc_ori = accuracy_all_numpy(array_predict_ori[:,:,d], image_label_ori[:,:,d])
        # print(f'current file: {file_name}, current piece: {d}/{depth}, acc: {tmp_acc}')
        acc_ori += tmp_acc_ori

    
    print('file: %s, depth: %d, TL acc: %.5f, FL acc: %.5f, acc: %.5f, acc_ori: %.5f' % (file_name, depth, acc_tl / depth, acc_fl / depth , acc / depth, acc_ori / depth))
    log.write('file: %s, depth: %d, TL acc: %.5f, FL acc: %.5f, acc: %.5f, acc_ori: %.5f\n' % (file_name, depth, acc_tl / depth, acc_fl / depth , acc / depth, acc_ori / depth))

    return



def test_experiment(image_path, log_path, model_weight_path, seeds_case = 0, window_transform_flag = True, sobel_flag = True, feature_flag = True, in_channels = 5, out_channels = 3, dice_coeff_thred = 0.75, clean_region_flag = False):
    """
    img_7 for test bidirectionally
    """
    log = open(log_path, "a+", buffering=1)

    for file_name in open(image_path, 'r'):
        file_name = file_name.replace("\n", "")
        file_image = h5py.File(file_name, 'r')

        acc = 0.0

        print("current file: ", file_name)

        image_data = (file_image['image'])[()]
        image_label = (file_image['label'])[()]

        image_data = image_data - image_data.min()
        
        image_label = np.uint8(image_label)
        height, width, depth = image_data.shape

        array_predict = np.zeros(image_data.shape, dtype=np.uint8)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        model = U_Net(in_channels, out_channels) 
        # model_weight_path = r'../training_files/two_class/train5_validate2/U_Net_1.pth'
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.to(device)
        model.eval()
        


        start_piece = int(depth / 4)
        
        start_label = image_label[:,:,start_piece]
        while start_label.max() < 0.5:
            start_piece += 1
            start_label = image_label[:,:,start_piece]
        cur_image = image_data[:,:,start_piece]
        last_image = image_data[:,:,start_piece]
        last_label = start_label

        for i in range(start_piece, depth):
            cur_image = image_data[:,:,i]
            flag, prediction,_ = get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, array_predict, i - start_piece, device, model, seeds_case, clean_region_flag=clean_region_flag)
            if not flag:
                break
            # print(np.unique(prediction, return_counts = True))
            # print(prediction.shape)
            array_predict[:,:,i] = prediction
            # tmp_acc = accuracy_all_numpy(prediction, image_label[:,:,i])
            # print(f'current file: {file_name}, current piece: {i}/{depth}, acc: {tmp_acc}')
            # acc += tmp_acc
            # acc_num += 1
            if prediction.max() < 0.5:
                break
            cur_piece = i
            cur_coeff = accuracy_all_numpy(array_predict[:,:,cur_piece-1], array_predict[:,:,cur_piece])
            while cur_piece > 0 and cur_coeff  < dice_coeff_thred:
                roll_flag, roll_prediction,_ = get_prediction_all_bidirectional(array_predict[:,:,cur_piece], image_data[:,:,cur_piece-1], image_data[:,:,cur_piece], window_transform_flag, feature_flag, sobel_flag, array_predict, 1, device, model, seeds_case, clean_region_flag=clean_region_flag)
                if not roll_flag:
                    break
                if accuracy_all_numpy(array_predict[:,:,cur_piece - 1], roll_prediction) < 0.98:
                    array_predict[:,:,cur_piece - 1] = roll_prediction
                    # tmp_acc = accuracy_all_numpy(roll_prediction, image_label[:,:,cur_piece - 1])
                    # print(f'current file: {file_name}, current piece: {cur_piece - 1}/{depth}, acc: {tmp_acc}')
                    # acc += tmp_acc
                    # acc_num += 1
                else:
                    break
                if roll_prediction.max() < 0.5:
                    break
                cur_piece = cur_piece - 1
                cur_coeff = accuracy_all_numpy(array_predict[:,:,cur_piece-1], array_predict[:,:,cur_piece])
            last_image = image_data[:,:,i]
            last_label = prediction
            
        cal_image_acc_experiment(array_predict_ori=array_predict, image_label_ori=image_label, log=log, file_name=file_name)
        
    log.close()




if __name__ == '__main__':
    # test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_loss_20_0.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_20.pth', True, False, True, True, 5, 3, 0.75, 0)
    # test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_loss_20_6.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_20.pth', True, False, True, True, 5, 3, 0.75, 6)
    # test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_acc_20_0.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_acc_20.pth', True, False, True, True, 5, 3, 0.75, 0)
    # test_all_bidirectional(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/validate_2_transform_sobel_scribble_acc_20_6.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_acc_20.pth', True, False, True, True, 5, 3, 0.75, 6)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_6.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_5.pth', True)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/notransform_sobel_scribble/validate_2_region_notransform_sobel_scribble_loss_5.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/notransform_sobel_scribble/U_Net_region_notransform_sobel_scribble_loss_5.pth', False)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_4.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_4.pth', True)
    # test_region(r'/data/xuxin/ImageTBAD_processed/two_class/2.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/validate_2_region_transform_sobel_scribble_loss_3.h5', r'/data/xuxin/ImageTBAD_processed/training_files/two_class/connected_region/transform_sobel_scribble/U_Net_region_transform_sobel_scribble_loss_3.pth', True)
    test_experiment(image_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/test.txt',log_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/AD_1/test_log_dice_loss_1.txt',model_weight_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/AD_1/UNet_dice_loss_1.pth')
    test_experiment(image_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/test.txt',log_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/AD_1/test_log_dice_acc_1.txt',model_weight_path=r'/data/xuxin/ImageTBAD_processed/training_files/experiment/datalist/AD_1/UNet_dice_acc_1.pth')
