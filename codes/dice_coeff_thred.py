import os
import torch
from train import accuracy_all
import h5py
import numpy as np


def dice_coeff_thred_all(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 161):
    """
    根据结果,最终确定为0.75
    """
    min_roi = []
    max_roi = []
    mean_roi = []
    log = open(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/Dice_coeff_thred_training.txt', "a+")

    for cur_file in range(start_file2, end_file2, 1):
        file_name = two_class_path + str(cur_file) + ".h5"
        if not os.path.exists(file_name):
            continue
        cur_roi = []
        file_image = h5py.File(file_name, 'r')
        label_data = (file_image['label'])[()]
        label_data = np.uint8(label_data)
        height, width, depth = label_data.shape
        for cur_piece in range(depth - 1):
            cur_pic = label_data[:,:,cur_piece]
            next_pic = label_data[:,:,cur_piece+1]
            if np.max(cur_pic) == 0 or np.max(next_pic) == 0:
                continue
            cur_roi.append(accuracy_all(torch.from_numpy(cur_pic), torch.from_numpy(next_pic)))
        max_value = max(cur_roi)
        min_value = min(cur_roi)
        mean_value = sum(cur_roi) / len(cur_roi)

        print('cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f' % 
              (cur_file, max_value, min_value, mean_value))
        log.write('cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f\n' % 
              (cur_file, max_value, min_value, mean_value))
        #log.writelines(str(cur_roi))
        
        min_roi.append(min_value)
        max_roi.append(max_value)
        mean_roi.append(mean_value)
    
    print('min of min dice coeff: ', min(min_roi))
    print('max of max dice coeff: ', max(max_roi))
    print('mean of mean dice coeff: ', sum(mean_roi) / len(mean_roi))

    log.writelines(str(min_roi))
    log.writelines(str(max_roi))
    log.writelines(str(mean_roi))

    log.close()


def dice_coeff_thred_region(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 161):
    """
    one/two:0.60, 0.7
    one/two:0.64, 0.7 -- currently prefer
    """
    min_roi_1 = []
    max_roi_1 = []
    mean_roi_1 = []
    min_roi_2 = []
    max_roi_2 = []
    mean_roi_2 = []
    log = open(r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/Dice_coeff_thred_training_for_oneRegion.txt', "a+")

    for cur_file in range(start_file2, end_file2, 1):
        file_name = two_class_path + str(cur_file) + ".h5"
        if not os.path.exists(file_name):
            continue
        cur_roi_1 = []
        cur_roi_2 = []
        file_image = h5py.File(file_name, 'r')
        label_data = (file_image['label'])[()]
        label_data = np.uint8(label_data)
        height, width, depth = label_data.shape
        array_one = np.zeros(label_data.shape)
        array_one = np.where(label_data == 1, 1, 0)
        array_two = np.zeros(label_data.shape)
        array_two = np.where(label_data == 2, 2, 0)
        for cur_piece in range(depth - 1):
            cur_pic = array_one[:,:,cur_piece]
            next_pic = array_one[:,:,cur_piece+1]
            if np.max(cur_pic) == 0 or np.max(next_pic) == 0:
                continue
            cur_roi_1.append(accuracy_all(torch.from_numpy(cur_pic), torch.from_numpy(next_pic)))
        for cur_piece in range(depth - 1):
            cur_pic = array_two[:,:,cur_piece]
            next_pic = array_two[:,:,cur_piece+1]
            if np.max(cur_pic) == 0 or np.max(next_pic) == 0:
                continue
            cur_roi_2.append(accuracy_all(torch.from_numpy(cur_pic), torch.from_numpy(next_pic)))    
        max_value_1 = max(cur_roi_1)
        min_value_1 = min(cur_roi_1)
        mean_value_1 = sum(cur_roi_1) / len(cur_roi_1)
        max_value_2 = max(cur_roi_2)
        min_value_2 = min(cur_roi_2)
        mean_value_2 = sum(cur_roi_2) / len(cur_roi_2)

        print('For one class -- cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f' % 
              (cur_file, max_value_1, min_value_1, mean_value_1))
        print('For two class -- cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f' % 
              (cur_file, max_value_2, min_value_2, mean_value_2))
        log.write('For one class -- cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f\n' % 
              (cur_file, max_value_1, min_value_1, mean_value_1))
        log.write('For two class -- cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f\n' % 
              (cur_file, max_value_2, min_value_2, mean_value_2))
        #log.writelines(str(cur_roi))
        
        min_roi_1.append(min_value_1)
        max_roi_1.append(max_value_1)
        mean_roi_1.append(mean_value_1)

        min_roi_2.append(min_value_2)
        max_roi_2.append(max_value_2)
        mean_roi_2.append(mean_value_2)
    
    print('--------------------for class one----------------------')
    print('min of min dice coeff: ', min(min_roi_1))
    print('max of max dice coeff: ', max(max_roi_1))
    print('mean of mean dice coeff: ', sum(mean_roi_1) / len(mean_roi_1))

    print('--------------------for class two----------------------')
    print('min of min dice coeff: ', min(min_roi_2))
    print('max of max dice coeff: ', max(max_roi_2))
    print('mean of mean dice coeff: ', sum(mean_roi_2) / len(mean_roi_2))

    log.writelines(str(min_roi_1))
    log.writelines(str(max_roi_1))
    log.writelines(str(mean_roi_1))
    log.writelines(str(min_roi_2))
    log.writelines(str(max_roi_2))
    log.writelines(str(mean_roi_2))

    log.close()


if __name__ == '__main__':
    dice_coeff_thred_region()
        