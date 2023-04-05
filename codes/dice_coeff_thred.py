import os
import torch
from train import accuracy_all
import h5py


def dice_coeff_thred(two_class_path = r'/data/xuxin/ImageTBAD_processed/two_class/', start_file2 = 139, end_file2 = 161):
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
        height, width, depth = label_data.shape
        for cur_piece in range(depth - 1):
            cur_roi.append(accuracy_all(torch.from_numpy(label_data[:,:,cur_piece]), torch.from_numpy(label_data[:,:,cur_piece+1])))
        max_value = max(cur_roi)
        min_value = min(cur_roi)
        mean_value = sum(cur_roi) / len(cur_roi)

        print('cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f' % 
              (cur_file, max_value, min_value, mean_value))
        log.write('cur file: %d, max dice coeff: %.5f , min dice coeff: %.5f, mean dice coeff: %.5f\n' % 
              (cur_file, max_value, min_value, mean_value))
        
        min_roi.append(min_value)
        max_roi.append(max_value)
        mean_roi.append(mean_value)
    
    print('min of min dice coeff: ', min(min_roi))
    print('max of max dice coeff: ', max(max_roi))
    print('mean of mean dice coeff: ', sum(mean_roi) / len(mean_roi))

    log.write(str(min_roi))
    log.write(str(max_roi))
    log.write(str(mean_roi))

    log.close()


if __name__ == '__main__':
    dice_coeff_thred()
        