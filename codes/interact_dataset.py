import numpy as np
import os
import cv2
import h5py
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
import random
# from region_grow import *


def get_seeds_based_seedscase(seeds_case_flag, num, quit_num, cur_label_ori, coord_ori):
    cur_label = cur_label_ori.copy()
    coord = coord_ori.copy()

    if seeds_case_flag == 0:
        cur_quit_num = 0
        while cur_quit_num < quit_num:
            boundaries = find_boundaries(cur_label, mode='inner').astype(np.uint8)
            cur_quit_num += np.sum(boundaries == 1)
            cur_label = np.where(boundaries == 1, 0, cur_label)
            
        if np.sum(cur_label == 1) == 0:
            coord = coord[0: max(1, int(num / 2)), :]
        else:
            coord = np.argwhere(cur_label > 0)
    elif seeds_case_flag == 1:
        """截取上面一部分"""
        coord = coord[0:num - quit_num, :]
    elif seeds_case_flag == 2:
        """截取下面一部分"""
        coord = coord[quit_num:, :]
    elif seeds_case_flag == 3:
        """截取左面一部分"""
        min_pos = min(coord[:, 1])
        max_pos = max(coord[:, 1])
        cur_pos = max_pos
        while cur_pos >= min_pos:
            delete_label = cur_label[:, cur_pos:max_pos + 1]
            if np.sum(delete_label > 0) >= quit_num:
                cur_label[:, cur_pos:max_pos + 1] = 0
                break
            cur_pos = cur_pos - 1
        if np.sum(cur_label == 1) == 0:
            coord = coord[0: max(1, int(num / 2)), :]
        else:
            coord = np.argwhere(cur_label > 0)
    elif seeds_case_flag == 4:
        """
        seeds_case == 4, 截取右面一部分
        """
        min_pos = min(coord[:, 1])
        max_pos = max(coord[:, 1])
        cur_pos = min_pos + 1
        while cur_pos <= max_pos + 1:
            delete_label = cur_label[:, min_pos:cur_pos]
            if np.sum(delete_label > 0) >= quit_num:
                cur_label[:, min_pos:cur_pos] = 0
                break
            cur_pos = cur_pos + 1
        if np.sum(cur_label == 1) == 0:
            coord = coord[0: max(1, int(num / 2)), :]
        else:
            coord = np.argwhere(cur_label > 0)

    return coord


def get_seeds(label, rate, thred, seeds_case, cur_image, last_image):
    """
    label只有一个种类，但是可能有多个连通分量
    需要对边界seeds进行训练
    """
    coords = np.zeros((0,2), int)
    if rate <= thred:
        label_unit8 = np.uint8(label)
        _, labels = cv2.connectedComponents(label_unit8)
        
        block_num = labels.max()
        seeds_case_flag_list = []
        # print(f'block_num: {block_num}')
        for cur_block in range(block_num, 0, -1):
            cur_label = np.where(labels > cur_block - 0.5,1,0)
            # print(f'cur_label size: {cur_label.shape}')
            labels[labels > cur_block - 0.5] = 0
            
            cur_label = np.uint8(cur_label)
            coord = np.argwhere(cur_label > 0)
            coord_cur_block = coord.copy()
            num, _ = coord.shape
            # num > 8, 否则没有quit_num
            quit_num = int((1-rate) * num)

            seeds_case_flag = seeds_case
            """
            seeds_case == 5, 则是随机挑选seeds组合（block num > 1）,但不能全部都是一样的，这样就与01234的情况重复了
            """
            if seeds_case == 5:
                if block_num > 1:
                    seeds_case_flag = random.randint(0,6)
                    while seeds_case_flag == 5:
                        seeds_case_flag = random.randint(0,6)
                    while cur_block == 1 and seeds_case_flag == max(seeds_case_flag_list) and seeds_case_flag == min(seeds_case_flag_list):
                        seeds_case_flag = random.randint(0,6)
                    seeds_case_flag_list.append(seeds_case_flag)
                else:
                    break

            # if seeds_case_flag == 0:
            #     cur_quit_num = 0
            #     while cur_quit_num < quit_num:
            #         boundaries = find_boundaries(cur_label, mode='inner').astype(np.uint8)
            #         cur_quit_num += np.sum(boundaries == 1)
            #         cur_label = np.where(boundaries == 1, 0, cur_label)
                    
            #     if np.sum(cur_label == 1) == 0:
            #         coord = coord[0: max(1, int(num / 2)), :]
            #     else:
            #         coord = np.argwhere(cur_label > 0)
            # elif seeds_case_flag == 1:
            #     """截取上面一部分"""
            #     coord = coord[0:num - quit_num, :]
            # elif seeds_case_flag == 2:
            #     """截取下面一部分"""
            #     coord = coord[quit_num:, :]
            # elif seeds_case_flag == 3:
            #     """截取左面一部分"""
            #     min_pos = min(coord[:, 1])
            #     max_pos = max(coord[:, 1])
            #     cur_pos = max_pos
            #     while cur_pos >= min_pos:
            #         delete_label = cur_label[:, cur_pos:max_pos + 1]
            #         if np.sum(delete_label > 0) >= quit_num:
            #             cur_label[:, cur_pos:max_pos + 1] = 0
            #             break
            #         cur_pos = cur_pos - 1
            #     if np.sum(cur_label == 1) == 0:
            #         coord = coord[0: max(1, int(num / 2)), :]
            #     else:
            #         coord = np.argwhere(cur_label > 0)
            # elif seeds_case_flag == 4:
            #     """
            #     seeds_case == 4, 截取右面一部分
            #     """
            #     min_pos = min(coord[:, 1])
            #     max_pos = max(coord[:, 1])
            #     cur_pos = min_pos + 1
            #     while cur_pos <= max_pos + 1:
            #         delete_label = cur_label[:, min_pos:cur_pos]
            #         if np.sum(delete_label > 0) >= quit_num:
            #             cur_label[:, min_pos:cur_pos] = 0
            #             break
            #         cur_pos = cur_pos + 1
            #     if np.sum(cur_label == 1) == 0:
            #         coord = coord[0: max(1, int(num / 2)), :]
            #     else:
            #         coord = np.argwhere(cur_label > 0)
            if seeds_case_flag < 5:
                coord_cur_block = get_seeds_based_seedscase(seeds_case_flag=seeds_case_flag, num=num, quit_num=quit_num, cur_label_ori=cur_label, coord_ori=coord)
            elif seeds_case_flag == 6:
                """
                一个连通区域有2-3处seeds， 其中一处在中间
                """
                coord_cur_block = get_seeds_based_seedscase(seeds_case_flag=0, num=num, quit_num=quit_num, cur_label_ori=cur_label, coord_ori=coord)

                old_seeds_case = random.randint(1,4)
                coord_tmp = get_seeds_based_seedscase(seeds_case_flag=old_seeds_case, num=num, quit_num=int((1-rate / 3) * num), cur_label_ori=cur_label, coord_ori=coord)
                coord_cur_block = np.concatenate((coord_cur_block, coord_tmp), axis=0)
                if random.random() < 0.5:
                    """
                    有三处seeds
                    """
                    new_seeds_case = random.randint(1,4)
                    while new_seeds_case == old_seeds_case:
                        new_seeds_case = random.randint(1,4)
                    coord_tmp = get_seeds_based_seedscase(seeds_case_flag=new_seeds_case, num=num, quit_num=int((1-rate / 3) * num), cur_label_ori=cur_label, coord_ori=coord)
                    coord_cur_block = np.concatenate((coord_cur_block, coord_tmp), axis=0) 
            
            clean_flag, coord_cur_block = clean_seeds(coord_cur_block, cur_image=cur_image, last_image=last_image)
            if clean_flag:
                coords = np.concatenate((coords, coord_cur_block), axis=0)
            # else:
            #     get_seeds(label, rate + step, thred, seeds_case, cur_image, last_image, step)
            else:
            #     if rate < thred:
            #         return False, coords
            #     else:
                continue
        coords = np.unique(coords, axis=0)
        if coords.shape[0] > 0:
            return True, coords
        else:
            return False, coords
    else:
        return False, coords

def clean_seeds(seeds, cur_image, last_image):
    """这里采用seeds而不是完整的last_label来估算last_image的均值和标准差, 目的是为了减少错误结果带来的误差"""
    ele_last = []
    for i in range(seeds.shape[0]):
        ele_last.append(last_image[seeds[i,0], seeds[i,1]])
    ele_last = np.array(ele_last)
    mean_last = ele_last.mean()
    std_last = np.sqrt(ele_last.var())
    min_last = ele_last.min()
    max_last = ele_last.max()

    new_seeds = []
    for i in range(seeds.shape[0]):
        test_value = cur_image[seeds[i,0], seeds[i,1]]
        if test_value >= min_last and test_value <= max_last:
            new_seeds.append(seeds[i])
    
    new_seeds = np.array(new_seeds)

    if new_seeds.shape[0]:
        return True, new_seeds
    else:
        return False, new_seeds

def get_right_seeds(label, cur_image, last_image, seeds_case, rate = 0.2, step = 0.1, thred = 0.4):
    label = np.uint8(label)
    if np.sum(label == 1) == 0:
        return False, None
    flag_find, seeds = get_seeds(label, rate, thred, seeds_case, cur_image=cur_image, last_image=last_image)
    if flag_find:
        return True, seeds
    else:
        print("ERROR!!! Large rate to get clean seeds!")
        rate = rate + step
        if rate > thred:
            return False, None
        return get_right_seeds(label, cur_image, last_image, seeds_case, rate, step, thred)
    # else:
    #     print("ERROR!!!! Rate exceeds threshold!! There is no seeds!!!")
    #     # print(type(seeds))
    #     # print(seeds.shape)
    #     # print(seeds)
    #     return False, seeds


def get_right_seeds_all(label, cur_image, last_image, seeds_case = 0, rate = 0.4, step = 0.1, thred = 0.6):
    label = np.uint8(label)
    if seeds_case == 0:
        rate = 0.4
        thred = 0.6
    elif seeds_case < 6:
        rate = 0.3
        thred = 0.5
    elif seeds_case == 6:
        rate = 0.2
        thred = 0.4
    seeds = np.zeros((0,2), int)
    seeds_map = np.zeros(label.shape).astype(np.uint8)
    for i in range(1, label.max() + 1):
        curkind_label = np.where(label == i, 1, 0)
        if curkind_label.max() == 0:
            continue
        flag, seed = get_right_seeds(curkind_label, cur_image, last_image, seeds_case, rate, step, thred)
        if flag:
            seeds = np.concatenate((seeds, seed), axis=0)
            for s in range(seed.shape[0]):
                seeds_map[seed[s, 0], seed[s, 1]] = i
        
    
    if seeds.shape[0] == 0:
        return False, None, None
    else:
        return True, seeds, seeds_map


def window_transform(image, windowWidth, windowCenter, normal = True):
    """
    return: trucated image according to window center and window width calculated by last label seeds
    and normalized to [0,1]
    """

    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newing = (image - minWindow)/float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    #将值域转到0-255之间
    if not normal:
        newing = (newing *255).astype('uint8')
    return newing


def get_curclass_label(label, cur_class):
    label = np.uint8(label)
    out = np.zeros(label.shape)
    out = np.where(label == cur_class, 1, 0)

    return np.uint8(out)


def get_multiclass_labels(label, out_channels):
    label = np.uint8(label)
    out = np.zeros((out_channels, label.shape[0], label.shape[1]), dtype=np.uint8)
    for i in range(out_channels):
        out[i,:,:] = get_curclass_label(label, i)

    return out



def generate_interact_dataset(father_path, dataset_data, dataset_label, dataset_len, start_file, end_file, window_transform_flag, FLT_flag, sobel_flag, seeds_flag, crop_size = 256, str_suffix = ".h5"):
    """
    最后生成的是三通道的图像-[原图(window transform)，原图sobel之后的图，seeds]
    大小-depth, height, width
    """
    for cur_file in range(start_file, end_file, 1): 
        """
        还没想好怎么划分训练集和测试集，可能会更改循环条件
        """
        file_name = father_path + str(cur_file) + str_suffix
        if not os.path.exists(file_name):
            continue

        file_image = h5py.File(file_name, 'r')
        image_data = (file_image['image'])[()]
        label_data = (file_image['label'])[()]
        # 让image data的值大于等于0
        image_data = image_data - image_data.min()
        label_data = np.uint8(label_data)
        height, width, depth = label_data.shape
        
        for cur_piece in range(depth):
            # 不考虑没有标注的帧
            if label_data[:,:,cur_piece].max() == 0:
                continue

            cur_image = image_data[:,:,cur_piece]
            cur_label = label_data[:,:,cur_piece]

            # sobel 算法
            # image_float = sitk.Cast(sitk.GetImageFromArray(cur_image), sitk.sitkFloat32)
            # sobel_op = sitk.SobelEdgeDetectionImageFilter()
            # sobel_sitk = sobel_op.Execute(image_float)
            # sobel_sitk = sitk.GetArrayFromImage(sobel_sitk)
            # sobel_sitk = sobel_sitk - sobel_sitk.min()
            # sobel_sitk = sobel_sitk / sobel_sitk.max()

            # for last_flag in [1,-1]:
            #     last_num = cur_piece - last_flag
            #     # last_num本身就不合法
            #     if last_num < 0 or last_num >= depth:
            #         continue
            #     # last_num对应的图片不存在label也不考虑
            #     last_image = image_data[:,:,last_num]
            #     last_label = label_data[:,:,last_num]
            #     if last_label.max() == 0:
            #         continue

            # array_zeros = np.zeros(cur_label.shape)
            # array_ones = np.ones(cur_label.shape)
            
            # label_class = cur_label.max()
            # 每次只分割同一种类的标注
            # for cur_class in range(int(label_class), 0, -1):
                
            #     cur_curkind_label_all = np.where(cur_label == cur_class, array_ones, array_zeros)
                # if cur_curkind_label.max() == 0:
                #     continue
                # last_curkind_label_all = np.where(last_label == cur_class, array_ones, array_zeros)
                # if last_curkind_label_all.max() == 0:
                #     continue
                
                # print(f'current image: {cur_file}, current piece: {cur_piece}, current_class: {cur_class}')
                
                # for class_chosen in range(int(last_label.max()), 0, -1):
                #     last_chosen_label = np.where(last_label == class_chosen, array_ones, array_zeros)
                #     if last_chosen_label.max() == 0:
                #         continue
                #     last_chosen_distance = last_chosen_label - cur_curkind_label
                #     last_curkind_distance = last_curkind_label - cur_curkind_label
                #     a_chosen, b_chosen = np.where(last_chosen_distance != 0)
                #     a_curkind, b_curkind = np.where(last_curkind_distance != 0)
                #     if a_chosen.shape[0] < a_curkind.shape[0]:
                #         print(f"ERROR!!!! Wrong label in image {cur_file}, piece {cur_piece}, label of current piece: {cur_class}, distance: {a_curkind.shape[0]}, better label: {class_chosen}, distance: {a_chosen.shape[0]}")
                #         last_curkind_label = last_chosen_label
                # get seeds from last label

            # 同一种类标注得到连通分量
            for cur_class in range(1, cur_label.max() + 1):
                cur_curclass_label = np.where(cur_label == cur_class, 1, 0)
                cur_connected_num, cur_connected_labels = cv2.connectedComponents(np.uint8(cur_curclass_label))
                cur_connected_labels = np.uint8(cur_connected_labels)

                for cur_region in range(1, cur_connected_num):
                    cur_curregion_label = np.where(cur_connected_labels == cur_region, 1, 0)
                
                    for seeds_case in range(7):
                        flag, seeds = get_right_seeds(cur_curregion_label, cur_image, cur_image, seeds_case)
                        if not flag:
                            print(f"ERROR!!!!! Cannot get right seeds! cur image: {cur_file}, cur piece: {cur_piece}, cur label class: {cur_region} -- there is no seed!")
                            continue
                        
                        # 调整窗位窗宽
                        ele = []
                        for i in range(seeds.shape[0]):
                            ele.append(cur_image[seeds[i,0], seeds[i,1]])
                        ele = np.array(ele)

                        cur_image_processed = window_transform(cur_image, max(ele.max() - ele.min() + 2 * np.sqrt(ele.var()), 255), (ele.max() + ele.min()) / 2) if window_transform_flag else cur_image

                        # 得到seeds图
                        seeds_image = np.zeros(cur_label.shape)
                        for i in range(seeds.shape[0]):
                            seeds_image[seeds[i,0], seeds[i,1]] = 1


                        # sobel 算法
                        sobel_sitk = get_sobel_image(cur_image)# if sobel_flag else last_label

                        # 将三者重叠起来
                        cur_curkind_data = np.stack((cur_image_processed, sobel_sitk, seeds_image))#  if seeds_flag else np.stack((cur_image_processed, sobel_sitk, region_grow(cur_image_processed, seeds)))
                        # cur_curkind_data = np.stack((cur_image, sobel_sitk, seeds_image))#  if feature_flag else np.stack((cur_image, seeds_image))
                        # cur_curkind_label 
                        """↑这是一对数据"""
                        dataset_data.append(cur_curkind_data)
                        dataset_label.append(cur_curregion_label)
                        dataset_len = dataset_len + 1

                        print(f'cur image: {cur_file}, cur piece: {cur_piece}, cur class: [{cur_class} / {cur_label.max()}] cur region: [{cur_region} / {cur_connected_num - 1}] cur seeds case: {seeds_case}')

                        # if not sobel_flag:
                        #     zero_array = np.zeros(cur_image.shape)
                        #     cur_curkind_data = np.stack((cur_image_processed, zero_array, seeds_image))
                        #     dataset_data.append(cur_curkind_data)
                        #     dataset_label.append(cur_curkind_label)
                        #     dataset_len = dataset_len + 1

    return dataset_data, dataset_label, dataset_len

class interact_dataset_image(Dataset):
    def __init__(self, three_class_path = None, start_file3 = None, end_file3 = None, two_class_path = None, start_file2 = None, end_file2 = None, window_transform_flag = True, FLT_flag = True, sobel_flag = True, seeds_flag = True) -> None:
        super(interact_dataset_image, self).__init__()
        self.dataset_data = []
        self.dataset_label = []
        self.dataset_len = 0
        
        if three_class_path != None:
            self.dataset_data, self.dataset_label, self.dataset_len = generate_interact_dataset(three_class_path, self.dataset_data, self.dataset_label, self.dataset_len, start_file3, end_file3, window_transform_flag, FLT_flag, sobel_flag, seeds_flag)
        if two_class_path != None:
            self.dataset_data, self.dataset_label, self.dataset_len = generate_interact_dataset(two_class_path, self.dataset_data, self.dataset_label, self.dataset_len, start_file2, end_file2, window_transform_flag, FLT_flag, sobel_flag, seeds_flag)

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.dataset_data[index], self.dataset_label[index]
    
def get_sobel_image(cur_image):
    image_float = sitk.Cast(sitk.GetImageFromArray(cur_image), sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.GetArrayFromImage(sobel_sitk)
    sobel_sitk = sobel_sitk - sobel_sitk.min()
    sobel_sitk = sobel_sitk / sobel_sitk.max()

    return sobel_sitk
        
        
def generate_interact_dataset_all(father_path, dataset_data, dataset_label, dataset_len, start_file, end_file, window_transform_flag, FLT_flag, sobel_flag, feature_flag, crop_size = 256, str_suffix = ".h5"):
    """
    最后生成的是三通道的图像-[原图(window transform)，原图sobel之后的图，seeds]
    大小-depth, height, width
    """
    n_classes = 3
    for cur_file in range(start_file, end_file, 1): 
        """
        还没想好怎么划分训练集和测试集，可能会更改循环条件
        """
        file_name = father_path + str(cur_file) + str_suffix
        if not os.path.exists(file_name):
            continue

        file_image = h5py.File(file_name, 'r')
        image_data = (file_image['image'])[()]
        label_data = (file_image['label'])[()]
        # 让image data的值大于等于0
        image_data = image_data - image_data.min()
        label_data = np.uint8(label_data)
        if not FLT_flag:
            label_data = np.where(label_data == 3, 0, label_data)
            

        height, width, depth = label_data.shape
        
        for cur_piece in range(depth):
            # 不考虑没有标注的帧
            if label_data[:,:,cur_piece].max() == 0:
                continue

            cur_image = image_data[:,:,cur_piece]
            cur_label = label_data[:,:,cur_piece].astype(np.uint8)

            for last_flag in [1,-1]:
                break_flag = False
                last_num = cur_piece - last_flag
                # last_num本身就不合法
                if last_num < 0 or last_num >= depth:
                    continue
                # last_num对应的图片不存在label也不考虑
                last_image = image_data[:,:,last_num]
                last_label = label_data[:,:,last_num]
                if last_label.max() == 0:
                    continue
                if last_label.max() < cur_label.max():
                    continue

                class_num = last_label.max()
                
                for cur_class in range(1, class_num + 1):
                    last_curkind_label = np.where(last_label == cur_class, cur_class, 0)
                    cur_curkind_label = np.where(cur_label == cur_class, cur_class, 0)
                    cur_connected_num, _ = cv2.connectedComponents(np.uint8(cur_curkind_label))
                    last_connected_num, _ = cv2.connectedComponents(np.uint8(last_curkind_label))
                    
                    if last_connected_num < cur_connected_num:
                        break_flag = True
                        break
                if break_flag:
                    continue

                for seeds_case in range(7):
                    flag, seeds, seeds_image = get_right_seeds_all(last_label, cur_image, last_image, seeds_case)
                    if not flag:
                        print(f"ERROR!!!!! Cannot get right seeds! cur image: {cur_file}, cur piece: {cur_piece} -- there is no seed!")
                        continue
                    else:
                        if len(np.unique(seeds_image)) < len(np.unique(cur_label)):
                            print(f"ERROR!!!!! Current label has more kinds of labels than seeds image!")
                            continue
                    
                    print(f'current image: {cur_file}, current piece: {cur_piece}, last piece: {last_num}, seeds case: {seeds_case}')
                    # 调整窗位窗宽
                    ele = []
                    for i in range(seeds.shape[0]):
                        ele.append(cur_image[seeds[i,0], seeds[i,1]])
                    ele = np.array(ele)

                    cur_image_processed = window_transform(cur_image, max(ele.max() - ele.min() + 2 * np.sqrt(ele.var()), 255), (ele.max() + ele.min()) / 2) if window_transform_flag else cur_image

                    # sobel 算法
                    sobel_sitk = get_sobel_image(cur_image) if sobel_flag else last_label

                    # 将三者重叠起来
                    cur_curkind_data = np.stack((cur_image_processed, sobel_sitk, seeds_image)) if feature_flag else np.stack((cur_image_processed, seeds_image))
                    # cur_curkind_label 
                    """↑这是一对数据"""
                    dataset_data.append(cur_curkind_data)
                    # dataset_label.append(get_multiclass_labels(cur_label, n_classes + 1))
                    dataset_label.append(cur_label)
                    dataset_len = dataset_len + 1

                    if not sobel_flag:
                        zero_array = np.zeros(cur_image.shape)
                        cur_curkind_data = np.stack((cur_image_processed, zero_array, seeds_image))
                        dataset_data.append(cur_curkind_data)
                        dataset_label.append(cur_label)
                        dataset_len = dataset_len + 1


    return dataset_data, dataset_label, dataset_len

class interact_dataset_image_all(Dataset):
    def __init__(self, three_class_path = None, start_file3 = None, end_file3 = None, two_class_path = None, start_file2 = None, end_file2 = None, window_transform_flag = True, FLT_flag = True, sobel_flag = True, feature_flag = 0) -> None:
        super(interact_dataset_image_all, self).__init__()
        self.dataset_data = []
        self.dataset_label = []
        self.dataset_len = 0
        
        if three_class_path != None:
            self.dataset_data, self.dataset_label, self.dataset_len = generate_interact_dataset_all(three_class_path, self.dataset_data, self.dataset_label, self.dataset_len, start_file3, end_file3, window_transform_flag, FLT_flag, sobel_flag, feature_flag)
        if two_class_path != None:
            self.dataset_data, self.dataset_label, self.dataset_len = generate_interact_dataset_all(two_class_path, self.dataset_data, self.dataset_label, self.dataset_len, start_file2, end_file2, window_transform_flag, FLT_flag, sobel_flag, feature_flag)

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.dataset_data[index], self.dataset_label[index]
        
    

class interact_dataset_file(Dataset):
    def __init__(self, datapath):
        super(interact_dataset_file, self).__init__()
        data_file = h5py.File(datapath, 'r')
        self.image = (data_file['image'])[()]
        self.label = (data_file['label'])[()]
        self.size = self.image.shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.image[index], self.label[index]
    

def save2h5(path, name_list, data_list):
    hf = h5py.File(path, 'w')
    for i in range(len(name_list)):
        hf.create_dataset(name_list[i], data=data_list[i])
    hf.close()
        

if __name__ == '__main__':
    # 67 for train dataset, 33 for test dataset
    # train_dataset = interact_dataset(r'/data/xuxin/ImageTBAD_processed/three_class/', 3, 128, r'/data/xuxin/ImageTBAD_processed/two_class/', 2, 94)
    """先改成一半来debug"""
    # train_dataset = interact_dataset_image(r'/data/xuxin/ImageTBAD_processed/three_class/', 47, 128, r'/data/xuxin/ImageTBAD_processed/two_class/', 40, 94)
    # test_dataset = interact_dataset_image(r'/data/xuxin/ImageTBAD_processed/three_class/', 129, 193, r'/data/xuxin/ImageTBAD_processed/two_class/', 99, 161)
    test_dataset = interact_dataset_image(r'/data/xuxin/ImageTBAD_processed/three_class/', 129, 140, r'/data/xuxin/ImageTBAD_processed/two_class/', 99, 110)
    # train_dataset = interact_dataset(r'/data/xuxin/ImageTBAD_processed/three_class/', 3, 7, r'/data/xuxin/ImageTBAD_processed/two_class/', 2, 3)

    # print(f'The number of train dataset is: {train_dataset.dataset_len}')
    print(f'The number of test dataset is: {test_dataset.dataset_len}')

    # train_data_path = r'/data/xuxin/ImageTBAD_processed/train_dataset_all.h5'
    test_data_path = r'/data/xuxin/ImageTBAD_processed/test_dataset_all.h5'

    # save2h5(train_data_path, ['image','label'], [np.array(train_dataset.dataset_data), np.array(train_dataset.dataset_label)])
    save2h5(test_data_path,  ['image','label'], [np.array(test_dataset.dataset_data), np.array(test_dataset.dataset_label)])