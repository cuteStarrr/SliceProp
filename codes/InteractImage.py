"""
数据格式记录：
image_src(h5文件)：height，width，depth的三维数组，取值范围[0,1]
每次选取特定depth，则处理二维矩阵，height——行数，width——列数——直接视作2D图像

anotation_output(用来表示标注数据)，depth，height，width，red，green，blue三通道（因为QImage是RGB三通道显示的, opencv是BGR三通道显示的）

"""

import os
from collections import OrderedDict
from os.path import join as opj
from turtle import left, shape
from typing import Any
from xml.sax.handler import DTDHandler

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import color, measure
import matplotlib.pyplot as plt
import h5py
from UNet import *
from interact_dataset import *
from train import accuracy_all_numpy
from test import get_prediction_all_bidirectional, get_network_input_all, get_prediction_all
from region_grow import *
from medpy.metric import binary


"""
NEED TO DO:
1. 更改训练集 -- 得到更好的pth
    1. 改变rate -- 0.1就行了 贴近实际用户分割 -- 改成上一次的分类rate
    2. scribble loss 交叉熵
    3. 增加一个seeds的种类 -- 雪花状噪声
2. refinement时确定不好的帧的标准可能需要改进 -- loss + region loss -- region loss不是特别对
"""


class InteractImage(object):
    def __init__(self, image_path):
        """
        add_seed的坐标对应方式与原始图像，也就是h5文件一致
        """
        self.FL_flag = False
        self.TL_flag = False
        self.background_flag = False
        self.TL_label = int(1)
        self.FL_label = int(2)
        self.background_label = int(3) # 与0做区分
        self.TL_color = (0, 0, 255)
        self.FL_color = (255, 0, 0) # BGR
        self.background_color = (0, 255, 0)
        if image_path[-3:] == '.h5':
            file = h5py.File(image_path, 'r')
            self.image = (file['image'])[()]
            self.label = (file['label'])[()]
            self.image = self.image - self.image.min()
            self.label = np.uint8(self.label)
        else:
            file_name_label = image_path[:-12] + "_seg.nii.gz"
            image_obj = nib.load(image_path)
            label_obj = nib.load(file_name_label)
            self.image = image_obj.get_fdata()
            self.label = label_obj.get_fdata()
            # 让image data的值大于等于0
            self.image = self.image - self.image.min()
            self.label = np.where(self.label > 1.5, 0, self.label)
            self.label = np.uint8(self.label)
        # self.label = np.uint8((file['label'])[()])
        # self.image = self.image / self.image.max()
        self.height, self.width, self.depth = self.image.shape
        self.depth_current = self.depth // 2
        self.depth_anotate = self.depth_current
        """一帧一帧地segment and refine"""
        self.uncertainty_pieces = np.zeros((self.depth))
        self.isrefine_flag = np.zeros((self.depth), dtype=np.uint8)
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        self.prediction = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
        self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.uncertainty_thred = 0
        self.depth_initseg = -1

        self.dice_coeff_thred = 0.75
        self.penthickness = 1

    def set_depth(self, depth):
        self.depth_current = depth

    def set_uncertainty(self, depth, uncertainty):
        self.uncertainty_pieces[depth] = uncertainty

    def set_anotate_depth(self, depth):
        self.depth_anotate = depth

    def get_refine_flag(self, depth):
        return self.isrefine_flag[depth]
    

    def get_test_evaluation(self):
        array_predict_tl = np.bool_(np.where(self.prediction == self.TL_label, 1, 0))
        image_label_tl = np.bool_(np.where(self.label == self.TL_label, 1, 0))
        array_predict_fl = np.bool_(np.where(self.prediction == self.FL_label, 1, 0))
        image_label_fl = np.bool_(np.where(self.label == self.FL_label, 1, 0))
        array_predict = np.bool_(np.where(self.prediction > 0, 1, 0))
        image_label = np.bool_(np.where(self.label > 0, 1, 0))
        
        dc1,dc2,dc3,hd1,hd2,hd3 = binary.dc(array_predict_tl, image_label_tl), binary.dc(array_predict_fl, image_label_fl) , binary.dc(array_predict, image_label), binary.hd(array_predict_tl, image_label_tl), binary.hd(array_predict_fl, image_label_fl), binary.hd(array_predict, image_label)
                
        return dc1,dc2,dc3,hd1,hd2,hd3

    def get_scribble_loss(self, prediction, seeds_map):
        """
        prediction: numpy
        seeds_map: numpy
        """
        prediction = np.uint8(prediction)
        seeds_map = np.uint8(seeds_map)
        add_array = prediction + seeds_map
        zero_num = np.sum(add_array == 0)
        right_num = np.sum(prediction == seeds_map) - zero_num
        total_num = np.sum(seeds_map > 0)

        return (total_num - right_num) / total_num
    

    def get_region_loss(self, prediction):
        prediction = np.uint8(prediction)
        total_loss = 0
        connected_array = np.uint8(np.where(prediction > 0, 1, 0))
        region_num, regions = cv2.connectedComponents(connected_array)
        for cur_region in range(region_num - 1, 0, -1):
            region_mask = regions > cur_region - 0.5
            regions[regions > cur_region - 0.5] = 0
            crop_prediction = np.uint8(np.where(region_mask, prediction, 0))
            TL_num = np.sum(crop_prediction == self.TL_label)
            FL_num = np.sum(crop_prediction == self.FL_label)
            # print(crop_prediction.shape)
            # print(TL_num.shape)
            # print(TL_num, FL_num, np.sum(crop_prediction > 0))
            total_loss = total_loss + min(TL_num, FL_num) / max(TL_num, FL_num)

        return total_loss / region_num


    def get_scribble_loss_plus_region_loss(self, prediction, seeds_map):
        if prediction.max() < 0.5:
            return 0
        return self.get_scribble_loss(prediction=prediction, seeds_map=seeds_map) + self.get_region_loss(prediction=prediction)

    def gray2BGRImage(self, gray_image_src):
        gray_image = gray_image_src.copy()
        gray_image = np.uint8(cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX))
        # print(gray_image.max())
        # print(gray_image.min())
        # print(type(gray_image))
        # gray_image = (gray_image * 255).astype(np.uint8)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def getImage2show(self):
        return cv2.addWeighted(self.gray2BGRImage(self.image[:, :, self.depth_current]), 0.9, self.anotation[self.depth_current], 0.7, 0.7)
    
    def tmpseedsCoords2map(self):
        # return np.uint8(self.TL_seeds[:,:,depth] * self.TL_label + self.FL_seeds[:,:,depth] * self.FL_label)
        return np.uint8(np.where(self.tmp_seeds == self.background_label, 0, self.tmp_seeds))
        
    def prediction2anotation(self, seeds_flag=False):
        for i in range(self.depth):
            """for test"""
            
            mask_TL = np.uint8(self.prediction[:,:,i] == self.TL_label)
            tmp_TL = self.gray2BGRImage(mask_TL)
            tmp_TL = np.where(tmp_TL == [0, 0, 0], [0, 0, 0], list(self.TL_color))
            mask_FL = np.uint8(self.prediction[:,:,i] == self.FL_label)
            tmp_FL = self.gray2BGRImage(mask_FL)
            tmp_FL = np.where(tmp_FL == [0, 0, 0], [0, 0, 0], list(self.FL_color))

            if seeds_flag:
                mask_TL_seeds = np.uint8(self.TL_seeds[:,:,i] == 1)
                tmp_TL_seeds = self.gray2BGRImage(mask_TL_seeds)
                tmp_TL_seeds = np.where(tmp_TL_seeds == [0, 0, 0], [0, 0, 0], list(self.background_color))
                mask_FL_seeds = np.uint8(self.FL_seeds[:,:,i] == 1)
                tmp_FL_seeds = self.gray2BGRImage(mask_FL_seeds)
                tmp_FL_seeds = np.where(tmp_FL_seeds == [0, 0, 0], [0, 0, 0], list(self.background_color))
                
                """for test"""
                self.anotation[i,:,:,:] = tmp_TL + tmp_FL + tmp_TL_seeds + tmp_FL_seeds
            else:
                self.anotation[i,:,:,:] = tmp_TL + tmp_FL
            # self.anotation[i,:,:] = np.where(self.prediction[:,:,i] == self.TL_label, np.array(self.TL_color), self.anotation[i,:,:])
            # self.anotation[i,:,:] = np.where(self.prediction[:,:,i] == self.FL_label, np.array(self.FL_color), self.anotation[i,:,:])

    
    def get_prediction_all_basedon_anotation(self, model, indata, seeds_map):
        """
        除了不确定性 还需要考虑 
        scribble loss
        同一连通区域不应该有多种label
        """
        prediction = model(indata).squeeze()
        # prediction = torch.softmax(prediction, dim=0)
        uncertainty =  -torch.sum(prediction * torch.log(prediction   + 1e-16), dim=0).cpu().detach().numpy()
        # print(uncertainty.shape)
        # prediction = torch.sigmoid(prediction).detach().numpy()
        prediction = prediction.detach().numpy()
        # prediction = prediction - prediction.min()
        # prediction = prediction / prediction.max()
        prediction = np.uint8(np.argmax(prediction, axis=0))
        prediction_mask = prediction > 0
        seeds_map_mask = seeds_map > 0
        cal_num = (np.sum(prediction_mask) - np.sum(seeds_map_mask))


        return prediction, np.sum(uncertainty[prediction_mask]) / cal_num if cal_num else 0
    


    def get_prediction_with_seeds_map(self, cur_image, seeds_map, window_transform_flag, model, device):
        indata = get_network_input_all(cur_image, np.argwhere(seeds_map > 0), seeds_map, window_transform_flag)
        indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        prediction, uncertainty = get_prediction_all(model, indata)
        prediction = np.uint8(prediction)

        return prediction, uncertainty
        
    def get_prediction_intergrate_known_seeds(self, last_label, cur_image, last_image, window_transform_flag, device, model, seeds_case, depth, clean_region_flag, clean_seeds_flag):
        """
        for init segment, seeds are all regarded right
        """
        flag, seeds, seeds_map = get_right_seeds_all(last_label, cur_image, last_image, seeds_case=seeds_case, clean_region_flag=clean_region_flag, clean_seeds_flag=clean_seeds_flag)
        if not flag:
            return False, None, None, None
        seeds_map = np.where(self.TL_seeds[:,:,depth] == 1, self.TL_label, seeds_map)
        seeds_map = np.where(self.FL_seeds[:,:,depth] == 1, self.FL_label, seeds_map)
        seeds = np.argwhere(seeds_map > 0)
        # plt.imshow(seeds_map, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # print("seeds")
        # if not flag:
        #     return False, None, None
        if seeds_map.max() < 0.5:
            return False, None, None, None
        indata = get_network_input_all(cur_image, seeds, seeds_map, window_transform_flag)
        # print("input")
        
        indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        prediction, uncertainty = get_prediction_all(model, indata)
        # print("prediction")
        prediction = np.uint8(prediction)

        return True, prediction, seeds_map, uncertainty

    
    def init_segment(self, model, device):
        window_transform_flag = True
        clean_seeds_flag = True
        clean_region_flag = False

        cur_image = self.image[:,:,self.depth_anotate]
        last_image = self.image[:,:,self.depth_anotate]
        last_label = self.tmpseedsCoords2map()
        # last_label = self.label[:,:,self.depth_current]
        seeds_map = self.tmpseedsCoords2map()
        if not self.TL_flag and not self.FL_flag and not self.background_flag:
            # change start_piece to get better performance
            start_piece = int(self.depth / 4)
        
            start_label = self.label[:,:,start_piece]
            while start_label.max() < 0.5:
                start_piece += 1
                start_label = self.label[:,:,start_piece]
            self.depth_anotate = start_piece
            cur_image = self.image[:,:,start_piece]
            last_image = self.image[:,:,start_piece]
            last_label = start_label
            _, _, seeds_map = get_right_seeds_all(last_label, cur_image, last_image, seeds_case=0, clean_region_flag=clean_region_flag)
        
        self.depth_initseg = self.depth_anotate
        for i in range(self.depth_anotate, self.depth):
            # print("start one piece")
            cur_image = self.image[:,:,i]
            """test"""
            flag = True
            prediction = last_label
            uncertainty = 0
            if i == self.depth_anotate:
                prediction, uncertainty = self.get_prediction_with_seeds_map(cur_image, seeds_map, window_transform_flag, model, device)
                # indata = get_network_input_all(cur_image, np.argwhere(seeds_map > 0), seeds_map, window_transform_flag)
                # indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                # prediction = get_prediction_all(model, indata)
                # prediction = np.uint8(prediction)
                # print("get prediction - 1")
            else:
                flag, prediction,seeds_map, uncertainty = self.get_prediction_intergrate_known_seeds(last_label, cur_image, last_image, window_transform_flag, device, model, seeds_case = 0, depth=i, clean_region_flag=clean_region_flag, clean_seeds_flag=clean_seeds_flag)
            # print("get prediction - 2")
            if not flag:
                break
            if prediction.max() < 0.5:
                self.uncertainty_pieces[i] = 0
                break
            # print(np.unique(prediction, return_counts = True))
            # print(prediction.shape)
            #prediction, uncertainty = self.mask_prediction_with_newadded_TLFL_seeds(prediction=prediction, seeds_map=seeds_map, uncertainty=uncertainty)
            self.prediction[:,:,i] = prediction
            uncertainty += self.get_scribble_loss_plus_region_loss(prediction=prediction, seeds_map=seeds_map)
            self.uncertainty_pieces[i] = uncertainty
            # if i == 157:
            #     plt.imshow(seeds_map, cmap='gray')
            #     plt.axis('off')
            #     plt.show()
            # if i != self.depth_current: 
            self.TL_seeds[:,:,i] = np.where(seeds_map == self.TL_label, 1, self.TL_seeds[:,:,i])
            self.FL_seeds[:,:,i] = np.where(seeds_map == self.FL_label, 1, self.FL_seeds[:,:,i])
            # print("get seeds for each piece - 1")
            cur_piece = i
            cur_coeff = accuracy_all_numpy(self.prediction[:,:,cur_piece-1], self.prediction[:,:,cur_piece])
            # print("cal acc - 1")
            while cur_piece > 0 and cur_coeff  < self.dice_coeff_thred:
                roll_flag, roll_prediction, roll_seeds_map, roll_uncertainty = self.get_prediction_intergrate_known_seeds(self.prediction[:,:,cur_piece], self.image[:,:,cur_piece-1], self.image[:,:,cur_piece], window_transform_flag, device, model, seeds_case = 0, depth=cur_piece-1, clean_region_flag=clean_region_flag, clean_seeds_flag=clean_seeds_flag)
                # plt.imshow(roll_seeds_map, cmap='gray')
                # plt.axis('off')
                # plt.show()
                # print("get prediction - 3")
                if not roll_flag:
                    break
                if roll_prediction.max() < 0.5:
                    self.uncertainty_pieces[cur_piece-1] = 0
                    break
                #roll_prediction, roll_uncertainty = self.mask_prediction_with_newadded_TLFL_seeds(prediction=roll_prediction, seeds_map=roll_seeds_map, uncertainty=roll_uncertainty)
                if accuracy_all_numpy(self.prediction[:,:,cur_piece - 1], roll_prediction) < 0.98:
                    self.prediction[:,:,cur_piece - 1] = roll_prediction
                    roll_uncertainty += self.get_scribble_loss_plus_region_loss(prediction=roll_prediction, seeds_map=roll_seeds_map)
                    self.uncertainty_pieces[cur_piece-1] = roll_uncertainty
                    # plt.imshow(roll_prediction, cmap='gray')
                    # plt.axis('off')
                    # plt.show()
                    # self.prediction2anotation(cur_piece-1)
                    # print("cal acc - 2")
                    self.TL_seeds[:,:,cur_piece - 1] = np.where(roll_seeds_map == self.TL_label, 1, self.TL_seeds[:,:,cur_piece - 1])
                    self.FL_seeds[:,:,cur_piece - 1] = np.where(roll_seeds_map == self.FL_label, 1, self.FL_seeds[:,:,cur_piece - 1])
                    # print("get seeds for each piece - 2")
                else:
                    break
                cur_piece = cur_piece - 1
                # """test"""
                # if cur_piece == 41:
                #     return
                cur_coeff = accuracy_all_numpy(self.prediction[:,:,cur_piece-1], self.prediction[:,:,cur_piece])
                print(f'cur piece: [{cur_piece}/{self.depth}]')
                # print("cal acc - 3")
            last_image = self.image[:,:,i]
            last_label = prediction
            print(f'cur piece: [{i}/{self.depth}]')
        """delete tmp_seeds"""
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        self.uncertainty_thred = self.uncertainty_pieces.mean()
        print("finish init segmentation")
        # dc1,dc2,dc3,hd1,hd2,hd3 = self.get_test_evaluation()
        # print('TL acc: %.5f, FL acc: %.5f, acc: %.5f, hd tl: %.5f, hd fl: %.5f, hd: %.5f' % (dc1,dc2,dc3,hd1,hd2,hd3))
        print("---------------- uncertainty info -----------------")
        print("max uncertainty: ", self.uncertainty_pieces.max())
        print("min uncertainty: ", self.uncertainty_pieces.min())
        print("mean uncertainty: ", self.uncertainty_pieces.mean())


    def get_max_uncertainty(self):
        return np.argmax(self.uncertainty_pieces)
    
    def get_min_uncertainty(self):
        return np.argmin(self.uncertainty_pieces)
    
    def get_uncertainty(self, depth):
        return self.uncertainty_pieces[depth]
    
    def seedsArray2map(self, depth):
        """TL FL seeds to seeds map"""
        return np.uint8(self.TL_seeds[:,:,depth] * self.TL_label + self.FL_seeds[:,:,depth] * self.FL_label)

    
    def integrate_tmp_seeds_nobackground(self, depth):
        """
        return refined seeds map based on newly added tmp_seeds
        """
        self.TL_seeds[:,:,depth] = np.where(self.tmp_seeds == self.TL_label, 1, self.TL_seeds[:,:,depth])
        self.FL_seeds[:,:,depth] = np.where(self.tmp_seeds == self.FL_label, 1, self.FL_seeds[:,:,depth])

        return self.seedsArray2map(depth=depth)
    

    def delete_badseeds_basedon_newadded_TLFL_seeds(self, depth, TL_seeds_new_mask, FL_seeds_new_mask):
        """针对FL_seeds"""
        old_seeds = self.FL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num-1, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if cur_block_seeds[TL_seeds_new_mask].any() == 1:
                """即FL_seesd与用户标注的TL_seeds冲突 应去掉原有的FL_seeds的这一个连通分量"""
                self.FL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.FL_seeds[:,:,depth])
        # self.FL_seeds[:,:,depth][TL_seeds_new_mask] = 0

        """针对TL_seeds"""
        old_seeds = self.TL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num-1, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if cur_block_seeds[FL_seeds_new_mask].any() == 1:
                """即TL_seesd与用户标注的FL_seeds冲突 应去掉原有的TL_seeds的这一个连通分量"""
                self.TL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.TL_seeds[:,:,depth])
        # self.TL_seeds[:,:,depth][FL_seeds_new_mask] = 0


        return self.seedsArray2map(depth=depth)
    

    def delete_badseeds_basedon_newadded_background_seeds(self, depth, background_seeds_new_mask, hard_flag = True):
        """可能需要再考虑一下 去掉连通分量还是只去掉background的部分"""
        if hard_flag:
            """针对FL_seeds"""
            old_seeds = self.FL_seeds[:,:,depth]
            block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
            for cur_block in range(block_num-1, 0, -1):
                cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
                seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
                if cur_block_seeds[background_seeds_new_mask].any() == 1:
                    """即FL_seesd与用户标注的TL_seeds冲突 应去掉原有的FL_seeds的这一个连通分量"""
                    self.FL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.FL_seeds[:,:,depth])

            """针对TL_seeds"""
            old_seeds = self.TL_seeds[:,:,depth]
            block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
            for cur_block in range(block_num-1, 0, -1):
                cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
                seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
                if cur_block_seeds[background_seeds_new_mask].any() == 1:
                    """即TL_seesd与用户标注的FL_seeds冲突 应去掉原有的TL_seeds的这一个连通分量"""
                    self.TL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.TL_seeds[:,:,depth])
        else:
            self.TL_seeds[:,:,depth][background_seeds_new_mask] = 0
            self.FL_seeds[:,:,depth][background_seeds_new_mask] = 0


        return self.seedsArray2map(depth=depth)
    

    def integrate_refine_seedsmap_with_oldseeds(self, refine_seeds_map_ori, depth):
        refine_seeds_map_ori = np.uint8(refine_seeds_map_ori)
        refine_seeds_map = refine_seeds_map_ori.copy()
        background_seeds_mask = self.background_seeds[:,:,depth] == 1
        if background_seeds_mask.any():
            self.delete_badseeds_basedon_newadded_background_seeds(depth=depth, background_seeds_new_mask=background_seeds_mask)
        """针对FL_seeds"""
        old_seeds = self.FL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num-1, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            cur_block_seeds_mask = seeds_blocks > cur_block - 0.5
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if refine_seeds_map[cur_block_seeds_mask].all() != self.TL_label:
                refine_seeds_map = np.where(cur_block_seeds == 1, self.FL_label, refine_seeds_map)
    

        """针对TL_seeds"""
        old_seeds = self.TL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num-1, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            cur_block_seeds_mask = seeds_blocks > cur_block - 0.5
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if refine_seeds_map[cur_block_seeds_mask].all() != self.FL_label:
                refine_seeds_map = np.where(cur_block_seeds == 1, self.TL_label, refine_seeds_map)

        refine_seeds_map = np.uint8(refine_seeds_map)

        if (refine_seeds_map == self.seedsArray2map(depth=depth)).all():
            return False, refine_seeds_map
        else:
            return True, refine_seeds_map
        

    
    def delete_prediction_basedon_backgroundseeds(self, anotate_prediction, background_seeds_new_mask, uncertainty):
        if (self.tmp_seeds == self.TL_label).any() or (self.tmp_seeds == self.FL_label).any():
            old_num = np.sum(anotate_prediction > 0)
            anotate_prediction[background_seeds_new_mask] = 0
            return np.uint8(anotate_prediction), uncertainty / old_num * np.sum(anotate_prediction > 0)
        else:
            old_num = np.sum(anotate_prediction > 0)
            num, blocks = cv2.connectedComponents((np.uint8(anotate_prediction)))
            for cur_block in range(num-1, 0, -1):
                cur_block_seeds = np.uint8(np.where(blocks > cur_block - 0.5, 1, 0))
                cur_block_seeds_mask = blocks > cur_block - 0.5
                blocks[blocks > cur_block - 0.5] = 0

                if cur_block_seeds[background_seeds_new_mask].any():
                    anotate_prediction[cur_block_seeds_mask] = 0
            return np.uint8(anotate_prediction), uncertainty / old_num * np.sum(anotate_prediction > 0)

    
    

    def mask_prediction_with_newadded_TLFL_seeds(self, prediction_ori, seeds_map, uncertainty):
        prediction = prediction_ori.copy()
        seeds_map_curkind = np.uint8(np.where(seeds_map == self.TL_label, 1, 0))
        prediction_curkind = np.uint8(np.where(prediction == self.FL_label, 1, 0))
        seeds_block_num, seeds_blocks = cv2.connectedComponents(seeds_map_curkind)
        labels_block_num, labels_blocks = cv2.connectedComponents(prediction_curkind)
        for cur_block in range(seeds_block_num-1, 0, -1):
            cur_block_seeds_mask = seeds_blocks > cur_block - 0.5
            # cur_block_seeds = np.uint8(np.where(cur_block_seeds_mask, 1, 0))
            seeds_blocks[cur_block_seeds_mask] = 0

            for cur_label in range(labels_block_num-1, 0, -1):
                cur_block_labels_mask = labels_blocks > cur_label - 0.5
                cur_block_labels = np.uint8(np.where(cur_block_labels_mask, 1, 0))
                labels_blocks[cur_block_labels_mask] = 0

                if cur_block_labels[cur_block_seeds_mask].any():
                    # tmp_prediction = prediction.copy()
                    prediction[cur_block_labels_mask] = self.TL_label
                    # if accuracy_all_numpy(tmp_prediction, prediction) > 0.85:
                    #     prediction = tmp_prediction
        
        seeds_map_curkind = np.uint8(np.where(seeds_map == self.FL_label, 1, 0))
        prediction_curkind = np.uint8(np.where(prediction == self.TL_label, 1, 0))
        seeds_block_num, seeds_blocks = cv2.connectedComponents(seeds_map_curkind)
        labels_block_num, labels_blocks = cv2.connectedComponents(prediction_curkind)
        for cur_block in range(seeds_block_num-1, 0, -1):
            cur_block_seeds_mask = seeds_blocks > cur_block - 0.5
            # cur_block_seeds = np.uint8(np.where(cur_block_seeds_mask, 1, 0))
            seeds_blocks[cur_block_seeds_mask] = 0

            for cur_label in range(labels_block_num-1, 0, -1):
                cur_block_labels_mask = labels_blocks > cur_label - 0.5
                cur_block_labels = np.uint8(np.where(cur_block_labels_mask, 1, 0))
                labels_blocks[cur_block_labels_mask] = 0

                if cur_block_labels[cur_block_seeds_mask].any():
                    # tmp_prediction = prediction.copy()
                    prediction[cur_block_labels_mask] = self.FL_label
                    # if accuracy_all_numpy(tmp_prediction, prediction) > 0.85:
                    #     prediction = tmp_prediction

        # prediction_new = np.where(seeds_map == self.TL_label, self.TL_label, prediction)
        # prediction_new = np.where(seeds_map == self.FL_label, self.FL_label, prediction_new)
        # if accuracy_all_numpy(prediction_ori, prediction) < 0.85:
        #     prediction = np.where(seeds_map == self.TL_label, self.TL_label, prediction_ori)
        #     prediction = np.where(seeds_map == self.FL_label, self.FL_label, prediction)
        # else:
        prediction = np.where(seeds_map == self.TL_label, self.TL_label, prediction)
        prediction = np.where(seeds_map == self.FL_label, self.FL_label, prediction)
        total_num = np.sum(prediction_ori > 0)
        sure_num = np.sum(seeds_map > 0)

        return np.uint8(prediction), uncertainty / total_num * (total_num - sure_num) if total_num > sure_num else 0
    

    def mask_prediction_with_newadded_TLFL_seeds_notregion(self, prediction, seeds_map, uncertainty):

        prediction_new = np.where(seeds_map == self.TL_label, self.TL_label, prediction)
        prediction_new = np.where(seeds_map == self.FL_label, self.FL_label, prediction_new)
        total_num = np.sum(prediction > 0)
        sure_num = np.sum(seeds_map > 0)

        return np.uint8(prediction), uncertainty / total_num * (total_num - sure_num) if total_num > sure_num else 0

    def stop_correct(self, cur_depth, delete_mask, next_depth = -10):
        # delete_mask = self.prediction[:,:,cur_depth] > 0
        self.prediction[:,:,cur_depth] = np.where(delete_mask, 0, self.prediction[:,:,cur_depth], dtype=np.uint8)
        self.uncertainty_pieces[cur_depth] = 0
        self.isrefine_flag[cur_depth] = 1
        if next_depth == -10:
            if cur_depth > 0:
                # left_seeds_map = self.delete_badseeds_basedon_newadded_background_seeds(depth=cur_depth-1, background_seeds_new_mask=delete_mask, hard_flag=True)
                self.delete_prediction_basedon_backgroundseeds(anotate_prediction=self.prediction[:,:,cur_depth-1], background_seeds_new_mask=delete_mask)
                if left_seeds_map.max() < 0.5:
                    self.stop_correct(cur_depth=cur_depth-1, next_depth=cur_depth-2)
                else:
                    return
            if cur_depth < self.depth-1:
                right_seeds_map = self.delete_badseeds_basedon_newadded_background_seeds(depth=cur_depth+1, background_seeds_new_mask=delete_mask, hard_flag=True)
                if right_seeds_map.max() < 0.5:
                    self.stop_correct(cur_depth=cur_depth+1, next_depth=cur_depth+2)
                else:
                    return
        else:
            if next_depth < 0 or next_depth >= self.depth:
                return
            if next_depth < cur_depth:
                if cur_depth > 0:
                    left_seeds_map = self.delete_badseeds_basedon_newadded_background_seeds(depth=cur_depth-1, background_seeds_new_mask=delete_mask, hard_flag=True)
                    if left_seeds_map.max() < 0.5:
                        self.stop_correct(cur_depth=cur_depth-1, next_depth=cur_depth-2)
                    else:
                        return
            if next_depth > cur_depth:
                if cur_depth < self.depth-1:
                    right_seeds_map = self.delete_badseeds_basedon_newadded_background_seeds(depth=cur_depth+1, background_seeds_new_mask=delete_mask, hard_flag=True)
                    if right_seeds_map.max() < 0.5:
                        self.stop_correct(cur_depth=cur_depth+1, next_depth=cur_depth+2)
                    else:
                        return

    
    def refinement(self, model, device):
        """get different kinds of seeds"""
        background_seeds_new_mask = self.background_seeds[:,:,self.depth_anotate] == 1
        TL_seeds_new_mask = self.tmp_seeds == self.TL_label
        FL_seeds_new_mask = self.tmp_seeds == self.FL_label

        """
        先不考虑background seeds
        background seeds的更新也是按照更新另外两种seeds的思路
        找到background seeds连接的所有标签
        先对当前帧进行分析 background标签传播到的seeds都去掉
        """
        # if background_seeds_new_mask.any():
        """
        background_seeds只影响当前帧 并不进行传播
        1. 先去掉与background_seeds相连接的seeds
        2. 最后分割结果不包含与background标签连接的部分
        """
        """background -- 1"""

        """refinement前 该帧的seeds map"""
        start_seeds_map = self.seedsArray2map(self.depth_anotate)
        seeds_map = self.delete_badseeds_basedon_newadded_background_seeds(depth=self.depth_anotate, background_seeds_new_mask=background_seeds_new_mask) if background_seeds_new_mask.any() else self.seedsArray2map(depth=self.depth_anotate)
            
        if TL_seeds_new_mask.any() or FL_seeds_new_mask.any():
            """得到anotate帧优化后的seeds -- 1. 去掉不对的seeds 2. 加上对的seeds newly added"""
            """1."""
            seeds_map = self.delete_badseeds_basedon_newadded_TLFL_seeds(self.depth_anotate, TL_seeds_new_mask=TL_seeds_new_mask, FL_seeds_new_mask=FL_seeds_new_mask)
            """2."""
            seeds_map = self.integrate_tmp_seeds_nobackground(self.depth_anotate)
        """即使seeds_map没有发生变化 prediction也有可能发生变化 -- mask prediction with seeds map"""
        # if (seeds_map == start_seeds_map).all():
        #     """seeds map与原来相同 则该帧的分割结果也与原来相同 则不用传播 只需要处理background"""
        #     if background_seeds_new_mask.any():
        #         old_prediction_num = np.sum(self.prediction[:,:,self.depth_anotate] > 0)
        #         anotate_prediction = self.delete_prediction_basedon_backgroundseeds(self.prediction[:,:,self.depth_anotate], background_seeds_new_mask)
        #         self.prediction[:,:,self.depth_anotate], self.uncertainty_pieces[self.depth_anotate] = anotate_prediction, self.uncertainty_pieces[self.depth_anotate] / old_prediction_num * np.sum(anotate_prediction > 0)
        # else:
        print(f'cur piece: [{self.depth_anotate}/{self.depth}]')
        if seeds_map.max() < 0.5:
            """如果新的seeds map全为0 则没有prediction 终止传播 且uncertainty为0"""
            if self.depth_anotate > self.depth_initseg:
                self.prediction[:,:,self.depth_anotate:] = np.zeros((self.height, self.width, self.depth - self.depth_anotate), dtype=np.uint8)
                self.uncertainty_pieces[self.depth_anotate:] = 0
                # self.stop_correct(cur_depth=self.depth_anotate)
                self.isrefine_flag[self.depth_anotate:] = 1
            else:
                self.prediction[:,:,:(self.depth_anotate+1)] = np.zeros((self.height, self.width, self.depth_anotate+1), dtype=np.uint8)
                self.uncertainty_pieces[:self.depth_anotate] = 0
                # self.stop_correct(cur_depth=self.depth_anotate)
                self.isrefine_flag[:self.depth_anotate] = 1
        else:
            anotate_prediction, anotate_uncertainty = self.get_prediction_with_seeds_map(self.image[:,:,self.depth_anotate], seeds_map, True, model, device)
            # anotate_prediction, anotate_uncertainty = self.prediction[:,:,self.depth_anotate], self.uncertainty_pieces[self.depth_anotate]
            if anotate_prediction.max() < 0.5:
                # self.prediction[:,:,self.depth_anotate] = np.zeros((self.height, self.width), dtype=np.uint8)
                # self.uncertainty_pieces[self.depth_anotate] = 0
                # self.stop_correct(cur_depth=self.depth_anotate)
                # self.isrefine_flag[self.depth_anotate] = 1
                """如果新的seeds map全为0 则没有prediction 终止传播 且uncertainty为0"""
                if self.depth_anotate > self.depth_initseg:
                    self.prediction[:,:,self.depth_anotate:] = np.zeros((self.height, self.width, self.depth - self.depth_anotate), dtype=np.uint8)
                    self.uncertainty_pieces[self.depth_anotate:] = 0
                    # self.stop_correct(cur_depth=self.depth_anotate)
                    self.isrefine_flag[self.depth_anotate:] = 1
                else:
                    self.prediction[:,:,:(self.depth_anotate+1)] = np.zeros((self.height, self.width, self.depth_anotate+1), dtype=np.uint8)
                    self.uncertainty_pieces[:self.depth_anotate] = 0
                    # self.stop_correct(cur_depth=self.depth_anotate)
                    self.isrefine_flag[:self.depth_anotate] = 1
            else:
                """去掉anotate_prediction中background seeds的部分"""
                """background -- 2"""
                """并不只是根据cur prediction来进行refine"""
                # anotate_prediction, anotate_uncertainty = self.prediction[:,:,self.depth_anotate], self.uncertainty_pieces[self.depth_anotate]
                if background_seeds_new_mask.any():
                    anotate_prediction, anotate_uncertainty = self.delete_prediction_basedon_backgroundseeds(anotate_prediction, background_seeds_new_mask, anotate_uncertainty)
                    #anotate_prediction, anotate_uncertainty = anotate_prediction, anotate_uncertainty / old_prediction_num * np.sum(anotate_prediction > 0)
                """考虑prediction要覆盖掉新加的seeds"""
                anotate_prediction, anotate_uncertainty = self.mask_prediction_with_newadded_TLFL_seeds_notregion(anotate_prediction, seeds_map, anotate_uncertainty)
                anotate_uncertainty = anotate_uncertainty + self.get_scribble_loss_plus_region_loss(prediction=anotate_prediction, seeds_map=seeds_map)
                """uncertainty更高也不管了"""
                # if anotate_uncertainty >= self.uncertainty_pieces[self.depth_anotate]:
                #     anotate_uncertainty = self.uncertainty_thred
                self.prediction[:,:,self.depth_anotate], self.uncertainty_pieces[self.depth_anotate] = anotate_prediction, anotate_uncertainty

                # self.prediction[:,:,self.depth_anotate], self.uncertainty_pieces[self.depth_anotate] = anotate_prediction, anotate_uncertainty + self.get_region_loss(prediction=anotate_prediction)
                self.isrefine_flag[self.depth_anotate] = 1
                cur_piece = self.depth_anotate - 1
                while cur_piece > 0:
                    if self.isrefine_flag[cur_piece]:
                        break
                    print(f'cur piece: [{cur_piece}/{self.depth}]') 
                    """
                    1. seedsmap进行传播
                    2. 保留原来好的seeds + 新传播得到的seeds
                    按照init segment的方法进行传播 跳出循环的方法有 传播的seeds就不要求mask prediction with seeds
                    1. 更新过后的seeds和原来相同 
                    2. 分割结果高度相似 0.98
                    3. uncertainty的值更高
                    4. 或者到达了一个一定正确的帧 -- uncertainty=0
                    """
                    refine_flag, refine_seeds, refine_seeds_map = get_right_seeds_all(self.prediction[:,:,cur_piece+1], self.image[:,:,cur_piece], self.image[:,:,cur_piece+1], seeds_case=6, clean_region_flag=False, clean_seeds_flag=True)
                    if not refine_flag:
                        break
                    # change_flag, refine_seeds_map = self.integrate_refine_seedsmap_with_oldseeds(refine_seeds_map, cur_piece)
                    # if not change_flag:
                    #     break
                    # refine_seeds = np.argwhere(refine_seeds_map > 0)
                    self.TL_seeds[:,:,cur_piece] = np.where(refine_seeds_map == self.TL_label, 1, 0)
                    self.FL_seeds[:,:,cur_piece] = np.where(refine_seeds_map == self.FL_label, 1, 0)
                    if refine_seeds_map.max() < 0.5:
                        self.prediction[:,:,cur_piece] = np.zeros((self.height, self.width), dtype=np.uint8)
                        self.uncertainty_pieces[cur_piece] = 0
                        self.isrefine_flag[cur_piece] = 1
                        break
                    indata = get_network_input_all(image=self.image[:,:,cur_piece], seeds=refine_seeds, seeds_image=refine_seeds_map, window_transform_flag=True)
                    indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                    refine_prediction, refine_uncertainty = get_prediction_all(model, indata)
                    refine_prediction = np.uint8(refine_prediction)
                    if refine_prediction.max() < 0.5:
                        self.prediction[:,:,cur_piece] = np.zeros((self.height, self.width), dtype=np.uint8)
                        self.uncertainty_pieces[cur_piece] = 0
                        self.isrefine_flag[cur_piece] = 1
                        break
                    refine_background_seeds_mask = self.background_seeds[:,:,cur_piece] == 1
                    if refine_background_seeds_mask.any():
                        refine_prediction, refine_uncertainty = self.delete_prediction_basedon_backgroundseeds(refine_prediction, refine_background_seeds_mask, refine_uncertainty)
                    #refine_prediction, refine_uncertainty = self.mask_prediction_with_newadded_TLFL_seeds_notregion(refine_prediction, refine_seeds_map, refine_uncertainty)
                    # refine_uncertainty += self.get_scribble_loss_plus_region_loss(prediction=refine_prediction, seeds_map=refine_seeds_map)
                    # refine_uncertainty += self.get_region_loss(prediction=refine_prediction)
                    refine_uncertainty += self.get_scribble_loss_plus_region_loss(prediction=refine_prediction, seeds_map=refine_seeds_map)
                    # if refine_uncertainty > self.uncertainty_pieces[cur_piece]:
                    #     break
                    # refine_prediction = np.uint8(refine_prediction)
                    if accuracy_all_numpy(self.prediction[:,:,cur_piece], refine_prediction) < 0.98:
                        self.prediction[:,:,cur_piece] = refine_prediction
                        self.uncertainty_pieces[cur_piece] = refine_uncertainty
                    else:
                        self.prediction[:,:,cur_piece] = refine_prediction
                        self.uncertainty_pieces[cur_piece] = refine_uncertainty
                        break
                    # if refine_prediction.max() < 0.5:
                    #     break
                    cur_piece = cur_piece-1

                cur_piece = self.depth_anotate + 1
                while cur_piece < self.depth:
                    if self.isrefine_flag[cur_piece]:
                        break
                    print(f'cur piece: [{cur_piece}/{self.depth}]') 
                    """
                    1. seedsmap进行传播
                    2. 保留原来好的seeds + 新传播得到的seeds
                    按照init segment的方法进行传播 跳出循环的方法有
                    1. 更新过后的seeds和原来相同
                    2. 分割结果高度相似 0.98
                    3. uncertainty的值更高
                    """
                    refine_flag, refine_seeds, refine_seeds_map = get_right_seeds_all(self.prediction[:,:,cur_piece-1], self.image[:,:,cur_piece], self.image[:,:,cur_piece-1], seeds_case=6, clean_region_flag=False, clean_seeds_flag=True)
                    if not refine_flag:
                        break
                    # change_flag, refine_seeds_map = self.integrate_refine_seedsmap_with_oldseeds(refine_seeds_map, cur_piece)
                    # if not change_flag:
                    #     break
                    # refine_seeds = np.argwhere(refine_seeds_map > 0)
                    self.TL_seeds[:,:,cur_piece] = np.where(refine_seeds_map == self.TL_label, 1, 0)
                    self.FL_seeds[:,:,cur_piece] = np.where(refine_seeds_map == self.FL_label, 1, 0)
                    if refine_seeds_map.max() < 0.5:
                        self.prediction[:,:,cur_piece] = np.zeros((self.height, self.width), dtype=np.uint8)
                        self.uncertainty_pieces[cur_piece] = 0
                        self.isrefine_flag[cur_piece] = 1
                        break
                    indata = get_network_input_all(image=self.image[:,:,cur_piece], seeds=refine_seeds, seeds_image=refine_seeds_map, window_transform_flag=True)
                    indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                    refine_prediction, refine_uncertainty = get_prediction_all(model, indata)
                    refine_prediction = np.uint8(refine_prediction)
                    if refine_prediction.max() < 0.5:
                        self.prediction[:,:,cur_piece] = np.zeros((self.height, self.width), dtype=np.uint8)
                        self.uncertainty_pieces[cur_piece] = 0
                        self.isrefine_flag[cur_piece] = 1
                        break
                    refine_background_seeds_mask = self.background_seeds[:,:,cur_piece] == 1
                    if refine_background_seeds_mask.any():
                        refine_prediction, refine_uncertainty = self.delete_prediction_basedon_backgroundseeds(refine_prediction, refine_background_seeds_mask, refine_uncertainty)
                    #refine_prediction, refine_uncertainty = self.mask_prediction_with_newadded_TLFL_seeds(refine_prediction, refine_seeds_map, refine_uncertainty)
                    # refine_uncertainty += self.get_region_loss(prediction=refine_prediction)
                    refine_uncertainty += self.get_scribble_loss_plus_region_loss(prediction=refine_prediction, seeds_map=refine_seeds_map)
                    # refine_uncertainty += self.get_scribble_loss_plus_region_loss(prediction=refine_prediction, seeds_map=refine_seeds_map)
                    # if refine_uncertainty > self.uncertainty_pieces[cur_piece]:
                    #     break
                    # refine_prediction = np.uint8(refine_prediction)
                    if accuracy_all_numpy(self.prediction[:,:,cur_piece], refine_prediction) < 0.98:
                        self.prediction[:,:,cur_piece] = refine_prediction
                        self.uncertainty_pieces[cur_piece] = refine_uncertainty
                    else:
                        self.prediction[:,:,cur_piece] = refine_prediction
                        self.uncertainty_pieces[cur_piece] = refine_uncertainty
                        break
                    # if refine_prediction.max() < 0.5:
                    #     break
                    cur_piece = cur_piece+1   
            
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        # self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        print("finish refinement")
        # dc1,dc2,dc3,hd1,hd2,hd3 = self.get_test_evaluation()
        # print('TL acc: %.5f, FL acc: %.5f, acc: %.5f, hd tl: %.5f, hd fl: %.5f, hd: %.5f' % (dc1,dc2,dc3,hd1,hd2,hd3))

        
    def Clear(self):
        self.uncertainty_pieces = np.zeros((self.depth))
        self.isrefine_flag = np.zeros((self.depth), dtype=np.uint8)
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        self.prediction = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
        self.uncertainty_thred = 0
        self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)


    def savePrediction(self, save_path):
        save2h5(save_path, ['image', 'prediction'], [self.image, self.prediction])



    def anotate(self, x, y):
        """
        need to do: 先得到img再求得坐标，需要搞清楚坐标之间的关系
        prediction也需要更改，和anotation output不一样，一个是保存预测结果，一个是保存渲染结果
        """
        self.depth_anotate = self.depth_current
        if self.TL_flag:
            # if not self.TL_seeds[y, x, self.depth_current]:
                # print("add seed")
                # self.TL_seeds.append((y, x, self.depth_current))
            cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x, y), self.TL_color, self.penthickness)
            self.tmp_seeds[y-1:y+1,x-1:x+1] = self.TL_label

        if self.FL_flag:
            # if not self.FL_seeds[y, x, self.depth_current]:
                # self.FL_seeds.append((y, x, self.depth_current))
            cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x, y), self.FL_color, self.penthickness)
            self.tmp_seeds[y-1:y+1,x-1:x+1] = self.FL_label
            # print(y,x)
        if self.background_flag:
            # if not self.background_seeds[y, x, self.depth_current]:
                # self.background_seeds.append((y, x, self.depth_current))
            cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x, y), self.background_color, self.penthickness)
            self.tmp_seeds[y-1:y+1,x-1:x+1] = self.background_label
            self.background_seeds[y-1:y+1,x-1:x+1,self.depth_anotate] = 1
