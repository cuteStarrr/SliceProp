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
from UNet_COPY import *
from interact_dataset import *
from train import accuracy_all_numpy
from test import get_prediction_all_bidirectional, get_network_input_all, get_prediction_all
from region_grow import *


"""
NEED TO DO:
1. self.prediction2anotation
2. background seeds
3. refinement
4. 初始分割的时候，更新每个depth对应的seeds
5. seeds clean需要修改，应该在得到seeds的时候就clean-即针对一个连通区域
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
        file = h5py.File(image_path, 'r')
        self.image = (file['image'])[()]
        self.image = self.image - self.image.min()
        # self.label = np.uint8((file['label'])[()])
        # self.image = self.image / self.image.max()
        self.height, self.width, self.depth = self.image.shape
        self.depth_current = self.depth // 2
        self.depth_anotate = self.depth_current
        """一帧一帧地segment and refine"""
        self.unceitainty_pieces = np.zeros((self.depth))
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        self.prediction = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
        self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)

        self.dice_coeff_thred = 0.75
        self.penthickness = 1

    def set_depth(self, depth):
        self.depth_current = depth

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
        
    def prediction2anotation(self):
        for i in range(self.depth):
            """for test"""
            
            mask_TL = np.uint8(self.prediction[:,:,i] == self.TL_label)
            tmp_TL = self.gray2BGRImage(mask_TL)
            tmp_TL = np.where(tmp_TL == [0, 0, 0], [0, 0, 0], list(self.TL_color))
            mask_FL = np.uint8(self.prediction[:,:,i] == self.FL_label)
            tmp_FL = self.gray2BGRImage(mask_FL)
            tmp_FL = np.where(tmp_FL == [0, 0, 0], [0, 0, 0], list(self.FL_color))

            # mask_TL_seeds = np.uint8(self.TL_seeds[:,:,i] == 1)
            # tmp_TL_seeds = self.gray2BGRImage(mask_TL_seeds)
            # tmp_TL_seeds = np.where(tmp_TL_seeds == [0, 0, 0], [0, 0, 0], list(self.background_color))
            # mask_FL_seeds = np.uint8(self.FL_seeds[:,:,i] == 1)
            # tmp_FL_seeds = self.gray2BGRImage(mask_FL_seeds)
            # tmp_FL_seeds = np.where(tmp_FL_seeds == [0, 0, 0], [0, 0, 0], list(self.background_color))
            
            """for test"""
            self.anotation[i,:,:,:] = tmp_TL + tmp_FL # + tmp_TL_seeds + tmp_FL_seeds
            # self.anotation[i,:,:] = np.where(self.prediction[:,:,i] == self.TL_label, np.array(self.TL_color), self.anotation[i,:,:])
            # self.anotation[i,:,:] = np.where(self.prediction[:,:,i] == self.FL_label, np.array(self.FL_color), self.anotation[i,:,:])
    
    def get_prediction_with_seeds_map(self, cur_image, seeds_map, window_transform_flag, model, device):
        indata = get_network_input_all(cur_image, np.argwhere(seeds_map > 0), seeds_map, window_transform_flag)
        indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
        prediction, unceitainty = get_prediction_all(model, indata)
        prediction = np.uint8(prediction)

        return prediction, unceitainty
        
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
        prediction, unceitainty = get_prediction_all(model, indata)
        # print("prediction")
        prediction = np.uint8(prediction)

        return True, prediction, seeds_map, unceitainty

    
    def init_segment(self, model, device):
        # print("start segmentation")
        # TL_seeds = np.argwhere(self.TL_seeds[:,:,self.depth_current] == 1)
        # FL_seeds = np.argwhere(self.FL_seeds[:,:,self.depth_current] == 1)
        # """
        # 这里有大问题！！！
        # """
        # print(TL_seeds.shape)
        # self.TL_seeds[:,:,self.depth_current] = seeds2map(TL_seeds, (self.height, self.width))
        # self.FL_seeds[:,:,self.depth_current] = seeds2map(FL_seeds, (self.height, self.width))
        # print("get init seeds")
        window_transform_flag = True
        clean_seeds_flag = True
        clean_region_flag = False

        cur_image = self.image[:,:,self.depth_anotate]
        last_image = self.image[:,:,self.depth_anotate]
        # last_label = region_grow(cur_image,TL_seeds) * self.TL_label + region_grow(cur_image, FL_seeds) * self.FL_label
        # self.prediction[:,:,self.depth_current] = last_label
        last_label = self.tmpseedsCoords2map()
        # last_label = self.label[:,:,self.depth_current]
        seeds_map = self.tmpseedsCoords2map()
        # plt.imshow(seeds_map, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # print("finish preparation")

        # self.prediction2anotation(self.depth_current)
        # print("finish anotation")
        # return 

        for i in range(self.depth_anotate, self.depth):
            # print("start one piece")
            cur_image = self.image[:,:,i]
            """test"""
            flag = True
            prediction = last_label
            unceitainty = 0
            if i == self.depth_anotate:
                prediction, unceitainty = self.get_prediction_with_seeds_map(cur_image, seeds_map, window_transform_flag, model, device)
                # indata = get_network_input_all(cur_image, np.argwhere(seeds_map > 0), seeds_map, window_transform_flag)
                # indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                # prediction = get_prediction_all(model, indata)
                # prediction = np.uint8(prediction)
                # print("get prediction - 1")
            else:
                flag, prediction,seeds_map, unceitainty = self.get_prediction_intergrate_known_seeds(last_label, cur_image, last_image, window_transform_flag, device, model, seeds_case = 0, depth=i, clean_region_flag=clean_region_flag, clean_seeds_flag=clean_seeds_flag)
            # print("get prediction - 2")
            if not flag:
                break
            # print(np.unique(prediction, return_counts = True))
            # print(prediction.shape)
            self.prediction[:,:,i] = prediction
            self.unceitainty_pieces[i] = unceitainty
            # if i == 157:
            #     plt.imshow(seeds_map, cmap='gray')
            #     plt.axis('off')
            #     plt.show()
            # if i != self.depth_current: 
            self.TL_seeds[:,:,i] = np.where(seeds_map == self.TL_label, 1, self.TL_seeds[:,:,i])
            self.FL_seeds[:,:,i] = np.where(seeds_map == self.FL_label, 1, self.FL_seeds[:,:,i])
            # print("get seeds for each piece - 1")
            if prediction.max() < 0.5:
                break
            cur_piece = i
            cur_coeff = accuracy_all_numpy(self.prediction[:,:,cur_piece-1], self.prediction[:,:,cur_piece])
            # print("cal acc - 1")
            while cur_piece > 0 and cur_coeff  < self.dice_coeff_thred:
                roll_flag, roll_prediction, roll_seeds_map, roll_unceitainty = self.get_prediction_intergrate_known_seeds(self.prediction[:,:,cur_piece], self.image[:,:,cur_piece-1], self.image[:,:,cur_piece], window_transform_flag, device, model, seeds_case = 0, depth=cur_piece-1, clean_region_flag=clean_region_flag, clean_seeds_flag=clean_seeds_flag)
                # plt.imshow(roll_seeds_map, cmap='gray')
                # plt.axis('off')
                # plt.show()
                # print("get prediction - 3")
                if not roll_flag:
                    break
                if accuracy_all_numpy(self.prediction[:,:,cur_piece - 1], roll_prediction) < 0.98:
                    self.prediction[:,:,cur_piece - 1] = roll_prediction
                    self.unceitainty_pieces[cur_piece-1] = roll_unceitainty
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
                if roll_prediction.max() < 0.5:
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
        print("finish init segmentation")
        print("---------------- unceitainty info -----------------")
        print("max unceitainty: ", self.unceitainty_pieces.max())
        print("min unceitainty: ", self.unceitainty_pieces.min())
        print("mean unceitainty: ", self.unceitainty_pieces.mean())


    def get_max_unceitainty(self):
        return np.argmax(self.unceitainty_pieces)
    
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


    def propagate_seeds_map_basedon_label(self, cur_label, TL_seeds_new_mask, FL_seeds_new_mask):
        """
        focus on TL FL seeds, don't regard background seeds
        将原有的seeds和new added seeds进行融合，去掉不符合条件的new added seeds
        """
        new_seeds_map = np.where(cur_label[TL_seeds_new_mask] > 0, 1, 0)
        new_seeds_map = np.where(cur_label[FL_seeds_new_mask] > 0, new_seeds_map, 0)

        return new_seeds_map
    

    def delete_badseeds_basedon_newadded_seeds(self, depth, TL_seeds_new_mask, FL_seeds_new_mask):
        """针对FL_seeds"""
        old_seeds = self.FL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if cur_block_seeds[TL_seeds_new_mask].any() == 1:
                """即FL_seesd与用户标注的TL_seeds冲突 应去掉原有的FL_seeds的这一个连通分量"""
                self.FL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.FL_seeds[:,:,depth])

        """针对TL_seeds"""
        old_seeds = self.TL_seeds[:,:,depth]
        block_num, seeds_blocks = cv2.connectedComponents(old_seeds)
        for cur_block in range(block_num, 0, -1):
            cur_block_seeds = np.uint8(np.where(seeds_blocks > cur_block - 0.5, 1, 0))
            seeds_blocks[seeds_blocks > cur_block - 0.5] = 0
            if cur_block_seeds[FL_seeds_new_mask].any() == 1:
                """即TL_seesd与用户标注的FL_seeds冲突 应去掉原有的TL_seeds的这一个连通分量"""
                self.TL_seeds[:,:,depth] = np.where(cur_block_seeds == 1, 0, self.TL_seeds[:,:,depth])


        return self.seedsArray2map(depth=depth)

    
    def refinement(self, model, device):
        """get different kinds of seeds"""
        background_seeds_new_mask = self.tmp_seeds == self.background_label
        TL_seeds_new_mask = self.tmp_seeds == self.TL_label
        FL_seeds_new_mask = self.tmp_seeds == self.FL_label

        """
        先不考虑background seeds
        background seeds的更新也是按照更新另外两种seeds的思路
        找到background seeds连接的所有标签
        先对当前帧进行分析 background标签传播到的seeds都去掉
        """
        if TL_seeds_new_mask.any() or FL_seeds_new_mask.any():
            """得到anotate帧优化后的seeds -- 1. 去掉不对的seeds 2. 加上对的seeds newly added"""
            """1."""
            seeds_map = self.delete_badseeds_basedon_newadded_seeds(self.depth_anotate, TL_seeds_new_mask=TL_seeds_new_mask, FL_seeds_new_mask=FL_seeds_new_mask)
            """2."""
            seeds_map = self.integrate_tmp_seeds_nobackground(self.depth_anotate)
            self.prediction[:,:,self.depth_anotate] = self.get_prediction_with_seeds_map(self.image[:,:,self.depth_anotate], seeds_map, True, model, device)
            """above checked"""
            cur_piece = self.depth_anotate - 1
            while cur_piece > 0 and ((TL_seeds_new_mask.any() and self.prediction[:,:,cur_piece][TL_seeds_new_mask].any() != 0) or (FL_seeds_new_mask.any() and self.prediction[:,:,cur_piece][FL_seeds_new_mask].any() != 0)):
                """
                1. 先将用户标注的seeds进行裁剪 裁剪到该帧的标签内
                2. 去掉原来的错误的seeds
                3. 加上裁剪之后的seeds
                按照init segment的方法进行传播 跳出循环的方法有
                1. 用户标注的seeds全在该帧的标签外面
                2. 更新过后的seeds和原来相同
                3. 分割结果高度相似 0.98
                """
                flag, cur_seeds_map = self.propagate_seeds_map_basedon_label(seeds_map=seeds_map, cur_label=self.prediction[:,:,cur_piece], TL_seeds_new_mask=TL_seeds_new_mask, FL_seeds_new_mask=FL_seeds_new_mask)


        print("finish refinement")

        
    def Clear(self):
        self.unceitainty_pieces = np.zeros((self.depth))
        self.tmp_seeds = np.zeros((self.height, self.width), dtype=np.uint8)
        self.prediction = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
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
