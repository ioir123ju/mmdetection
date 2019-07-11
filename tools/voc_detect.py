#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   voc_detect.py    
@Contact :   juzheng@hxdi.com
@License :   (C)Copyright 2018-2019, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/7/11 14:10   juzheng      1.0         None
'''


import mmcv
from mmdet.apis import init_detector, inference_detector, show_result
import os
import cv2
import numpy as np
import os

def listDir(path, list_name):
    """
    :param path: 路径
    :param list_name: path下所有文件的绝对路径名的列表
    :return:
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listDir(file_path, list_name)
        else:
            list_name.append(file_path)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg = mmcv.Config.fromfile('configs/voc_faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# YangHE/mmdetection-master/work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_1.pth
# construct the model and load checkpoint

model = init_detector(cfg, 'work_dirs/faster_rcnn_r50_fpn_1x/latest.pth', device='cuda:0')


path = "data/my_grain"
# path = "data/VOCdevkit/VOC2007/JPEGImages"
imgs = []
listDir(path, imgs)
for i, result in enumerate(inference_detector(model, imgs)):
    show_result(imgs[i], result, model.CLASSES, out_file='result/grain/result_{}.jpg'.format(i))
# print(result[0].shape)
#
# img = cv2.imread('test.jpg')
# sores = result[0][:,-1]
# ind = sores > 0.01
# bboxes = result[0][ind,:-1]
# for bbox in bboxes:
#     bbox_int = bbox.astype(np.int32)
#     left_top = (bbox_int[0], bbox_int[1])
#     right_bottom = (bbox_int[2], bbox_int[3])
#     cv2.rectangle(
#         img, left_top, right_bottom,color=(0, 255, 0))
# cv2.imshow("s", img)
# cv2.waitKey(0)

