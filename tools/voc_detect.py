#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   voc_detect.py    
@Contact :   JZ
@License :   (C)Copyright 2018-2019, JZ All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/7/11 14:10   JZ      1.0         None
'''


import mmcv
from mmdet.apis import init_detector, inference_detector, show_result
import os
import cv2
import numpy as np
import os
from torch2trt import torch2trt
import torch

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
cfg = mmcv.Config.fromfile('configs/reppoints/voc_reppoints_moment_r50_fpn_2x_mt.py')
cfg.model.pretrained = None

# YangHE/mmdetection-master/work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_1.pth
# construct the model and load checkpoint

model = init_detector(cfg, 'work_dirs/reppoints_moment_r50_fpn_2x_mt/latest.pth', device='cuda:0')
x = torch.ones((1, 3, 800, 800)).cuda()
cap = cv2.VideoCapture("/home/juzheng/dataset/granary/29-1.avi")
i = 0
while True:

    res, image1 = cap.read()
    i += 1
    if i % 15 != 0:
        continue
    image1 = cv2.resize(image1, (1024, 600))
    result = inference_detector(model, image1)

    show_result(image1, result, model.CLASSES, out_file='result/mask/result_{}.jpg'.format(i))



imgs = []
path = "data/VOCdevkit/VOC2007/JPEGImages"
with open('data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'r') as file:
    for line in file:
        imgs.append(os.path.join(path, line.rstrip('\n') + ".jpg"))

# listDir(path, imgs)
for i, img in enumerate(imgs):
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, out_file='result/mask/result_{}.jpg'.format(i))
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='result/mask/result_{}.jpg'.format(i))
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

#
# from mmdet.datasets import dali_transforms
# import time
#
#
# def test_dali():
#     # eii = ExternalInputIterator(batch_size)
#     # iterator = iter(eii)
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     cfg = mmcv.Config.fromfile('config/voc_2_libra_faster_rcnn_r50_fpn_1x.py')
#     cfg.model.pretrained = None
#     # granary_model = init_detector(cfg, 'model/libra.pth', device='cuda:0')
#     person_model = init_detector(cfg, 'model/libra_person.pth', device='cuda:0')
#     ann_file = cfg.data.test.ann_file
#     image_ids = mmcv.list_from_file(ann_file)
#     img_prefix = cfg.data.test.img_prefix
#     batch_size = 32
#
#     num = len(image_ids)
#     eii = dali_transforms.ExternalInputIterator(img_prefix, ann_file, batch_size, device_id=0, num_gpus=1)
#     pipe = dali_transforms.ExternalSourcePipeline(batch_size=batch_size, num_threads=2, device_id=0, external_data=eii,
#                                                   cfg=cfg.img_norm_cfg)
#     pipe.build()
#     start = time.clock()
#
#     for i in range(num // batch_size + 1):
#         try:
#             images, labels = pipe.run()
#             print('frame :', len(images))
#
#         except Exception as e:
#             print(e)
#             # cv2.imshow("{}".format(image_id), imageout1)
#             # cv2.waitKey()
#             # pass
#             # cv2.destroyWindow("{}".format(image_id))
#             break
#     elapsed = (time.clock() - start)
#     print('NUM:{}, TIME USE:{}'.format(num, elapsed))
