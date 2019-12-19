#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dali_transforms.py.py    
@Contact :   JZ

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/4 15:09   JZ      1.0         None
"""


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import mmcv
import os
from random import shuffle


batch_size = 16


class ExternalInputIterator(object):
    def __init__(self, img_prefix, ann_file, batch_size, device_id, num_gpus):
        self.images_dir = img_prefix
        self.batch_size = batch_size
        self.files = mmcv.list_from_file(ann_file)
        self.data_set_len = len(self.files)
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = self.data_set_len

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        if self.i >= self.n:
            raise StopIteration
        for _ in range(self.batch_size):

            filename = 'JPEGImages/{}.jpg'.format(self.files[self.i])
            url = os.path.join(self.images_dir, filename)
            f = open(url, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([1], dtype=np.uint8))
            self.i = (self.i + 1)
            if self.i >= self.n:
                break
        return (batch, labels)

    @property
    def size(self, ):
        return self.data_set_len

    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, cfg):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           output_dtype=types.FLOAT,
                                           # crop=(227, 227),
                                           image_type=types.RGB,
                                           mean=cfg.mean,
                                           std=cfg.std)
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.Uniform(range=(256, 480))
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.resize(images, resize_shorter=self.resize_rng())
        output = self.cmn(images, crop_pos_x=self.uniform(),
                          crop_pos_y=self.uniform())

        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, cfg):
        super(CommonPipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           output_dtype=types.FLOAT,
                                           # crop=(227, 227),
                                           image_type=types.RGB,
                                           mean=cfg.mean,
                                           std=cfg.std)
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.Uniform(range=(512, 512))
        self.cast = ops.Cast(device="gpu",
                             dtype=types.UINT8)

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter=self.resize_rng())
        output = self.cmn(images, crop_pos_x=self.uniform(),
                          crop_pos_y=self.uniform())
        return (output, labels)


class FileReadPipeline(CommonPipeline):
    def __init__(self, image_dir, batch_size, num_threads, device_id, cfg):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id, cfg)
        self.input = ops.FileReader(file_root=image_dir)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)



batch_size = 2
sequence_length = 8
initial_prefetch_size = 11


class VideoPipe(CommonPipeline):
    def __init__(self, image_dir, batch_size, num_threads, device_id, num_gpus, shuffle, cfg):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, num_gpus, cfg)
        self.input = ops.VideoReader(device="gpu", file_root=image_dir, sequence_length=sequence_length,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_prefetch_size)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)
