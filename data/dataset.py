import torch
import copy
import json
import cv2
import os
from glob import glob
import blobfile as bf
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random

from . import tools as Tls
from . import transformers as Tfs




class InjectDataset(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.preprocess = cfg['preprocess']
        self.data_aug = cfg['data_aug']
        f=open(cfg['data_json'], 'r')
        annotations = json.load(f)["annotations"]

        # 用于存储转换后的数据
        self.data_list = []

        # 遍历每个标注，构造新字典并转换为 JSON 字符串
        for ann in annotations:
            image_id = ann["image_id"]
            # 将 image_id 格式化为 12 位数字的文件名
            file_name = f"{image_id:012d}.jpg"
            new_entry = {
                "img_path": f"./datafiles/coco2017/train2017/{file_name}",
                "txt": ann["caption"],
                "label": 1,
                "IN_label": -1
            }
            # 将新字典转换为 JSON 格式字符串并追加到列表中
            self.data_list.append(json.dumps(new_entry) + "\n")
        self.data_list=random.sample(self.data_list,50000)


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        info = json.loads(self.data_list[index])

        img = Image.open(info['img_path']).convert("RGB")
        img = np.array(img)

        assert img is not None, "Img read error at {}".format(info['img_path'])

        for name, kwarg in self.data_aug.items():
            img = getattr(Tfs, name)(img, **kwarg)
        for name, kwarg in self.preprocess.items():
            img = getattr(Tls, name)(img, **kwarg)

        info['imgs'] = img

        Tls.cvtTensor(info)
        return info
