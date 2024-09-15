
import objaverse.xl as oxl

import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from transformers import PreTrainedTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import pickle
from transformers import CLIPTextModel, CLIPTokenizer
class CustomDataset(Dataset):
    def __init__(self, dataframe, trans=None,max_length=77, num_views=4, guidance_scale=7.5, is_train=True):
        """
        初始化数据集
        """
        self.dataframe = dataframe
        self.max_length = max_length
        self.num_views = num_views
        self.is_train = is_train
        self.guidance_scale = guidance_scale

        # 图像预处理
        self.transform = trans
        '''
        # 下载和渲染步骤
        self.download()
        self.render()
        '''
        # 预先加载所有图像
        self.image_data_dict = self.load_all_images()

    def generate_image_paths(self, sample_id):
        image_paths = []
        for i in range(0, self.num_views):
            image_path = os.path.join(f"./render/Cap3D_imgs/Cap3D_imgs_view{i}/", f"{sample_id}_{i}.png")
            image_paths.append(image_path)
        return image_paths

    def download(self):
        # 下载所需文件
        oxl.download_objects(objects=self.dataframe, download_dir='.')
        root_dir = "./hf-objaverse-v1"
        glb_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".glb"):
                    full_path = os.path.join(dirpath, filename)
                    glb_files.append(full_path)

        with open("glb_files.pkl", "wb") as pkl_file:
            pickle.dump(glb_files, pkl_file)

    def render(self):
        # 运行Blender渲染
        os.system('git clone https://github.com/crockwell/Cap3D.git && wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip && unzip blender.zip')
        os.system("./blender-3.4.1-linux-x64/blender -b -P ./Cap3D/captioning_pipeline/render_script.py -- --object_path_pkl './glb_files.pkl' --parent_dir './render'")

    def load_all_images(self):
        """
        预加载所有图像并进行预处理，将图像保存在字典中
        """
        image_data_dict = {}
        for idx in tqdm(range(len(self.dataframe))):
            row = self.dataframe.iloc[idx]
            sample_id = row['id']
            image_paths = self.generate_image_paths(sample_id)
            images = []
            for image_path in image_paths:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')  # 加载并转为RGB模式
                    images.append(image)
                else:
                    images.append(Image.open('text1.png'))  # 若图像不存在，则添加一个全0张量

            image_data_dict[sample_id] = images

        return image_data_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx:int):
        row = self.dataframe.iloc[idx]
        sample_id = row['id']

        images = self.image_data_dict[sample_id]  # 获取预处理后的图像
        input_ids = row['caption']
        neg_ids = row['desciption']
        rank=row['score']
        example={
            'images': images,  # 返回预处理后的图像
            'input_ids': input_ids,
            'neg_ids': neg_ids,
            'rank':rank
        }
        return self.transform(example)

