import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import itertools
import numpy as np
from PIL import Image

class Tiny_imagenet_dataset(Dataset):
    def __init__(self, root='data/train', use_box=False, box_norm=True):
        '''
            example: 
                root='data/train'
                use_box: use bbox or not
                box_norm: bbox div by 64 or not
        '''
        self.use_box = use_box
        self.box_norm = box_norm
        folder_list = glob(os.path.join(root, '*'))
        self.image_paths = []
        self.labels = []
        self.boxes = []
        self.ohe = OneHotEncoder(sparse=False)
        class_l = np.array([i.split('/')[2] for i in folder_list])
        class_l = class_l.reshape((len(class_l), 1))
        self.ohe.fit(class_l)
        
        for folder in folder_list:
            loc = open(os.path.join(folder, folder.split('/')[2] + '_boxes.txt'))
            l_loc = loc.readlines()
            file_paths = [folder +  '/' + 'images/' + i.split('\t')[0] for i in l_loc]
            labels = [i.split('_')[0].split('/')[-1] for i in file_paths]
            boxes = [(int(i[0]), int(i[1]), int(i[2]), int(i[3])) for i in [j.split('\t')[1:] for j in l_loc]]
            
            self.image_paths.extend(file_paths)
            self.labels.extend(labels)
            self.boxes.extend(boxes)
        
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))
        self.labels = self.ohe.transform(self.labels)
        self.boxes = np.array(self.boxes)
            
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
            
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        if img.mode == 'L':
            img = img.convert('RGB')
            
        labels = np.argmax(self.labels[index])
        if self.use_box:
            if self.box_norm:
                box = self.boxes[index] / 64
            else:
                box = self.boxes[index]
            return self.transform(img), labels, box
        else:
            return self.transform(img), labels
        
    def __len__(self):
        return len(self.image_paths)
    
    