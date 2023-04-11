import os, json, cv2, time
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from torch.utils.data import Dataset
# from utils.factorization import *

# 가로 / 세로 랜덤 수 추출 후 데이터셋 추출
# inference에서는 600px정도로 등분

class UnetData(Dataset):
    def __init__(self, data_path:str, total_epoch:int, pad:int, mode:str="train", eval_num:int=3):
        super(UnetData,self).__init__()

        tree_file = "tree.json"
        with open(tree_file) as f: paths = json.load(f)

        data_path = os.path.join(data_path, mode)
        tif_folder = "surface_volume"

        if mode.lower() == "train":
            a = list(map(int, paths["train"]))
            a.remove(eval_num)
            self.order = list(np.random.choice(a, total_epoch) - 1)
        elif mode.lower() == "test": print()

        self.input = []
        self.label = []

        # train = [1,2,3] / test = [a, b]
        for i in tqdm(paths[mode.lower()], total=len(paths[mode]), desc="dataset", ascii=" =", position=0, leave=True):
            img = []

            # file list
            tif_path = sorted(glob(os.path.join(data_path, i, tif_folder, "*.tif")))

            # read file
            for j in tqdm(tif_path, total=len(tif_path), desc=f"image {i}", ascii=" =", position=1, leave=False):
                input_ = self.input_resize(mat=cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE), pad=pad)
                img.append(input_)
            
            label_path = os.path.join(data_path, i, "inklabels.png")
            label_ = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

            self.input.append(np.array(img, dtype=np.uint8))
            self.label.append(label_//255)

        self.shuffle_data()


    def __len__(self):
        return len(self.coord)
    

    def __getitem__(self, index):
        return super().__getitem__(index)
        

    def input_resize(self, mat:np.array, pad:int):
        return cv2.copyMakeBorder(mat, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
    
    
    def shuffle_data(self):
        self.img_num = self.order.pop()

        max_width = self.label[self.img_num].shape[1] - 572 + 1
        max_height = self.label[self.img_num].shape[0] - 572 + 1

        np.random.seed(int(time.time()))
        step = np.random.randint(3, 11)
        x = np.arange(0, max_width, step)
        np.random.shuffle(x)

        np.random.seed(int(time.time()))
        step = np.random.randint(3, 11)
        y = np.arange(0, max_height, step)
        np.random.shuffle(y)

        if x.shape[0] < y.shape[0] : y = y[:x.shape[0]]
        elif x.shape[0] > y.shape[0] : x = x[:y.shape[0]]
        
        self.coord = list(zip(y,x))
        print(np.unique(self.coord))
    

dataset = UnetData(".", 50, 4 * (2**4 + sum([2**(d+1) for d in range(4)])) // 2)