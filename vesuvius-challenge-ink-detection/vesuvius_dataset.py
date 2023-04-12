import os, json, cv2, time
import numpy as np
from glob import glob
from tqdm import tqdm

from torch.utils.data import Dataset
# from utils.factorization import *

# 가로 / 세로 랜덤 수 추출 후 데이터셋 추출
# inference에서는 600px정도로 등분

def calc_pad(depth:int=4):
    return 4 * (2**depth + sum([2**(d+1) for d in range(depth)])) // 2

class UnetData(Dataset):
    def __init__(self, data_path:str, total_epoch:int, pad:int, size:int, mode:str="train", eval_num:int=3):
        super(UnetData,self).__init__()
        self.pad = pad
        self.size = size
        self.iter = 0
        self.eval_num = eval_num
        self.total_epoch = total_epoch

        tree_file = "tree.json"
        with open(tree_file) as f: paths = json.load(f)
        dir_list = paths[mode]

        self.input = []
        self.label = []

        if mode == "train":
            data_path_ = os.path.join(data_path, mode)
            tif_folder = "surface_volume"
            self.generate_order(dir_list)
            self.generate_train_dataset(dir_list, data_path_, tif_folder)
            
        elif mode == "test": print()

        self.shuffle_data()


    def __len__(self):
        return len(self.coord)
    

    def __getitem__(self, index):
        y, x = self.coord[index]
        input = self.input[self.img_num][:, y:y+self.size+self.pad*2, x:x+self.size+self.pad*2]
        label = self.label[self.img_num][y:y+self.size, x:x+self.size]
        return input, label
        

    def input_resize(self, mat:np.array):
        return cv2.copyMakeBorder(mat, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT101)
    

    def generate_order(self, dir_list):
        a = list(map(int, dir_list))
        a.remove(self.eval_num)
        q, r = divmod(self.total_epoch, 2)
        a = a * (q+r)
        np.random.shuffle(a)
        self.order = list(np.array(a, dtype=np.uint8)-1)

    
    def generate_train_dataset(self, dir_list:list, data_path_:str, tif_folder:str):
        # train = [1,2,3] / test = [a, b]
        for i in tqdm(dir_list, total=len(dir_list), desc="dataset", ascii=" =", position=0, leave=True):
            img = []

            # file list
            tif_path = sorted(glob(os.path.join(data_path_, i, tif_folder, "*.tif")))

            # read file
            for j in tqdm(tif_path, total=len(tif_path), desc=f"image {i}", ascii=" =", position=1, leave=False):
                input_ = self.input_resize(mat=cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE))
                img.append(input_)
            
            label_path = os.path.join(data_path_, i, "inklabels.png")
            label_ = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

            self.input.append(np.array(img, dtype=np.uint8))
            self.label.append(label_//255)
    
    
    def shuffle_data(self):
        self.img_num = self.order.pop()

        max_width = self.label[self.img_num].shape[1] - self.size + 1
        max_height = self.label[self.img_num].shape[0] - self.size + 1

        np.random.seed(int(time.time()))
        step = np.random.randint(2, 4)
        x = np.arange(0, max_width, step)
        np.random.shuffle(x)

        np.random.seed(int(time.time()))
        step = np.random.randint(2, 4)
        y = np.arange(0, max_height, step)
        np.random.shuffle(y)

        if x.shape[0] < y.shape[0] : y = y[:x.shape[0]]
        elif x.shape[0] > y.shape[0] : x = x[:y.shape[0]]
        
        coord = list(zip(y,x))
        self.coord = []
        
        print(f" - img {self.img_num+1} - shuffle data : check empty label")
        for y, x in tqdm(coord, total=len(coord), desc="coord", ascii=" =", leave=True):
            if np.count_nonzero(self.label[self.img_num][y:y+self.size, x:x+self.size]):
                self.coord.append([y,x])

    

dataset = UnetData(".", 50, calc_pad(), 388)
print(len(dataset))
i, l = dataset[2]
print(i.shape, type(i))
print(l.shape, type(l))