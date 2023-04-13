import os, json, cv2, time
import numpy as np
from glob import glob
from tqdm import tqdm
from itertools import product
from torch.utils.data import Dataset, DataLoader


class UnetData(Dataset):
    def __init__(self, data_path:str, total_epoch:int, pad:int, size:int, mode:str="train", eval_num:int=3):
        super(UnetData,self).__init__()
        self.pad = pad
        self.size = size
        self.mode = mode
        self.eval_num = eval_num
        self.total_epoch = total_epoch
        tif_folder = "surface_volume"

        tree_file = os.path.join(data_path, "tree.json")
        with open(tree_file) as f: paths = json.load(f)

        self.input = []
        self.label = []

        if mode == "train":
            self.iter = 0
            train_dir = paths[mode]
            train_dir.remove(str(self.eval_num))
            data_path_ = os.path.join(data_path, "train")

            self.generate_train_dataset(train_dir, data_path_, tif_folder)
            self.generate_train_order(train_dir)
            self.generate_train_coord()
            
        elif mode == "eval":
            data_path_ = os.path.join(data_path, "train")
            self.generate_eval_dataset(data_path_, tif_folder)
            self.generate_eval_coord()

        elif mode == "test":
            test_dir = paths[mode]
            data_path_ = os.path.join(data_path, "test")
            self.img_num = -1

            self.generate_test_dataset(test_dir, data_path_, tif_folder)
            self.generate_test_coord()

        else : raise Exception("Dataset : Wrong word in mode")



    def __len__(self):
        return len(self.coord)
    


    def __getitem__(self, index):
        if self.mode == "train":
            y, x = self.coord[index]
            input = self.input[self.img_num][:, y:y+self.size+self.pad*2, x:x+self.size+self.pad*2]
            label = self.label[self.img_num][y:y+self.size, x:x+self.size]
            self.iter += 1
            return np.array(input/255, dtype=np.float32), np.array([label], dtype=np.float32)
        else:
            y, x = self.coord[index]
            input = self.input[self.img_num][:, y:y+self.size+self.pad*2, x:x+self.size+self.pad*2]
            return np.array(input/255, dtype=np.float32)



    def input_resize(self, mat:np.array):
        return cv2.copyMakeBorder(mat, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT101)
    


    def generate_train_order(self, dir_list):
        a = [i for i in range(len(dir_list))]
        q, r = divmod(self.total_epoch, 2)
        a = a * (q+r)
        np.random.shuffle(a)
        self.order = list(np.array(a, dtype=np.uint8))


    
    def generate_train_dataset(self, dir_list:list, data_path_:str, tif_folder:str):
        print(f" - load train dataset image : {dir_list}")
        for bottom_dir in tqdm(dir_list, total=len(dir_list), desc=" - dataset", ascii=" =", position=0, leave=True):
            img = []

            # file list
            tif_path = sorted(glob(os.path.join(data_path_, bottom_dir, tif_folder, "*.tif")))

            # read file
            for tif_file in tqdm(tif_path, total=len(tif_path), desc=f" - image {bottom_dir}", ascii=" =", position=1, leave=False):
                input_ = self.input_resize(mat=cv2.imread(tif_file, flags=cv2.IMREAD_GRAYSCALE))
                img.append(input_)
            
            label_path = os.path.join(data_path_, bottom_dir, "inklabels.png")
            label_ = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

            self.input.append(np.array(img, dtype=np.uint8))
            self.label.append(label_//255)



    def generate_eval_dataset(self, data_path_:str, tif_folder:str):
        print(f" - load eval dataset image : {self.eval_num}")
        img = []

        # file list
        tif_path = sorted(glob(os.path.join(data_path_, str(self.eval_num), tif_folder, "*.tif")))

        # read file
        for j in tqdm(tif_path, total=len(tif_path), desc=" - dataset", ascii=" =", leave=True):
            input_ = self.input_resize(mat=cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE))
            img.append(input_)
        
        label_path = os.path.join(data_path_, str(self.eval_num), "inklabels.png")
        label_ = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

        self.input.append(np.array(img, dtype=np.uint8))
        self.label.append(label_//255)


    
    def generate_test_dataset(self, dir_list:list, data_path_:str, tif_folder:str):
        print(f" - load test dataset image : {dir_list}")
        self.test_shape = []
        for i in tqdm(dir_list, total=len(dir_list), desc=" - dataset", ascii=" =", position=0, leave=True):
            img = []

            # file list
            tif_path = sorted(glob(os.path.join(data_path_, i, tif_folder, "*.tif")))

            # read file
            flag_ = True
            for j in tqdm(tif_path, total=len(tif_path), desc=f"image {i}", ascii=" =", position=1, leave=False):
                input_ = cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE)

                if flag_ : 
                    self.test_shape.append(input_.shape)
                    flag_ = False

                input_ = self.input_resize(input_)
                img.append(input_)

            self.input.append(np.array(img, dtype=np.uint8))

    

    def generate_train_coord(self):
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
        
        print(f" - img {self.img_num+1} - generate train coord : check empty label")
        for y, x in tqdm(coord, total=len(coord), desc="coord", ascii=" =", leave=False):
            if np.count_nonzero(self.label[self.img_num][y:y+self.size, x:x+self.size]):
                self.coord.append([y,x])



    def generate_eval_coord(self):

        # image를 H, W 각각 tile사이즈로 나눈다.
        #
        # H, W 각각 좌표 리스트 생성
        #
        # - 나머지가 있을 경우
        # -- 마지막 좌표값 구해서 리스트에 추가
        #
        # y, x좌표 경우의수로 전부 순서대로 뽑기
        self.img_num = 0
        label_shape = self.label[0].shape

        yq, self.y_remainder = divmod(label_shape[0], self.size)
        xq, self.x_remainder = divmod(label_shape[1], self.size)

        self.y_list = [self.size*i for i in range(yq)]
        if self.y_remainder : self.y_list.append(label_shape[0]-self.size)

        self.x_list = [self.size*i for i in range(xq)]
        if self.x_remainder : self.x_list.append(label_shape[1]-self.size)

        self.coord = list(product(self.y_list, self.x_list))
        print(f" - generate eval coord")
        # print(f" - label_shape = {label_shape}")
        # print(f" - yq = {yq} / y_remainder = {self.y_remainder} / y_list = {len(self.y_list)}")
        # print(f" - xq = {xq} / x_remainder = {self.x_remainder} / x_list = {len(self.x_list)}")



    def generate_test_coord(self):
        # main.py에서 test image개수만큼 반복시킨다.

        self.img_num += 1
        label_shape = self.test_shape[self.img_num]

        yq, self.y_remainder = divmod(label_shape[0], self.size)
        xq, self.x_remainder = divmod(label_shape[1], self.size)

        y_list = [self.size*i for i in range(yq)]
        if self.y_remainder : y_list.append(label_shape[0]-self.size)

        x_list = [self.size*i for i in range(xq)]
        if self.x_remainder : x_list.append(label_shape[1]-self.size)

        self.coord = list(product(y_list, x_list))



    def get_eval_label(self):
        return self.label[0]




def get_dataloader(data_path:str, 
                   total_epoch:int, 
                   pad:int, size:int, 
                   mode:str="train", 
                   eval_num:int=3, 
                   batch_size:int=1,
                   drop:bool=True,
                   num_workers:int=0):
    
    dataset = UnetData(data_path=data_path, total_epoch=total_epoch, pad=pad, size=size, mode=mode, eval_num=eval_num)

    if mode == "train":
        loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop, shuffle=False, num_workers=num_workers)
    elif mode == "eval" or mode == "test":
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return dataset, loader


def calc_pad(depth:int=4):
    return 4 * (2**depth + sum([2**(d+1) for d in range(depth)])) // 2