from net.Unet import Unet
from net.vesuvius_dataset import *
from utils.pth_control import *
from utils.IOU import *

import json
import numpy as np

import torch
import torch.backends.cudnn as cudnn


class train:
    def __init__(self, cfg:json):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Available Device = {self.device}")
        cudnn.enabled = cfg["cudnn_enable"]

        load = cfg["load"]
        dataset_path = cfg["dataset_path"]
        if not os.path.exists(dataset_path): raise Exception("Can't found dataset dir")

        self.pth_load_path = cfg["pth_load_path"]
        if not os.path.exists(self.pth_load_path): os.mkdir(self.pth_load_path)

        result_save_path = cfg["result_save_path"]
        if not os.path.exists(result_save_path): os.mkdir(result_save_path)

        self.epoch = cfg["epoch"]
        lr = cfg["lr"]
        size = cfg["size"]
        eval_num = cfg["eval_num"]
        batch_size = cfg["batch_size"]
        drop_last = cfg["drop_last"]
        num_workers = cfg["num_workers"]

        self.model = Unet(input_ch_=65, output_ch_=1).to(self.device)
        self.loss_func = DiceLoss_BIN(device=self.device).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.start_epoch = 0

        print(f" * Data init")
        self.train_data, self.train_loader = \
            get_dataloader(data_path=dataset_path,
                           total_epoch=self.epoch,
                           pad=calc_pad(),
                           size=size,
                           mode="train".lower(),
                           eval_num=eval_num,
                           batch_size=batch_size,
                           drop=drop_last,
                           num_workers=num_workers)
        
        self.eval_data, self.eval_loader = \
            get_dataloader(data_path=dataset_path,
                           total_epoch=self.epoch,
                           pad=calc_pad(),
                           size=size,
                           mode="eval".lower(),
                           eval_num=eval_num,
                           batch_size=1,
                           drop=drop_last,
                           num_workers=num_workers)        
        print(f" * Data init complete\n")

        if load : 
            train_pth = cfg["train_pth"]
            if not train_pth : raise Exception ("Fill in train_pth in config.json")


    
    def run(self):
        print(f" * Start train : Total epoch = {self.epoch}")
        pbar_epoch = tqdm(range(self.start_epoch+1, self.start_epoch+self.epoch+1),
                          desc=f"{self.start_epoch+1} / {self.start_epoch+self.epoch}",
                          ascii=" =", position=0, leave=True)
        
        for e in pbar_epoch:
            pbar_epoch.desc = f"{e} / {self.start_epoch+self.epoch}"

            # self.model.train()
            # self.iterate_train_data(loader=self.train_loader)
            
            self.model.eval()
            with torch.no_grad():
                output, loss, IOU = self.iterate_eval_data(loader=self.eval_loader,
                                                           dataset=self.eval_data)
            save_model(self.pth_load_path, 
                       self.model, 
                       self.optim, 
                       e, 
                       self.eval_data.eval_num, 
                       self.train_data.iter)
            
            self.train_data.generate_train_coord()


                
    def iterate_train_data(self, loader:DataLoader):
        loss_arr = []
        pbar_loader = tqdm(enumerate(loader), 
                        desc=f"loss : 0.00000 / IOU : 0.00%", 
                        total=len(loader), ascii=" =", 
                        position=1, leave=False)

        for idx, i in pbar_loader:
            input = i[0].to(self.device)
            label = i[1].to(self.device)
            output = self.model(input)

            loss, IOU = self.loss_func(output, label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            loss_arr += [loss.item()]
            pbar_loader.desc = f"loss : {np.mean(loss_arr):.5f} / IOU : {IOU.item()*100:.2f}%"



    def iterate_eval_data(self, loader:DataLoader, dataset:UnetData):
        eval_output:torch.Tensor
        pbar_loader = tqdm(enumerate(loader), 
                        desc=f"loss : 0.00000 / IOU : 0.00%", 
                        total=len(loader), ascii=" =", 
                        position=1, leave=False)

        for idx, i in pbar_loader:
            input = i.to(self.device)
            output = self.model(input)

            if not idx: eval_output = output
            else: eval_output = torch.cat([eval_output, output], dim=3)

        eval_output = self.reshape_eval_output(dataset=dataset, eval_output=eval_output)
        eval_label = dataset.get_eval_label()
        eval_label = torch.Tensor(np.array([[eval_label]])).to(self.device)
        eval_loss, IOU = self.loss_func(eval_output, eval_label)

        return eval_output, eval_loss.item(), IOU.item()*100

            
    
    def reshape_eval_output(self, dataset:UnetData, eval_output:torch.Tensor):
        eval_split = torch.chunk(eval_output, chunks=len(dataset.y_list), dim=3)
        eval_split = torch.cat(eval_split, dim=2)

        out_shape = eval_split.shape
        x_r = dataset.size - dataset.x_remainder
        y_r = dataset.size - dataset.y_remainder
        dim2_point = out_shape[2] - dataset.size
        dim3_point = out_shape[3] - dataset.size

        x_r_tensor = eval_split[:,:, :dim2_point, dim3_point+x_r:]
        y_r_tensor = eval_split[:,:, dim2_point+y_r:, :dim3_point]
        xy_r_tensor = eval_split[:,:, dim2_point+y_r:, dim3_point+x_r:]
        eval_split = eval_split[:,:, :dim2_point, :dim3_point]

        eval_split = torch.concat([eval_split, x_r_tensor], dim=3)
        del x_r_tensor
        y_r_tensor = torch.concat([y_r_tensor, xy_r_tensor], dim=3)
        del xy_r_tensor
        eval_split = torch.concat([eval_split, y_r_tensor], dim=2)
        del y_r_tensor

        return eval_split
    


    def save_result(self):
        return