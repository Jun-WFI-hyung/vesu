from net.Unet import Unet
from net.UnetData import UnetData
# from utils.save_load import *
# from utils.IOU import *
# from utils.read_arg import *

import json, time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms

# main
# 1. argparse
# 2. json load
# 3. run train / inference

if __name__ == "__main__":
    # self.pad = 4 * (2**depth + sum([2**(d+1) for d in range(depth)])) // 2
    print("")