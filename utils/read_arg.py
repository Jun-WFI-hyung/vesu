import argparse
from utils.IOU import *

def read_train_arg():
    parser = argparse.ArgumentParser(description="Start Training U-net")
    parser.add_argument("load", help="T[true] or F[false]")
    parser.add_argument("-p", "--pth", help="Select *.pth file")
    parser.add_argument("-c", "--category", help="B[Binary] or C[Category]")

    return parser.parse_args()



def read_infer_arg():
    # mode = check mean of full time / check once time & write image
    parser = argparse.ArgumentParser(description="Start U-net Inference")
    parser.add_argument("pth", help="Select *.pth file")

    return parser.parse_args()



def check_mode(args, class_num_, device_):
    if args.load.upper() == 'T' and args.pth is not None: load_ = True
    elif args.load.upper() == 'T': raise Exception("Put in pth filename")
    elif args.load.upper() == 'F' :
        load_ = False

        if args.category is None : 
            raise Exception("Put in what kind of classify type")
        
        elif args.category.upper() == 'B': 
            loss_func = DiceLoss_BIN(class_num_, device_).to(device_)

        elif args.category.upper() == 'C': 
            loss_func = DiceLoss_BIN(class_num_, device_).to(device_)

        else : raise Exception("Put in right kind of classify type")
        
    else : raise Exception("Put in T or F at first arg")

    return loss_func, load_