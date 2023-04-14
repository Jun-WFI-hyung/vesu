import json
from train import *
from test import *

# main
# 1. json load
# 2. run train / test

if __name__ == "__main__":
    with open("config.json") as f: cfg = json.load(f)
    mode = cfg["mode"].lower()
    
    if mode == "train": 
        train_ = train(cfg)
        train_.run()
        
    elif mode == "test": print()
        # test_pth1 = cfg["test_pth1"]
        # test_pth2 = cfg["test_pth2"]
        # test_pth3 = cfg["test_pth3"]
