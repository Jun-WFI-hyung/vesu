import torch, os


def save_model(pth_path, model, optim, epoch, eval_num, iter):
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    cp_path = os.path.join(pth_path, "eval%02d.pth" % eval_num)
    torch.save({"model" : model.state_dict(), 
                "optim" : optim.state_dict(), 
                "iter" : iter,
                "epoch" : epoch}, cp_path)
    
    print(" - save model\n")