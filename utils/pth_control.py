import torch, os



def save_model(pth_path, model, optim, epoch, eval_num, iter):
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    cp_path = os.path.join(pth_path, "eval%02d.pth" % eval_num)
    torch.save({"model" : model.state_dict(), 
                "optim" : optim.state_dict(), 
                "iter" : iter,
                "epoch" : epoch}, cp_path)
    


def load_model(pth_path, pth_name, model, optim=None):
    if not os.path.exists(pth_path):
        raise Exception("Please check pth_path")
    
    data = torch.load(os.path.join(pth_path, pth_name))

    model.load_state_dict(data["model"])

    if optim is not None: 
        optim.load_state_dict(data["optim"])
        epoch = data["epoch"]
        iter = data["iter"]
        return model, optim, epoch, iter
    
    else: return model