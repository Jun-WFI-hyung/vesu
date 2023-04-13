import torch, os


def save_model(pth_path, model, optim, epoch, eval_num, iter):
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    cp_path = os.path.join(pth_path, "eval%02d.pth" % eval_num)
    torch.save({"model" : model.state_dict(), 
                "optim" : optim.state_dict(), 
                "iter" : iter,
                "epoch" : epoch}, cp_path)
    
    # log_path = os.path.join(pth_path, "log")
    # log_file_name = "log_%04d.txt"%epoch
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)

    # with open(os.path.join(log_path, log_file_name), "w", encoding="utf-8") as f:
    #     f.write(f"epoch : {epoch} / ")
    #     f.write(f"eval_num : {eval_num} / ")
    #     f.write(f"loss : {loss:.5f} / ")
    #     f.write(f"IOU : {IOU:.2f}\n")

    print(" - save model\n")