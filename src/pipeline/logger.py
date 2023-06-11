import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.pipeline.visualization import plot_history

def load_cp(cp_path, model, optimizer, device):
    # load state dict
    cp_model_path = os.path.join(cp_path, "model.pt")
    state_dict = torch.load(cp_model_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    # load history
    cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
    print(f"loaded checkpoint from {cp_path}\n")
    return cp_history
    
class Logger:
    def __init__(self, arglist, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        save_path = os.path.join(arglist["exp_path"], arglist["model"], date_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(arglist, f)
        
        self.save_path = save_path
        self.cp_history = cp_history
        self.cp_every = arglist["cp_every"]
        self.iter = 0

    def __call__(self, model, df_hisotry):
        self.iter += 1
        if self.iter % self.cp_every == 0:
            self.save_history(df_hisotry)
            self.save_checkpoint(model)
    
    def save_history(self, df_history):
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)

        # plot history
        fig_history = plot_history(df_history)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)
        plt.clf()
        plt.close()
    
    def save_checkpoint(self, model):
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # optimizer_state_dict = {
        #     k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in optimizer.state_dict().items()
        # }
        
        model_path = os.path.join(self.save_path, "model.pt")
        torch.save({
            "model_state_dict": model_state_dict,
            # "optimizer_state_dict": optimizer_state_dict,
        }, model_path)
        print(f"\ncheckpoint saved at: {model_path}\n")