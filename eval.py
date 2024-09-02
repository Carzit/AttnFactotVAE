import os
import sys
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm
from safetensors.torch import save_file, load_file
import numpy as np
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard.writer import SummaryWriter

from matplotlib import pyplot as plt
import plotly.graph_objs as go

from dataset import StockDataset, StockSequenceDataset
from nets import AttnFactorVAE
from loss import ObjectiveLoss, MSE_Loss, KL_Div_Loss, PearsonCorr, SpearmanCorr
from utils import str2bool


class FactorVAEEvaluator:
    def __init__(self,
                 model:AttnFactorVAE,
                 device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        
        self.model:AttnFactorVAE = model # FactorVAE 模型实例
        self.test_loader:DataLoader
        
        self.pred_eval_func:Union[nn.Module, Callable]
        self.latent_eval_func:Union[nn.Module, Callable] = KL_Div_Loss()
        self.pred_scores:List[float] = []
        self.latent_scores:List[float] = []
        
        self.y_true_list:List[torch.Tensor] = []
        self.y_hat_list:List[torch.Tensor] = []
        self.y_pred_list:List[torch.Tensor] = []
        
        self.log_folder:str = "log"
        self.device = device # 运算设备，默认为 CUDA（如果可用，否则为CPU）

        self.save_folder:str = "."
        self.plotter = Plotter()
    
    def load_dataset(self, test_set:StockSequenceDataset, num_workers:int = 4):
        self.test_loader = DataLoader(dataset=test_set,
                                        batch_size=None, 
                                        shuffle=False,
                                        num_workers=num_workers)

    def load_checkpoint(self, model_path:str):
        if model_path.endswith(".pt"):
            self.model.load_state_dict(torch.load(model_path))
        elif model_path.endswith(".safetensors"):
            self.model.load_state_dict(load_file(model_path))
    
    def calculate_icir(self, ic_list:List[float]):
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list, ddof=1)  # Use ddof=1 to get the sample standard deviation
        n = len(ic_list)
    
        if ic_std == 0:
            return float('inf') if ic_mean != 0 else 0
        
        icir = (ic_mean / ic_std) * np.sqrt(n)
        return icir
    
    def eval(self, metric:Literal["MSE", "IC", "Rank_IC", "ICIR", "Rank_ICIR"]="IC"):
        if metric == "MSE":
            self.pred_eval_func = MSE_Loss(scale=1)
        elif metric == "IC" or "ICIR":
            self.pred_eval_func = PearsonCorr()
        elif metric == "Rank_IC" or "Rank_ICIR":
            self.pred_eval_func = SpearmanCorr()
        
        self.eval_scores = []
        model = self.model.to(device=self.device)
        model.eval() # set eval mode to frozen layers like dropout
        with torch.no_grad(): 
            for batch, (quantity_price_feature, fundamental_feature, label) in enumerate(tqdm(self.test_loader)):
                if fundamental_feature.shape[0] <= 2:
                    continue
                quantity_price_feature = quantity_price_feature.to(device=self.device)
                fundamental_feature = fundamental_feature.to(device=self.device)
                label = label.to(device=self.device)
                y_pred, mu_posterior, sigma_posterior, mu_prior, sigma_prior = model(fundamental_feature, 
                                                                                     quantity_price_feature, 
                                                                                     label)
                y_hat, mu_prior, sigma_prior  = model.predict(fundamental_feature, 
                                                              quantity_price_feature,)
                pred_score = self.pred_eval_func(y_hat, label)
                latent_score = self.latent_eval_func(mu_prior, sigma_prior, mu_posterior, sigma_posterior)
                
                self.pred_scores.append(pred_score.item())
                self.latent_scores.append(latent_score.item())
                
                self.y_true_list.append(label)
                self.y_hat_list.append(y_hat)
                self.y_pred_list.append(y_pred)
        if metric == "MSE" or "IC" or "Rank_IC":
            y_pred_score = sum(self.pred_scores) / len(self.pred_scores)
        elif metric == "ICIR" or "Rank_ICIR":
            y_pred_score = self.calculate_icir(self.pred_scores)
        latent_kl_div = sum(self.latent_scores) / len(self.latent_scores)
        logging.info(f"y pred score: {y_pred_score}")
        logging.info(f"latent kl divergence: {latent_kl_div}")
    
    def visualize(self, idx:int=0, save_folder:Optional[str]=None):
        if save_folder is not None:
            self.save_folder = save_folder
        self.plotter.plot_score(self.pred_scores, 
                                self.latent_scores)
        self.plotter.save_fig(os.path.join(self.save_folder, "Scores"))
        self.plotter.plot_pred_sample(self.y_true_list, 
                                      self.y_hat_list, 
                                      self.y_pred_list,
                                      idx=idx)
        self.plotter.save_fig(os.path.join(self.save_folder, f"Trace {idx}"))

class Plotter:
    def __init__(self) -> None:
        pass
    
    def plot_score(self, pred_scores, latent_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(pred_scores, label='pred scores', marker='', color="b")
        plt.plot(latent_scores, label='latent scores', marker='', color="r")

        plt.legend()
        plt.title('Evaluation Scores')
        plt.xlabel('Index')
        plt.ylabel('Value')
    
    def plot_pred_sample(self, y_true_list, y_hat_list, y_pred_list, idx=0):
        y_true_list = [y_true[idx].item() for y_true in y_true_list]
        y_hat_list = [y_hat[idx].item() for y_hat in y_hat_list]
        y_pred_list = [y_pred[idx].item() for y_pred in y_pred_list]

        plt.figure(figsize=(10, 6))
        plt.plot(y_true_list, label='y true', marker='', color="g")
        plt.plot(y_hat_list, label='y rec', marker='', color="b")
        plt.plot(y_pred_list, label='y pred', marker='', color="r")

        plt.legend()
        plt.title('Comparison of y_true, y_hat, and y_pred')
        plt.xlabel('Index')
        plt.ylabel('Value')

    def save_fig(self, filename:str):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        plt.savefig(filename)
def parse_args():
    parser = argparse.ArgumentParser(description="FactorVAE Training.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path of dataset .pt file")
    parser.add_argument("--subset", type=str, default="test", help="Subset of dataset, literally `train`, `val` or `test`. Default `test`")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path of checkpoint")

    parser.add_argument("--quantity_price_feature_size", type=int, required=True, help="Input size of quantity-price feature")
    parser.add_argument("--fundamental_feature_size", type=int, required=True, help="Input size of fundamental feature")
    parser.add_argument("--num_gru_layers", type=int, required=True, help="Num of GRU layers in feature extractor.")
    parser.add_argument("--gru_hidden_size", type=int, required=True, help="Hidden size of each GRU layer. num_gru_layers * gru_hidden_size i.e. the input size of FactorEncoder and Factor Predictor.")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of FactorVAE(Encoder, Pedictor and Decoder), i.e. num of portfolios.")
    parser.add_argument("--latent_size", type=int, required=True, help="Latent size of FactorVAE(Encoder, Pedictor and Decoder), i.e. num of factors.")
    parser.add_argument("--std_activation", type=str, default="exp", help="Activation function for standard deviation calculation, literally `exp` or `softplus`. Default `exp`")
    
    parser.add_argument("--num_workers", type=int, default=4, help="Num of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default 4")
    parser.add_argument("--metric", type=str, default="IC", help="Eval metric type, literally `MSE`, `IC`, `Rank_IC`, `ICIR` or `Rank_ICIR`. Default `IC`. ")

    parser.add_argument("--visualize", type=str2bool, default=True, help="Whether to shuffle dataloader. Default True")
    parser.add_argument("--index", type=int, default=0, help="Stock index to plot Comparison of y_true, y_hat, and y_pred. Default 0")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save plot figures")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.log_folder, args.log_name)), logging.StreamHandler()])
    
    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")
    
    datasets:Dict[str, StockSequenceDataset] = torch.load(args.dataset_path)
    test_set = datasets[args.subset]

    model = AttnFactorVAE(fundamental_feature_size=args.fundamental_feature_size, 
                          quantity_price_feature_size=args.quantity_price_feature_size,
                          num_gru_layers=args.num_gru_layers, 
                          gru_hidden_size=args.gru_hidden_size, 
                          hidden_size=args.hidden_size, 
                          latent_size=args.latent_size,
                          gru_drop_out=0,
                          std_activ=args.std_activation)
    
    evaluator = FactorVAEEvaluator(model=model)
    evaluator.load_checkpoint(args.checkpoint_path)
    evaluator.load_dataset(test_set, num_workers=args.num_workers)
    
    evaluator.eval(metric=args.metric)
    if args.visualize:
        evaluator.visualize(idx=args.index, save_folder=args.save_folder)
        




                    
            
                

    

