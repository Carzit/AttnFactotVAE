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
import pandas as pd
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


class AttnFactorVAEOutput:
    def __init__(self,
                 model:AttnFactorVAE,
                 device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        
        self.model:AttnFactorVAE = model # FactorVAE 模型实例
        self.test_loader:DataLoader
        
        self.pred_eval_func:Union[nn.Module, Callable]
        self.latent_eval_func:Union[nn.Module, Callable] = KL_Div_Loss()
        self.pred_scores:List[float] = []
        self.latent_scores:List[float] = []
        
        self.dates:List[str]
        self.stock_codes:List[str]
        self.seq_len:int
        
        self.log_folder:str = "log"
        self.device = device # 运算设备，默认为 CUDA（如果可用，否则为CPU）

        self.save_folder:str = "."

    
    def load_dataset(self, test_set:StockSequenceDataset, num_workers:int = 4):
        self.stock_codes = ...
        self.dates = [f.removesuffix(".pkl") for f in os.listdir(test_set.stock_dataset.fundamental_feature_dir)]
        self.seq_len = test_set.seq_len
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
        
        icir = (ic_mean / ic_std) #* np.sqrt(n)
        return icir
    
    def eval(self):
        model = self.model.to(device=self.device)
        model.eval() # set eval mode to frozen layers like dropout
        with torch.no_grad(): 
            for batch, (quantity_price_feature, fundamental_feature, label, valid_indices) in enumerate(tqdm(self.test_loader)):
                if fundamental_feature.shape[0] <= 2:
                    continue
                quantity_price_feature = quantity_price_feature.to(device=self.device)
                fundamental_feature = fundamental_feature.to(device=self.device)
                label = label.to(device=self.device)
                y_pred, *_ = model.predict(fundamental_feature, quantity_price_feature)

                date = self.dates[batch + self.seq_len - 1]
                self.save_predictions_with_nan_handling(y_pred, valid_indices, date)

    def save_predictions_with_nan_handling(self, 
                                           predictions:torch.Tensor, 
                                           valid_indices:torch.Tensor,  
                                           file_path:str) -> None:
        """
        保存包含原始NaN值的预测结果CSV。对于NaN被drop掉的行，预测结果为0；否则保存模型的预测值。

        Args:
            stock_codes (torch.Tensor): 股票代码的张量 (num_stocks)。
            predictions (torch.Tensor): 模型计算后的预测结果 (num_valid_stocks)。
            valid_indices (torch.Tensor): 有效股票索引的布尔张量 (num_stocks)，用于标记哪些股票代码有效。
            date (str): 日期，作为CSV文件名。
            file_path (str): 保存CSV的文件路径。
        """
        num_stocks = len(self.stock_codes)
        full_predictions = torch.zeros(num_stocks)  # 初始化为0
        full_predictions[valid_indices] = predictions  # 有效行填充预测结果

        # 创建 DataFrame，包含股票代码和预测结果
        df = pd.DataFrame({
            'stock_code': self.stock_codes, 
            'prediction': full_predictions.cpu().numpy()
        })
        
        # 保存为CSV文件，文件名为日期
        df.to_pickle(f"{file_path}.csv")


    
 
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

    parser.add_argument("--visualize", type=str2bool, default=True, help="Whether to visualize the result. Default True")
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
        




                    
            
                

    

