import os
import sys
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm
from safetensors.torch import save_file, load_file
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import StockDataset, StockSequenceDataset, RandomSampleSampler
from nets import AttnRet, AttnFactorVAE
from loss import MSE_Loss
from utils import str2bool, str2dtype, str2device, get_optimizer, get_lr_scheduler, read_config, save_config

class AttnRetTrainer:
    """FactorVAE Trainer，用于训练和评估一个基于因子变分自编码器（FactorVAE）的模型"""
    def __init__(self,
                 model:AttnRet,
                 loss_func:MSE_Loss = None,
                 optimizer:torch.optim.Optimizer = None,
                 lr_scheduler:torch.optim.lr_scheduler.LRScheduler = None,
                 dtype:torch.dtype = torch.float32,
                 device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        
        self.model: AttnRet = model
        self.loss_func:MSE_Loss = loss_func
        self.optimizer: torch.optim.Optimizer = optimizer

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = lr_scheduler

        self.train_loader:DataLoader
        self.val_loader:DataLoader

        self.writer:SummaryWriter = None

        self.max_epoches:int
        self.hparams:Optional[dict]

        self.log_folder:str = "log"
        self.sample_per_batch:int = 0
        self.report_per_epoch:int = 1
        self.save_per_epoch:int = 1
        self.save_folder:str = os.curdir
        self.save_name:str = "Model"
        self.save_format:Literal[".pt", ".safetensors"] = ".pt"
        
        self.dtype:torch.dtype = dtype
        self.device:torch.device = device
        
    def load_dataset(self, 
                     train_set:StockSequenceDataset, 
                     val_set:StockSequenceDataset,
                     batch_size:Optional[int] = None,
                     sampler:Optional[Sampler] = None,
                     shuffle:bool = True,
                     num_workers:int = 4):
        # 数据集加载
        if sampler is not None:
            self.train_loader = DataLoader(dataset=train_set,
                                        batch_size=batch_size, 
                                        sampler=sampler,
                                        num_workers=num_workers,)
        else:
            self.train_loader = DataLoader(dataset=train_set,
                                        batch_size=batch_size, 
                                        shuffle=shuffle,
                                        num_workers=num_workers)
        self.val_loader = DataLoader(dataset=val_set, 
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers)
        
    def save_checkpoint(self, 
                        save_folder:str, 
                        save_name:str, 
                        save_format:Literal[".pt",".safetensors"]=".pt"):
        # 模型保存
        save_path = os.path.join(save_folder, save_name+save_format)
        if save_format == ".pt":
            torch.save(self.model.state_dict(), save_path)
        elif save_format == ".safetensors":
            save_file(self.model.state_dict(), save_path)

    def load_checkpoint(self,
                        model_path:str):
        # 模型加载
        if model_path.endswith(".pt"):
            self.model.load_state_dict(torch.load(model_path))
        elif model_path.endswith(".safetensors"):
            self.model.load_state_dict(load_file(model_path))
        else:
            pass

    def set_configs(self,
                    max_epoches:int,
                    grad_clip:float = None,
                    hparams:Optional[dict] = None,
                    log_folder:str = "log",
                    sample_per_batch:int = 0,
                    report_per_epoch:int=1,
                    save_per_epoch:int=1,
                    save_folder:str=os.curdir,
                    save_name:str="Model",
                    save_format:str=".pt"):
        # 配置设置
        self.max_epoches = max_epoches
        self.grad_clip = grad_clip
        self.hparams = hparams
        
        self.log_folder = log_folder
        self.sample_per_batch = sample_per_batch
        self.report_per_epoch = report_per_epoch
        self.save_per_epoch = save_per_epoch
        self.save_folder = save_folder
        self.save_name = save_name
        self.save_format = save_format

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.writer = SummaryWriter(
            os.path.join(
                self.log_folder, f"TRAIN_{self.save_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ))
        
    def train(self):
        writer = self.writer
        model = self.model.to(device=self.device, dtype=self.dtype)
        loss_func = self.loss_func
        optimizer = self.optimizer
        
        # 主训练循环
        for epoch in range(self.max_epoches):
            train_loss_list = []
            val_loss_list = []
            
            model.train()
            for batch, (quantity_price_feature, fundamental_feature, label) in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.max_epoches}] Train")):    
                if fundamental_feature.shape[0] <= 2:
                    continue
                quantity_price_feature = quantity_price_feature.to(device=self.device)
                fundamental_feature = fundamental_feature.to(device=self.device)
                label = label.to(device=self.device)
                
                optimizer.zero_grad()

                out = model(fundamental_feature, quantity_price_feature)
                train_loss = loss_func(out, label)
                train_loss.backward() # 梯度反向传播
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.grad_clip)
                optimizer.step()
                
                train_loss_list.append(train_loss.item())
                
                # 训练时抽样检查
                if self.sample_per_batch:
                    if (batch+1) % self.sample_per_batch == 0:
                        logging.debug(f"<Batch {batch+1}>  loss:{train_loss.item()}")
                        logging.debug(f"<Batch {batch+1}>  y_pred:{out} label:{label}")
              
            # Tensorboard写入当前完成epoch的训练损失
            train_loss_epoch = sum(train_loss_list)/len(train_loss_list)
            writer.add_scalar("Train Loss", train_loss_epoch, epoch+1)
            
            # 交叉验证集上验证（无梯度）
            model.eval() # 设置为eval模式以冻结dropout
            with torch.no_grad(): 
                for batch, (quantity_price_feature, fundamental_feature, label) in enumerate(tqdm(self.val_loader, desc=f"Epoch [{epoch+1}/{self.max_epoches}] Val")):
                    if fundamental_feature.shape[0] <= 2:
                        continue
                    quantity_price_feature = quantity_price_feature.to(device=self.device)
                    fundamental_feature = fundamental_feature.to(device=self.device)
                    label = label.to(device=self.device)
                    out = model(fundamental_feature, quantity_price_feature)
                    val_loss = loss_func(out, label)

                    val_loss_list.append(val_loss.item())

                val_loss_epoch = sum(val_loss_list) / len(val_loss_list)  
                writer.add_scalar("Validation Loss", val_loss_epoch, epoch+1)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss_epoch, "Validation Loss": val_loss_epoch}, epoch+1)
                writer.add_scalars("Loss", {"Train": sum(train_loss_list)/len(train_loss_list), "Val": sum(val_loss_list)/len(val_loss_list)}, epoch+1)
                

            # 如果有超参数字典传入，Tensorboard记录超参数
            if self.hparams:
                writer.add_hparams(hparam_dict=self.hparams, metric_dict={"hparam/TrainLoss":train_loss_epoch, "hparam/ValLoss":val_loss_epoch})

            # 如果有学习率调度器传入，则更新之。
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch+1)
            self.lr_scheduler.step()
            
            # Tensorboard写入磁盘
            writer.flush()

            # 打印每个epoch训练结果
            if self.report_per_epoch:
                if (epoch+1) % self.report_per_epoch == 0:
                    logging.info('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, self.max_epoches, train_loss_epoch, val_loss_epoch))
            
            # 保存模型
            if self.save_per_epoch:
                if (epoch+1) % self.save_per_epoch == 0:
                    model_name = f"{self.save_name}_epoch{epoch+1}"
                    self.save_checkpoint(save_folder=self.save_folder,
                                         save_name=model_name,
                                         save_format=self.save_format)
                    logging.info(f"Epoch {epoch+1} Model weights saved to {os.path.join(self.save_folder)}")

        writer.close()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AttnRet Training.")
    
    parser.add_argument("--config_file", type=str, default=None, help="Path of config file. Optional")
    parser.add_argument("--output_config", type=str, default=None, help="Path of output config file. Default saved to save_folder as `config.toml`")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    # dataloader config
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of dataset .pt file")
    parser.add_argument("--num_workers", type=int, default=4, help="Num of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default 4")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Whether to shuffle dataloader. Default True")
    parser.add_argument("--num_batches_per_epoch", type=int, default=-1, help="Num of batches sampled from all batches to be trained per epoch. Note that sampler option is mutually exclusive with shuffle. Specify -1 to disable (use all batches). Default -1")
    
    # model config
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path of checkpoint. Optional")
    parser.add_argument("--quantity_price_feature_size", type=int, required=True, help="Input size of quantity-price feature")
    parser.add_argument("--fundamental_feature_size", type=int, required=True, help="Input size of fundamental feature")
    parser.add_argument("--num_gru_layers", type=int, required=True, help="Num of GRU layers in feature extractor.")
    parser.add_argument("--gru_hidden_size", type=int, required=True, help="Hidden size of each GRU layer. num_gru_layers * gru_hidden_size i.e. the input size of FactorEncoder and Factor Predictor.")
    parser.add_argument("--gru_dropout", type=float, default=0.1, help="Dropout probs in gru layers. Default 0.1")
    parser.add_argument("--num_fc_layers", type=int, required=True, help="Num of full connected layers in MLP")

    # optimizer config
    parser.add_argument("--optimizer_type", type=str, default="Lion", choices=["Adam", "AdamW", "Lion", "SGDNesterov", "DAdaptation", "Adafactor"], help="Optimizer for training. Literally `Adam`, `AdamW`, `Lion`, `SGDNesterov`, `DAdaptation` or `Adafactor`. Default `Lion`")
    parser.add_argument("--optimizer_kwargs", type=str, default=None, nargs="+", help="Key arguments for optimizer. e.g. `betas=(0.9, 0.99) weight_decay=0.0 use_triton=False decoupled_weight_decay=False` for optimizer Lion by default")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer of AttnRet. Default 0.001")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"], help="Learning rate scheduler for optimizer. Literally `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `adafactor`. Default `constant`.")
    parser.add_argument("--lr_scheduler_warmup_steps", type=int, default=0, help="Number of steps for the warmup phase in the learning rate scheduler. Default 0")
    parser.add_argument("--lr_scheduler_num_cycles", type=float, default=0.5, help="Number of cycles (for cosine scheduler) or factor in polynomial scheduler. Default 0.5")
    parser.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power factor for polynomial learning rate scheduler. Default 1.0")
    
    # loss configs
    parser.add_argument("--scale", type=float, default=100, help="Scale for MSE Loss. Default 100")
    
    # train configs
    parser.add_argument("--max_epoches", type=int, default=20, help="Max Epoches for train loop")
    parser.add_argument("--grad_clip", type=float, default=-1, help="Value of gradient clipping. Specify -1 to disable. Default -1")
    parser.add_argument("--detect_anomaly", type=str2bool, default=False, help="Debug option. When enabled, PyTorch detects unusual operations (such as NaN or inf values) in the computation graph and throws exceptions to help locate the source of the problem. But it will greatly reduce the training performance. Default False")
    parser.add_argument("--dtype", type=str, default="FP32", choices=["FP32", "FP64", "FP16", "BF16"], help="Dtype of data and weight tensor. Literally `FP32`, `FP64`, `FP16` or `BF16`. Default `FP32`")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="Device to take calculation. Literally `cpu` or `cuda`. Default `cuda`")
    parser.add_argument("--sample_per_batch", type=int, default=0, help="Check X, y and all kinds of outputs per n batches in one epoch. Specify 0 to disable. Default 0")
    parser.add_argument("--report_per_epoch", type=int, default=1, help="Report train loss and validation loss per n epoches. Specify 0 to disable. Default 1")
    parser.add_argument("--save_per_epoch", type=int, default=1, help="Save model weights per n epoches. Specify 0 to disable. Default 1")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save model")
    parser.add_argument("--save_name", type=str, default="Model", help="Model name. Default `Model`")
    parser.add_argument("--save_format", type=str, default=".pt", help="File format of model to save, literally `.pt` or `.safetensors`. Default `.pt`")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.config_file:
        args = read_config(args.config_file, parser=parser)

    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)
    
    if args.output_config:
        save_config(args, args.output_config)
    else:
        save_config(args, os.path.join(args.save_folder, "config.toml"))

    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.log_folder, args.log_name)), logging.StreamHandler()])
    
    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    datasets:Dict[str, StockSequenceDataset] = torch.load(args.dataset_path)
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]
    if args.num_batches_per_epoch != -1:
        train_sampler = RandomSampleSampler(train_set, args.num_batches_per_epoch)
    else:
        train_sampler = None

    model = AttnRet(fundamental_feature_size=args.fundamental_feature_size, 
                          quantity_price_feature_size=args.quantity_price_feature_size,
                          num_gru_layers=args.num_gru_layers, 
                          gru_hidden_size=args.gru_hidden_size, 
                          gru_drop_out=args.gru_dropout,
                          num_fc_layers=args.num_fc_layers)
    
    pretrained_attnvae = AttnFactorVAE(quantity_price_feature_size=101,
                                        fundamental_feature_size=31,
                                        num_gru_layers=4,
                                        gru_hidden_size=32,
                                        hidden_size=100,
                                        latent_size=48)
    pretrained_attnvae.load_state_dict(torch.load(r"C:\Users\21863\Desktop\temp\AttnFactorVAE_test2_epoch40.pt"))
    feature_extractor_state_dict = {k.removeprefix("feature_extractor."): v for k, v in pretrained_attnvae.state_dict().items() if k.startswith("feature_extractor")}
    model.feature_extractor.load_state_dict(feature_extractor_state_dict)
    
    loss_func = MSE_Loss(scale=args.scale)

    optimizer = get_optimizer(args, 
                              trainable_params=model.parameters(), 
                              lr=args.learning_rate)
    lr_scheduler = get_lr_scheduler(args, 
                                    optimizer=optimizer, 
                                    lr=args.learning_rate)

    hparams = {"fundamental_feature_size":args.fundamental_feature_size, 
               "quantity_price_feature_size":args.quantity_price_feature_size,
               "num_gru_layers": args.num_gru_layers, 
               "gru_hidden_size": args.gru_hidden_size, 
               "gru_drop_out": args.gru_dropout,
               "num_fc_layers":  args.num_fc_layers,
               "checkpoint": args.checkpoint_path,
               "num_batches_per_epoch": args.num_batches_per_epoch}

    trainer = AttnRetTrainer(model=model,
                               loss_func=loss_func, 
                               optimizer=optimizer,lr_scheduler=lr_scheduler, 
                               dtype=str2dtype(args.dtype),
                               device=str2device(args.device))
    trainer.load_dataset(train_set=train_set, 
                         val_set=val_set, 
                         shuffle=args.shuffle, 
                         sampler=train_sampler,
                         num_workers=args.num_workers)
    if args.checkpoint_path is not None:
        trainer.load_checkpoint(args.checkpoint_path)
    trainer.set_configs(max_epoches=args.max_epoches,
                        grad_clip=args.grad_clip,
                        log_folder=args.log_folder,
                        sample_per_batch=args.sample_per_batch,
                        report_per_epoch=args.report_per_epoch,
                        save_per_epoch=args.save_per_epoch,
                        save_folder=args.save_folder,
                        save_name=args.save_name,
                        save_format=args.save_format,
                        hparams=hparams)
    logging.info("Training start...")
    trainer.train()
    logging.info("Training complete.")
