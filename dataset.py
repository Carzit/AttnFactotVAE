import os
import sys
import random
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from functools import lru_cache

import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np

def save_dataframe(df:pd.DataFrame, path:str, format:Literal["csv", "pkl", "parquet", "feather"])->None:
    if format == "csv":
        df.to_csv(path)
    elif format ==  "pkl":
        df.to_pickle(path)
    elif format == "parquet":
        df.to_parquet(path)
    elif format == "feather":
        df.to_feather(path)
    else:
        raise NotImplementedError()

def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"])->pd.DataFrame:
    if format == "csv":
        df = pd.read_csv(path, index_col=0)
    elif format ==  "pkl":
        df = pd.read_pickle(path)
    elif format == "parquet":
        df = pd.read_parquet(path)
    elif format == "feather":
        df = pd.read_feather(path)
    else:
        raise NotImplementedError()
    return df

def translate_dtype(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return torch.float32
    elif dtype == "FP64":
        return torch.float64
    elif dtype == "FP16":
        return torch.float16
    elif dtype == "BF16":
        return torch.bfloat16

@torch.no_grad()
def drop_nan_inf(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor]:
    a_valid_mask = torch.all(~torch.isnan(a) & ~torch.isinf(a), dim=(0, 2))
    b_valid_mask = torch.all(~torch.isnan(b) & ~torch.isinf(b), dim=1)
    c_valid_mask = ~torch.isnan(c) & ~torch.isinf(c)
    
    valid_indices = a_valid_mask & b_valid_mask & c_valid_mask
    
    a_filtered = a[:, valid_indices, :]
    b_filtered = b[valid_indices, :]
    c_filtered = c[valid_indices]
    
    return a_filtered, b_filtered, c_filtered

@torch.no_grad()
def convert_nan_inf(tensor:torch.Tensor) -> torch.Tensor:
    converted_tensor = tensor.nan_to_num()
    return converted_tensor

class StockDataset(Dataset):
    """继承自 Dataset，用于加载和预处理股票数据集。支持序列化分割和随机分割，并提供了数据集的信息。"""
    def __init__(self, 
                 quantity_price_feature_dir:str, 
                 fundamental_feature_dir:str,
                 label_dir:str, 
                 label_name:str, 
                 format:Literal["csv", "pkl", "parquet", "feather"] = "pkl",
                 cache_size:int = 10, 
                 dtype:Literal["FP32", "FP64", "FP16", "BF16"] = "FP32") -> None:
        super().__init__()
        self.quantity_price_feature_dir:str = quantity_price_feature_dir
        self.fundamental_feature_dir:str = fundamental_feature_dir
        self.label_dir:str = label_dir
        
        self.format:str = format
        self.label_name:str = label_name

        self.quantity_price_feature_file_paths:List[str] = [os.path.join(quantity_price_feature_dir, f) for f in os.listdir(quantity_price_feature_dir) if f.endswith(format)]
        self.fundamental_feature_file_paths:List[str] = [os.path.join(fundamental_feature_dir, f) for f in os.listdir(fundamental_feature_dir) if f.endswith(format)]
        self.label_file_paths:List[str] = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(format)]

        self.cache_size:int = cache_size
        self.dtype:torch.dtype = translate_dtype(dtype=dtype)
        #self.load_df:Callable = self.create_cached_loader() 

    @lru_cache(maxsize=10)
    def load_dataframe(self, path:str, format:Literal["csv", "pkl", "parquet", "feather"])->pd.DataFrame:
        if format == "csv":
            df = pd.read_csv(path, index_col=0)
        elif format ==  "pkl":
            df = pd.read_pickle(path)
        elif format == "parquet":
            df = pd.read_parquet(path)
        elif format == "feather":
            df = pd.read_feather(path)
        else:
            raise NotImplementedError()
        return df

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        """根据索引 index 从文件路径列表中读取对应的 CSV 文件。将读取的数据转换为 PyTorch 张量，并处理 NaN 值。
        注意，这是一种惰性获取数据的协议，数据集类并不事先将所有数据载入内存，而是根据索引再做读取。能较好地应对大量数据的情形，节约内存；但训练时由于IO瓶颈速度较慢，需要花费大量时间在csv的读取上面。"""
        quantity_price_feature = self.load_dataframe(self.quantity_price_feature_file_paths[index], format=self.format).iloc[:,1:]
        quantity_price_feature = torch.from_numpy(quantity_price_feature.values).to(self.dtype)
        
        fundamental_feature = self.load_dataframe(self.fundamental_feature_file_paths[index], format=self.format).iloc[:,1:]
        fundamental_feature = torch.from_numpy(fundamental_feature.values).to(self.dtype)

        label = self.load_dataframe(self.label_file_paths[index], format=self.format).iloc[:,1:][self.label_name]
        label = torch.from_numpy(label.values).to(self.dtype)
        return quantity_price_feature, fundamental_feature, label
    
    def __len__(self):          
        """获取数据集长度"""
        return min(len(self.quantity_price_feature_file_paths), 
                   len(self.fundamental_feature_file_paths), 
                   len(self.label_file_paths))
    
    def serial_split(self, ratios:List[Number], mask:int=0) -> List["StockDataset"]:
        """序列化分割。根据给定的比例 ratios 将数据集分割成多个子数据集，且保持原顺序。返回一个StockDataset类的列表。"""
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(quantity_price_feature_dir=self.quantity_price_feature_dir, 
                                            fundamental_feature_dir=self.fundamental_feature_dir,
                                            label_dir=self.label_dir, 
                                            label_name=self.label_name, 
                                            format=self.format, 
                                            cache_size=self.cache_size)
            splitted_dataset.quantity_price_feature_file_paths = splitted_dataset.quantity_price_feature_file_paths[i:i+j]
            splitted_dataset.fundamental_feature_file_paths = splitted_dataset.fundamental_feature_file_paths[i:i+j]
            splitted_dataset.label_file_paths = splitted_dataset.label_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += (j+mask)

        return splitted_datasets
    
    def random_split(self, ratios:List[Number]) -> List["StockDataset"]:
        """随机分割。根据给定的比例 ratios 将数据集分割成多个子数据集，且顺序随机。返回一个StockDataset类的列表。"""
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        base_names = [os.path.basename(file_path) for file_path in self.quantity_price_feature_file_paths]
        random.shuffle(base_names)
        shuffled_quantity_price_feature_file_paths = [os.path.join(self.quantity_price_feature_dir, base_name) for base_name in base_names]
        shuffled_fundamental_feature_file_paths = [os.path.join(self.fundamental_feature_dir, base_name) for base_name in base_names]
        shuffled_label_file_paths = [os.path.join(self.label_dir, base_name) for base_name in base_names]

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(quantity_price_feature_dir=self.quantity_price_feature_dir, 
                                            fundamental_feature_dir=self.fundamental_feature_dir,
                                            label_dir=self.label_dir, 
                                            label_name=self.label_name, 
                                            format=self.format, 
                                            cache_size=self.cache_size)
            splitted_dataset.quantity_price_feature_file_paths = shuffled_quantity_price_feature_file_paths[i:i+j]
            splitted_dataset.fundamental_feature_file_paths = shuffled_fundamental_feature_file_paths[i:i+j]
            splitted_dataset.label_file_paths = shuffled_label_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += j
            
        return splitted_datasets

class StockSequenceDataset(Dataset):
    """StockSequenceDataset 类继承自 Dataset，将 StockDataset 中的单日数据堆叠为日期序列数据"""
    def __init__(self, 
                 stock_dataset:StockDataset, 
                 seq_len:int,
                 mode:Literal["convert", "drop"] = "convert") -> None:
        super().__init__()
        self.stock_dataset:StockDataset = stock_dataset # 原StockDataset对象
        self.seq_len:int = seq_len # 日期序列长度
        self.mode:str = mode

    def __len__(self):
        """获取数据集长度"""
        return len(self.stock_dataset) - self.seq_len + 1
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        """根据索引 index 从 StockDataset 中获取一个长度为 seq_len 的序列数据。"""
        quantity_price_feature = torch.stack([self.stock_dataset[i][0] for i in range(index, index+self.seq_len)], dim=0) # (seq_len, num_stock, num_feature1)
        fundamental_feature = self.stock_dataset[index+self.seq_len-1][1] # (num_stock, num_feature2)
        label = self.stock_dataset[index+self.seq_len-1][2] # (num_stock)
        if self.mode == "convert":
            quantity_price_feature = convert_nan_inf(quantity_price_feature)
            fundamental_feature = convert_nan_inf(fundamental_feature)
            label = convert_nan_inf(label)
        elif self.mode == "drop":
            quantity_price_feature, fundamental_feature, label = drop_nan_inf(quantity_price_feature, fundamental_feature, label)
        return quantity_price_feature, fundamental_feature, label

class RandomSampleSampler(Sampler):
    def __init__(self, data_source:Dataset, num_samples_per_epoch:int):
        self.data_source:Dataset = data_source
        self.num_samples_per_epoch:int = num_samples_per_epoch

    def __iter__(self):
        indices = random.sample(range(len(self.data_source)), self.num_samples_per_epoch)
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_epoch

class RandomBatchSampler(Sampler):
    def __init__(self, data_source:Dataset, num_batches_per_epoch:int, batch_size:int):
        self.data_source:Dataset = data_source
        self.num_batches_per_epoch:int = num_batches_per_epoch
        self.batch_size:int = batch_size

    def __iter__(self):
        # 计算总的batch数量
        num_total_batches = len(self.data_source) // self.batch_size
        # 随机选择 num_batches_per_epoch 个批次的起始索引
        selected_batches = random.sample(range(num_total_batches), self.num_batches_per_epoch)
        # 生成这些批次的所有索引
        indices = []
        for batch_idx in selected_batches:
            start_idx = batch_idx * self.batch_size
            indices.extend(range(start_idx, start_idx + self.batch_size))
        return iter(indices)

    def __len__(self):
        return self.num_batches_per_epoch * self.batch_size

# TODO
def parse_args():
    parser = argparse.ArgumentParser(description="Data acquisition and dataset generation.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--quantity_price_feature_dir", type=str, required=True, help="Path of folder for quantity-price feature data files")
    parser.add_argument("--fundamental_feature_dir", type=str, required=True, help="Path of folder for fundamental feature data files")
    parser.add_argument("--label_dir", type=str, required=True, help="Path of folder for label data files")

    parser.add_argument("--file_format", type=str, default="pkl", help="File format to read, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")
    parser.add_argument("--label_name", type=str, required=True, help="Target label name (col name in y files)")

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Split ratio for train-validation-test. Default 0.7, 0.2, 0.1")
    parser.add_argument("--mask_len", type=int, default=0, help="Mask seq length to avoid model cheat(get information from future), e.g. mask 20 days when training prediction for ret20. Default 0")
    parser.add_argument("--mode", type=str, default="convert", help="Mode of processing missing value and infinite value. Literally `convert`(convert Nan to 0 and Inf to the greatest finite value representable by input’s dtype) or `drop`(drop stock code containing Nan or Inf in the whole sequence). Default `convert`")
    parser.add_argument("--dtype", type=str, default="FP32", help="Dtype of data tensor. Literally `FP32`, `FP64`, `FP16` or `BF16`. Default `FP32`")
    parser.add_argument("--train_seq_len", type=int, required=True, help="Sequence length (num of days) for train dataset")
    parser.add_argument("--val_seq_len", type=int, default=None, help="Sequence length (num of days) for validation dataset. If not specified, default equal to train_seq_len.")
    parser.add_argument("--test_seq_len", type=int, default=None, help="Sequence length (num of days) for test dataset. If not specified, default equal to train_seq_len.")

    parser.add_argument("--save_path", type=str, required=True, help="Path to save the dataset dictionary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_folder = args.log_folder
    log_name = args.log_name
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_folder, log_name)), logging.StreamHandler()])
    
    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    dataset = StockDataset(quantity_price_feature_dir=args.quantity_price_feature_dir,
                           fundamental_feature_dir=args.fundamental_feature_dir,
                           label_dir=args.label_dir,
                           label_name=args.label_name, 
                           format=args.file_format,
                           dtype=args.dtype)
    train_set, val_set, test_set = dataset.serial_split(ratios=args.split_ratio, mask=args.mask_len)
    train_set = StockSequenceDataset(train_set, seq_len=args.train_seq_len, mode=args.mode)
    val_set = StockSequenceDataset(val_set, seq_len=args.val_seq_len or args.train_seq_len, mode=args.mode)
    test_set = StockSequenceDataset(test_set, seq_len=args.test_seq_len or args.train_seq_len, mode=args.mode)
    logging.debug(f"train_set length: {len(train_set)}, val_set length: {len(val_set)}, test_set length: {len(test_set)}")

    torch.save({"train": train_set, "val": val_set, "test": test_set}, args.save_path)
    logging.debug(f"Dataset saved to {args.save_path}")

# python dataset.py --x_folder "D:\PycharmProjects\SWHY\data\preprocess\alpha" --y_folder "D:\PycharmProjects\SWHY\data\preprocess\label" --label_name "ret10" --train_seq_len 20 --save_path "D:\PycharmProjects\SWHY\data\preprocess\dataset.pt"

# python dataset.py --x_folder "D:\PycharmProjects\SWHY\data\preprocess\alpha_cs_zscore" --y_folder "D:\PycharmProjects\SWHY\data\preprocess\label" --label_name "ret10" --train_seq_len 20 --save_path "D:\PycharmProjects\SWHY\data\preprocess\dataset_cs_zscore.pt"

# python dataset.py --x_folder "D:\PycharmProjects\SWHY\data\preprocess\alpha" --y_folder "D:\PycharmProjects\SWHY\data\preprocess\label" --label_name "ret10" --train_seq_len 20 --save_path "D:\PycharmProjects\SWHY\data\preprocess\dataset.pt" --mask_len 10

# python dataset.py --quantity_price_feature_dir "D:\PycharmProjects\SWHY\AttnFactorVAE\preprocess_new\quantity_price_feature" --fundamental_feature_dir "D:\PycharmProjects\SWHY\AttnFactorVAE\preprocess_new\fundamental_feature" --label_dir "D:\PycharmProjects\SWHY\AttnFactorVAE\preprocess_new\label" --label_name "ret10" --train_seq_len 20 --save_path "dataset.pt" --mask_len 10 --mode "drop"





