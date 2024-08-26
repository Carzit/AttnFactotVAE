"""
数据构建
对原Alpha和Label的pkl文件进行缺失值处理、数据对齐和重新组织
"""

import os
import sys
import logging
import argparse
from numbers import Number
from typing import List, Literal, Optional
from dataclasses import dataclass
from functools import reduce

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 定义一个FileData类，包含path和dataframe两个属性。作为基类，描述每个数据文件。
@dataclass
class FileData:
    path: str
    dataframe: pd.DataFrame

@dataclass
class FactorData(FileData):
    codes: List[str]
    dates: List[str]
    factor: str

@dataclass
class LabelData(FileData):
    codes: List[str]
    dates: List[str]
    label: str

def save_dataframe(df:pd.DataFrame, path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
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

def load_dataframe(path:str, format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> pd.DataFrame:
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

class Processor:
    def __init__(self) -> None:
        pass
    
    def process(self):
        raise NotImplementedError()

class FactorProcessor(Processor):
    """
    因子处理管线
    """
    def __init__(self, 
                 folder_path:str):
        self.folder_path:str = folder_path #待处理数据所在文件夹的路径

        self.common_dates:List = None #共有日期（用于后续做数据对齐）
        self.common_codes:List = None #共有股票代码（用于后续做数据对齐）
        self.factor_names:List = [] #因子名

        self.factor_data_list:List[FactorData] = [] #所有Alpha数据

    def read_and_process_file(self, file_path, file_format) -> FactorData:
        df = load_dataframe(path=file_path, format=file_format)
        return FileData(path=file_path, 
                        dataframe=df)
    
    def load_data(self, file_format:str) -> None:
        # 读取指定文件夹中的所有pkl文件，处理数据并存储在 alpha_data_list 中，同时计算所有文件中共同的日期和股票代码。
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith(file_format)]
        
        for file_name in tqdm(file_list):
            file_path = os.path.join(self.folder_path, file_name)
            factor_data = self.read_and_process_file(file_path, file_format)
            
            if self.common_dates is None:
                self.common_dates = pd.Index(factor_data.dates)
            else:
                self.common_dates = self.common_dates.intersection(factor_data.dates)

            if self.common_codes is None:
                self.common_codes = pd.Index(factor_data.codes)
            else:
                self.common_codes = self.common_codes.intersection(factor_data.codes)

            self.factor_data_list.append(factor_data)

        self.common_dates = list(self.common_dates)
        self.common_codes = list(self.common_codes)
    
    def common_filter(self) -> None:
        # 过滤数据，只保留所有文件中共同的日期和股票代码。
        for factor_data in tqdm(self.factor_data_list):
            factor_data.dataframe = factor_data.dataframe.loc[self.common_dates, self.common_codes]
            factor_data.codes = self.common_codes
            factor_data.dates = self.common_dates
    
    def merge_date(self, date:str) -> pd.DataFrame:
        # 根据不同的日期合并数据，并返回一个以股票代码为行名，以因子为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for factor_data in self.factor_data_list:
            if date in factor_data.dates:
                series = factor_data.dataframe.loc[date]
                series.name = factor_data.factor
                merged_df = merged_df.join(series, how='left')

        #merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True, drop=False)
        return merged_df

    def process(self, 
                save_folder:Optional[str]=os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
        """x
        logging.debug("recording extra info")
        with open(os.path.join(save_folder, "alpha_extra_info.json"), "w") as f:
            json.dump({"dates": self.common_dates,
                       "stock_codes": self.common_codes}, f)
        """
        logging.debug("merging data...")
        for date in tqdm(self.common_dates):
            merged_df = self.merge_date(date)
            save_dataframe(df=merged_df,
                           path=os.path.join(save_folder, f"{date}.{save_format}"),
                           format=save_format)

class QuantityPriceFeature_FactorProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path, file_format) -> FactorData:
        df = load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor_name = os.path.basename(file_path).replace("OHO_", "").replace(".pkl", "")
        self.factor_names.append(factor_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=factor_name)
    
class FundamentalFeature_FactorProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path, file_format) -> FactorData:
        df = load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor_name = os.path.basename(file_path).replace(".pkl", "")
        self.factor_names.append(factor_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=factor_name)
    
class LabelProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path, file_format) -> FactorData:
        df = load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        label_name = os.path.basename(file_path).replace("label_", "").replace(".pkl", "")
        self.factor_names.append(label_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=label_name)
        
class ProcessorAlignment:
    def __init__(self, 
                 processors:List[Processor],  
                 align_attr_name:str) -> None:
        self.processors = processors
        self.align_attr_name = align_attr_name
        
    def align(self, lst1:List, lst2:List) -> List:
        set1 = set(lst1)
        set2 = set(lst2)
        common_set = set1.intersection(set2)
        return list(common_set)
    
    def process(self):
        attrs = [getattr(processor, self.align_attr_name) for processor in self.processors]
        align_attr = reduce(self.align, attrs)
        for processor in self.processors:
            setattr(processor, self.align_attr_name, align_attr)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--quantity_price_factor_folder", type=str, required=True, help="Path of folder for Quantity-Price Factor pickle files")
    parser.add_argument("--fundamental_factor_folder", type=str, required=True, help="Path of folder for Fundamental Factor pickle files")
    parser.add_argument("--label_folder", type=str, required=True, help="Path of folder for label pickle files")
    parser.add_argument("--save_folder", type=str, required=True, help="Path of folder for Processor to save processed result in subdir `alpha` and `label`")
    parser.add_argument("--read_format", type=str, default="pkl", help="File format to read, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")
    parser.add_argument("--save_format", type=str, default="pkl", help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")

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
    if not os.path.exists(args.save_folder):    
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, "quantity_price_feature"))
        os.makedirs(os.path.join(args.save_folder, "fundamental_feature"))
        os.makedirs(os.path.join(args.save_folder, "label"))

    quantity_price_processor = QuantityPriceFeature_FactorProcessor(args.quantity_price_factor_folder)
    fundamental_processor = FundamentalFeature_FactorProcessor(args.fundamental_factor_folder)
    label_processor = LabelProcessor(args.label_folder)

    logging.debug("Loading Quantity-Price Factor Data...")
    quantity_price_processor.load_data(file_format=args.read_format)
    
    logging.debug("Loading Fundamental Factor data...")
    fundamental_processor.load_data(file_format=args.read_format)

    logging.debug("Loading Label Data...")
    label_processor.load_data(file_format=args.read_format)

    logging.debug("Doing Data Alignment...")
    date_aligner = ProcessorAlignment([quantity_price_processor, fundamental_processor, label_processor], "common_dates")
    date_aligner.process()
    code_aligner = ProcessorAlignment([quantity_price_processor, fundamental_processor, label_processor], "common_codes")
    code_aligner.process()

    quantity_price_processor.common_filter()
    fundamental_processor.common_filter()
    label_processor.common_filter()

    logging.debug("Doing Quantity-Price Factor Data Split...")
    quantity_price_processor.process(save_folder=os.path.join(args.save_folder, "quantity_price_feature"), 
                                     save_format=args.save_format)
    logging.debug(f"Quantity-Price Date Data Saved to `{os.path.join(args.save_folder, 'quantity_price_feature')}`")
    
    logging.debug("Doing Fundamental Factor Data Split...")
    fundamental_processor.process(save_folder=os.path.join(args.save_folder, "fundamental_feature"), 
                                     save_format=args.save_format)
    logging.debug(f"Fundamental Date Data Saved to `{os.path.join(args.save_folder, 'fundamental_feature')}`")
    
    logging.debug("Doing Label Data Split...")
    label_processor.process(save_folder=os.path.join(args.save_folder, "label"), 
                            save_format=args.save_format)
    logging.debug(f"Label Date Data Saved to `{os.path.join(args.save_folder, 'label')}`")
    
    logging.debug("Data Construct Accomplished")

# python data_construct.py --alpha_folder "data\Alpha101" --label_folder "data\label" --save_folder "data\preprocess" --merge_mode "date" --log_folder "log"

# python data_construct.py --quantity_price_factor_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\demo\alpha" --fundamental_factor_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\demo\basic" --label_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\demo\label" --save_folder "preprocess" --log_folder "log"