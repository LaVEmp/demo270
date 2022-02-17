# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""Make dataset dataloader."""

__all__ = ['DatasetBase',
           'LSTMDatasetBase',
           'get_dataloader',
           'get_lstm_dataloader',
           'get_dataset_class']

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')
os.chdir(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')

from utils.common_utils import cal_input_mean_std
from utils.common_utils import cal_label_mean_std
from utils.common_utils import cal_sample_weight
from utils.common_utils import parse_config
from utils.common_utils import process_input
from utils.common_utils import process_label
from utils.common_utils import save_config


class DatasetBase(Dataset):
    """ Dataset Base Class. """

    def __init__(self, cfg: dict, mode: str):
        super(DatasetBase, self).__init__()
        self.cfg = cfg

        self.write_to_yaml_file = self.cfg['write_to_yaml_file'] if mode == 'train' else False
        self.yaml_file_path = self.cfg['yaml_file_path']

        random.seed(self.cfg['fixed_seed'])
        start = time.time()

        print('DatasetBase class init started.')
        self.f_pdf = pd.read_csv(self.cfg['train_dataset_path']
                                 if mode == 'train' else self.cfg['test_dataset_path'])
        if self.cfg['balance_by_delta_and_shuffle']:
            from utils.dataset_process import balance_by_delta_and_shuffle
            self.f_pdf = balance_by_delta_and_shuffle(df_train=self.f_pdf,
                                                      label_column_name=self.cfg['label_column_name'],
                                                      thres=self.cfg['balance_by_delta_thres'],
                                                      seed=self.cfg['fixed_seed'])

        if self.cfg['input_data_columns']:
            assert self.cfg['label_column'] is not None
            self.input_data_array = self.f_pdf.loc[:, self.f_pdf.columns[self.cfg['input_data_columns']]].values
            self.input_columns_name = list(self.f_pdf.columns[self.cfg['input_data_columns']].values)  # list of str.
            self.labels_array = self.f_pdf.loc[:, self.f_pdf.columns[self.cfg['label_column']]].values
        else:
            self.input_data_array = self.f_pdf.loc[:, self.f_pdf.columns[:-1]].values
            self.input_columns_name = list(self.f_pdf.columns[:-1].values)  # list of str.
            self.labels_array = self.f_pdf.loc[:, self.f_pdf.columns[-1]].values

        self._cal_input_mean_std()
        self._cal_label_mean_std()

        self.input_data = process_input(self.input_data_array, enable_normalize=True)
        self.input_data = torch.from_numpy(self.input_data)
        self.labels = process_label(self.labels_array)
        self.labels = torch.from_numpy(self.labels)

        self.sample_weight = cal_sample_weight(labels=self.labels.numpy())
        self.sample_weight = torch.from_numpy(self.sample_weight)

        print('get_dataset init done.')
        print('self.input_data.shape: ', list(self.input_data.shape))
        print('====== use %.2f s ======' % (time.time() - start))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x_data = (index, self.input_data[index], self.sample_weight[index])
        y_data = self.labels[index]
        return x_data, y_data

    def get_data(self) -> np.ndarray:
        return self.input_data.numpy()

    def get_weight(self) -> np.ndarray:
        return self.sample_weight.numpy()

    def get_label(self) -> np.ndarray:
        return self.labels.numpy()

    def get_meta_pdf(self) -> pd.DataFrame:
        return self.f_pdf

    def _cal_input_mean_std(self):
        input_data_mean, input_data_std = cal_input_mean_std(self.input_data_array)
        if self.write_to_yaml_file:
            assert self.yaml_file_path is not None
            tar_configs = parse_config(self.yaml_file_path)
            tar_configs['DEFAULT_VALUE']['input_data_mean'] = list(map(float, input_data_mean))
            tar_configs['DEFAULT_VALUE']['input_data_std'] = list(map(float, input_data_std))
            save_config(tar_configs, cfg_path=self.yaml_file_path)

    def _cal_label_mean_std(self):
        _mean, _std = cal_label_mean_std(self.labels_array)
        if self.write_to_yaml_file:
            assert self.yaml_file_path is not None
            tar_configs = parse_config(self.yaml_file_path)
            tar_configs['DEFAULT_VALUE']['label_mean'] = _mean
            tar_configs['DEFAULT_VALUE']['label_std'] = _std
            save_config(tar_configs, cfg_path=self.yaml_file_path)


class LSTMDatasetBase(DatasetBase):
    """ Dataset Base Class. """

    def __init__(self, cfg: dict, mode: str):
        super(LSTMDatasetBase, self).__init__(cfg=cfg, mode=mode)
        self.seq_len = self.cfg['seq_len']

        assert len(self.input_data.shape) == 2
        self.input_data_ = torch.cat([self.input_data,
                                      self.input_data[-1].repeat(self.seq_len - 1, 1)], dim=0)
        self.labels_ = torch.cat([self.labels,
                                  self.labels[-1].repeat(self.seq_len - 1, 1).squeeze()], dim=0)
        self.sample_weight_ = torch.cat([self.sample_weight,
                                         self.sample_weight[-1].repeat(self.seq_len - 1, 1).squeeze()], dim=0)

    def __getitem__(self, index):
        x_data = (index,
                  self.input_data_[index: index + self.seq_len],
                  self.sample_weight_[index: index + self.seq_len])
        y_data = self.labels_[index: index + self.seq_len]
        return x_data, y_data


def get_dataloader(cfg: dict, mode: str = 'train'):
    """
    Get dataloader of type torch.utils.data.DataLoader.

    :param cfg: dict
        dataset_cfg.
    :param mode: str
        'train' or 'test'.

    :return: A torch.utils.data.DataLoader object contains __len__ and __getitem__ methods.
    """
    return DataLoader(
        DatasetBase(cfg=cfg, mode=mode),
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'] if mode == 'train' else cfg['shuffle_valid'],
        drop_last=True if mode == 'train' else False
    )


def get_lstm_dataloader(cfg: dict, mode: str = 'train'):
    """
    Get dataloader of type torch.utils.data.DataLoader.

    :param cfg: dict
        dataset_cfg.
    :param mode: str
        'train' or 'test'.

    :return: A torch.utils.data.DataLoader object contains __len__ and __getitem__ methods.
    """
    return DataLoader(
        LSTMDatasetBase(cfg=cfg, mode=mode),
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'] if mode == 'train' else cfg['shuffle_valid'],
        drop_last=True if mode == 'train' else False
    )


def get_dataset_class(cfg: dict, mode: str = 'train'):
    """
    Get lightgbm dataset by 'lgb_model_cfg'.

    :param cfg: dict, 'lgb_model_cfg' import from lgb_model.py.
    :param mode: select in ('train', 'test')

    :return: A DatasetBase object contains get_data() and get_labels() methods.
    """
    return DatasetBase(cfg=cfg, mode=mode)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_cfg', type=str, default='./configs/dataset_config.yaml')
    parser.add_argument('-a', '--axis', type=str, default='x')
    parser.add_argument('-i', '--index', type=int, default=0)
    args = parser.parse_args()

    dataset_config = parse_config(args.dataset_cfg)
    # dataset_config = parse_config('./configs/dataset_config.yaml')

    if args.axis in ('y', 'Y', '1'):
        dataset_config['meta_dataset_path'] = dataset_config['meta_dataset_path'].replace('_x', '_y')
        dataset_config['train_dataset_path'] = dataset_config['train_dataset_path'].replace('_x', '_y')
        dataset_config['test_dataset_path'] = dataset_config['test_dataset_path'].replace('_x', '_y')
        dataset_config['dataset_cleaned_path'] = dataset_config['dataset_cleaned_path'].replace('_x', '_y')
        dataset_config['result_save_path'] = dataset_config['result_save_path'].replace('_x', '_y')

    if dataset_config['dataset_process']:
        from utils.dataset_process import train_test_dataset_split_by_line
        from utils.dataset_process import dataset_clean_by_line

        dataset_clean_by_line(meta_dataset_path=dataset_config['meta_dataset_path'],
                              dataset_cleaned_path=dataset_config['dataset_cleaned_path'])
        train_test_dataset_split_by_line(dataset_cleaned_path=dataset_config['dataset_cleaned_path'],
                                         train_dataset_path=dataset_config['train_dataset_path'],
                                         test_dataset_path=dataset_config['test_dataset_path'],
                                         test_size=dataset_config['test_size'])

    train_loader = get_lstm_dataloader(dataset_config, 'train')

    for i, data in enumerate(train_loader):
        print(i)
        print('---------------')
        (index, input_data, sample_weight), labels = data
        print('index: ', index.shape)
        print('input_data: ', input_data.shape)
        print('sample_weight: ', sample_weight.shape)
        print('---------------')
        print('labels: ', labels.shape)
        break
