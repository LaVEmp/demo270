# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""common_utils.py"""

__all__ = ['list_to_dict',
           'parse_config',
           'save_config',
           'rand', 'rand_',
           'train_test_dataset_split',
           'process_input',
           'resume_input',
           'process_label',
           'resume_label',
           'cal_sample_weight',
           'cal_input_mean_std',
           'cal_label_mean_std',
           'evaluate_score',
           'AverageMeter',
           ]

import os
import sys
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import torch
from sklearn.model_selection import train_test_split
import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')
os.chdir(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')

FEATURE_CONFIG_PATH = './configs/feature_config.yaml'


def list_to_dict(list_str: str) -> str:
    """ for personal use. """
    # list_str = '[id, M, K, Tg, Cn, Cc, Tc, Ic, g, Tr, If, I_il, I_ip, I_id, Ip]'
    elements = list_str[1:-1].replace(' ', '').split(',')
    elements_dict = list(map(lambda x: "'" + x + "': " + x + ', ', elements))
    result = '{' + ''.join(elements_dict)[:-2] + '}'
    print(result)
    return result


def parse_config(cfg_path: str) -> dict:
    """ Parse .yaml config. """
    with open(cfg_path, 'r', encoding='utf-8') as fi:
        yaml_config = yaml.load(fi, Loader=yaml.Loader)
    return yaml_config


def save_config(configs: dict, cfg_path: str):
    """ Save .yaml config. """
    with open(cfg_path, 'w', encoding='utf-8') as fo:
        yaml.dump(configs, fo)


def rand():
    return np.random.rand()


def rand_():
    return (np.random.rand() - 0.5) * 2


def train_test_dataset_split(meta_dataset_path: str,
                             train_dataset_path: str,
                             test_dataset_path: str,
                             test_size: float = 0.2):
    f_pdf = pd.read_csv(meta_dataset_path)
    if f_pdf.columns[0].startswith('Unnamed'):
        f_pdf = f_pdf.loc[:, f_pdf.columns[1:]]
    x_train, x_test = train_test_split(f_pdf, test_size=test_size)
    x_train.to_csv(train_dataset_path, index=False)
    x_test.to_csv(test_dataset_path, index=False)


def process_input(x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]],
                  enable_normalize: bool = True) -> np.ndarray:
    """  Given x_input, expand its dim to 2, and return its normalized value if enable_normalize is True.

    :param x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]
                len(x_input.shape) should be 1 or 2.
    :param enable_normalize: default=True.

    :return: nd(batch_size, num_fea).
    """
    if isinstance(x_input, torch.Tensor):
        x_input = x_input.cpu().numpy()
    x_input = np.asarray(x_input).astype(float)
    if len(x_input.shape) == 1:
        x_input = np.expand_dims(x_input, axis=0)

    if enable_normalize:
        default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
        input_data_mean = default_value['input_data_mean']
        input_data_std = default_value['input_data_std']
        x_input = (x_input - input_data_mean) / input_data_std
    return x_input


def process_label(labels: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]) -> np.ndarray:
    """  Given labels, and return its normalized value times 100.

    :param labels: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]
                e.q. nd(b, )
    :return: nd(b, ). With same shape as labels.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).astype(float)

    default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
    label_mean = default_value['label_mean']
    label_std = default_value['label_std']
    labels = (labels - label_mean) / label_std * 100
    return labels


def resume_input(x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]) -> np.ndarray:
    """ Given x_input, expand its dim to 2, and return its un-normalized value.

    :param x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]
                len(x_input.shape) should be 1 or 2.
    :return: nd(batch_size, num_fea).
    """
    if isinstance(x_input, torch.Tensor):
        x_input = x_input.cpu().numpy()
    x_input = np.asarray(x_input).astype(float)
    if len(x_input.shape) == 1:
        x_input = np.expand_dims(x_input, axis=0)

    default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
    input_data_mean = default_value['input_data_mean']
    input_data_std = default_value['input_data_std']

    x_input = (x_input * input_data_std) + input_data_mean
    return x_input


def resume_label(labels: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]) -> np.ndarray:
    """  Given pred_batch, return its un-normalized value.

    :param labels: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]].
                e.q. T(b, 1) or nd(b,)
    :return: nd(batch_size,).
    """
    if isinstance(labels, torch.Tensor):
        if len(labels.shape) == 2:
            labels = labels.squeeze(dim=1)
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if len(labels.shape) == 2 and labels.shape[1] == 1:
        labels = np.squeeze(labels, axis=1)

    default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
    label_mean = default_value['label_mean']
    label_std = default_value['label_std']
    result = (labels * label_std / 100) + label_mean
    return result


def _cal_class_weight(labels: np.ndarray, num_bins: int = 10):
    """ Calculate the class weight. """
    label_hist = np.histogram(labels, bins=num_bins)
    label_counts = label_hist[0]
    label_edge = label_hist[1]

    class_weight = 1 / torch.tensor(list(label_counts))
    class_weight = (class_weight / sum(class_weight)).clip(0.03, 0.7) * num_bins
    return class_weight, label_edge


def _cal_label_hist_index(labels: np.ndarray, label_edge: np.ndarray):
    """ Calculate the label hist index. """
    label_hist_index = np.zeros_like(labels).astype(int)
    for i in range(len(label_edge) - 1):
        label_hist_index[np.where((labels >= label_edge[i]) & (labels <= label_edge[i + 1]))] = i
    return label_hist_index


def cal_sample_weight(labels: np.ndarray) -> np.ndarray:
    """ Calculate the sample weight.    """
    class_weight, label_edge = _cal_class_weight(labels, num_bins=10)

    label_hist_index = _cal_label_hist_index(labels, label_edge=label_edge)
    sample_unbalanced_weight = np.array([class_weight[i] for i in label_hist_index])

    sample_weight = sample_unbalanced_weight
    return sample_weight


def cal_input_mean_std(x_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate mean and std for x_input. """
    input_data_mean = x_input.mean(axis=0)
    input_data_std = x_input.std(axis=0)
    return input_data_mean, input_data_std


def cal_label_mean_std(labels: np.ndarray) -> Tuple[float, float]:
    """   Calculate mean and std for labels.    """
    labels_mean = labels.mean()
    labels_std = labels.std()
    return float(labels_mean), float(labels_std)


def _mean_relative_error(label, y_pred):
    """ An additional error detector.

    :param label: np.ndarray_like
        Should support element-wise DIV.
    :param y_pred: np.ndarray_like
    :return: np.ndarray
    """
    return np.abs((y_pred - label) / label).mean()


def evaluate_score(label, y_pred):
    """ Evaluate the score between label and y_pred.

    Return a dict that contains:
        - 'mean_absolute_error'
        - 'mean_relative_error'
        - 'mean_squared_error'
        - 'r2_score'

    :param label: ndarray-like.
    :param y_pred: Same shape as label. ndarray-like.
    :return: Dict[str, float]
    """
    return {'mean_absolute_error': mean_absolute_error(label, y_pred),
            'mean_relative_error': _mean_relative_error(label, y_pred),
            'mean_squared_error': mean_squared_error(label, y_pred),
            'r2_score': r2_score(label, y_pred)}


class AverageMeter:
    """
    Computes and stores the average and current value.

    :ivar val: value.
    :ivar avg: average of value.
    :ivar sum: sum of value.
    :ivar count: count of value.
    """

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        """reset vals."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, num=1):
        """update vals."""
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
