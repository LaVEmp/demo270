# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_append_point(dataset) -> list:
    """
    获取行line转折点。
    """
    assert 'line' in dataset.columns
    line = dataset['line'].values

    append_point = [0]
    for j in range(1, line.shape[0]):
        if line[j] != line[j - 1]:
            append_point.append(j)
    print('append_point_shape: ', np.array(append_point).shape)
    return append_point


def dataset_clean_by_line(meta_dataset_path: str,
                          dataset_cleaned_path: str):
    """
    通过line列清洗数据集，只保留加工圆的路径。
    """
    dataset = pd.read_csv(meta_dataset_path)
    if dataset.columns[0].startswith('Unnamed'):
        dataset = dataset.loc[:, dataset.columns[1:]]

    append_point = get_append_point(dataset)

    for i in range(len(append_point) - 1, 0, -1):
        if append_point[i] - append_point[i - 1] < 1000:
            dataset.drop(index=range(append_point[i - 1], append_point[i]), inplace=True)
    dataset.drop((dataset[dataset['line'] == 0.0]).index, inplace=True)
    dataset.drop((dataset[dataset['line'] == 17.0]).index, inplace=True)
    dataset.drop((dataset[dataset['line'] == 4.0]).index, inplace=True)
    dataset.drop((dataset[dataset['line'] == 5.0]).index, inplace=True)
    dataset.drop((dataset[dataset['line'] == 2.0]).index, inplace=True)
    dataset.to_csv(dataset_cleaned_path, index=False)


def train_test_dataset_split_by_line(dataset_cleaned_path: str,
                                     train_dataset_path: str,
                                     test_dataset_path: str,
                                     test_size: float = 0.2):
    """
    通过line列划分测试集和训练集，使测试集包含完整加工圆的路径，以便将测试结果可视化。
    """
    dataset_cleaned = pd.read_csv(dataset_cleaned_path)
    line = dataset_cleaned['line'].values
    line_dict = [(line[0], 0)]
    for i in range(len(line) - 1):
        if line[i] != line[i + 1]:
            line_dict.append((line[i + 1], i))

    circle_num = len(line_dict)
    print('Circle num: ', circle_num)
    test_circle_num = round(circle_num * test_size)
    print('Test Circle num: ', test_circle_num)

    x_train = dataset_cleaned.loc[:line_dict[circle_num - test_circle_num][1], :]
    x_test = dataset_cleaned.loc[line_dict[circle_num - test_circle_num][1]:, :]
    x_train.to_csv(train_dataset_path, index=False)
    x_test.to_csv(test_dataset_path, index=False)


def balance_by_delta_and_shuffle(df_train: pd.DataFrame,
                                 label_column_name: str = 'delta',
                                 thres: float = 0.005,
                                 seed: int = 2021):
    """
    训练集权重修正，加大误差大的部分所占权重。
    """
    length = df_train.shape[0]
    df_train = df_train.iloc[np.where(np.abs(df_train[label_column_name]) < 0.015)]
    j = df_train[label_column_name]

    df_mark = df_train.iloc[np.where(np.abs(j) > thres)]
    df_not_mark = df_train.iloc[np.where(np.abs(j) <= thres)]
    df_mark_sample = df_mark.sample(round(length / 3.0), replace=True)
    df_not_mark_sample = df_not_mark.sample(round(length / 3.0 * 2.0), replace=True)
    df = pd.concat([df_mark_sample, df_not_mark_sample])
    df = shuffle(df, random_state=seed)
    return df
