# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""
The predict py file.

This .py file should run on MLU220 device.

driver version: 4.1.2
cntoolkit version: 1.3.0
pycnrt: python3.5
"""

from __future__ import division
import pandas as pd
import joblib
import math
import time
import numpy as np
import os
import sys
import pycnrttools
import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

FEATURE_CONFIG_PATH = './configs/feature_config.yaml'


def parse_config(cfg_path: str) -> dict:
    """ Parse .yaml config. """
    with open(cfg_path, 'r', encoding='utf-8') as fi:
        yaml_config = yaml.load(fi, Loader=yaml.Loader)
    return yaml_config


def _smooth9(list_):
    for i in range(4, len(list_) - 4):
        list_[i] = (list_[i - 4] + list_[i - 3] + list_[i - 2] + list_[i - 1] + list_[i] + list_[i + 1] + list_[i + 2] +
                    list_[
                        i + 3] + list_[i + 4]) / 9
    return list_


def predict(input_dataset_path, model_save_path, result_save_path, model_select='cnn'):
    """
    :param input_dataset_path: type(str). Input data path.
    :param model_save_path: type(str). select Cambricon model file by this.
    :param result_save_path: type(str). Output result path.
    :param model_select: select in ('cnn', 'dnn', 'lgb', 'lstm')
    :return: NoneType.
    """
    assert model_select in ('cnn', 'dnn', 'lgb', 'lstm')

    test_data_set = pd.read_csv(input_dataset_path)
    print(input_dataset_path)
    X_test = test_data_set.loc[:, test_data_set.columns[[0, 1, 3, 4, 5, 6, 8]]].values
    # X_test = process_input(X_test, enable_normalize=True)
    X_test = np.asarray(X_test).astype(float)
    if len(X_test.shape) == 1:
        X_test = np.expand_dims(X_test, axis=0)

    default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
    input_data_mean = default_value['input_data_mean']
    input_data_std = default_value['input_data_std']
    X_test = (X_test - input_data_mean) / input_data_std

    labels = test_data_set.loc[:, test_data_set.columns[7]].values

    # # ---------LGB-----------
    if model_select == 'lgb':
        model = joblib.load(model_save_path)
        pred_test = model.predict(X_test)
    # ---------LSTM_NN-----------
    else:  # model_select in ('cnn', 'dnn', 'lstm'):
        runner = pycnrttools.inferencer()
        task_num = 1
        runner.loadModel(model_save_path, task_num)
        batch_size = 16

        iters = int(math.ceil(X_test.shape[0] / batch_size))  # n / batch_size
        print('Predict start... (Iters = %d)' % iters)
        pred_test = np.empty(0)
        start = time.time()
        append_size = 0
        for i in range(iters):
            x_batch = X_test[i * batch_size: (i + 1) * batch_size]
            if i == iters - 1:
                assert x_batch.shape[0] <= batch_size
                append_size = batch_size - x_batch.shape[0]
                x_batch = np.concatenate((x_batch, np.zeros((append_size, x_batch.shape[1]))), axis=0)

            # -- new --
            x_batch = np.concatenate([x_batch for _ in range(task_num)], axis=0)
            x_batch = np.expand_dims(x_batch, axis=1)
            x_batch = np.expand_dims(x_batch, axis=1)
            # -- new end. --

            pred_batch = runner.infer(x_batch.flatten().tolist())
            # pred_batch = offline_runner.run(x_batch.flatten().tolist())

            # -- new --
            pred_batch = pred_batch[0]
            pred_batch = pred_batch.reshape(task_num, -1)  # (task_num, batch_size)
            pred_batch = pred_batch.mean(axis=0)  # (batch_size, )
            # -- new end. --

            pred_test = np.append(pred_test, np.array(pred_batch))
        if append_size != 0:
            pred_test = pred_test[:- append_size]
        end = time.time()
        print("use time : %.2f s." % (end - start))

    # pred_test = resume_label(pred_test)
    labels = np.asarray(labels)
    if len(labels.shape) == 2 and labels.shape[1] == 1:
        labels = np.squeeze(labels, axis=1)

    default_value = parse_config(FEATURE_CONFIG_PATH)['DEFAULT_VALUE']
    label_mean = default_value['label_mean']
    label_std = default_value['label_std']
    labels = (labels * label_std / 100) + label_mean

    pred_test = _smooth9(pred_test)

    result = pd.DataFrame({'labels': labels,
                           'preds': pred_test.astype(float)})
    result.to_csv(result_save_path, index=False)
    print("End!!!!!")


if __name__ == "__main__":
    axis = 'x'
    model_select = 'dnn'

    if model_select == 'lgb':
        model_path = './model_save/model_%s_lgb_20211227.txt' % 'X' if axis == 'x' else 'Y'
    else:
        model_path = './model_save/model_%s_%s_20211227_offline.cambricon' % (
            'X' if axis == 'x' else 'Y', model_select)

    input_path = './dataset_offline/test_%s.csv' % axis
    out_path = './result/result_%s_%s_cambricon.csv' % (axis, model_select)

    predict(input_path, model_path, out_path, model_select=model_select)
