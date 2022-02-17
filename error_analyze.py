# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from utils.dataset_process import get_append_point
from utils.error_analyze import estimate_contour_error
from utils.error_analyze import err_amplification


def process_result_pdf(test_dataset_path, result_save_path):
    test_dataset = pd.read_csv(test_dataset_path)
    result_pdf = pd.read_csv(result_save_path)
    assert len(test_dataset) == len(result_pdf)
    result_pdf = pd.concat([test_dataset.loc[:, ['cmd_s', 'act_s', 'sim_s', 'error_s', 'line']],
                            result_pdf.loc[:, ['labels', 'preds']]],
                           axis=1)
    assert max(result_pdf['error_s'] - result_pdf['labels']) < 1e-5

    result_pdf['preds_s'] = result_pdf['preds'] + result_pdf['sim_s']

    result_pdf.to_csv(result_save_path, index=False)
    return result_pdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dnn')
    args = parser.parse_args()

    test_dataset_path = './dataset_offline/test_x.csv'
    result_save_path = './result/result_x_' + args.model + '.csv'
    df_x = process_result_pdf(test_dataset_path, result_save_path)
    test_dataset_path = './dataset_offline/test_y.csv'
    result_save_path = './result/result_y_' + args.model + '.csv'
    df_y = process_result_pdf(test_dataset_path, result_save_path)

    picture_save_path = './result/picture_%s/' % args.model
    if not os.path.exists(picture_save_path):
        os.makedirs(picture_save_path)

    append_point = get_append_point(df_x)

    cmd_x = df_x['cmd_s'].values.reshape(-1, )
    act_x = df_x['act_s'].values.reshape(-1, )
    sim_x = df_x['sim_s'].values.reshape(-1, )
    pred_x = df_x['preds_s'].values.reshape(-1, )

    cmd_y = df_y['cmd_s'].values.reshape(-1, )
    act_y = df_y['act_s'].values.reshape(-1, )
    sim_y = df_y['sim_s'].values.reshape(-1, )
    pred_y = df_y['preds_s'].values.reshape(-1, )

    cnt, length, flag_r = 0, 0, 0
    error_array = np.array([])
    max_error = {}
    f = [1000, 2000, 3000, 4000, 5000]
    for i in range(1, len(append_point)):
        delta = 0
        cmd_x_ = cmd_x[append_point[i - 1] + delta: append_point[i]]
        cmd_y_ = cmd_y[append_point[i - 1] + delta: append_point[i]]
        pred_x_ = pred_x[append_point[i - 1] + delta: append_point[i]]
        pred_y_ = pred_y[append_point[i - 1] + delta: append_point[i]]
        sim_x_ = sim_x[append_point[i - 1] + delta: append_point[i]]
        sim_y_ = sim_y[append_point[i - 1] + delta: append_point[i]]
        act_x_ = act_x[append_point[i - 1] + delta: append_point[i]]
        act_y_ = act_y[append_point[i - 1] + delta: append_point[i]]

        d_x = (max(cmd_x_) + min(cmd_x_)) / 2
        d_y = (max(cmd_y_) + min(cmd_y_)) / 2
        cmd_x_ = cmd_x_ - d_x
        cmd_y_ = cmd_y_ - d_y
        pred_x_ = pred_x_ - d_x
        pred_y_ = pred_y_ - d_y
        sim_x_ = sim_x_ - d_x
        sim_y_ = sim_y_ - d_y
        act_x_ = act_x_ - d_x
        act_y_ = act_y_ - d_y

        r = int(max(cmd_x_) - min(cmd_x_) + 1) / 2
        print('r :', r)
        if r < 40:
            flag_r += 1
            continue

        cut = 500

        error, _ = estimate_contour_error(act_x_[cut:], act_y_[cut:], pred_x_[cut:], pred_y_[cut:])
        error_array = np.concatenate((error_array, error), axis=0)

        act_x_, act_y_ = err_amplification(act_x_, act_y_, r, 1000)
        pred_x_, pred_y_ = err_amplification(pred_x_, pred_y_, r, 1000)
        sim_x_, sim_y_ = err_amplification(sim_x_, sim_y_, r, 1000)

        cnt += 1
        if cnt == 10:
            length = error_array.shape[0]
            print('length: ', length)
        if cnt % 2 == 1:
            picture_name = picture_save_path + 'R' + str(r) + 'F' + str(f[int((cnt - 1) % 10 / 2)]) + '.svg'
            max_error['R' + str(r) + 'F' + str(f[int((cnt - 1) % 10 / 2)])] = np.max(np.abs(error)) * 1000
        else:
            picture_name = picture_save_path + 'R' + str(r) + 'F' + str(f[int((cnt - 1) % 10 / 2)]) + '_' + '.svg'
            max_error['R' + str(r) + 'F' + str(f[int((cnt - 1) % 10 / 2)]) + '_'] = np.max(np.abs(error)) * 1000

        plt.figure(1, figsize=[6, 6])
        plt.plot(cmd_x_, cmd_y_, label='Cmd')
        plt.plot(act_x_, act_y_, label='Act')
        plt.plot(pred_x_, pred_y_, label='Pred')
        plt.plot(sim_x_, sim_y_, label='Sim')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.grid()
        plt.legend()
        plt.savefig(picture_name)
        plt.close()
    print('flag_r: ', flag_r)

    arr1 = error_array[length:]
    arr2 = error_array[: length]
    error_array = np.concatenate((arr1, arr2), axis=0)
    for i, j in max_error.items():
        print(i, j)
    # plt.figure(2)
    # plt.plot(error_array * 1000)
    # plt.xlabel('Time(ms)')
    # plt.ylabel('Error(um)')
    # plt.show()
