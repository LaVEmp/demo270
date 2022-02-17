# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

import numpy as np


def _calcu_distance(x1, y1, x2, y2):
    """
    x0, y0 到 x1, y1 点的距离
    """
    x = x2 - x1
    y = y2 - y1
    return np.sqrt(x * x + y * y)


def _get_cross_point(x1, y1, x2, y2, x0, y0):
    """
    计算点到直线的最短距离
    return: x0, y0 到 x1, y1, x2, y2 直线的距离
    """
    numerator = (y2 - y1) * x0 + (x1 - x2) * y0 + ((x2 * y1) - (x1 * y2))
    assignment = np.sqrt(pow(y2 - y1, 2) + pow(x1 - x2, 2))
    if assignment == 0:
        assignment = 0.00001

    b = numerator / assignment
    return b


def err_amplification(x, y, r, k):
    """
    圆的轮廓误差放大函数
    return: newx, newy
    """
    assert len(x) == len(y)
    err1 = np.sqrt(x * x + y * y) - r
    newx = x + x * err1 * k / (r + err1)
    newy = y + y * err1 * k / (r + err1)
    return newx, newy


def estimate_contour_error(CmdS_X, CmdS_Y, ActS_X, ActS_Y):
    """
    轮廓误差计算函数
    in:   (n,1)*4, ndarray
    out:  np.array(error) : nd(n,),
          target_point : (n,nd(4,))
    """
    cmd_point = np.c_[CmdS_X, CmdS_Y]  # (n,2)
    act_point = np.c_[ActS_X, ActS_Y]  # (n,2)

    error = []
    target_point = []
    for p in range(len(CmdS_X)):  # 为什么不用enumerate n,
        i = p  # i表示当前求解点的位置

        direction = 1
        while direction > 0:  # 先沿负方向寻址
            if i < 1:  # 起始位置的情况排除
                i = 1
                break
            # x0,y0, x1,y1
            d1 = _calcu_distance(cmd_point[i, 0], cmd_point[i, 1], act_point[p, 0], act_point[p, 1])
            d2 = _calcu_distance(cmd_point[i - 1, 0], cmd_point[i - 1, 1], act_point[p, 0], act_point[p, 1])
            direction = d1 - d2
            i = i - 1
        while direction <= 0:  # 沿相反方向寻找
            i = i + 1
            if i > len(CmdS_X) - 2:
                i = len(CmdS_X) - 2
                break
            d1 = _calcu_distance(cmd_point[i, 0], cmd_point[i, 1], act_point[p, 0], act_point[p, 1])
            d2 = _calcu_distance(cmd_point[i + 1, 0], cmd_point[i + 1, 1], act_point[p, 0], act_point[p, 1])
            direction = d2 - d1

        target_point1 = cmd_point[i - 2]  # 获得离目标点距离最近的三个点      i=1时，i-2=-1，但此点后续可排除
        target_point2 = cmd_point[i - 1]
        target_point3 = cmd_point[i]

        # 判断实际轮廓在指令轨迹内侧还是外侧的标志
        error1 = _get_cross_point(target_point1[0], target_point1[1], target_point2[0], target_point2[1],
                                  act_point[p, 0], act_point[p, 1])
        error2 = _get_cross_point(target_point2[0], target_point2[1], target_point3[0], target_point3[1],
                                  act_point[p, 0], act_point[p, 1])
        if (error1 < 0) & (error2 < 0):  # 此处a的作用为解决轮廓内外的计算精度
            a = 1
        else:
            a = -1
        # error_tmp = min(abs(error1), abs(error2))*a
        # err_tmp = 0
        if abs(error1) < abs(error2):
            err_tmp = abs(error1) * a
            target_point_tmp = np.r_[target_point1, target_point2]
        else:
            err_tmp = abs(error2) * a
            target_point_tmp = np.r_[target_point2, target_point3]
        error.append(err_tmp)
        target_point.append(target_point_tmp)
    # err[]
    # print(cmd_point[12747], cmd_point[12747], cmd_point[12747])
    # print(act_point[12747])
    return np.array(error), target_point  # 返回距离目标点最近的两个点
