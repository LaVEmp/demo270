# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""
Trans Torch file to Cambricon file.

This .py file should run on MLU270 device.

driver version: 4.8.0
cntoolkit version: 1.6.0
"""

import argparse
import os
import sys

import torch
import torch_mlu
import torch_mlu.core.mlu_model as mm
import torch_mlu.core.mlu_quantize as mq

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from model.nn_model import DNNModel, CNNModel, LSTMModel
from utils.common_utils import parse_config

os.environ['TORCH_MIN_CNLOG_LEVEL'] = '-1'
os.environ['DISABLE_MLU_FUSION'] = '0'

run_quantization = True
use_mlu = False if run_quantization else True


def run_quantize(model: DNNModel,
                 batch_size: int,
                 quantize_precision: str = 'int8') -> str:
    """
    Run quantize on given model and save to model_int_save_path.

    :param model: A DNNModel-class object defined in model.nn_model.
    :param batch_size: BATCH_SIZE defined in configs.dataset_config.yaml file.
    :param quantize_precision: str, default='int8'

    :return: model_int_save_path
    """
    x_batch = torch.rand(batch_size, model.cfg['in_dim'], dtype=torch.float)
    _s = model.cfg['nn_model_path']
    model_int_save_path = _s[:_s.rfind('.')] + '_' + quantize_precision + _s[_s.rfind('.'):]

    model.load_model()
    model.eval()

    config = {
        'iteration': 1,
        'use_avg': False,
        'data_scale': 1.0,
        'mean': [0, 0, 0],
        'std': [1, 1, 1],
        'firstconv': False,
        'per_channel': False
    }
    model = mq.quantize_dynamic_mlu(model, config, dtype=quantize_precision, gen_quant=True)
    model(x_batch)
    torch.save(model.state_dict(), model_int_save_path)
    return model_int_save_path


def trans_model_to_cambricon(model: DNNModel, model_int_save_path: str, batch_size: int = 16):
    """     Trans pytorch model to cambricon.

    :param model: DNNModel-class object.
    :param model_int_save_path: str.
    :param batch_size: int, default=16.
    """
    x_batch = torch.rand(batch_size, model.cfg['in_dim'], dtype=torch.float)
    example_batch = x_batch.to(mm.mlu_device())

    _s = model.cfg['nn_model_path']
    _model_save_cambricon_name = _s[:_s.rfind('.')] + '_offline'

    torch.set_grad_enabled(False)  # for mlu270 jit.trace fuse
    torch_mlu.core.mlu_model.set_core_number(4)
    torch_mlu.core.mlu_model.set_core_version('MLU220')

    model = mq.quantize_dynamic_mlu(model)
    model.load_state_dict(torch.load(model_int_save_path))
    model.to(mm.mlu_device())
    model.eval()

    # print('model(example_batch).device: ', model(example_batch).device)
    # print('model(example_batch).cpu().shape: ', model(example_batch).cpu().shape)

    # ==========================START===============================
    torch_mlu.core.mlu_model.save_as_cambricon(_model_save_cambricon_name)  # ================
    traced_model = torch.jit.trace(model, example_batch, check_trace=False)
    traced_model(example_batch)
    print('Done!')
    # print(traced_model(x_batch).to('cpu'))
    torch_mlu.core.mlu_model.save_as_cambricon("")  # ==================
    # ===========================END================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dnn')
    parser.add_argument('--nn_config', type=str, default='./configs/nn_config.yaml')
    parser.add_argument('--dataset_cfg', type=str, default='./configs/dataset_config.yaml')
    parser.add_argument('--quantize', '-q', '--qp', type=str, default='int8')
    args = parser.parse_args()

    nn_config = parse_config(args.nn_config)
    dataset_config = parse_config(args.dataset_cfg)
    quantize_precision = args.quantize

    model_cfg = nn_config['MODEL']
    model_cfg['device'] = 'cpu'  # cause of MLU270, there is usually no GPU on device.
    if args.model == 'dnn':
        model = DNNModel(model_cfg)
    elif args.model == 'lstm':
        model = LSTMModel(model_cfg)
    else:
        model = CNNModel(model_cfg)

    model_int_save_path = run_quantize(model=model,
                                       batch_size=dataset_config['batch_size_mlu'],
                                       quantize_precision=quantize_precision
                                       )
    trans_model_to_cambricon(model, model_int_save_path, batch_size=dataset_config['batch_size_mlu'])
