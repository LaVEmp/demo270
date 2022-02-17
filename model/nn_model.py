# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""NN Model py file."""

__all__ = ['DNNModel', 'LSTMModel', 'CNNModel']

import argparse
import datetime
import os
import random
import sys
import time
from typing import Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
# from torch.jit import ScriptModule, script_method
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')
os.chdir(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')

from utils.common_utils import AverageMeter
from utils.common_utils import evaluate_score
from utils.common_utils import parse_config
from utils.common_utils import process_input
from utils.common_utils import resume_label
from utils.dataset_utils import get_dataloader
from utils.dataset_utils import get_lstm_dataloader


class DNNModel(nn.Module):
    """
    Main class for the nn model.

    :argument cfg: The nn model config dict. Type: dict
        Always contains keys like:
            'in_dim': Doc size per batch.
            'hidden_dim': Extracted feature num when perform extract_fea_from_numpy method.
            'num_fea': The num of features of the meta data.
            'n_classes': Num of classes.
            'device': Device of this model.
            'nn_model_path': Path of model to save to and load from.
    """

    def __init__(self, cfg: dict):
        super(DNNModel, self).__init__()
        self.cfg = cfg
        self.stop_train_flag = False
        self.subclass_flag = 'dnn'

        self.layer_table = []
        for i in range(len(self.cfg['hidden_dim'])):
            self.layer_table.append(
                nn.Linear(self.cfg['hidden_dim'][i - 1] if i > 0 else self.cfg['in_dim'],
                          self.cfg['hidden_dim'][i],
                          bias=False))
            # self.layer_table.append(nn.ReLU6(True))
            self.layer_table.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*self.layer_table)

        # self.process_fn = lambda out: out.view(out.shape[0], -1)

        self.classifier = nn.Sequential(
            nn.Linear(self.cfg['hidden_dim'][-1], self.cfg['n_classes'], bias=False),
        )

        self.init_device()

    def init_random_seed(self):
        """ init random seed. """
        if self.cfg['fixed_seed'] is not None:
            seed = self.cfg['fixed_seed']
            torch.manual_seed(seed)
            if 'cuda' in self.cfg['device']:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            print('fixed seed %d has been set... ' % seed, flush=True)

    def init_weight_bias(self):
        """ init weight and bias. """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(module.weight, mode='fan_in')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
        print('init_weight_bias done.')

    def init_device(self):
        if 'cuda' in self.cfg['device']:
            assert torch.cuda.is_available()
        self.to(self.cfg['device'])

    # @script_method
    def forward(self, x_input: Tensor) -> Tensor:
        """
        The network of this nn model.

        :param x_input: Tensor(batch_size, num_fea)
        :return: Tensor(batch_size, 1)
        """
        # input: T(batch_size, num_fea)
        out = self.encoder(x_input)  # _out: T(b, hidden_dim[-1])
        # out = self.process_fn(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)  # output: T(b, 1)
        return out  # output: T(b, 1)

    # @torch.jit.ignore
    def fit(self, data_loader: DataLoader, train_cfg: dict):
        """
        Train the nn model with configs defined in train_cfg dict.

        :param data_loader : DataLoader
            T((_, input_data, doc_cover), label)
        :param train_cfg : dict
            If train_cfg['class_weight_list'] is given,
                then: dataset_cfg[''with_class_weight'] should be False.
        """
        self.init_random_seed()
        self.init_weight_bias()
        print(self)

        if 'optimizer' not in train_cfg or train_cfg['optimizer'] is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=train_cfg['lr'])
            # optimizer = torch.optim.Adam(self.parameters(), lr=train_cfg['lr'])
        else:
            optimizer = train_cfg['optimizer']
        if 'criterion' not in train_cfg or train_cfg['criterion'] is None:
            criterion = torch.nn.SmoothL1Loss(reduction='none')
            # criterion = torch.nn.MSELoss(reduction='none')
        else:
            criterion = train_cfg['criterion']

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=train_cfg['milestones'],
                                                         gamma=train_cfg['gamma'])

        print('Training start.')
        loss_for_visdom = []
        start = time.time()
        self.stop_train_flag = False
        for epoch in range(train_cfg['epochs']):
            iters = len(data_loader)
            for batch_i, data in enumerate(data_loader):
                if self.stop_train_flag:
                    break
                if 'max_step_per_epoch' in train_cfg \
                        and train_cfg['max_step_per_epoch'] \
                        and batch_i > train_cfg['max_step_per_epoch']:
                    break

                # (list(b), T(b, 100, 107), T(b,)), T(b,)
                # # -- for test --
                # temp = iter(data_loader)
                # for i in range(100):
                # data = next(temp)
                # # -- for test end. --
                (_, input_data, sample_weight), label = data
                input_data = input_data.float().to(self.cfg['device'])  # T(b, num_fea)
                label = label.float().detach().to(self.cfg['device'])
                if label.dim() == 1:
                    label = torch.unsqueeze(label, dim=1)  # T(b, 1)
                sample_weight = sample_weight.float().detach().to(self.cfg['device'])
                if sample_weight.dim() == 1:
                    sample_weight = torch.unsqueeze(sample_weight, dim=1)  # T(b, 1)

                pred = self.forward(input_data)  # T(b, num_fea) -> T(b, 1)

                loss = criterion(pred, label)
                if train_cfg['use_sample_weight']:
                    loss = loss * sample_weight  # element wise
                if train_cfg['use_relative_loss']:
                    loss = loss / label * 100
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                scheduler.step()

                loss_for_visdom.append(loss.mean().detach().cpu().numpy())
                if (batch_i + 1) % 200 == 0:
                    process_cur = epoch * iters + batch_i
                    process_total = iters * train_cfg['epochs']
                    time_left = (time.time() - start) * (process_total - process_cur) / process_cur
                    print('%6.2f s. ' % (time.time() - start),
                          'Time Left:%6.1f s,' % time_left,
                          '%3d of %3d epochs.   ' % (epoch + 1, train_cfg['epochs']),
                          'lr=%r  ' % optimizer.param_groups[0]['lr'],
                          '%5d steps of %5d.' % ((batch_i + 1), iters),
                          'Loss: %8.4f' % loss.mean().item())
                    if train_cfg['print_to_file']:
                        f = open('./train_log_one_line.txt', 'w')
                        print('%6.2f s. ' % (time.time() - start),
                              'Time Left:%6.1f s,' % time_left,
                              '%3d of %3d epochs.   ' % (epoch + 1, train_cfg['epochs']),
                              'lr=%r  ' % optimizer.param_groups[0]['lr'],
                              '%5d steps of %5d.' % ((batch_i + 1), iters),
                              'Loss: %8.4f' % loss.mean().item(), file=f)
        print('Training done.')
        print('===============================')
        return loss_for_visdom

    def stop_train(self):
        """ Stop training when stop_train_flag is set to True by another thread. """
        self.stop_train_flag = True

    # @torch.jit.ignore
    def predict_once(self,
                     x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]],
                     enable_normalize: bool = True,
                     device=None) -> np.ndarray:
        """
        Single sample predict.

        :param x_input: Union[np.ndarray, torch.Tensor, List[Union[int, float, np.ndarray]]]
        :param enable_normalize: True almost any time except debug from Dataset.
        :param device: Choose the device when predict.
        :return: nd(b, )
        """
        self.to(self.cfg['device'] if device is None else device)
        self.eval()

        x_input = process_input(x_input, enable_normalize=enable_normalize)
        x_input = torch.from_numpy(x_input).float()
        x_input = x_input.to(self.cfg['device'] if device is None else device)

        pred_batch = self.forward(x_input)  # T(b, 1)

        pred_batch = resume_label(pred_batch)
        return pred_batch  # nd(b, )

    # @torch.jit.ignore
    def validate_by_dataloader(self, val_loader: DataLoader,
                               valid_cfg: dict,
                               device=None):
        """
        Validate the nn model by DataLoader.

        :param val_loader: torch.utils.data.DataLoader
        :param valid_cfg: dict
        :param device: str or torch.device('XX')

        :return: preds, labels, score_valid. HERE 'drop_last=False' !!!
            - preds: nd(valid_dataset_size,)
            - labels: nd(valid_dataset_size,)
            - score_valid: Dict[str, float]
        """
        # self.load_model()
        self.to(self.cfg['device'] if device is None else device)
        self.eval()

        if 'criterion' not in valid_cfg or valid_cfg['criterion'] is None:
            criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            criterion = valid_cfg['criterion']

        losses = AverageMeter()
        preds_list, tars_list = [], []
        start = time.time()
        print('Validating start.')
        with torch.no_grad():
            for batch_i, data in enumerate(val_loader):
                # (list(b), T(b, 100, 107), T(b,)), T(b,)
                (_, input_data, sample_weight), label = data
                input_data = input_data.float().to(self.cfg['device'] if device is None else device)  # T(b, num_fea)
                label = label.float().to(self.cfg['device'] if device is None else device)  # T(b,)
                if label.dim() == 1:
                    label = torch.unsqueeze(label, dim=1)  # T(b, 1)
                sample_weight = sample_weight.float().to(self.cfg['device'] if device is None else device)  # T(b,)
                if sample_weight.dim() == 1:
                    sample_weight = torch.unsqueeze(sample_weight, dim=1)  # T(b, 1)

                pred = self.forward(input_data)  # T(b, num_fea) -> T(b,)

                loss = criterion(pred, label)
                if valid_cfg['use_sample_weight']:
                    loss = loss * sample_weight  # element wise
                if valid_cfg['use_relative_loss']:
                    loss = loss / label * 100
                losses.update(loss.mean())

                if batch_i % 100 == 0:
                    print('Test_Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          '{dt}'.format(valid_cfg['epoch'], batch_i, len(val_loader), loss=losses,
                                        dt=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                preds_list.append(pred.squeeze(dim=1).detach().cpu().numpy())
                tars_list.append(label.squeeze().cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        labels = np.concatenate(tars_list, axis=0)

        preds = resume_label(preds)
        labels = resume_label(labels)
        score_valid = evaluate_score(labels, preds)

        print('Validating done. Use time: %.2fs.' % (time.time() - start))
        print('===============================')
        return preds, labels, score_valid

    def save_model(self, device=None):
        """Save the nn model after training."""
        try:
            self.to(self.cfg['device'] if device is None else device)
            torch.save(self.state_dict(),
                       self.cfg['nn_model_path'],
                       _use_new_zipfile_serialization=False)
        except NameError:
            print('Please init model first!')
            raise
        print('model save done. Save path is %s' % self.cfg['nn_model_path'])

    def save_offline_model(self):
        """Save the nn offline model after training."""
        _s = self.cfg['nn_model_path']
        _model_offline_path = _s[:_s.rfind('.')] + '_offline' + _s[_s.rfind('.'):]
        try:
            self.to('cpu')
            model = torch.jit.script(self)
            torch.jit.save(model, _model_offline_path)
        except NameError:
            print('Please init model first!')
            raise
        print('offline_model save done. Save path is %s' % _model_offline_path)

    def load_model(self):
        """Load the nn model from nn_model_path."""
        assert os.path.exists(self.cfg['nn_model_path'])
        # print('os.path.abspath(''): ', os.path.abspath(''))
        self.load_state_dict(torch.load(self.cfg['nn_model_path'], map_location=torch.device(self.cfg['device'])))
        print('load model from ckpt done. \nPath: %s' % self.cfg['nn_model_path'])

    def load_offline_model(self):
        """
        Load the nn model from nn_offline_model_path.

        Returns:
            A :class:`ScriptModule` object.
        """
        _s = self.cfg['nn_model_path']
        _model_offline_path = _s[:_s.rfind('.')] + '_offline' + _s[_s.rfind('.'):]
        assert os.path.exists(_model_offline_path)
        _offline_model = torch.jit.load(_model_offline_path)
        print('load offline_model from ckpt done. \nPath: %s' % _model_offline_path)
        return _offline_model


class LSTMModel(DNNModel):
    def __init__(self, cfg: dict):
        super(LSTMModel, self).__init__(cfg=cfg)
        self.cfg['nn_model_path'] = self.cfg['nn_model_path'].replace('dnn', 'lstm')
        self.subclass_flag = 'LSTMModel'

        self.encoder = nn.LSTM(input_size=self.cfg['in_dim'],
                               hidden_size=self.cfg['lstm_hidden_dim'],
                               num_layers=self.cfg['lstm_num_layers'],
                               batch_first=True,
                               bidirectional=False)  # input: (batch, seq, feature) (N, L, H_{in})

        # self.process_fn = lambda out: out[0].squeeze(dim=0)

        self.classifier = nn.Sequential(
            nn.Linear(self.cfg['lstm_hidden_dim'], self.cfg['n_classes'], bias=False),
        )

        self.init_device()

    # @script_method
    def forward(self, x_input: Tensor) -> Tensor:
        """
        The network of this nn model.

        :param x_input: Tensor(batch, seq, feature)
            e.g. input: T(16, 14, 7)
                encoder(x_input)[0] -> T(batch, seq_len, hidden_size)  e.q. T(16, 14, 32)
        :return: Tensor(batch_size, seq_len)
        """
        out = self.encoder(x_input)[0]  # T(batch, seq_len, hidden_size)
        # _, (h_n, _) = self.encoder(x_input)[0]
        # out = h_n[-1, :, :]  # T(batch, hidden_size)
        out = self.classifier(out)  # output: T(b, 1)
        out = out.squeeze(dim=2)
        return out  # output: Tensor(batch_size, seq_len)


class CNNModel(DNNModel):
    def __init__(self, cfg: dict):
        super(CNNModel, self).__init__(cfg=cfg)
        self.cfg['nn_model_path'] = self.cfg['nn_model_path'].replace('dnn', 'cnn')
        self.subclass_flag = 'CNNModel'

        self.layer_table = []
        # 1 * 1
        for i in range(len(self.cfg['convT_kernel_size'])):
            self.layer_table.append(
                nn.ConvTranspose2d(in_channels=self.cfg['convT_hidden_dim'][i - 1] if i > 0 else self.cfg['in_dim'],
                                   out_channels=self.cfg['convT_hidden_dim'][i],
                                   bias=False,
                                   kernel_size=self.cfg['convT_kernel_size'][i],
                                   stride=self.cfg['convT_stride'][i],
                                   output_padding=self.cfg['convT_output_padding'][i],
                                   padding=self.cfg['convT_padding'][i]))
            self.layer_table.append(nn.Tanh())
            # self.layer_table.append(nn.ReLU(True))
        # 224 * 224
        for i in range(len(self.cfg['conv_kernel_size'])):
            self.layer_table.append(
                nn.Conv2d(in_channels=self.cfg['conv_hidden_dim'][i - 1] if i > 0 else self.cfg['convT_hidden_dim'][-1],
                          out_channels=self.cfg['conv_hidden_dim'][i],
                          bias=False,
                          kernel_size=self.cfg['conv_kernel_size'][i],
                          stride=self.cfg['conv_stride'][i],
                          padding=self.cfg['conv_padding'][i]))
            self.layer_table.append(nn.BatchNorm2d(self.cfg['conv_hidden_dim'][i]))
            self.layer_table.append(nn.Tanh())
            # self.layer_table.append(nn.ReLU(True))
            self.layer_table.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # b * self.cfg['conv_hidden_dim'][-1] * 4 * 4
        self.encoder = nn.Sequential(*self.layer_table)

        self.classifier = nn.Sequential(nn.Linear(self.cfg['conv_hidden_dim'][-1] * 4 * 4,
                                                  self.cfg['hidden_dim'][-1]),
                                        # nn.Dropout(0.5),
                                        nn.Sigmoid(),
                                        nn.Linear(self.cfg['hidden_dim'][-1],
                                                  self.cfg['n_classes']))

        self.init_device()

    # @script_method
    def forward(self, x_input: Tensor) -> Tensor:
        """
        The network of this nn model.

        :param x_input: Tensor(batch_size, num_fea)
        :return: Tensor(batch_size, 1)
        """
        x_input = torch.unsqueeze(x_input, dim=-1)
        x_input = torch.unsqueeze(x_input, dim=-1)

        out = self.encoder(x_input)  # _out: T(b, hidden_dim[-1])
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)  # output: T(b, 1)
        # return super(CNNModel, self).forward(x_input)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dnn')
    parser.add_argument('--model_cfg', type=str, default='./configs/nn_config.yaml')
    parser.add_argument('--dataset_cfg', type=str, default='./configs/dataset_config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('-a', '--axis', type=str, default='x')
    parser.add_argument('-i', '--index', type=int, default=1)
    args = parser.parse_args()

    nn_config = parse_config(args.model_cfg)
    # nn_config = parse_config('./configs/nn_config.yaml')
    dataset_config = parse_config(args.dataset_cfg)
    # dataset_config = parse_config('./configs/dataset_config.yaml')
    # from model.nn_model import DNNModel, CNNModel, LSTMModel

    if args.axis in ('y', 'Y', '1'):
        nn_config['MODEL']['nn_model_path'] = nn_config['MODEL']['nn_model_path'].replace('_X', '_Y')
        dataset_config['meta_dataset_path'] = dataset_config['meta_dataset_path'].replace('_x', '_y')
        dataset_config['train_dataset_path'] = dataset_config['train_dataset_path'].replace('_x', '_y')
        dataset_config['test_dataset_path'] = dataset_config['test_dataset_path'].replace('_x', '_y')
        dataset_config['dataset_cleaned_path'] = dataset_config['dataset_cleaned_path'].replace('_x', '_y')
        dataset_config['result_save_path'] = dataset_config['result_save_path'].replace('_x', '_y')

    if args.model == 'dnn':
        model = DNNModel(nn_config['MODEL'])
    elif args.model == 'lstm':
        model = LSTMModel(nn_config['MODEL'])
        dataset_config['shuffle'] = False
        dataset_config['shuffle_valid'] = False
        dataset_config['balance_by_delta_and_shuffle'] = False
    else:
        model = CNNModel(nn_config['MODEL'])

    # ====================  <<  --  TRAIN  --  >>  ====================
    if args.train:
        if args.model == 'lstm':
            train_loader = get_lstm_dataloader(dataset_config, 'train')
        else:
            train_loader = get_dataloader(dataset_config, 'train')
        # train_cfg = nn_config['TRAIN']
        # data_loader = train_loader
        # self = model
        loss_for_visdom = model.fit(train_loader, nn_config['TRAIN'])
        if dataset_config['save_loss']:
            loss_for_visdom_set = pd.DataFrame(columns=['loss'], data=loss_for_visdom)
            loss_for_visdom_set.to_csv(dataset_config['train_log_save_path'], index=False)
        model.save_model()
        model.save_offline_model()

    # ====================  <<  --  VALID  --  >>  ====================
    if args.valid:
        dataset_config['balance_by_delta_and_shuffle'] = False

        model.load_model()
        if args.model == 'lstm':
            valid_loader = get_lstm_dataloader(dataset_config, 'valid')
        else:
            valid_loader = get_dataloader(dataset_config, 'valid')

        preds, labels, score_valid = model.validate_by_dataloader(valid_loader, nn_config['VALID'])
        if args.model == 'lstm':
            preds, labels = preds[:, -1], labels[:, -1]
        print('score_valid: ', score_valid)

        if dataset_config['save_result']:
            result = pd.DataFrame({'labels': labels,
                                   'preds': preds.astype(float)})
            _s = dataset_config['result_save_path']
            result_save_path = _s[:_s.rfind('.')] + '_' + args.model + _s[_s.rfind('.'):]
            result.to_csv(result_save_path, index=False)
        print('========================================================')

        # ===================  <<  --  Predict  --  >>  ===================
        for _ in range(3):
            (_, x_input, _), labels_example = next(iter(valid_loader))

            result = model.predict_once(x_input, enable_normalize=False)
            true_label = resume_label(labels_example)

            if args.model == 'lstm':
                result, true_label = result[:, -1], true_label[:, -1]

            print('predict: ', result)
            print('label: ', true_label)

        # =================================================================
        # ===========  <<  --  Predict by Offline Model.  --  >>  =========
        # =================================================================
        _s = nn_config['MODEL']['nn_model_path']
        offline_model_path = _s[:_s.rfind('.')] + '_offline' + _s[_s.rfind('.'):]
        offline_model = torch.jit.load(offline_model_path)
        offline_model.cpu()

        meta_input_data = []
        for _ in range(3):
            (_, x_input, _), labels_example = next(iter(valid_loader))

            x_input = process_input(x_input, enable_normalize=False)
            x_input = torch.from_numpy(x_input).float().cpu()
            result = offline_model(x_input)
            result = resume_label(result)
            true_label = resume_label(labels_example)

            if args.model == 'lstm':
                result, true_label = result[:, -1], true_label[:, -1]

            print('predict offline: ', result)
            print('label offline: ', true_label)
