# -*- coding:utf-8 -*-
# __author__ = 'lvluqiang'#
# (C)Copyright 2018-2030, HuaZhongCNC Inc. All rights reserved

"""Lightgbm Model py file."""

__all__ = ['LGBModel']

import argparse
import os
import sys
import time
from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')
os.chdir(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../')

from utils.common_utils import evaluate_score
from utils.common_utils import parse_config
from utils.common_utils import process_input
from utils.common_utils import resume_label
from utils.dataset_utils import DatasetBase
from utils.dataset_utils import get_dataset_class


class LGBModel:
    """
    Main class for the lightgbm model.

    :argument cfg: The lightgbm model config dict. Type: dict
        Always contains keys:
            params: The parameters used to create the LGB model.
            num_boost_round_list: List of the num_boost_round parameter in params.
            lr_list: List of the learning_rate parameter in params.
            lgb_model_path: Path of model to save to and load from.
            test_size: Proportion of test sets
            random_state: The random seed when making dataset.
            use_sample_weight: Use sample weight or not when training.
            free_raw_data: Set True for continue training.
                The raw data will not be freed when True.
            extract_fea_by_nn_model: Extract features by nn_model or not.
    """

    def __init__(self,
                 cfg: dict):
        super(LGBModel, self).__init__()
        self.cfg = cfg
        assert 'params' in self.cfg and self.cfg['params'] is not None
        self.params = self.cfg['params']
        self.model = None
        self.model_ = None  # for grid_search.

        if self.cfg['train_by_grid_search']:
            self.cfg['lgb_model_path'] = self.cfg['lgb_model_path'].split('.')[0] + '_train_by_grid_search.' + \
                                         self.cfg['lgb_model_path'].split('.')[1]
        # print('model: ', self)

    def train_with_lr_decay(self, train_dataset: lgb.Dataset, test_dataset: lgb.Dataset):
        """
        Train the lightgbm model with lr decay from lgb.Dataset.

        :param train_dataset: lgb.Dataset
        :param test_dataset: lgb.Dataset
        :return: None
        """
        assert self.cfg['num_boost_round_list']
        assert self.cfg['lr_list']

        print('=======================')
        print('Training start.')
        start = time.time()
        for i in range(len(self.cfg['num_boost_round_list'])):
            self.params['num_boost_round'] = self.cfg['num_boost_round_list'][i]
            self.params['learning_rate'] = self.cfg['lr_list'][i]
            print('training epoch %d of %d start...' % (i, len(self.cfg['num_boost_round_list'])))
            self.model = lgb.train(self.params,
                                   train_dataset,
                                   valid_sets=[train_dataset, test_dataset],
                                   init_model=self.model,
                                   # verbose_eval=False,
                                   keep_training_booster=True)
            score_train = {s[1]: s[2] for s in self.model.eval_train()}
            print(score_train)
            score_train = {s[1]: s[2] for s in self.model.eval_valid()}
            print(score_train)
        print('Training done. Use time: %.2fs.' % (time.time() - start))
        print('=======================')

    def train_by_grid_search(self, param_grid, x_train, y_train, x_test, y_test):
        """
        Train the lightgbm model by grid search from lgb.Dataset.
        """
        print('param_grid is: ', param_grid)
        print('=======================')
        print('Training start.')
        start = time.time()
        params = self.cfg['params']
        self.model_ = lgb.LGBMClassifier(boosting_type=params['boosting_type'],
                                         num_leaves=params['num_leaves'],
                                         max_depth=params['max_depth'],
                                         learning_rate=params['learning_rate'],
                                         n_estimators=params['num_boost_round'],
                                         objective=params['objective'],
                                         class_weight='balanced',
                                         min_split_gain=0.0,
                                         min_child_samples=params['min_data_in_leaf'],
                                         min_child_weight=params['min_sum_hessian_in_leaf'],
                                         subsample=params['bagging_fraction'],
                                         subsample_freq=params['bagging_freq'],
                                         colsample_bytree=params['feature_fraction'],
                                         reg_alpha=params['lambda_l1'],
                                         reg_lambda=params['lambda_l2'],
                                         random_state=params['seed'],
                                         silent=True)

        gsearch = GridSearchCV(self.model_,
                               param_grid,
                               scoring=['f1_macro', 'precision_macro', 'recall_macro'],
                               refit='recall_macro',
                               cv=3)
        gsearch.fit(x_train,
                    y_train,
                    eval_set=(x_test, y_test),
                    eval_metric=params['metric'],
                    # init_model=self.model_,
                    verbose=params['verbosity'],
                    early_stopping_rounds=params['early_stopping_rounds'])
        print('Training done. Use time: %.2fs.' % (time.time() - start))
        print('=======================')

        print('gsearch.best_params_:{0}'.format(gsearch.best_params_))
        print('gsearch.best_score_:{0}'.format(gsearch.best_score_))
        # print(gsearch.cv_results_['mean_test_score'])
        # print(gsearch.cv_results_['params'])

    def train_by_dataset_with_lr_decay(self, dataset_base: DatasetBase):
        """
        Train the lightgbm model with lr decay from DatasetBase class.

        :param dataset_base: DatasetBase
        :return: None.
        """
        input_data = dataset_base.get_data()
        weight = dataset_base.get_weight()
        labels = dataset_base.get_label()
        x_train, x_test, y_train, y_test, w_train, _ = train_test_split(input_data,
                                                                        labels,
                                                                        weight,
                                                                        test_size=0.2,
                                                                        random_state=2021)
        print('train_by_grid_search', self.cfg['train_by_grid_search'])

        if self.cfg['train_by_grid_search']:
            assert self.cfg['param_grid'] is not None
            self.train_by_grid_search(self.cfg['param_grid'], x_train, y_train, x_test, y_test)
        else:
            if 'use_sample_weight' in self.cfg and self.cfg['use_sample_weight']:
                train_data_set = lgb.Dataset(x_train,
                                             y_train,
                                             weight=w_train,
                                             free_raw_data=self.cfg['free_raw_data'])
            else:
                train_data_set = lgb.Dataset(x_train,
                                             y_train,
                                             free_raw_data=self.cfg['free_raw_data'])
            test_data_set = lgb.Dataset(x_test,
                                        y_test,
                                        reference=train_data_set,
                                        free_raw_data=self.cfg['free_raw_data'])
            self.train_with_lr_decay(train_data_set, test_data_set)

    def predict_once(self, x_input: Union[list, np.ndarray], enable_normalize: bool = True):
        """
        Calculate the predicted value of the input data.

        :param x_input: list or np.ndarray
            with shape (fea_num, )
                or shape (batch_size, fea_num).
        :param enable_normalize:  True almost any time except debug from Dataset.
        :return: with shape (1, ) or (batch_size, )
        """
        if not self.model:
            raise Exception('Please init model first!')
        print('input_data: ', x_input)

        x_input = process_input(x_input, enable_normalize=enable_normalize)
        predict = self.model.predict(x_input,
                                     num_iteration=self.model.best_iteration)
        predict = resume_label(predict)
        return predict  # nd(b, )

    def validate_by_dataset(self,
                            dataset_base: DatasetBase
                            ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Validate the lightgbm model after training.

        :param dataset_base: DatasetBase
        :return: preds, labels, score_valid
        """
        if not self.model:
            raise Exception('Please init model first!')

        start = time.time()
        print('Validating start.')

        self.load_model()
        valid_data = dataset_base.get_data()
        labels = dataset_base.get_label()
        print('valid_data.shape: ', valid_data.shape)
        print('self.model.best_iteration', self.model.best_iteration)
        preds = self.model.predict(valid_data, num_iteration=self.model.best_iteration)
        print('preds.shape: ', preds.shape)

        preds = resume_label(preds)
        labels = resume_label(labels)
        score_valid = evaluate_score(labels, preds)

        print('Validating done. Use time: %.2fs.' % (time.time() - start))
        print('===============================')
        return preds, labels, score_valid

    def save_model(self):
        """Save the lightgbm model after training."""
        if isinstance(self.model, lgb.LGBMClassifier):
            self.model = self.model.booster_
        self.model.save_model(self.cfg['lgb_model_path'])
        print('model save done. Save path is %s' % self.cfg['lgb_model_path'])
        print('===============================')

    def load_model(self):
        """Load the lightgbm model from lgb_model_path."""
        if self.model is None:
            self.model = lgb.Booster(model_file=self.cfg['lgb_model_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, default='./configs/lgb_config.yaml')
    parser.add_argument('--dataset_cfg', type=str, default='./configs/dataset_config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('-a', '--axis', type=str, default='x')
    parser.add_argument('-i', '--index', type=int, default=1)
    args = parser.parse_args()

    # class Args:
    #     def __init__(self):
    #         self.model_cfg = './configs/lgb_config.yaml'
    #         self.dataset_cfg = './configs/dataset_config.yaml'
    # args = Args()

    lgb_config = parse_config(args.model_cfg)
    dataset_config = parse_config(args.dataset_cfg)

    if args.axis in ('y', 'Y', '1'):
        lgb_config['MODEL']['lgb_model_path'] = lgb_config['MODEL']['lgb_model_path'].replace('_X', '_Y')
        dataset_config['meta_dataset_path'] = dataset_config['meta_dataset_path'].replace('_x', '_y')
        dataset_config['train_dataset_path'] = dataset_config['train_dataset_path'].replace('_x', '_y')
        dataset_config['test_dataset_path'] = dataset_config['test_dataset_path'].replace('_x', '_y')
        dataset_config['dataset_cleaned_path'] = dataset_config['dataset_cleaned_path'].replace('_x', '_y')
        dataset_config['result_save_path'] = dataset_config['result_save_path'].replace('_x', '_y')

    model = LGBModel(lgb_config['MODEL'])

    # ========  <<  --  train_by_grid_search or not  --  >>  =========
    if args.train:
        dataset = get_dataset_class(dataset_config, 'train')
        model.train_by_dataset_with_lr_decay(dataset)
        model.save_model()

    # ====================  <<  --  VALID  --  >>  ====================
    if args.valid:
        dataset_config['balance_by_delta_and_shuffle'] = False

        # dataset = get_lgb_dataset_class(dataset_config, 'train')
        # model.load_model()
        # result = model.validate_by_dataset(dataset)
        # print('score_valid: ', result[-1])
        # print('========================================================')
        model.load_model()
        dataset = get_dataset_class(dataset_config, 'valid')
        preds, labels, score_valid = model.validate_by_dataset(dataset)
        print('score_valid: ', score_valid)

        if dataset_config['save_result']:
            result = pd.DataFrame({'labels': labels,
                                   'preds': preds.astype(float)})
            _s = dataset_config['result_save_path']
            result_save_path = _s[:_s.rfind('.')] + '_lgb' + _s[_s.rfind('.'):]
            result.to_csv(result_save_path, index=False)
        print('========================================================')

        # ===================  <<  --  Predict  --  >>  ===================
        input_data = dataset.get_data()
        labels_example = dataset.get_label()
        true_label = resume_label(labels_example)
        for i in range(3):
            x_input = input_data.tolist()[args.index + i]
            print('predict: ', model.predict_once(x_input, enable_normalize=False))
            print('labels: ', true_label[args.index + i])
