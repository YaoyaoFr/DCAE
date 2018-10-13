#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Data preparation

Usage:
  prepare_data.py
  prepare_data.py (-h | --help)

Options:
  -h --help           Show this screen
"""

import h5py
import random
import pandas as pd
import numpy as np
from docopt import docopt
from Data.load_data import load_patients_to_file, load_phenotypes
from sklearn.model_selection import StratifiedKFold, train_test_split
from Data.utils_prepare_data import *


def load_datas(cover: bool = False) -> h5py.Group:
    hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
    hdf5 = hdf5_handler(hdf5_path, 'a')

    features = ['reho', 'falff', 'vmhc']
    datasets = ['ABIDE', 'ADHD', 'ABIDE II', 'FCP']

    if cover:
        load_patients_to_file(hdf5, features, datasets)
    return hdf5


def prepare_folds(data_hdf5: h5py.Group, arguments: dict, folds_hdf5: h5py.Group = None):
    if folds_hdf5 is None:
        folds_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_folds.hdf5'
        folds_hdf5 = hdf5_handler(folds_hdf5_path, 'a')

    datasets = arguments['datasets']
    fold_nums = arguments['fold_nums']
    features = arguments['features']
    groups = arguments['groups']
    for group, dataset, fold_num in zip(groups, datasets, fold_nums):
        pheno = load_phenotypes(dataset=dataset)

        # choose subjects according to group
        dx_group = {'health': 1, 'patient': 0, 'all': 2}[group]
        pheno = pheno[pheno['DX_GROUP'] != dx_group]
        ids = pheno['FILE_ID']
        ids_encode = list()
        for id in ids:
            ids_encode.append(id.encode())
        ids_encode = pd.Series(ids_encode)

        for feature in features:
            exp = folds_hdf5.require_group('{:s}/{:s}'.format(dataset, feature))
            exp.attrs['feature'] = [feature.encode(encoding='utf-8')]

            if fold_num == 0:
                fold = exp.require_group(str(fold_num))

                # Subject list
                if 'train' in fold:
                    fold.pop('train')
                fold['train'] = ids_encode.tolist()

                # Train data
                data = load_fold(subjs=data_hdf5[dataset + '/subjects'],
                                 fold=fold,
                                 experiment=exp,
                                 )

                for flag in ['data', 'label']:
                    tvt_flag = 'train {:s}'.format(flag)
                    create_dataset_hdf5(group=fold,
                                        name=tvt_flag,
                                        data=data[tvt_flag])
                continue

            else:
                skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
                for i, (train_index, test_index) in enumerate(skf.split(ids_encode, pheno['STRAT'])):
                    train_index, valid_index = train_test_split(train_index, test_size=0.2)
                    fold = exp.require_group(str(i + 1))

                    for flag, index in zip(['train', 'test', 'valid'],
                                           [train_index, test_index, valid_index]):
                        if flag in fold:
                            fold.pop(flag)
                        fold[flag] = ids_encode[index].tolist()

                    data = load_fold(subjs=data_hdf5[dataset + '/subjects'],
                                     experiment=exp,
                                     fold=fold)

                    for tvt in ['train', 'valid', 'test']:
                        for flag in ['data', 'label']:
                            tvt_flag = '{:s} {:s}'.format(tvt, flag)
                            create_dataset_hdf5(group=fold,
                                                name=tvt_flag,
                                                dtype=float,
                                                data=data[tvt_flag])
                continue


def prepare_scheme(arguments: dict, folds_hdf5: h5py.Group = None, scheme_hdf5: h5py.Group = None):
    if folds_hdf5 is None:
        folds_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_folds.hdf5'
        folds_hdf5 = hdf5_handler(folds_hdf5_path, 'a')
    if scheme_hdf5 is None:
        scheme_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
        scheme_hdf5 = hdf5_handler(scheme_hdf5_path, 'a')

    scheme = arguments['scheme']
    features = arguments['features']
    feature = '_'.join(features)
    scheme_group = scheme_hdf5.require_group('scheme {:d}/{:s}'.format(scheme, feature))
    dataset = arguments['dataset']

    if scheme == 1:
        folds = folds_hdf5['ABIDE/{:s}'.format(feature)]
        basic_fold = folds_hdf5['ADHD/{:s}/0'.format(feature)]
        basic_data = np.array(basic_fold['train data'])
        basic_label = np.array(basic_fold['train label'])

        for fold_index in folds:
            fold = folds[fold_index]
            fold_new = scheme_group.require_group('fold {:s}'.format(fold_index))

            # Pre-training
            data = np.array(fold['train data'])
            label = np.array(fold['train label'])
            data = data[label == 0]
            data = np.concatenate((basic_data, data), axis=0)
            data = data_normalization(data=data, axis=0)
            data = split_slices(data)
            create_dataset_hdf5(group=fold_new,
                                name='pre training data',
                                data=data,
                                )

            for tvt in ['train', 'valid', 'test']:
                data = np.array(fold['{:s} data'.format(tvt)])
                label = np.array(fold['{:s} label'.format(tvt)])
                data = data_normalization(data=data, axis=0)
                data = split_slices(data)

                create_dataset_hdf5(group=fold_new,
                                    name='{:s} data'.format(tvt),
                                    data=data,
                                    )
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} label'.format(tvt),
                                    data=label,
                                    )
    elif scheme == 3:
        folds = folds_hdf5['ABIDE/{:s}'.format(feature)]
        pass
        for fold_index in folds:
            fold = folds[fold_index]
            fold_new = scheme_group.require_group('fold {:s}'.format(fold_index))
            for tvt in ['train', 'valid', 'test']:
                data = np.array(fold['{:s} data'.format(tvt)])
                label = np.array(fold['{:s} label'.format(tvt)])
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} data'.format(tvt),
                                    data=data,
                                    )
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} label'.format(tvt),
                                    data=label,
                                    )
    elif scheme == 4:
        folds = folds_hdf5['{:s}/{:s}'.format(dataset, feature)]
        datasets = ['ABIDE', 'ABIDE II', 'ADHD', 'FCP']
        datasets.remove(dataset)

        if 'pre train data' not in scheme_group:
            basic_folds = [folds_hdf5['{:s}/{:s}/0'.format(d, feature)] for d in datasets]
            basic_data = np.concatenate([np.array(basic_fold['train data']) for basic_fold in basic_folds], 0)
            data = data_normalization(data=basic_data, axis=0)
            data = data[:, :, :, :, 0]
            create_dataset_hdf5(group=scheme_group,
                                name='pre train data',
                                data=data,
                                )
        for fold_index in np.arange(start=1, stop=6):
            fold = folds[str(fold_index)]
            fold_new = scheme_group.require_group('fold {:d}'.format(fold_index))

            # # Pre-training
            # data = np.array(fold['train data'])
            # label = np.array(fold['train label'])
            # data = data[label == 0]
            # data = np.concatenate((basic_data, data), axis=0)
            # data = data_normalization(data=data, axis=0)
            # data = data[:, :, :, :, 0]
            # create_dataset_hdf5(group=fold_new,
            #                     name='pre train data',
            #                     data=data,
            #                     )

            for tvt in ['train', 'valid', 'test']:
                data = np.array(fold['{:s} data'.format(tvt)])
                label = np.array(fold['{:s} label'.format(tvt)])
                data = data_normalization(data=data, axis=0)
                data = data[:, :, :, :, 0]

                create_dataset_hdf5(group=fold_new,
                                    name='{:s} data'.format(tvt),
                                    data=data,
                                    )
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} label'.format(tvt),
                                    data=label,
                                    )


def main(arguments):
    hdf5 = load_datas()
    # prepare_folds(data_hdf5=hdf5, arguments=arguments)
    # prepare_scheme(arguments=arguments)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    # Debug Code
    arguments['features'] = ['falff']
    arguments['groups'] = ['all', 'all', 'all', 'health', 'health', 'health', 'all']
    arguments['dataset'] = 'ABIDE'
    arguments['fold_nums'] = [5, 5, 5, 0]
    arguments['scheme'] = 4

    main(arguments=arguments)
