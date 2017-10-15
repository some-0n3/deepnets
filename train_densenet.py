#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime

from lasagne.layers import count_params

from densenet import DenseNet
from trainers import CIFAR_DenseNetTrainer, SVHN_DenseNetTrainer
from utils.data.cifar_lee14 import CIFAR10, CIFAR100


def main(config=None, init_path='', out_path='', batchsize=64, dataset='C10',
         n=31, growth=40, bottleneck=True, neck_size=None, compression=1,
         dropout=0):
    # network
    assert dataset in ('C10', 'C100', 'SVHN')
    classes = 100 if dataset == 'C100' else 10
    model = DenseNet.cifar_model(
        n=n, growth=growth, bottleneck=bottleneck, neck_size=neck_size,
        compression=compression, dropout=dropout, classes=classes
    )
    # trainer
    if dataset == 'SVHN':
        trainer_cls = SVHN_DenseNetTrainer
    else:
        trainer_cls = CIFAR_DenseNetTrainer
    if init_path:
        trainer = trainer_cls.load_state(model, init_path, batchsize=batchsize)
    else:
        trainer = trainer_cls(model, batchsize=batchsize)
    # dataset
    if not trainer.dataset:
        if dataset == 'C10':
            trainer.dataset = CIFAR10(testsplit=0.1)
        elif dataset == 'C100':
            trainer.dataset = CIFAR100(testsplit=0.1)
        else:
            raise NotImplementedError(
                'The SVHN dataset is not yet implemented.')

    # training the network
    print('Training model ({} parameters) ...'.format(
        count_params(model, trainable=True)))
    trainer.train(config)

    # save the network, the updates and the journal
    if not out_path:
        _, acc = trainer.validate()
        date = datetime.now().strftime('%Y-%m-%d_%H:%M')
        bn_str = 'bottleneck' if bottleneck else 'no_bottleneck'
        tmpl = 'densenet-{}_-_n_{}_-_k_{}_-_{}_-_t_{:.2f}_-_acc_{:.2f}_{}'
        out_path = tmpl.format(dataset, n, growth, bn_str, compression,
                               acc * 100, date)
    trainer.save_state(out_path, resume=True)


parser = argparse.ArgumentParser(description='Train a DenseNet')
parser.add_argument('out', metavar='out', type=str, nargs='?', default='',
                    help='prefix for the output files')
parser.add_argument('-i', '--init', type=str, nargs=1, default=['', ],
                    help='prefix for the initialization files')
parser.add_argument('-c', '--config-file', type=str, nargs=1, default=[None, ],
                    help='configuration file the training')
parser.add_argument('-b', '--batch-size', type=int, nargs=1, default=[64, ],
                    help='mini-batch size for training')
parser.add_argument('-n', type=int, nargs=1, default=[31, ],
                    help='the parameter n from the paper')
parser.add_argument('-k', '--growth', type=int, nargs=1, default=[40, ],
                    help='the groth parameter "k" from the paper')
parser.add_argument('-l', '--no-bottleneck', dest='bottleneck',
                    action='store_false', default=True,
                    help='turn off the bottleneck approach')
parser.add_argument('--neck-size', type=int, nargs=1, default=[None, ],
                    help='the number of channels in a bottleneck, if not'
                    ' specified it is 4 times the growth factor "k"')
parser.add_argument('-t', '--compression', type=float, default=[0.5, ],
                    nargs=1, help='the compression factor theta')
parser.add_argument('-d', '--dropout', type=float, default=[0], nargs=1,
                    help='the dropout probability ("0" for no dropout)')
parser.add_argument('-s', '--dataset', type=str, nargs=1, default=['C10', ],
                    choices=['C10', 'C100', 'SVHN'],
                    help='The dataset to use for training.')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.config_file[0] is None:
        config = None
    else:
        config = json.load(open(args.config_file[0], 'r'))

    sys.setrecursionlimit(1000000)

    main(init_path=args.init[0], out_path=args.out, config=config,
         batchsize=args.batch_size[0], dataset=args.dataset[0],
         n=args.n[0], growth=args.growth[0], bottleneck=args.bottleneck,
         neck_size=args.neck_size[0], compression=args.compression[0],
         dropout=args.dropout[0])
