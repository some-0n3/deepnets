#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime

from lasagne.layers import count_params

from resnet import PreResNet, ResNet, StochasticDepth
from trainers import CIFAR_ResNetTrainer, CIFAR_SDTrainer, SVHN_SDTrainer
from utils.data.cifar_lee14 import CIFAR10, CIFAR100


def main(config=None, init_path='', out_path='', batchsize=128, dataset='C10',
         pre=False, stoch_depth=False, n=5, stype='A', bottleneck=False,
         dim_inc_meth='1x1'):
    """Train a resnet on the CIFAR-10 data set.

    Parameters
    ----------
    config : list of dictionaries or ``None`` (``None``)
        The configuration for the training.
    init_model : string (``''``)
        The path (prefix) to the initial model parameters, updates and
        journal files. This is for continuing a previous training.
    out_path : string (``''``)
        The path (prefix) for the output file. This function will save
        the trained model, a journal with training statistics the updates
        for the optimizer and the create data set (this is needed to
        continue the training).
    batch_size : integer (``128``)
        The batch size for training.
    dataset : ``'C10'``, ``'C100'`` or ``'SVHN'`` (``'C10'``)
        The data set to use for training. The options are: CIFAR-10,
        CIFAR-100 and the "Street View House Number" data sets.
    pre : boolean (``False``)
        If ``True`` use the pre-action order.
    stoch_depth: boolean (``False``)
        If ``True`` use the stochastic depth approach, with a linear
        decay and $p_L = 0..5$.
    n : integer (``5``)
        The parameter 'n' from the paper.
    stype : ``'A'``, ``'B'`` or ``'C'`` (``'A'``)
        The type of shortcut.
    bottleneck : boolean (``False``)
        Use bottleneck approach with 3 layers per stack.
    dim_inc : ``'1x1'``, ``'2x2'``, ``'max'``, ``'sum'`` or ``'avg'``
        The method to deal with the increase in dimensions. '1x1' will
        perform a 1x1 convolution and ignore 3/4 of the input. '2x2'
        will perform a 2x2 convolution. This will add some parameters,
        but won't ignore any inputs. 'max', 'sum' and 'avg' will perform
        a 1x1 convolution with a 1x1 followed by the corresponding
        pooling operation, followed by a 1x1 convolution. This will not
        ignore any inputs nor add any parameters to the model.
        NOTE: This argument is ignored if the shortcut type is 'A'.
    """
    # network
    assert dataset in ('C10', 'C100', 'SVHN')
    classes = 109 if dataset == 'C100' else 10
    bases = (PreResNet, ResNet) if pre else (ResNet, )
    if stoch_depth:
        bases = (StochasticDepth, ) + bases
    model_cls = type('ModelClass', bases, {})
    model = model_cls.cifar_model(n=n, type=stype, bottleneck=bottleneck,
                                  dim_inc_meth=dim_inc_meth, classes=classes)
    # trainer
    if dataset == 'SVHN':
        trainer_cls = SVHN_SDTrainer
    else:
        trainer_cls = CIFAR_SDTrainer if stoch_depth else CIFAR_ResNetTrainer
    if init_path:
        trainer = trainer_cls.load_state(model, init_path, batchsize=batchsize)
    else:
        trainer = trainer_cls(model, batchsize=batchsize)
    # dataset
    if not trainer.dataset:
        if dataset == 'SVHN':
            raise NotImplementedError(
                'The SVHN dataset is not yet implemented.')
        elif dataset == 'C10':
            trainer.dataset = CIFAR10(testsplit=0.1)
        elif dataset == 'C100':
            trainer.dataset = CIFAR100(testsplit=0.1)

    # training the network
    print('Training model ({} parameters) ...'.format(
        count_params(model, trainable=True)))
    trainer.train(config)

    # save the network, the updates and the journal
    if not out_path:
        _, acc = trainer.validate()
        date = datetime.now().strftime('%Y-%m-%d_%H:%M')
        bn_str = 'bottleneck' if bottleneck else 'no_bottleneck'
        _type = 'A' if stype == 'A'else '{}_{}'.format(stype, dim_inc_meth)
        mdl_str = 'pre-resnet' if pre else 'resnet'
        if stoch_depth:
            mdl_str += '-sd'
        tmpl = '{}-{}__-__n_{}_-_{}_-_{}__-__acc_{:.2f}_{}'
        out_path = tmpl.format(mdl_str, dataset, n, _type, bn_str, acc * 100,
                               date)
    trainer.save_state(out_path, resume=True)


parser = argparse.ArgumentParser(description='Train a ResNet')
parser.add_argument('out', metavar='out', type=str, nargs='?', default='',
                    help='prefix for the output files')
parser.add_argument('-i', '--init', type=str, nargs=1, default=['', ],
                    help='prefix for the initialization files')
parser.add_argument('-c', '--config-file', type=str, nargs=1, default=[None, ],
                    help='configuration file the training')
parser.add_argument('-b', '--batch-size', type=int, nargs=1, default=[128, ],
                    help='mini-batch size for training')
parser.add_argument('-d', '--dataset', type=str, nargs=1, default=['C10', ],
                    choices=['C10', 'C100', 'SVHN'],
                    help='The dataset to use for training.')
parser.add_argument('-p', '--pre-activation', dest='pre_activation',
                    action='store_true', default=False,
                    help='use the pre-activation approach')
parser.add_argument('-s', '--stochastic-depth', dest='stochastic_depth',
                    action='store_true', default=False,
                    help='use a network with stochastic depth')
parser.add_argument('-n', type=int, nargs=1, default=[9, ],
                    help='the parameter n from the paper')
parser.add_argument('-t', '--type', type=str, nargs=1, default=['A', ],
                    choices=['A', 'B', 'C'], help='shortcut method')
parser.add_argument('-l', '--bottleneck', dest='bottleneck',
                    action='store_true', default=False,
                    help='use bottleneck approach')
parser.add_argument('-m', '--method', type=str, nargs=1, default=['1x1', ],
                    choices=['1x1', '2x2', 'max', 'avg', 'sum'],
                    help='method for handling dimension increase in shortcut\
                          projections.')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.config_file[0] is None:
        config = None
    else:
        config = json.load(open(args.config_file[0], 'r'))

    sys.setrecursionlimit(1000000)

    main(init_path=args.init[0], out_path=args.out, config=config,
         dataset=args.dataset[0], pre=args.pre_activation,
         stoch_depth=args.stochastic_depth, n=args.n[0], stype=args.type[0],
         bottleneck=args.bottleneck, dim_inc_meth=args.method[0],
         batchsize=args.batch_size[0])
