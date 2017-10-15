from collections import OrderedDict
from itertools import product

import numpy
from lasagne.layers.helper import get_output
from lasagne.utils import floatX
from parameterized import param, parameterized
from theano import function, tensor

from .densenet import DenseNet, MyDenseNet
from .test_resnet import name_func


KWARGS = {
    'growth': (4, 12, 40),
    'bottleneck': (True, False),
    'neck_size': (None, 16),
    'compression': (1, 0.5),
    'dropout': (0.0, 0.2),
}

FRWRD = [
    param(DenseNet, n=4, growth=4, bottleneck=False, compression=1,
          dropout=0.5),
    param(DenseNet, n=4, growth=4, bottleneck=True, compression=1,
          dropout=0.5),
    param(DenseNet, n=4, growth=4, bottleneck=True, compression=1,
          neck_size=4),
    param(DenseNet, n=4, growth=12, bottleneck=True, compression=1),
    param(DenseNet, n=16, growth=12, bottleneck=True, compression=1,
          neck_size=16),
    param(DenseNet, n=4, growth=12, bottleneck=True, compression=0.5),
    param(MyDenseNet, n=4, growth=4, bottleneck=True, compression=1),
    param(MyDenseNet, n=16, growth=12, bottleneck=True, compression=1,
          neck_size=16),
]


def iter_args(*names, **kwargs):
    values = OrderedDict((n, KWARGS[n]) for n in names)
    values.update((k, v) for k, v in kwargs.items() if v)
    names = tuple(values.keys())
    for combi in product(*values.values()):
        yield dict(zip(names, combi))


def cifar_models(cls, *extra_params, n=(2, 3, 6, 16, 18, 27), **kwargs):
    for kwargs in iter_args('growth', 'bottleneck', 'compression',
                            *extra_params, n=n, **kwargs):
        yield cls.cifar_model(**kwargs)


def imagenet_models(cls, *extra_params, **kwargs):
    for kwargs in iter_args('compression', *extra_params, **kwargs):
        yield cls.imagenet_121(**kwargs)
        yield cls.imagenet_169(**kwargs)
        yield cls.imagenet_201(**kwargs)
        yield cls.imagenet_161(**kwargs)


@parameterized.expand([
    param(DenseNet, 'neck_size', 'dropout'),
    param(MyDenseNet, ),
], testcase_func_name=name_func)
def test_model_create(cls, *extra_params, **kwargs):
    for _ in cifar_models(cls, *extra_params, **kwargs):
        pass
    for _ in imagenet_models(cls, *extra_params, **kwargs):
        pass


@parameterized.expand(FRWRD, testcase_func_name=name_func)
def test_cifar_deterministic(cls, **kwargs):
    model = cls.cifar_model(**kwargs)
    data = floatX(numpy.random.normal(0.0, 1.0, (64, 3, 32, 32)))

    input_var = tensor.tensor4('inputs')
    activation = get_output(model, input_var, deterministic=True)

    func = function([input_var], activation)
    output = func(data)
    del output


@parameterized.expand(FRWRD, testcase_func_name=name_func)
def test_cifar_nondeterministic(cls, **kwargs):
    model = cls.cifar_model(**kwargs)
    data = floatX(numpy.random.normal(0.0, 1.0, (64, 3, 32, 32)))

    input_var = tensor.tensor4('inputs')
    activation = get_output(model, input_var, deterministic=False)

    func = function([input_var], activation)
    output = func(data)
    del output
