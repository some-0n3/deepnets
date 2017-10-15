import sys
from collections import OrderedDict
from itertools import permutations, product, chain

import numpy
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.helper import get_all_layers, get_output
from lasagne.layers.noise import DropoutLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers.pool import GlobalPoolLayer, Pool2DLayer
from lasagne.layers.special import BiasLayer, NonlinearityLayer, ScaleLayer
from lasagne.utils import floatX
from parameterized import param, parameterized
from pytest import skip
from theano import function, tensor

from .resnet import (BaseResNet, PreResNet, ResNet, RoR, StochasticDepth,
                     WeightedResNet, WideResNet, KnownDepth, MultiResNet)


sys.setrecursionlimit(int(1e6))


def iter_args(**kwargs):
    values = OrderedDict(kwargs)
    names = tuple(values.keys())
    for combi in product(*values.values()):
        yield dict(zip(names, combi))


def cifar_models(cls, n=(1, 3, 5, 7, 9, 18), **kwargs):
    """Yield all CIFAR models."""
    for kwargs in iter_args(n=n, **kwargs):
        yield cls.cifar_model(**kwargs)


def imagenet_models(cls, **kwargs):
    """Yield all ImageNet models."""
    for kwargs in iter_args(**kwargs):
        methods = ['imagenet_18', 'imagenet_34', 'imagenet_50', 'imagenet_101',
                   'imagenet_152']
        if issubclass(cls, WideResNet):
            methods.remove('imagenet_101')
            methods.remove('imagenet_152')
            if kwargs['width'] >= 4:
                methods.remove('imagenet_50')
            if kwargs['width'] >= 10:
                methods.remove('imagenet_34')
        if issubclass(cls, RoR) and kwargs['shortcut_lvl'] > 2:
            methods.remove('imagenet_18')
            try:
                methods.remove('imagenet_34')
            except ValueError:
                pass
            try:
                methods.remove('imagenet_50')
            except ValueError:
                pass
            try:
                methods.remove('imagenet_152')
            except ValueError:
                pass

        for method in methods:
            yield getattr(cls, method)(**kwargs)


def iter_cls(*classes, blacklist=tuple()):
    """Yield all combinations of classes.

    This iterator will yield every class constructed by using the
    classes in ``classes`` as bases. Therefor every permutation is
    used,except for those in ``blacklist``.
    """
    for bases in permutations(classes):
        if bases not in blacklist:
            yield type('_'.join(c.__name__ for c in bases), bases, {})


def check_shape(layer1, layer2, attr):
    """Check if the shape of two layer's attributes are equal."""
    attr1 = getattr(layer1, attr, None)
    attr2 = getattr(layer2, attr, None)
    if not attr1:
        return not attr2
    return all(attr1.shape.eval() == attr2.shape.eval())


def check_values(layer1, layer2, attr):
    """Check if the attributes of two layers have the same values."""
    attr1 = getattr(layer1, attr, None)
    attr2 = getattr(layer2, attr, None)
    if not attr1:
        return not attr2
    return numpy.allclose(attr1.eval(), attr2.eval())


def check_layer(layer1, layer2, values=False):
    """Check if two layers are roughly the same."""
    def check(name):
        assert check_shape(layer1, layer2, name)
        if values:
            assert check_values(layer1, layer2, name)

    assert type(layer1) is type(layer2)
    if hasattr(layer1, 'input_shape'):
        assert layer1.input_shape == layer2.input_shape
    if hasattr(layer2, 'output_shape'):
        assert layer1.output_shape == layer2.output_shape
    if isinstance(layer1, (Conv2DLayer, DenseLayer)):
        assert check_shape(layer1, layer2, 'W')
        check('b')
        assert layer1.nonlinearity == layer2.nonlinearity
    if isinstance(layer1, NonlinearityLayer):
        assert layer1.nonlinearity == layer2.nonlinearity
    if isinstance(layer1, BatchNormLayer):
        check('mean')
        check('inv_std')
        check('gamma')
        check('beta')
    if isinstance(layer1, DropoutLayer):
        assert layer1.p == layer2.p
        assert layer1.rescale == layer2.rescale
        assert layer1.shared_axes == layer2.shared_axes
    if isinstance(layer1, ScaleLayer):
        check('scales')
    if isinstance(layer1, BiasLayer):
        check('b')
    if isinstance(layer1, GlobalPoolLayer):
        assert layer1.pool_function is layer2.pool_function
    if isinstance(layer1, Pool2DLayer):
        assert layer1.ignore_border == layer2.ignore_border
        assert layer1.mode == layer2.mode
        assert layer1.pad == layer2.pad
        assert layer1.pool_size == layer2.pool_size
        assert layer1.stride == layer2.stride
    return True


def check_models(models):
    """Check if all models in the list are roughly the same."""
    layers_list = [get_all_layers(m) for m in models]
    n = len(layers_list[0])
    assert all(n == len(l) for l in layers_list)
    for layers in zip(*layers_list):
        first, *rest = layers
        assert all(check_layer(first, c) for c in rest)


def name_func(testcase_func, param_num, param):
    try:
        return '{}_{:03n}_{}'.format(
            testcase_func.__name__,
            param_num,
            parameterized.to_safe_name(
                "_".join(c.__name__ for c in param.args)),
        )
    except AttributeError:
        pass
    try:
        return '{}_{:03n}_{}'.format(
            testcase_func.__name__,
            param_num,
            parameterized.to_safe_name(
                "_".join(c.__name__ for c in param.args[0])),
        )
    except (TypeError, AttributeError):
        pass
    return '{}_{:03n}_{}'.format(
        testcase_func.__name__,
        param_num,
        param.args[0].__name__
    )


# #############################################################################
# ################################### TESTS ###################################
# #############################################################################


@parameterized.expand([
    (BaseResNet, ),
    (ResNet, ),
    (PreResNet, ),
    (PreResNet, ResNet),
    (StochasticDepth, ),
    (StochasticDepth, ResNet),
    (StochasticDepth, PreResNet),
    (StochasticDepth, ResNet, PreResNet),
    (WideResNet, ),
    (WideResNet, ResNet),
    (WideResNet, PreResNet),
    (WideResNet, StochasticDepth),
    (WideResNet, PreResNet, StochasticDepth),
    (WeightedResNet, ),
    (WeightedResNet, ResNet),
    (WeightedResNet, StochasticDepth),
    (WeightedResNet, ResNet, StochasticDepth),
    (WeightedResNet, WideResNet),
    (WeightedResNet, WideResNet, StochasticDepth),
    (WeightedResNet, WideResNet, ResNet),
    (WeightedResNet, WideResNet, StochasticDepth, ResNet),
    (RoR, ),
    (RoR, ResNet),
    (RoR, PreResNet),
    (RoR, StochasticDepth),
    (RoR, WideResNet),
    (RoR, WideResNet, PreResNet),
    (RoR, StochasticDepth, PreResNet),
    (RoR, StochasticDepth, WideResNet),
    (RoR, StochasticDepth, WideResNet, PreResNet),
    (MultiResNet, ),
    (MultiResNet, ResNet),
    (MultiResNet, PreResNet),
    (MultiResNet, WeightedResNet),
    (MultiResNet, StochasticDepth),
    (MultiResNet, StochasticDepth, PreResNet),
    (MultiResNet, StochasticDepth, WeightedResNet),
    (StochasticDepth, MultiResNet),
    (StochasticDepth, MultiResNet, PreResNet),
    (StochasticDepth, MultiResNet, WeightedResNet),
    (MultiResNet, WideResNet),
    (MultiResNet, WideResNet, PreResNet),
], testcase_func_name=name_func)
def test_cls_create(*classes):
    # test if the (builder-) class can be created
    for _ in iter_cls(*classes):
        pass


@parameterized.expand([
    # resnet
    param(BaseResNet, type=tuple('ABC'), bottleneck=(True, False)),
    param(ResNet, n=(3, 5, 9), type=('B', ), bottleneck=(True, False),
          dim_inc_meth=ResNet.dim_inc_methods),
    # pre-resnet
    param(PreResNet, type=tuple('ABC'), bottleneck=(True, False)),
    param(ResNet, PreResNet, type=('B', ), dim_inc_meth=['1x1', '2x2', 'avg'],
          n=(3, 5, 9)),
    # Weighted
    param(WeightedResNet, n=(3, 9), bottleneck=(True, False)),
    # SD
    param(StochasticDepth, type=tuple('AB'), bottleneck=(True, False),
          n=(3, 5, 9), final_prob=(0.2, 0.5), decay=(True, False)),
    param(StochasticDepth, ResNet, type=('B', ), n=(3, 5, 9)),
    param(StochasticDepth, PreResNet, type=tuple('AB'),
          bottleneck=(True, False), n=(3, 9)),
    param(StochasticDepth, WeightedResNet, n=(3, 9), bottleneck=(True, False)),
    # Wide
    param(WideResNet, type=tuple('ABC'), bottleneck=(True, False), n=(3, 5, 9),
          width=(2, 4), dropout=(0, 0.3)),
    param(WideResNet, type=tuple('AB'), bottleneck=(True, ), n=(3, 5),
          width=(2, ), block_config=((1, 3, 1), (1, 3, 3)), dropout=(0, 0.3)),
    param(WideResNet, StochasticDepth, n=(3, 5), width=(2, )),
    param(WideResNet, PreResNet, type=tuple('ABC'), bottleneck=(True, False),
          n=(3, 9), width=(2, )),
    param(WideResNet, StochasticDepth, PreResNet, n=(3, 9), width=(2, )),
    # RoR
    param(RoR, type=('B', 'A', 'BA', ), bottleneck=(True, False),
          shortcut_lvl=(2, ), n=(3, 5, 9)),
    param(RoR, type=('B', 'BA', 'ABB'), bottleneck=(True, False),
          shortcut_lvl=(3, ), n=(3, 9)),
    param(RoR, PreResNet, type=('BA', ), shortcut_lvl=(2, ), n=(3, )),
    param(RoR, StochasticDepth, type=('BA', ), shortcut_lvl=(2, ), n=(3, )),
    param(RoR, StochasticDepth, PreResNet, type=('BA', 'AB'),
          shortcut_lvl=(2, ), n=(3, )),
    param(RoR, WideResNet, type=('BA', ), width=(2, ), shortcut_lvl=(2, ),
          n=(3, )),
    param(RoR, WideResNet, StochasticDepth, PreResNet, type=('BA', ),
          shortcut_lvl=(2, ), n=(3, ), block_config=((1, 3, 1), ),
          width=(2, )),
    # multi-resnet
    param(MultiResNet, type=tuple('ABC'), bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, ResNet, type=('B', ), dim_inc_meth=('avg', ),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, PreResNet, type=tuple('AB'), bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, WeightedResNet, bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, StochasticDepth, multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, StochasticDepth, PreResNet, multiplicity=(2, 4),
          n=(3, 5)),
    param(MultiResNet, StochasticDepth, WeightedResNet, multiplicity=(2, 4),
          n=(3, 5)),
    param(StochasticDepth, MultiResNet, multiplicity=(2, 4), n=(3, 5)),
    param(StochasticDepth, MultiResNet, PreResNet, multiplicity=(2, 4),
          n=(3, 5)),
    param(StochasticDepth, MultiResNet, WeightedResNet, multiplicity=(2, 4),
          n=(3, 5)),
    param(MultiResNet, WideResNet, type=tuple('AB'), bottleneck=(False, True),
          multiplicity=(2, 4), width=(2, ), n=(3, 5)),
    param(MultiResNet, WideResNet, PreResNet, type=tuple('AB'),
          bottleneck=(False, True), multiplicity=(2, 4), width=(2, ),
          n=(3, 5)),
], testcase_func_name=name_func)
def test_model_create(*classes, **kwargs):
    # test if the model can be created
    if len(classes) == 1:
        cls = classes[0]
    else:
        cls = type('_'.join(c.__name__ for c in classes), classes, {})

    for _ in cifar_models(cls, **kwargs):
        pass

    if 'n' in kwargs:
        del kwargs['n']
    if 'bottleneck' in kwargs:
        del kwargs['bottleneck']
    for _ in imagenet_models(cls, **kwargs):
        pass


FRWRD = [
    # resnet + pre-resnet
    param(BaseResNet, bottleneck=True, n=2, type='A'),
    param(BaseResNet, bottleneck=False, n=2, type='A'),
    param(BaseResNet, bottleneck=True, n=2, type='B'),
    param(BaseResNet, bottleneck=False, n=2, type='B'),
    param(BaseResNet, bottleneck=True, n=2, type='C'),
    param(BaseResNet, bottleneck=False, n=2, type='C'),
    param(ResNet, bottleneck=False, n=2, type='B', dim_inc_meth='2x2'),
    param(ResNet, bottleneck=False, n=2, type='B', dim_inc_meth='avg'),
    param(ResNet, bottleneck=True, n=2, type='B', dim_inc_meth='avg'),
    param(PreResNet, bottleneck=True, n=2, type='A'),
    param(PreResNet, bottleneck=False, n=2, type='A'),
    param(PreResNet, bottleneck=True, n=2, type='B'),
    param(PreResNet, bottleneck=False, n=2, type='B'),
    # SD
    param(StochasticDepth, bottleneck=False, n=2),
    param(StochasticDepth, PreResNet, bottleneck=False, n=2),
    param(StochasticDepth, PreResNet, ResNet, bottleneck=False, n=2,
          type='B', dim_inc_meth='avg'),
    # Wide
    param(WideResNet, width=2, bottleneck=False, n=2),
    param(WideResNet, ResNet, width=2, bottleneck=False, n=2,
          dim_inc_meth='avg'),
    param(WideResNet, PreResNet, width=2, bottleneck=False, n=2),
    param(WideResNet, StochasticDepth, width=2, bottleneck=False, n=2),
    param(WideResNet, StochasticDepth, PreResNet, width=2, bottleneck=False,
          n=2),
    # Weighted
    param(WeightedResNet, n=2, bottleneck=False),
    param(StochasticDepth, WeightedResNet, n=2, bottleneck=False),
    # RoR
    param(RoR, StochasticDepth, PreResNet, ResNet, shortcut_lvl=2, n=3,
          bottleneck=False, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, ResNet, shortcut_lvl=3, n=9,
          bottleneck=False, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, ResNet, shortcut_lvl=3, n=9,
          bottleneck=True, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, shortcut_lvl=2, n=3,
          bottleneck=False, type='BA'),
    param(RoR, PreResNet, shortcut_lvl=2, n=3, bottleneck=False, type='BA'),
    param(RoR, StochasticDepth, shortcut_lvl=2, n=3, bottleneck=False,
          type='BA'),
    param(RoR, shortcut_lvl=2, n=3, bottleneck=False, type='BA'),
    # TODO : more test cases
    # TODO : is RoR + Weighted passible
    # multi-resnet
    param(MultiResNet, multiplicity=3, n=3, bottleneck=True, type='A'),
    param(MultiResNet, multiplicity=3, n=3, bottleneck=True, type='B'),
    param(MultiResNet, multiplicity=3, n=3, bottleneck=False, type='A'),
    param(MultiResNet, ResNet, multiplicity=3, n=3, dim_inc_meth='avg',
          type='B'),
    param(MultiResNet, PreResNet, multiplicity=3, n=5, bottleneck=True),
    param(MultiResNet, PreResNet, multiplicity=3, n=5, bottleneck=False),
    param(MultiResNet, WeightedResNet, multiplicity=3, n=3, bottleneck=True),
    param(MultiResNet, WeightedResNet, multiplicity=3, n=3, bottleneck=False),
    param(MultiResNet, StochasticDepth, multiplicity=3, n=3),
    param(MultiResNet, StochasticDepth, PreResNet, multiplicity=3, n=3),
    param(MultiResNet, StochasticDepth, WeightedResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, PreResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, WeightedResNet, multiplicity=3, n=3),
    param(MultiResNet, WideResNet, multiplicity=2, n=3, width=2),
    param(MultiResNet, WideResNet, PreResNet, multiplicity=2, n=3, width=2),
]


@parameterized.expand(FRWRD, testcase_func_name=name_func)
def test_cifar_deterministic(*cls, **kwargs):
    # test a deterministic forward pass
    if len(cls) == 1:
        cls = cls[0]
    else:
        cls = type('_'.join(c.__name__ for c in cls), cls, {})
    model = cls.cifar_model(**kwargs)
    data = floatX(numpy.random.normal(0.0, 1.0, (100, 3, 32, 32)))

    input_var = tensor.tensor4('inputs')
    activation = get_output(model, input_var, deterministic=True)

    func = function([input_var], activation)
    output = func(data)
    del output


@parameterized.expand(FRWRD, testcase_func_name=name_func)
def test_cifar_nondeterministic(*cls, **kwargs):
    # test a non-deterministic forward pass
    if len(cls) == 1:
        cls = cls[0]
    else:
        cls = type('_'.join(c.__name__ for c in cls), cls, {})
    model = cls.cifar_model(**kwargs)
    data = floatX(numpy.random.normal(0.0, 1.0, (100, 3, 32, 32)))

    input_var = tensor.tensor4('inputs')
    activation = get_output(model, input_var, deterministic=False)

    func = function([input_var], activation)
    output = func(data)
    del output


MRO = [
    StochasticDepth,
    RoR,
    KnownDepth,
    MultiResNet,
    WideResNet,
    ResNet,
    WeightedResNet,
    PreResNet,
    BaseResNet,
]


def mro_key(cls):
    return MRO.index(cls)


VALID_MODELS = [
    (BaseResNet, ),
    (PreResNet, ),
    (WeightedResNet, ),
    (ResNet, ),
    (ResNet, PreResNet),
    (WideResNet, ),
    (WideResNet, PreResNet),
    (WideResNet, WeightedResNet),
    (WideResNet, ResNet),
    (WideResNet, ResNet, PreResNet),
    (RoR, ),
    (RoR, PreResNet),
    (RoR, WeightedResNet),
    (RoR, ResNet),
    (RoR, ResNet, PreResNet),
    (RoR, WideResNet),
    (RoR, WideResNet, PreResNet),
    (RoR, WideResNet, WeightedResNet),
    (RoR, WideResNet, ResNet),
    (RoR, WideResNet, ResNet, PreResNet),
    (StochasticDepth, RoR, ),
    (StochasticDepth, RoR, PreResNet),
    (StochasticDepth, RoR, WeightedResNet),
    (StochasticDepth, RoR, ResNet),
    (StochasticDepth, RoR, ResNet, PreResNet),
    (StochasticDepth, RoR, WideResNet),
    (StochasticDepth, RoR, WideResNet, PreResNet),
    (StochasticDepth, RoR, WideResNet, WeightedResNet),
    (StochasticDepth, RoR, WideResNet, ResNet),
    (StochasticDepth, RoR, WideResNet, ResNet, PreResNet),
    (StochasticDepth, MultiResNet),
    (StochasticDepth, MultiResNet, ResNet),
    (StochasticDepth, MultiResNet, PreResNet),
    (StochasticDepth, MultiResNet, ResNet, PreResNet),
    (StochasticDepth, MultiResNet, WideResNet),
    (StochasticDepth, MultiResNet, WideResNet, ResNet),
    (StochasticDepth, MultiResNet, WideResNet, PreResNet),
    (StochasticDepth, MultiResNet, WideResNet, ResNet, PreResNet),
    (StochasticDepth, RoR, MultiResNet),
    (StochasticDepth, RoR, MultiResNet, ResNet),
    (StochasticDepth, RoR, MultiResNet, PreResNet),
    (StochasticDepth, RoR, MultiResNet, WideResNet),
    (StochasticDepth, RoR, MultiResNet, WideResNet, ResNet),
    (StochasticDepth, RoR, MultiResNet, WideResNet, PreResNet),
    (StochasticDepth, RoR, MultiResNet, WideResNet, ResNet, PreResNet),
]


@parameterized.expand(VALID_MODELS, testcase_func_name=name_func)
def test_mro_man(*bases):
    if len(bases) > 1:
        cls = type('_'.join(c.__name__ for c in bases), bases, {})
    else:
        cls = bases[0]
    mro = cls.mro()
    mro = [c for c in mro if c in MRO]
    mro_s = sorted(mro, key=mro_key)
    assert all(c1 is c2 for c1, c2 in zip(mro, mro_s))


CLASSES = (StochasticDepth, RoR, WideResNet, ResNet, BaseResNet, PreResNet,
           WeightedResNet)
ALL_MODELS = list(chain.from_iterable(permutations(CLASSES, i)
                                      for i in range(1, len(CLASSES))))


@parameterized.expand(ALL_MODELS, testcase_func_name=name_func)
def _test_mro_opt(*bases):
    if len(bases) > 1:
        try:
            cls = type('_'.join(c.__name__ for c in bases), bases, {})
        except TypeError:
            skip()
    else:
        cls = bases[0]
    mro = cls.mro()
    mro = [c for c in mro if c in MRO]
    mro_s = sorted(mro, key=mro_key)
    assert all(c1 is c2 for c1, c2 in zip(mro, mro_s))


BASES = (ResNet, PreResNet, WeightedResNet)
BASE_MODELS = list(chain.from_iterable(permutations(BASES, i)
                                       for i in range(1, len(BASES))))


@parameterized.expand(BASE_MODELS, testcase_func_name=name_func)
def _test_mro_base(*bases):
    if len(bases) > 1:
        try:
            cls = type('_'.join(c.__name__ for c in bases), bases, {})
            raise Exception('Expected a TypeError.')
        except TypeError:
            return
    else:
        cls = bases[0]
    mro = cls.mro()
    mro = [c for c in mro if c in MRO]
    mro_s = sorted(mro, key=mro_key)
    assert all(c1 is c2 for c1, c2 in zip(mro, mro_s))


def test_methods_StochasticDepth_PreResNet():
    class Temp(StochasticDepth, PreResNet):
        pass

    assert Temp.shortcut is PreResNet.shortcut
    assert Temp.residual is StochasticDepth.residual
    assert Temp.add_residual_block is PreResNet.add_residual_block


def test_methods_StochasticDepth_WeightedResNet():
    class Temp(StochasticDepth, WeightedResNet):
        pass

    assert Temp.shortcut is WeightedResNet.shortcut
    assert Temp.residual is StochasticDepth.residual
    assert Temp.add_residual_block is WeightedResNet.add_residual_block


# TODO : test PReLuResNet

# #############################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TODO : fix RoR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!! TODO : fix PreResNet + StochasticDepth !!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!! TODO : fix WeightedResNet + StochasticDepth !!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!! TODO : RoR for imagenet_18, 34 and 50, 152 !!!!!!!!!!!!!!!!!
# #############################################################################
