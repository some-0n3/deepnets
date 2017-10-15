import sys
from collections import OrderedDict
from itertools import chain, permutations, product

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
from theano import function, tensor

from .resnet import (ResNet, PreResNet, WeightedResNet, StochasticDepth, RoR,
                     MultiResNet)


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
        if 'width' in kwargs and kwargs['width'] > 1:
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
    (ResNet, ),
    (PreResNet, ),
    (PreResNet, ResNet),
    (StochasticDepth, ),
    (StochasticDepth, PreResNet),
    (WeightedResNet, ),
    (WeightedResNet, StochasticDepth),
    (RoR, ResNet),
    (RoR, PreResNet),
    (RoR, StochasticDepth),
    (RoR, StochasticDepth, PreResNet),
    (MultiResNet, ),
    (MultiResNet, PreResNet),
    (MultiResNet, WeightedResNet),
    (MultiResNet, StochasticDepth),
    (MultiResNet, StochasticDepth, PreResNet),
    (MultiResNet, StochasticDepth, WeightedResNet),
    (StochasticDepth, MultiResNet),
    (StochasticDepth, MultiResNet, PreResNet),
    (StochasticDepth, MultiResNet, WeightedResNet),
], testcase_func_name=name_func)
def test_cls_create(*classes):
    # test if the (builder-) class can be created
    # TODO : some were suppost to crash ...
    for _ in iter_cls(*classes):
        pass


@parameterized.expand([
    # basic
    param(ResNet, type='ABC', bottleneck=(True, False), n=(3, 5)),
    param(ResNet, type='B', bottleneck=(True, False),
          dim_inc_meth=ResNet.dim_inc_methods, n=(3, 5)),
    param(PreResNet, type='ABC', bottleneck=(True, False), n=(3, 5)),
    param(WeightedResNet, bottleneck=(True, False), n=(3, 9)),
    # Wide
    param(ResNet, type='ABC', bottleneck=(True, False), n=(3, 5),
          width=(2, )),
    param(ResNet, type='B', bottleneck=(True, False),
          dim_inc_meth=ResNet.dim_inc_methods, width=(2, )),
    param(PreResNet, type='ABC', bottleneck=(True, False), n=(3, 5),
          width=(2, )),
    param(PreResNet, type='B', bottleneck=(True, False),
          dim_inc_meth=ResNet.dim_inc_methods, width=(2, )),
    param(WeightedResNet, bottleneck=(True, False), n=(3, 9), width=(2, )),
    # SD
    param(StochasticDepth, ResNet, bottleneck=(True, False),
          final_prob=(0.5, 0.3), decay=(True, False), n=(3, 5, 9)),
    param(StochasticDepth, PreResNet, type='AB', bottleneck=(True, False),
          n=(3, 5, 9)),
    param(StochasticDepth, WeightedResNet, bottleneck=(True, False), n=(3, 9)),
    # RoR
    param(RoR, ResNet, type=('B', 'A', 'BA', ), bottleneck=(True, False),
          shortcut_lvl=(2, ), n=(3, 5, 9)),
    param(RoR, ResNet, type=('B', 'BA', 'ABB'), bottleneck=(True, False),
          shortcut_lvl=(3, ), n=(3, 9)),
    param(RoR, PreResNet, type=('BA', ), bottleneck=(True, False),
          shortcut_lvl=(2, ), n=(3, )),
    param(RoR, StochasticDepth, ResNet, type=('BA', ),
          bottleneck=(True, False), shortcut_lvl=(2, ), n=(3, )),
    param(RoR, StochasticDepth, PreResNet, type=('BA', ),
          bottleneck=(True, False), n=(3, ), shortcut_lvl=(2, )),
    param(RoR, ResNet, type=('B', ), bottleneck=(True, False),
          shortcut_lvl=(2, ), n=(3, 5), dim_inc_meth=ResNet.dim_inc_methods),
    # TODO : RoR + WeightedResNet is not (yet) supported
    # param((RoR, WeightedResNet), m=(2, ), n=(3, )),
    # param((RoR, StochasticDepth, WeightedResNet), m=(2, ), n=(3, )),
    # multi-resnet
    param(MultiResNet, ResNet, type='ABC', bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, ResNet, type='B', bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5), dim_inc_meth=('avg', )),
    param(MultiResNet, PreResNet, type='ABC', bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, WeightedResNet, bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, StochasticDepth, ResNet, type='ABC',
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, StochasticDepth, PreResNet, type='ABC',
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, StochasticDepth, WeightedResNet,
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(StochasticDepth, MultiResNet, ResNet, type='ABC',
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(StochasticDepth, MultiResNet, PreResNet, type='ABC',
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(StochasticDepth, MultiResNet, WeightedResNet,
          bottleneck=(True, False), multiplicity=(2, 4), n=(3, 5)),
    param(MultiResNet, ResNet, type='ABC', bottleneck=(True, False),
          multiplicity=(2, 4), n=(3, 5), width=(2, )),
    param(MultiResNet, PreResNet, type='ABC', multiplicity=(2, 4), n=(3, 5),
          width=(2, )),
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
    # resnet-bases
    param(ResNet, bottleneck=True, n=2, type='A'),
    param(ResNet, bottleneck=False, n=2, type='A'),
    param(ResNet, bottleneck=True, n=2, type='B'),
    param(ResNet, bottleneck=False, n=2, type='B'),
    param(ResNet, bottleneck=True, n=2, type='C'),
    param(ResNet, bottleneck=False, n=2, type='C'),
    param(ResNet, bottleneck=False, n=2, type='B', dim_inc_meth='2x2'),
    param(ResNet, bottleneck=False, n=2, type='B', dim_inc_meth='avg'),
    param(ResNet, bottleneck=True, n=2, type='B', dim_inc_meth='avg'),
    param(PreResNet, bottleneck=True, n=2, type='A'),
    param(PreResNet, bottleneck=False, n=2, type='A'),
    param(PreResNet, bottleneck=True, n=2, type='B'),
    param(PreResNet, bottleneck=False, n=2, type='B'),
    param(WeightedResNet, n=2, bottleneck=False),
    param(WeightedResNet, n=2, bottleneck=True),
    # SD
    param(StochasticDepth, ResNet, bottleneck=False, n=2),
    param(StochasticDepth, PreResNet, bottleneck=False, n=2),
    param(StochasticDepth, PreResNet, bottleneck=False, n=2,
          type='B', dim_inc_meth='avg'),
    param(StochasticDepth, WeightedResNet, n=2, bottleneck=False),
    # Wide
    param(ResNet, n=2, width=2, bottleneck=False),
    param(ResNet, n=2, width=2, bottleneck=False, dim_inc_meth='avg'),
    param(PreResNet, n=2, width=2, bottleneck=False),
    param(WeightedResNet, n=2, width=2, bottleneck=False),
    param(StochasticDepth, ResNet, n=2, width=2, bottleneck=False),
    param(StochasticDepth, PreResNet, n=2, width=2, bottleneck=False),
    param(StochasticDepth, WeightedResNet, n=2, width=2, bottleneck=False),
    # RoR
    param(RoR, StochasticDepth, PreResNet, shortcut_lvl=2, n=3,
          bottleneck=False, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, shortcut_lvl=3, n=9,
          bottleneck=False, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, shortcut_lvl=3, n=9,
          bottleneck=True, type='B', dim_inc_meth='avg'),
    param(RoR, StochasticDepth, PreResNet, shortcut_lvl=2, n=3,
          bottleneck=False, type='BA'),
    param(RoR, PreResNet, shortcut_lvl=2, n=3, bottleneck=False, type='BA'),
    param(RoR, StochasticDepth, ResNet, shortcut_lvl=2, n=3, bottleneck=False,
          type='BA'),
    param(RoR, ResNet, shortcut_lvl=2, n=3, bottleneck=False, type='BA'),
    # TODO : more test cases
    # TODO : is RoR + Weighted passible
    # multi-resnet
    param(MultiResNet, ResNet, multiplicity=3, n=3, bottleneck=True, type='A'),
    param(MultiResNet, ResNet, multiplicity=3, n=3, bottleneck=True, type='B'),
    param(MultiResNet, ResNet, multiplicity=3, n=3, bottleneck=False,
          type='A'),
    param(MultiResNet, ResNet, multiplicity=3, n=3, dim_inc_meth='avg',
          type='B'),
    param(MultiResNet, PreResNet, multiplicity=3, n=5, bottleneck=True),
    param(MultiResNet, PreResNet, multiplicity=3, n=5, bottleneck=False),
    param(MultiResNet, WeightedResNet, multiplicity=3, n=3, bottleneck=True),
    param(MultiResNet, WeightedResNet, multiplicity=3, n=3, bottleneck=False),
    param(MultiResNet, StochasticDepth, ResNet, multiplicity=3, n=3),
    param(MultiResNet, StochasticDepth, PreResNet, multiplicity=3, n=3),
    param(MultiResNet, StochasticDepth, WeightedResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, ResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, PreResNet, multiplicity=3, n=3),
    param(StochasticDepth, MultiResNet, WeightedResNet, multiplicity=3, n=3),
    param(MultiResNet, ResNet, multiplicity=2, n=3, width=2),
    param(MultiResNet, PreResNet, multiplicity=2, n=3, width=2),
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


def test_methods_StochasticDepth_ResNet():
    # manually test the methods
    class Temp(StochasticDepth, ResNet):
        pass

    assert Temp.shortcut is ResNet.shortcut
    assert Temp.residual is StochasticDepth.residual
    assert Temp.add_residual_block is ResNet.add_residual_block


def test_methods_StochasticDepth_PreResNet():
    # manually test the methods
    class Temp(StochasticDepth, PreResNet):
        pass

    assert Temp.shortcut is PreResNet.shortcut
    assert Temp.residual is StochasticDepth.residual
    assert Temp.add_residual_block is PreResNet.add_residual_block


def test_methods_StochasticDepth_WeightedResNet():
    # manually test the methods
    class Temp(StochasticDepth, WeightedResNet):
        pass

    assert Temp.residual is StochasticDepth.residual
    assert Temp.shortcut is WeightedResNet.shortcut
    assert Temp.add_residual_block is WeightedResNet.add_residual_block


MRO = [
    StochasticDepth,
    RoR,
    MultiResNet,
    WeightedResNet,
    PreResNet,
    ResNet,
]


def mro_key(cls):
    """Return the index from ``MRO``."""
    return MRO.index(cls)


VALID_MODELS = [
    (ResNet, ),
    (PreResNet, ),
    (WeightedResNet, ),
    (RoR, ResNet),
    (RoR, PreResNet),
    (RoR, WeightedResNet),
    (StochasticDepth, ResNet),
    (StochasticDepth, PreResNet),
    (StochasticDepth, WeightedResNet),
    (StochasticDepth, RoR, ResNet),
    (StochasticDepth, RoR, PreResNet),
    (StochasticDepth, RoR, WeightedResNet),
    (StochasticDepth, MultiResNet),
    (StochasticDepth, MultiResNet, PreResNet),
    (StochasticDepth, RoR, MultiResNet),
    (StochasticDepth, RoR, MultiResNet, PreResNet),
]


@parameterized.expand(VALID_MODELS, testcase_func_name=name_func)
def test_mro_man(*bases):
    # those classes should have a valid mro
    if len(bases) > 1:
        cls = type('_'.join(c.__name__ for c in bases), bases, {})
    else:
        cls = bases[0]
    mro = cls.mro()
    mro = [c for c in mro if c in MRO]
    mro_s = sorted(mro, key=mro_key)
    assert all(c1 is c2 for c1, c2 in zip(mro, mro_s))


CLASSES = (StochasticDepth, RoR, ResNet, PreResNet, WeightedResNet)
ALL_MODELS = list(chain.from_iterable(permutations(CLASSES, i)
                                      for i in range(1, len(CLASSES))))


@parameterized.expand(ALL_MODELS, testcase_func_name=name_func)
def _test_mro_opt(*bases):
    # TODO : test if all models give of a valid mro
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
    # TODO : mixin different basic types should cause TypeError
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


# TODO : test PReLuResNet

# #############################################################################
# !!!!!!!!!!!!!!!!! TODO : fix PreResNet + StochasticDepth !!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!! TODO : fix WeightedResNet + StochasticDepth !!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!! TODO : RoR for imagenet_18, 34 and 50, 152 !!!!!!!!!!!!!!!!!
# #############################################################################
