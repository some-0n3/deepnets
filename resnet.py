"""Residual Neural Networks.

This module implements multiple residual neural networks mainly for
image classification.
The module provides builder classes to conveniently create deep
residual networks. Those builder come with methods to create the models
used by He et al. for the CIFAR and ImageNet data sets.

The module provides network architectures from the following papers:

* The original ResNet from `Deep Residual Learning for Image Recognition
  <https://arxiv.org/abs/1512.03385>`_.
* The version with a pre-activation order from `Identity Mappings in
  Deep Residual Networks <https://arxiv.org/abs/1603.05027>`_, which
  moves the batch normalization and nonlinearity to improve the
  information flow (backwards) through the network.
* The stochastic depth version, where some residual blocks are dropped
  to decrease the effective depth of the network while training, from
  `Deep Networks with Stochastic Depth
  <https://arxiv.org/abs/1603.09382>`_.
* The wide residual network approach that increased the number of
  filters in all residual blocks as described in `Wide Residual Networks
  <https://arxiv.org/abs/1605.07146>`_.
* The weighted residual approach from `Weighted Residuals for Very Deep
  Networks <https://arxiv.org/abs/1605.08831>`_ that adds learned weights
  for every residual block.
* The "resnet of resnet" approach that added more skip connections to
  the network as proposed in `Residual Networks of Residual Networks:
  Multilevel Residual Networks <https://arxiv.org/abs/1608.02908>`_.
* The "Multi-Residual Networks" approach by Abdi et al. from the paper
  `Multi-Residual Networks: Improving the Speed and Accuracy of Residual
  Networks <https://arxiv.org/abs/1609.05672>`_. They increased the
  number of residuals per block by a constant factor.


Please note that all builder classes use the (first) basic ResNets
as basis. To get a ResNet of ResNet with stochastic depth and the
pre-activation order use something like the following code:

>>> class DummyClass(RoR, StochasticDepth, PreResNet):
>>>     pass
>>> 
>>> model = DummyClass.cifar_model(n=27, bottleneck=True, m=3)
"""
import numpy
from lasagne.init import Constant, HeNormal
from lasagne.layers import BatchNormLayer, Conv2DLayer, DenseLayer,\
    DropoutLayer, ElemwiseSumLayer, ExpressionLayer, GlobalPoolLayer,\
    InputLayer, MaxPool2DLayer, NonlinearityLayer, PadLayer,\
    ParametricRectifierLayer, Pool2DLayer, ScaleLayer
from lasagne.nonlinearities import rectify, softmax


__all__ = ('ResNet', 'PreResNet', 'WeightedResNet', 'StochasticDepth',
           'WideResNet', 'RoR', 'MultiResNet', 'PReLuResNet')


def get_dimensions(incoming, num_filters=None, dim_inc=False,
                   bottleneck=False):
    """Figure out the filter sizes and stride."""
    assert not (bottleneck and num_filters is None)

    in_filters = getattr(incoming, 'output_shape', incoming)[1]
    if num_filters is None:
        num_filters = in_filters * 2 if dim_inc else in_filters
    stride = (2, 2) if dim_inc else (1, 1)
    out_filters = num_filters * 4 if bottleneck else num_filters
    return num_filters, stride, out_filters


class BaseResNet(object):
    """Basic (Original) ResNet as in `He et al.
    <https://arxiv.org/abs/1512.03385>`_.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    type : ``'A'``, ``'B'`` or ``'C'`` (``'A'``)
        The type of shortcuts to use.
    bottleneck : boolean (``False``)
        Use the 3 layer bottleneck approach if ``True``.
    """

    def __init__(self, incoming, type='A', bottleneck=False):
        self.model = incoming
        self.type = type
        self.bottleneck = bottleneck

        self.residuals = []
        self.shortcuts = []
        self.resblocks = []

    @staticmethod
    def convolution(model, num_filters, filter_size=(3, 3), stride=(1, 1),
                    init_gain='relu', pad='same'):
        """Standard convolution method."""
        return BatchNormLayer(Conv2DLayer(
            model, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=None, pad=pad,
            W=HeNormal(gain=init_gain), b=None, flip_filters=False
        ))

    @staticmethod
    def nonlinearity(incoming):
        """Apply the standard nonlinearity."""
        return NonlinearityLayer(incoming, nonlinearity=rectify)

    def projection(self, model, num_filters, stride=(1, 1), init_gain=1.0,
                   **kwargs):
        """Standard projection method."""
        return self.convolution(model, num_filters, (1, 1), stride=stride,
                                init_gain=init_gain, **kwargs)

    def residual(self, model, num_filters=None, dim_inc=False):
        """Return a residual block."""
        num_filters, first_stride, out_filters = get_dimensions(
            model, num_filters, dim_inc, self.bottleneck)

        if self.bottleneck:
            residual = self.convolution(model, num_filters, filter_size=(1, 1),
                                        stride=first_stride)
            residual = self.nonlinearity(residual)
            residual = self.convolution(residual, num_filters)
            residual = self.nonlinearity(residual)
            residual = self.convolution(residual, out_filters,
                                        filter_size=(1, 1))
        else:
            residual = self.convolution(model, num_filters,
                                        stride=first_stride)
            residual = self.nonlinearity(residual)
            residual = self.convolution(residual, num_filters)
        return residual

    def shortcut(self, incoming, residual, type=None):
        """Create a shortcut from ``incoming`` to ``residual``."""
        type = type or self.type
        in_shape = getattr(incoming, 'output_shape', incoming)
        out_shape = getattr(residual, 'output_shape', residual)
        in_filters = in_shape[1]
        out_filters = out_shape[1]
        stride = (in_shape[-2] // out_shape[-2], in_shape[-1] // out_shape[-1])

        if type == 'C':
            # all shortcuts are projections
            return self.projection(incoming, out_filters, stride=stride)
        elif in_filters == out_filters:
            # A and B use identity shortcuts (if the dimensions stay)
            return incoming
        elif type == 'B':
            # if dimensions increase, B uses projections
            return self.projection(incoming, out_filters, stride=stride)
        elif type == 'A':
            shortcut = ExpressionLayer(
                incoming, lambda x: x[:, :, ::stride[0], ::stride[1]],
                in_shape[:2] + out_shape[2:])
            side = (out_filters - in_filters) // 2
            return PadLayer(shortcut, [side, 0, 0], batch_ndim=1)

    def add_residual_block(self, num_filters=None, dim_inc=False):
        """Add a residual block (residual + shortcut) to the network."""
        residual = self.residual(self.model, num_filters, dim_inc)
        self.residuals.append(residual)

        shortcut = self.shortcut(self.model, residual)
        self.shortcuts.append(shortcut)

        self.model = self.nonlinearity(ElemwiseSumLayer([residual, shortcut]))
        self.resblocks.append(self.model)

    @classmethod
    def cifar_model(cls, n=9, incoming=None, classes=10, **kwargs):
        """Create model for the CIFAR data set like in section 4.2.

        Parameters
        ----------
        n : integer (``9``)
            A parameter used to govern the size of the network as
            described in the paper.
        incoming :  a :class:`Layer` instance or ``None``
            The input layer, if it is ``None`` a new one will be created.
        classes : integer (`10``)
            The number of classes to train, usually ``10`` or ``100``.
        kwargs : key-word arguments
            The key-word arguments that get passed down to the constructor.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        model = incoming or InputLayer(shape=(None, 3, 32, 32))
        builder = cls(model, **kwargs)

        # first layer, output is 16 x 32 x 32
        model = builder.convolution(model, 16, init_gain=1.0)
        model = builder.nonlinearity(model)
        builder.model = model

        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            builder.add_residual_block(16)

        # second stack of residual blocks, output is 32 x 16 x 16
        builder.add_residual_block(32, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(32)

        # third stack of residual blocks, output is 64 x 8 x 8
        builder.add_residual_block(64, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(64)

        # average pooling
        model = GlobalPoolLayer(builder.model)

        # fully connected layer
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def image_net_model(cls, config, incoming=None, classes=1000, **kwargs):
        """Create a network for the ImageNet data set.

        This is a helper function to create multiple networks like the
        ones in the paper. What specific model is used is determined by
        the ``config`` parameter. This parameter is a list of tuples
        (``num_blocks``, ``num_filters``). For each tuple there will be
        ``num_blocks`` residual block created with ``num_filters``
        channels at each layer (except the bottleneck layers).
        Except for the first tuple in the configuration, the dimensions
        are increased at the beginning of every entry in ``config``.

        Parameters
        ----------
        config : list of integer pairs
            The configuration as a list of tuples.
        incoming :  a :class:`Layer` instance or ``None`` (``None``)
            The input layer, if it is ``None`` a new one will be created.
        classes : integer (``1000``)
            The number of classes to train, usually ``1000`'.
        kwargs : key-word arguments
            The key-word arguments that get passed down to the constructor.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        # staring block
        model = incoming or InputLayer(shape=(None, 3, 224, 224))
        builder = cls(model, **kwargs)

        model = builder.convolution(model, 64, filter_size=(7, 7),
                                    stride=(2, 2), init_gain=1.0)
        model = builder.nonlinearity(model)
        model = MaxPool2DLayer(model, pool_size=(3, 3), stride=(2, 2),
                               ignore_border=False)
        builder.model = model

        config = iter(config)

        # no increasing dimensions on the first chuck
        n, num_filters = next(config)
        for _ in range(n):
            builder.add_residual_block(num_filters)

        # add other residual blocks
        for n, num_filters in config:
            builder.add_residual_block(num_filters, dim_inc=True)
            for _ in range(1, n):
                builder.add_residual_block(num_filters)

        # final part of the network
        model = Pool2DLayer(builder.model, pool_size=(7, 7),
                            stride=(1, 1), mode='average_exc_pad',
                            ignore_border=False)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def imagenet_18(cls, bottleneck=False, **kwargs):
        """The 18 layer network for the ImageNet data set."""
        return cls.image_net_model([(2, 64), (2, 128), (2, 256), (2, 512), ],
                                   bottleneck=bottleneck, **kwargs)

    @classmethod
    def imagenet_34(cls, bottleneck=False, **kwargs):
        """The 34 layer network for the ImageNet data set."""
        return cls.image_net_model([(3, 64), (4, 128), (6, 256), (3, 512), ],
                                   bottleneck=bottleneck, **kwargs)

    @classmethod
    def imagenet_50(cls, bottleneck=True, **kwargs):
        """The 50 layer network for the ImageNet data set."""
        return cls.image_net_model([(3, 64), (4, 128), (6, 256), (3, 512), ],
                                   bottleneck=bottleneck, **kwargs)

    @classmethod
    def imagenet_101(cls, bottleneck=True, **kwargs):
        """The 101 layer network for the ImageNet data set."""
        return cls.image_net_model([(3, 64), (4, 128), (23, 256), (3, 512), ],
                                   bottleneck=bottleneck, **kwargs)

    @classmethod
    def imagenet_152(cls, bottleneck=True, **kwargs):
        """The 152 layer network for the ImageNet data set."""
        return cls.image_net_model([(3, 64), (8, 128), (36, 256), (3, 512), ],
                                   bottleneck=bottleneck, **kwargs)


class PreResNet(BaseResNet):
    """The revisited version of the ResNet (PreResNet).

    This is the version of a residual network as described in
    `Identity Mappings in Deep Residual Networks by He et al.
    <https://arxiv.org/abs/1603.05027>`_. It uses the identity mapping
    as skip connections and after-addition activation.
    This improved generalization and made is possible to train a
    network with a depth of over 1000 layer.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    type : ``'A'``, ``'B'`` or ``'C'`` (``'A'``)
        The type of shortcuts to use.
    bottleneck : boolean (``False``)
        Use the 3 layer bottleneck approach if ``True``.
    """

    def convolution(self, *args, **kwargs):
        result = super(PreResNet, self).convolution(*args, **kwargs)
        assert isinstance(result, BatchNormLayer)
        return result.input_layer

    def residual(self, model, num_filters=None, dim_inc=False):
        num_filters, first_stride, out_filters = get_dimensions(
            model, num_filters, dim_inc, self.bottleneck)

        residual = self.nonlinearity(BatchNormLayer(model))
        if self.bottleneck:
            residual = self.convolution(residual, num_filters,
                                        filter_size=(1, 1),
                                        stride=first_stride)
            residual = self.nonlinearity(BatchNormLayer(residual))
            residual = self.convolution(residual, num_filters)
            residual = self.nonlinearity(BatchNormLayer(residual))
            residual = self.convolution(residual, out_filters,
                                        filter_size=(1, 1))
        else:
            residual = self.convolution(residual, num_filters,
                                        stride=first_stride)
            residual = self.nonlinearity(BatchNormLayer(residual))
            residual = self.convolution(residual, num_filters)
        return residual

    def add_residual_block(self, num_filters=None, dim_inc=False):
        residual = self.residual(self.model, num_filters, dim_inc)
        self.residuals.append(residual)

        shortcut = self.shortcut(self.model, residual)
        self.shortcuts.append(shortcut)

        self.model = ElemwiseSumLayer([residual, shortcut])
        self.resblocks.append(self.model)

    @classmethod
    def cifar_model(cls, n=9, incoming=None, classes=10, **kwargs):
        model = incoming or InputLayer(shape=(None, 3, 32, 32))
        builder = cls(model, **kwargs)

        # first layer, output is 16 x 32 x 32
        builder.model = builder.convolution(model, 16, init_gain=1.0)

        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            builder.add_residual_block(16)

        # second stack of residual blocks, output is 32 x 16 x 16
        builder.add_residual_block(32, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(32)

        # third stack of residual blocks, output is 64 x 8 x 8
        builder.add_residual_block(64, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(64)

        model = builder.nonlinearity(BatchNormLayer(builder.model))

        # average pooling
        model = GlobalPoolLayer(model)

        # fully connected layer
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def image_net_model(cls, config, incoming=None, classes=1000, **kwargs):
        # staring block
        model = incoming or InputLayer(shape=(None, 3, 224, 224))
        builder = cls(model, **kwargs)

        model = builder.convolution(model, 64, filter_size=(7, 7),
                                    stride=(2, 2), init_gain=1.0)
        model = MaxPool2DLayer(model, pool_size=(3, 3), stride=(2, 2),
                               ignore_border=False)
        builder.model = model

        config = iter(config)

        # no increasing dimensions on the first chuck
        n, num_filters = next(config)
        for _ in range(n):
            builder.add_residual_block(num_filters)

        # add other residual blocks
        for n, num_filters in config:
            builder.add_residual_block(num_filters, dim_inc=True)
            for _ in range(1, n):
                builder.add_residual_block(num_filters)

        model = builder.nonlinearity(BatchNormLayer(builder.model))

        # final part of the network
        model = Pool2DLayer(model, pool_size=(7, 7),
                            stride=(1, 1), mode='average_exc_pad',
                            ignore_border=False)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model


class WeightedResNet(BaseResNet):
    """The residual network with weighted residual blocks.

    It follows `Weighted Residuals for Very Deep Networks by Shen et al.
    <https://arxiv.org/abs/1605.08831>`_. The network scales every
    residual block with a weight between -1 and 1. It also uses a
    different way of increasing the dimensions in the network.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    bottleneck : boolean (``False``)
        Use the 3 layer bottleneck approach if ``True``.

    Note: The ``type`` parameter was removed due to the change in
    handling the increasing dimensions.
    """

    def __init__(self, incoming, bottleneck=False):
        super(WeightedResNet, self).__init__(incoming, type=None,
                                             bottleneck=bottleneck)
        self.weights = []

    def shortcut(self, incoming, num_filters=None, dim_inc=False):
        num_filters, stride, out_filters = get_dimensions(
            self.model, num_filters, dim_inc, self.bottleneck)
        if not dim_inc:
            return incoming
        return self.nonlinearity(self.convolution(
            self.model, out_filters, stride=stride))

    def residual(self, model, num_filters=None, dim_inc=False):
        residual = super(WeightedResNet, self).residual(
            model, num_filters=num_filters, dim_inc=dim_inc)
        residual = self.nonlinearity(residual)
        self.residuals.append(residual)
        shared_axes = tuple(range(len(residual.output_shape)))
        residual = ScaleLayer(residual, Constant(0), shared_axes=shared_axes)
        residual.params[residual.scales].add('layer_weight')
        self.weights.append(residual)
        return residual

    def add_residual_block(self, num_filters=None, dim_inc=False):
        shortcut = self.shortcut(self.model, num_filters=num_filters,
                                 dim_inc=dim_inc)
        self.shortcuts.append(shortcut)
        residual = self.residual(shortcut, num_filters, dim_inc=False)
        self.model = ElemwiseSumLayer([residual, shortcut])
        self.resblocks.append(self.model)

    @classmethod
    def cifar_model(cls, n=9, incoming=None, classes=10, **kwargs):
        """Create model for the CIFAR data set like in section 4.2.

        Parameters
        ----------
        n : integer (``9``)
            A parameter used to govern the size of the network as
            described in the paper.
        incoming :  a :class:`Layer` instance or ``None``
            The input layer, if it is ``None`` a new one will be created.
        classes : integer (`10``)
            The number of classes to train, usually ``10`` or ``100``.
        kwargs : key-word arguments
            The key-word arguments that get passed down to the constructor.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        model = incoming or InputLayer(shape=(None, 3, 32, 32))
        builder = cls(model, **kwargs)

        num_filters = 64 if builder.bottleneck else 16
        model = builder.convolution(model, num_filters, init_gain=1.0)
        model = builder.nonlinearity(model)
        builder.model = model
        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            builder.add_residual_block(16)

        # second stack of residual blocks, output is 32 x 16 x 16
        builder.add_residual_block(32, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(32)

        # third stack of residual blocks, output is 64 x 8 x 8
        builder.add_residual_block(64, dim_inc=True)
        for _ in range(1, n):
            builder.add_residual_block(64)

        # average pooling
        model = GlobalPoolLayer(builder.model)

        # fully connected layer
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def image_net_model(cls, config, incoming=None, classes=1000, **kwargs):
        """Create a network for the ImageNet data set.

        This is a helper function to create multiple networks like the
        ones in the paper. What specific model is used is determined by
        the ``config`` parameter. This parameter is a list of tuples
        (``num_blocks``, ``num_filters``). For each tuple there will be
        ``num_blocks`` residual block created with ``num_filters``
        channels at each layer (except the bottleneck layers).
        Except for the first tuple in the configuration, the dimensions
        are increased at the beginning of every entry in ``config``.

        Parameters
        ----------
        config : list of integer pairs
            The configuration as a list of tuples.
        incoming :  a :class:`Layer` instance or ``None`` (``None``)
            The input layer, if it is ``None`` a new one will be created.
        classes : integer (``1000``)
            The number of classes to train, usually ``1000`'.
        kwargs : key-word arguments
            The key-word arguments that get passed down to the constructor.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        # staring block
        model = incoming or InputLayer(shape=(None, 3, 224, 224))
        builder = cls(model, **kwargs)

        num_filters = 256 if builder.bottleneck else 64
        model = builder.convolution(model, num_filters, filter_size=(7, 7),
                                    stride=(2, 2), init_gain=1.0)
        model = builder.nonlinearity(model)
        model = MaxPool2DLayer(model, pool_size=(3, 3), stride=(2, 2),
                               ignore_border=False)
        builder.model = model

        config = iter(config)

        # no increasing dimensions on the first chuck
        n, num_filters = next(config)
        for _ in range(n):
            builder.add_residual_block(num_filters)

        # add other residual blocks
        for n, num_filters in config:
            builder.add_residual_block(num_filters, dim_inc=True)
            for _ in range(1, n):
                builder.add_residual_block(num_filters)

        # final part of the network
        model = Pool2DLayer(builder.model, pool_size=(7, 7),
                            stride=(1, 1), mode='average_exc_pad',
                            ignore_border=False)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model


class WideResNet(BaseResNet):
    """The residual network with wider residual blocks.

    The network increases the number of channels of every residual block
    in the network by a factor ``k``. The approach is described by
    Zagoruyko et al. in the paper `Wide Residual Networks
    <https://arxiv.org/abs/1605.07146>`_.
    They also introduced dropout between the convolutional layers in the
    residual blocks and a mechanism to vary the convulsions in a
    residual block.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    type : ``'A'``, ``'B'`` or ``'C'`` (``'A'``)
        The type of shortcuts to use.
    bottleneck : boolean (``False``)
        Use the 3 layer bottleneck approach if ``True``.
    width : integer (`10``)
        The growth rate as described by the parameter $k$ in the paper.
        The network will have ``width`` times as many filters in every
        layer (except the first and last ones) as the original approach.
    block_config : list of integers, integers pairs or ``None``.
        The list of convolutions in a residual block. The list describes
        the filter size of each convolution in the block. The filter
        size can be given as a tuple if integers or as a single integer,
        in which case a quadratic filter is used. If ``None`` it will be
        either ``(3, 3)`` or ``(1, 3, 1)`` depending on bottlenecking.
    dropout : a scalar between 0 and 1 (``0.3``)
        The dropout probability for each dropout layer. In case the
        dropout probability is ``0`` no dropout is applied.

    Note: Please make sure :class:`WideResNet` is before
    :class:`BaseResNet` and :class:`PreResNet` in the MRO. This class
    does not (yet) work together with :class:`WeightedResNet`.
    """

    def __init__(self, *args, width=10, block_config=None, dropout=0.3,
                 bottleneck=False, **kwargs):
        super(WideResNet, self).__init__(*args, bottleneck=bottleneck,
                                         **kwargs)
        self.width = width
        if block_config is None:
            block_config = (1, 3, 1) if bottleneck else (3, 3)
        self.block_config = [b if isinstance(b, tuple) else (b, b)
                             for b in block_config]
        self.dropout = dropout
        if bottleneck and len(block_config) < 3:
            raise ValueError('In case of bottlenecking, a residual block'
                             ' should have at least three layers.')

    def _base_residual_(self, model, num_filters=None, dim_inc=False):
        """Basic residual block (not pre-activation resnet)."""
        num_filters *= self.width
        num_filters, first_stride, out_filters = get_dimensions(
            model, num_filters, dim_inc, self.bottleneck)

        residual = self.convolution(model, num_filters, stride=first_stride,
                                    filter_size=self.block_config[0])
        if self.bottleneck:
            for filter_size in self.block_config[1:-1]:
                residual = self.nonlinearity(residual)
                if self.dropout > 0:
                    residual = DropoutLayer(residual, self.dropout)
                residual = self.convolution(residual, num_filters,
                                            filter_size=filter_size)
            residual = self.nonlinearity(residual)
            if self.dropout > 0:
                residual = DropoutLayer(residual, self.dropout)
            residual = self.convolution(residual, out_filters,
                                        filter_size=self.block_config[-1])
        else:
            for filter_size in self.block_config[1:]:
                residual = self.nonlinearity(residual)
                if self.dropout > 0:
                    residual = DropoutLayer(residual, self.dropout)
                residual = self.convolution(residual, num_filters,
                                            filter_size=filter_size)
        return residual

    def _pre_residual_(self, model, num_filters=None, dim_inc=False):
        """Residual block for pre-resnet."""
        num_filters *= self.width
        num_filters, first_stride, out_filters = get_dimensions(
            model, num_filters, dim_inc, self.bottleneck)

        residual = self.nonlinearity(BatchNormLayer(model))
        residual = self.convolution(residual, num_filters, stride=first_stride,
                                    filter_size=self.block_config[0])
        if self.bottleneck:
            for filter_size in self.block_config[1:-1]:
                residual = self.nonlinearity(BatchNormLayer(residual))
                if self.dropout > 0:
                    residual = DropoutLayer(residual, self.dropout)
                residual = self.convolution(residual, num_filters,
                                            filter_size=filter_size)
            residual = self.nonlinearity(BatchNormLayer(residual))
            if self.dropout > 0:
                residual = DropoutLayer(residual, self.dropout)
            residual = self.convolution(residual, out_filters,
                                        filter_size=self.block_config[-1])
        else:
            for filter_size in self.block_config[1:]:
                residual = self.nonlinearity(BatchNormLayer(residual))
                if self.dropout > 0:
                    residual = DropoutLayer(residual, self.dropout)
                residual = self.convolution(residual, num_filters,
                                            filter_size=filter_size)
        return residual

    def residual(self, *args, **kwargs):
        if isinstance(self, PreResNet):
            return self._pre_residual_(*args, **kwargs)
        return self._base_residual_(*args, **kwargs)


class ResNet(BaseResNet):
    """Basic (Original) ResNet as in He et al. with some extras.

    This class implements some additional approaches to deal with
    increasing the dimensions.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    type : ``'A'``, ``'B'`` or ``'C'`` (``'A'``)
        The type of shortcuts to use.
    bottleneck : boolean (``False``)
        Use the 3 layer bottleneck approach if ``True``.
    dim_inc_meth : ``'1x1'``, ``'2x2'``, ``'avg'``, ``'max'`` or ``'sum'``
        The type of projection method to use for increasing the
        dimensions. This parameter will be ignored if the type of
        shortcut is ``'A'`` because it uses padding. For the types
        ``'B'`` and ``'C'`` there are multiple ways to change the
        dimensions:

        ``'1x1'`` will perform a 1x1 convolution with a 2x2 stride. This
        is the original approach from the paper, but will discard 3/4 of
        the input.

        ``'2x2'`` will perform a 2x2 convolution with a 2x2 stride, which
        will increase the amount of free parameters, but won't discard
        any inputs.

        ``'sum'``, ``'avg'``, ``'max'`` will perform a 1x1 convolution
        (with a 1x1 stride) followed by either sum, average or max
        pooling respectively. This will not lead to any increase in free
        parameters and (hopefully) not discard any important information.

        Note: In case of a RoR like structure the ``'2x2'`` method may
        also perform a ``'4x4'`` or ``'8x8'`` convolution.

    Note: Please make sure this class is before :class:`BaseResNet` and
    :class:`PreResNet` in the MRO.
    """

    dim_inc_methods = ('1x1', '2x2', 'max', 'avg', 'sum')

    def __init__(self, *args, dim_inc_meth='1x1', **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

        if dim_inc_meth not in self.dim_inc_methods:
            raise ValueError(
                'The method "{}" is not supported.'.format(dim_inc_meth))
        self.pooling = dim_inc_meth in {'max', 'avg', 'sum'}
        if dim_inc_meth == 'avg':
            dim_inc_meth = 'average_exc_pad'
        self.dim_inc = dim_inc_meth

    def projection(self, model, num_filters, stride=(1, 1)):
        if stride == (1, 1):
            return super(ResNet, self).projection(model, num_filters,
                                                  stride=stride)
        filter_size = 1, 1
        pad = 'same'
        if self.dim_inc == '2x2':
            filter_size = stride
            pad = 'valid'
        elif self.pooling:
            oldstride = stride
            stride = 1, 1

        model = self.convolution(model, num_filters, filter_size,
                                 stride=stride, pad=pad, init_gain=1.0)
        if self.pooling:
            model = Pool2DLayer(model, oldstride, ignore_border=False,
                                mode=self.dim_inc)
        return model


class PReLuResNet(BaseResNet):
    """A ResNet with the parametric rectifier as nonlinearity.

    This builder replaces the normal rectifying nonlinearity with the
    parametric one.

    Parameters
    ----------
    args : arguments
        The arguments that gets passed down to the super class.
    alpha : scalar (``0.25``)
        The initial value for the parameter of the parametric rectifier.
    kwargs : key-word arguments
        The key-word arguments that gets passed down to the super class.
    """

    def __init__(self, *args, alpha=0.25, **kwargs):
        super(PReLuResNet, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.init_gain = numpy.sqrt(2 / (1 + alpha**2))

    def convolution(self, model, *args, init_gain=None, **kwargs):
        if init_gain is None:
            init_gain = self.init_gain
        return super(PReLuResNet, self).convolution(
            model, *args, init_gain=init_gain, **kwargs)

    def nonlinearity(self, incoming):
        return ParametricRectifierLayer(incoming, alpha=Constant(self.alpha))


def drop_layer(incoming, *args, **kwargs):
    """Convenience function for dropping a whole layer."""
    ndim = len(getattr(incoming, 'output_shape', incoming))
    kwargs['shared_axes'] = tuple(range(ndim))
    return DropoutLayer(incoming, *args, **kwargs)


class KnownDepth(BaseResNet):
    """A Network that has a predefined depth.

    The depth is here described by the number of residual blocks in
    the network.
    """

    def __init__(self, *args, length=None, **kwargs):
        super(KnownDepth, self).__init__(*args, **kwargs)
        self.length = length

    @classmethod
    def cifar_model(cls, *args, n=9, **kwargs):
        return super(KnownDepth, cls).cifar_model(
            *args, n=n, length=(3 * n), **kwargs)

    @classmethod
    def image_net_model(cls, config, **kwargs):
        config = tuple(config)
        length = sum(n for n, _ in config)
        return super(KnownDepth, cls).image_net_model(
            config, length=length, **kwargs)


class StochasticDepth(KnownDepth):
    """A residual network with stochastic depth.

    It implements the approach from `Deep Networks with Stochastic Depth
    by Huang et al. <https://arxiv.org/abs/1603.09382>`_. The network
    reduces it's effective depth by randomly dropping some residual
    blocks.

    Parameters
    ----------
    args : arguments
        The arguments that gets passed down to the super class.
    length : integer (``None``)
        The total number of residual blocks.
    final_prob : float in [0, 1] (``0.5``)
        The final survival probability.
    decay : boolean (``True``)
        If ``True``  the decay approach is used, otherwise all survival
        probabilities are constantly ``p_L`` for all layers.
    rescale : boolean (``True``)
        If ``True``, the output of the residual block will be scaled by
        ``1 / p_l`` while training (when ``deterministic=False``).
    kwargs : key-word arguments
        The key-word arguments that gets passed down to the super class.

    Note 1: Huang et al. just dropped the residual block in training and
    scaled the output while testing. Like lasagne's ``DropoutLayer``
    we rescale the output while training and leave it unchanged in the
    test phase.

    Note 2: Please make sure this class is before :class:`BaseResNet`,
    :class:`PreResNet`, :class:`WeightedResNet` and most other classes
    in the MRO.
    """

    def __init__(self, *args, length=None, final_prob=0.5, decay=True,
                 rescale=True, **kwargs):
        if decay and length is None:
            raise ValueError('If decay is to use specify the length.')
        super(StochasticDepth, self).__init__(*args, **kwargs, length=length)
        self.final_prob = final_prob
        self.decay = decay
        self.rescale = rescale

    @property
    def current_prob(self):
        """The survival probability for the current block."""
        if not self.decay:
            return self.final_prob
        return (len(self.resblocks) + 1) / self.length * (1 - self.final_prob)

    def residual(self, *args, **kwargs):
        residual = super(StochasticDepth, self).residual(*args, **kwargs)
        return drop_layer(residual, p=self.current_prob, rescale=self.rescale)


class RoR(KnownDepth):
    """The Residual Networks or Residual Networks.

    This class implements the "ResNet or ResNet" approach described in
    the paper `Residual Networks of Residual Networks: Multilevel
    Residual Networks <https://arxiv.org/abs/1608.02908>`_ by Zhang et
    al. It adds extra shortcuts from other (previous) residual blocks.
    It thereby creates shortcuts between residual blocks that are
    further away.

    Parameters
    ----------
    incoming : instance of :class:`Layer`
        The network's input layer.
    sortcut_lvl : integer (``3``)
        The shortcut level number to determine the shortcuts. If this
        is ``1`` it will yield the original (underling) resnet approach.
        If it is ``2`` it  will add a shortcut connection from the root
        (the input of the first residual block) to the last one. For
        higher values it will create mid-level shortcut connection.
    type : a sequence of ``'A'``, ``'B'`` and ``'C'`` (``'ABB'``)
        The types for the shortcuts to use. The sequence should contain
        one type for each shortcut level. The first element describes
        the type for last shortcut level (the normal shortcut connections
        from the original resnet). The last type in the sequence is used
        for the  first level (the connection from the input to the
        output). If the sequence would be too short it will be filled
        with the last type in the sequence. So passing ``type='B'`` will
        use shortcuts of type B for all connections.
    length : integer (``None``)
        The total number of residual blocks.
    splits : tuple of integers (``(3, )``)
        In case ``m`` is bigger than 2 this class will add mid-level
        shortcut connections. While doing so it will try to group all
        residual blocks into an number of sections (usually it tries
        to build and connect 3 groups of residual blocks). This
        parameter let you define different numbers of groups. For example
        ``(4, 3)`` will first try to separate the residual blocks into
        4 groups and if that does not work it will try to separate them
        into 3 groups. Shortcut connections are then added to those
        groups.
    kwargs : key-word arguments
        The key-word arguments that get passed down to the super class.

    Note: Please make sure this class is before :class:`BaseResNet`,
    :class:`PreResNet` and :class:`WideResNet` in the MRO. This class
    does not (yet) work together with :class:`WeightedResNet`.
    """

    def __init__(self, *args, shortcut_lvl=3, type='ABB', length=None,
                 splits=(3, ), **kwargs):
        # in the paper every level is split in three sublevels
        # via splits you can also try to split it into four or five
        # parts if the number of residual block are dividable
        if len(type) > shortcut_lvl:
            raise ValueError(
                'Too many types for {} shortcusts: "{}".'.format(shortcut_lvl,
                                                                 type))
        if len(type) < shortcut_lvl:
            type += type[-1:] * (shortcut_lvl - len(type))
        super(RoR, self).__init__(*args, **kwargs, type=type[0], length=length)
        self.shortcut_lvl = shortcut_lvl
        self.othertypes = type[1:]
        self.root = None

        # translate the shortcut level number into a list of "steps"
        # every "step" residual blocks we will add a shortcut connection
        breakpoints = []
        last = length
        if shortcut_lvl > 1:
            breakpoints.append(last)
        for _ in range(shortcut_lvl - 2):
            try:
                floats = [last / s for s in splits]
                last = next(i for i, f in zip((int(i) for i in floats), floats)
                            if i == f)
                if last <= 1:
                    raise ValueError(
                        'Your network is to shallow to have such a high level'
                        ' number {} (length={}).'.format(shortcut_lvl, length))
                breakpoints.append(last)
            except StopIteration:
                raise ValueError(
                    'Cannot figure out the shortcuts with length '
                    '{} and level number {}.'.format(length, shortcut_lvl))
        self.steps = breakpoints[::-1]
        assert len(self.steps) == len(self.othertypes)

    def add_residual_block(self, *args, **kwargs):
        # set the root (if not done before)
        if self.root is None:
            self.root = self.model
        # add a residual block
        super(RoR, self).add_residual_block(*args, kwargs)

        # get the layer with the element wise sum
        model = self.model
        last = None
        while not isinstance(model, ElemwiseSumLayer):
            last = model
            model = model.input_layer
        layers = []

        # create shortcut from all underling layers
        current = len(self.resblocks)
        for typ, step in zip(self.othertypes, self.steps):
            if not current % step:
                i = current - step - 1
                layer = self.resblocks[i] if i >= 0 else self.root
                layers.append(self.shortcut(layer, model.output_shape, typ))
            else:
                break

        # apply the changes to the network (if any)
        if not layers:
            return

        model = ElemwiseSumLayer(model.input_layers + layers)
        if last is not None:
            last.input_layer = model
        else:
            self.model = model
            self.resblocks[-1] = model


class MultiResNet(BaseResNet):
    """The Multi-Residual Networks.

    This class implements the "Multi-Residual Network" approach that
    increases the number of residuals per block by a constant factor
    ``m``. So the  output of a residual is
    $x_1 = x_0 + f_1(x_0) + ... + f_m(x_0)$. The approach is described
    by Abdi et al. in the paper `Multi-Residual Networks: Improving the
    Speed and Accuracy of Residual Networks
    <https://arxiv.org/abs/1609.05672>`_.

    Parameters
    ----------
    args : arguments
        The arguments that gets passed down to the super class.
    multiplicity : integer (``1``)
        The number of residuals per block.
    kwargs : key-word arguments
        The key-word arguments that get passed down to the super class.

    Note 1: Please make sure this class is before :class:`BaseResNet`,
    :class:`PreResNet` :class:`WideResNet` or :class:`WeightedResNet`
    in the MRO.
    Note 2: If this class is mixed with :class:`StochasticDepth` the
    order of those to classes is crucial. If :class:`StochasticDepth`
    is called first all residuals in a block share the same random
    variable and are thereby all dropped simultaneously. If
    :class:`MultiResNet` if first some residuals in a block might be
    dropped while some are not.
    """

    def __init__(self, *args, multiplicity=1, **kwargs):
        super(MultiResNet, self).__init__(*args, **kwargs)
        self.multiplicity = multiplicity

    def residual(self, *args, **kwargs):
        residuals = [super(MultiResNet, self).residual(*args, **kwargs)
                     for _ in range(self.multiplicity)]
        return ElemwiseSumLayer(residuals)
