"""This module provides implementations for densely connected networks.

This module provides a builder class to create multiple densenets as
described by `Huang et al. <https://arxiv.org/abs/1608.06993>`_. The
builder can construct the models for the CIFAR, SVHN and ImageNet
datasets. It also provides a densenet architecture that skips the
convolution in a transition if no compression is done.
"""
from lasagne.init import HeNormal
from lasagne.layers import BatchNormLayer, BiasLayer, ConcatLayer,\
    Conv2DLayer, DenseLayer, DropoutLayer, GlobalPoolLayer, InputLayer,\
    MaxPool2DLayer, NonlinearityLayer, Pool2DLayer, ScaleLayer
from lasagne.layers.noise import dropout_channels
from lasagne.nonlinearities import rectify, softmax


class DenseNet(object):
    r"""Implements the original densenet approach.

    This class provides building functions to create densenet as
    described by Huang et al. in the paper ` Densely Connected
    Convolutional Networks <https://arxiv.org/abs/1608.06993>`_. The
    class can create models for the CIFAR, SVHN and ImageNet datasets.
    If can also be used to create custom densenets for any other dataset.

    Parameters
    ----------
    incoming : a :class:`Layer` instance
        The input layer for the densenet.
    growth : integer (``40``)
        The growth factor of the densenet. This is equivalent to the
        parameter $k$ in the paper.
    bottleneck : boolean (``True``)
        If ``True`` the two layer bottleneck approach is used.
    neck_size : integer or ``None`` (``None``)
        In case of bottlenecking the first layer will return
       ``neck_size`` number of channels. If it is ``None`` the neck size
       will be set to four times the growth rate ($4 \cdot k$).
    compression : number (float) in [0, 1] (``1``)
        The compression level that is used, it is equal to the parameter
        $\theta$ in the paper. If it is ``1`` no compression will be
        done, if it is ``0.5`` the transition will output half as many
        channels as it's input.
    dropout : number (float) in [0, 1] (``0``)
        the dropout probability, If it is ``0`` no  dropout is performed.
    """

    def __init__(self, incoming, growth=40, bottleneck=True, neck_size=None,
                 compression=1, dropout=0):
        self.current = incoming
        self.growth = growth
        self.bottleneck = bottleneck
        if bottleneck:
            self.neck_size = neck_size or 4 * growth
        self.compression = compression
        self.p_drop = dropout

    @staticmethod
    def convolution(incoming, num_filters, filter_size=(3, 3), stride=(1, 1),
                    init_gain='relu'):
        """Standard convolution method."""
        return Conv2DLayer(
            incoming, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=None, pad='same',
            W=HeNormal(gain=init_gain), b=None, flip_filters=False
        )

    @staticmethod
    def batchnorm_pt1(incoming):
        """1st part of batch normalization: normalization."""
        return BatchNormLayer(incoming, beta=None, gamma=None)

    @staticmethod
    def batchnorm_pt2(incoming):
        """2nd part of batch normalization: scaling + biases."""
        return BiasLayer(ScaleLayer(incoming))

    @staticmethod
    def nonlinearity(incoming):
        """Apply the nonlinearity to the current layer."""
        return NonlinearityLayer(incoming, nonlinearity=rectify)

    def dropout(self, incoming):
        """Apply dropout to the current layer."""
        if self.p_drop:
            return DropoutLayer(incoming, p=self.p_drop)
        return incoming

    def dense_layer(self):
        """Add a dense layer (incl. bottleneck layer)."""
        model = self.batchnorm_pt2(self.current)
        model = self.nonlinearity(model)

        if self.bottleneck:
            model = self.convolution(model, self.neck_size,
                                     filter_size=(1, 1))
            model = self.dropout(model)
            model = BatchNormLayer(model)
            model = self.nonlinearity(model)

        model = self.convolution(model, self.growth)
        model = self.dropout(model)
        model = self.batchnorm_pt1(model)
        self.current = ConcatLayer([self.current, model], axis=1)

    def add_dense_block(self, layers):
        """Add a block of multiple dense layers."""
        for _ in range(layers):
            self.dense_layer()

    def transition(self):
        """Add a transition."""
        model = self.batchnorm_pt2(self.current)
        model = self.nonlinearity(model)
        model = self.convolution(
            model, int(self.current.output_shape[1] * self.compression),
            filter_size=(1, 1))
        model = self.dropout(model)
        model = Pool2DLayer(model, 2, mode='average_inc_pad')
        self.current = self.batchnorm_pt1(model)

    @classmethod
    def cifar_model(cls, n=31, incoming=None, classes=10, **kwargs):
        r"""Create a densenet for the CIFAR or SVHN datasets.

        Parameters
        ----------
        n : integer (``31``)
            A parameter to control the length of the network.
        classes :integer (``10``)
            Number of classes, usually ``10`` or ``100``.
        incoming : a :class:`Layer` instance or ``None``
            The input layer for the densenet, if ``None`` a new layer
            will be created.
        growth : integer (``40``)
            The growth factor of the densenet. This is equivalent to the
            parameter $k$ in the paper.
        bottleneck : boolean (``True``)
            If ``True`` the two layer bottleneck approach is used.
        neck_size : integer or ``None`` (``None``)
            In case of bottlenecking the first layer will return
            ``neck_size`` number of channels. If it is ``None`` the neck
            size will be set to four times the growth rate ($4 \cdot k$).
        compression : number (float) in [0, 1] (``1``)
            The compression level that is used, it is equal to the
            parameter $\theta$ in the paper. If it is ``1`` no
            compression will be done, if it is ``0.5`` the transition
            will output half as many channels as it's input.
        dropout : number (float) in [0, 1] (``0``)
            The dropout probability, If it is ``0`` no  dropout is
            performed.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        model = incoming or InputLayer(shape=(None, 3, 32, 32))
        builder = cls(model, **kwargs)

        if builder.bottleneck and builder.compression < 1:
            model = builder.convolution(model, 2 * builder.growth,
                                        init_gain=1.0)
        else:

            model = builder.convolution(model, 16, init_gain=1.0)
        builder.current = builder.batchnorm_pt1(model)

        builder.add_dense_block(n)
        builder.transition()
        builder.add_dense_block(n)
        builder.transition()
        builder.add_dense_block(n)

        model = builder.batchnorm_pt2(builder.current)
        model = builder.nonlinearity(model)

        model = GlobalPoolLayer(model)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def image_net_model(cls, config, incoming=None, bottleneck=True,
                        classes=1000, **kwargs):
        r"""Create a densenet for the ImageNet dataset.

        Parameters
        ----------
        config : list of integers
            For each number in  this list the network will have as many
            dense layers followed by a transition.
        incoming : a :class:`Layer` instance or ``None`` (``None``)
            The input layer for the densenet, if ``None`` a new layer
            will be created.
        classes :integer (``1000``)
            Number of classes, usually ``1000``.
        growth : integer
            The growth factor of the densenet. This is equivalent to the
            parameter $k$ in the paper.
        bottleneck : boolean (``True``)
            If ``True`` the two layer bottleneck approach is used.
        neck_size : integer or ``None`` (``None``)
            In case of bottlenecking the first layer will return
            ``neck_size`` number of channels. If it is ``None`` the neck
            size will be set to four times the growth rate ($4 \cdot k$).
        compression : number (float) in [0, 1] (``1``)
            The compression level that is used, it is equal to the
            parameter $\theta$ in the paper. If it is ``1`` no
            compression will be done, if it is ``0.5`` the transition
            will output half as many channels as it's input.
        dropout : number (float) in [0, 1] (``0``)
            The dropout probability, If it is ``0`` no  dropout is
            performed.

        Returns
        -------
        a :class:`DenseLayer` instance
            The model in form of its last layer.
        """
        model = incoming or InputLayer(shape=(None, 3, 224, 224))
        builder = cls(model, bottleneck=bottleneck, **kwargs)

        model = builder.convolution(builder.current, 2 * builder.growth,
                                    filter_size=(7, 7), stride=(2, 2),
                                    init_gain=1.0)
        model = BatchNormLayer(model)
        model = builder.nonlinearity(model)
        model = MaxPool2DLayer(model, pool_size=(3, 3), stride=(2, 2),
                               ignore_border=False)
        builder.current = builder.batchnorm_pt1(model)

        for n in config[:-1]:
            builder.add_dense_block(n)
            builder.transition()
        builder.add_dense_block(config[-1])

        model = builder.batchnorm_pt2(builder.current)
        model = builder.nonlinearity(model)

        model = Pool2DLayer(model, pool_size=(7, 7), stride=(1, 1),
                            mode='average_exc_pad', ignore_border=False)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model

    @classmethod
    def imagenet_121(cls, growth=32, **kwargs):
        """The 121 layer network for the ImageNet data set."""
        return cls.image_net_model([6, 12, 24, 16], growth=growth, **kwargs)

    @classmethod
    def imagenet_169(cls, growth=32, **kwargs):
        """The 169 layer network for the ImageNet data set."""
        return cls.image_net_model([6, 12, 32, 32], growth=growth, **kwargs)

    @classmethod
    def imagenet_201(cls, growth=32, **kwargs):
        """The 201 layer network for the ImageNet data set."""
        return cls.image_net_model([6, 12, 48, 32], growth=growth, **kwargs)

    @classmethod
    def imagenet_161(cls, growth=48, **kwargs):
        """The 161 layer network for the ImageNet data set."""
        return cls.image_net_model([6, 12, 36, 24], growth=growth, **kwargs)


class MyDenseNet(DenseNet):
    r"""A slightly different version of the densenet.

    This builder differs from the original implementation
    (:class:`DenseNet`) in the following regards:
    * If there is no compression the transition will not perform a
      convolution (or batch normalization, ...). The transition is then
      without parameters.
    * It performs spacial dropout.
    * The first nonlinearity in a ImageNet-model is skipped (to make it
      more similar to the pre-activation order resnet).

    Parameters
    ----------
    incoming : a :class:`Layer` instance
        The input layer for the densenet.
    growth : integer (``40``)
        The growth factor of the densenet. This is equivalent to the
        parameter $k$ in the paper.
    bottleneck : boolean (``True``)
        If ``True`` the two layer bottleneck approach is used.
    neck_size : integer or ``None`` (``None``)
        In case of bottlenecking the first layer will return
       ``neck_size`` number of channels. If it is ``None`` the neck size
       will be set to four times the growth rate ($4 \cdot k$).
    compression : number (float) in [0, 1] (``1``)
        The compression level that is used, it is equal to the parameter
        $\theta$ in the paper. If it is ``1`` no compression will be
        done, if it is ``0.5`` the transition will output half as many
        channels as it's input.
    dropout : number (float) in [0, 1] (``0``)
        the dropout probability, If it is ``0`` no  dropout is performed.
    """

    def dropout(self, incoming):
        if self.p_drop:
            return dropout_channels(incoming, p=self.p_drop)
        return incoming

    def transition(self):
        if self.compression < 1:
            super(MyDenseNet, self).transition()
        else:
            self.current = Pool2DLayer(self.current, 2, mode='average_inc_pad')

    @classmethod
    def image_net_model(cls, config, incoming=None, bottleneck=True,
                        classes=1000, **kwargs):
        model = incoming or InputLayer(shape=(None, 3, 224, 224))
        builder = cls(model, **kwargs)

        model = builder.convolution(builder.current, 2 * builder.growth,
                                    filter_size=(7, 7), stride=(2, 2),
                                    init_gain=1.0)
        # don't do batchnorm + nonlinearity
        builder.current = MaxPool2DLayer(model, pool_size=(3, 3),
                                         stride=(2, 2), ignore_border=False)
        builder.current = builder.batchnorm_pt1(model)

        for n in config[:-1]:
            builder.add_dense_block(n)
            builder.transition()
        builder.add_dense_block(config[-1])

        model = builder.batchnorm_pt2(builder.current)
        model = builder.nonlinearity(model)

        model = Pool2DLayer(model, pool_size=(7, 7), stride=(1, 1),
                            mode='average_exc_pad', ignore_border=False)
        model = DenseLayer(model, num_units=classes, W=HeNormal(gain='relu'),
                           nonlinearity=softmax)
        return model
