Collection of some Deepnets
===========================

This repository offers builder classes to create densenets [8]_ and
multiple types of resnets [1]_, [2]_, [3]_, [4]_, [5]_, [6]_,  [7]_ via the
`lasagne <https://github.com/Lasagne/Lasagne>`_ framework. The packages offers
classes to build the CIFAR, SVHN, ImageNet and other models. Combining (resnet)
approaches is done via class inheritance. To create a multi-residual with
pre-activation order for the CIFAR10 dataset use something like the code below.

.. code-block:: python

    from deepnets import MultiResNet, PreResnet

    class Model(MultiResNet, PreResNet):
         pass

    model = Model.cifar_model(n=18, multiplicity=4, type='B')


Please note that by default all classes like ``RoR`` or ``StochasticDepth``
will use the original resnet (not the pre-activation order).

To build a network for a CIFAR or SVHN model use the class method
``cifar_model``. For the ImageNet models there are multiple class methods for
the standard models (e.g. ``imagenet_50`` for resnets or ``imagenet_169``
for densenet). There is also a method for a more parametric model creation and
the class can also be used to create custom networks.


ResNets Classes
---------------

``BaseResNet``
    implements the original resnet as in [1]_.

``PreResNet``
    implements the pre-activation order version from [2]_.

``WeightedResNet``
    is to build a network with weighted residuals as in [3]_.

``ResNet``
    uses a slightly different projection for (original and
    pre-activation) resnets, when the shortcut type is ``'B'`` or ``'C'`` and
    the dimensions increase. Similar to the original it uses a 1x1 convolution,
    but performs a pooling operation before that. This will thereby use all the
    inputs (not just 1 out of 4 pixels) when changing dimensions.

``StochasticDepth``
    will create a resnet (orig., pre, or weighted) with a stochastic depth
    similar to [4]_. The difference in this implementation is that the output
    of a residual is scaled in training rather than while testing.

``WideResNet``
    implements the wide approach as described by [5]_. This class works (thus
    far) not together with ``WeightedResNet``.

``PReluResNet``
    replaces the ReLU nonlinearity with a parametric rectifying unit.

``RoR``
    create (original and pre-activation) resnets with multilevel shortcut as
    described in [6]_.

``MultiResNet``
    creates multi-residual (original, pre-activation or weighted) networks as
    described in [7]_. Those networks have multiple residuals added to the
    shortcuts.


Densenets Classes
-----------------

``DenseNet``
    implements the densenet approach by [8]_.

``MyDenseNet``
    if no compression is done the network will skip the convolution (and also
    the normalization, scaling and nonlinearity) in the transition.


References
----------

.. [1] He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian (2015)
       Deep Residual Learning for Image Recognition. arXiv eprint arXiv:1512.03385
.. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)
       Identity Mappings in Deep Residual Networks. arXiv eprint arXiv:1603.05027
.. [3] Shen, Falong; Zeng, Gang (2016)
       Weighted Residuals for Very Deep Networks. arXiv eprint arXiv:1605.08831
.. [4] Huang, Gao; Sun, Yu; Liu, Zhuang; Sedra, Daniel; Weinberger, Kilian (2016)
       Deep Networks with Stochastic Depth. arXiv eprint arXiv:1603.09382
.. [5] Zagoruyko, Sergey; Komodakis, Nikos (2016)
       Wide Residual Networks. arXiv eprint arXiv:1605.07146
.. [6] Zhang, Ke; Sun, Miao; Han, Tony X.; Yuan, Xingfang; Guo, Liru; Liu, Tao (2016)
       Residual Networks of Residual Networks: Multilevel Residual Networks. arXiv eprint arXiv:1608.02908
.. [7] Abdi, Masoud; Nahavandi, Saeid (2016)
       Multi-Residual Networks: Improving the Speed and Accuracy of Residual Networks. arXiv eprint arXiv:1609.05672
.. [8] Huang, Gao; Liu, Zhuang; Weinberger, Kilian Q.; van der Maaten, Laurens (2016)
       Densely Connected Convolutional Networks. arXiv eprint arXiv:1608.06993
