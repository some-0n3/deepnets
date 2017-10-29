from collections import OrderedDict

from lasagne.layers import get_all_layers, get_all_params
from lasagne.regularization import l2, regularize_layer_params
from lasagne.updates import momentum, nesterov_momentum
from theano import tensor

from utils import loss_acc
from utils.training import EpochTrainer, IterationTrainer, Trainer


class BaseTrainer(Trainer):
    """Implement the basic training for most of the residual networks.

    It trains the network with a weight decay of 0.0001 and a
    momentum of 0.9.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to train.
    dataset : a :class:`Dataset` instance or ``None`` (``None``)
        The data set used for training and testing.
    batchsize : integer (``256``)
        The batch size used for training.
    val_batchsize : integer (``500``)
        The batch size used for testing and validation.

    Note: This class implements the training on the CIFAR-10 data set.
    """

    momentum = 0.9
    weight_decay = 0.0001

    def __init__(self, *args, **kwargs):
        super(BaseTrainer, self).__init__(*args, **kwargs)
        self.default_training()

    def default_training(self):
        """Set the training (updates) for this trainer."""
        input_var = tensor.tensor4('inputs')
        target_var = tensor.ivector('targets')
        errors = OrderedDict()

        loss, acc = loss_acc(self.model, input_var, target_var,
                             deterministic=False)
        errors['train_acc'] = acc
        errors['classification error'] = loss
        layers = get_all_layers(self.model)
        decay = regularize_layer_params(layers, l2) * self.weight_decay
        errors['weight decay'] = decay

        loss = loss + decay

        params = get_all_params(self.model, trainable=True)
        updates = self.momentum_method(loss, params, momentum=self.momentum,
                                       learning_rate=self.learning_rate)
        self.set_training(input_var, target_var, loss, updates, values=errors)

    @staticmethod
    def _log_error_(values, valid_err, valid_acc):
        """Print loss and accuracy for training and test set."""
        print("    Training Loss:          {:>10.6f}".format(values[0].mean()))
        print("    Validation Loss:        {:>10.6f}".format(valid_err))
        print("    Validation Accuracy: {:>10.2%}".format(valid_acc))
        print("    Classification Loss:    {:>10.6f}".format(values[2].mean()))
        print("    Weight decay:           {:>10.6f}".format(values[3][-1]))


class ResNetTrainer(BaseTrainer, IterationTrainer):
    """Basic training for original the residual networks."""

    momentum_method = staticmethod(momentum)

    @staticmethod
    def _log_error_(values, valid_err, valid_acc):
        """Print loss and accuracy for training and test set."""
        print("    Training Loss:          {:>10.6f}".format(values[0].mean()))
        print("    Validation Loss:        {:>10.6f}".format(valid_err))
        print("    Validation Accuracy: {:>10.2%}".format(valid_acc))
        print("    Classification Loss:    {:>10.6f}".format(values[2].item()))
        print("    Weight decay:           {:>10.6f}".format(values[3].item()))


class CIFAR_ResNetTrainer(ResNetTrainer):
    """Implement the basic training schedule for the CIFAR image set.

    This is the training schedule for the basic residual network
    as proposed by `He et al. <https://arxiv.org/abs/1512.03385>`_.

    It trains the network for 64,000 iterations with weight decay of
    0.0001 and a momentum of 0.9. The learning rate starts rages from
    0.1 to 0.001.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to train.
    dataset : a :class:`Dataset` instance or ``None`` (``None``)
        The data set used for training and testing.
    batchsize : integer (``128``)
        The batch size used for training.
    val_batchsize : integer (``500``)
        The batch size used for testing and validation.
    """

    def __init__(self, *args, batchsize=128, **kwargs):
        super(CIFAR_ResNetTrainer, self).__init__(
            *args, batchsize=batchsize, **kwargs)

    def train(self, config=None):
        if config is None:
            config = [{'iterations': 32000, 'learning rate': 0.1},
                      {'iterations': 16000, 'learning rate': 0.01},
                      {'iterations': 16000, 'learning rate': 0.001}]
        super(CIFAR_ResNetTrainer, self).train(config)


class WideTrainer(BaseTrainer, EpochTrainer):
    # TODO : doc-string

    momentum_method = staticmethod(nesterov_momentum)
    weight_decay = 0.0005

    def __init__(self, *args, batchsize=128, **kwargs):
        super(WideTrainer, self).__init__(
            *args, batchsize=batchsize, **kwargs)


class CIAFR_WideTrainer(WideTrainer):
    # TODO : doc-string

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 60, 'learning rate': 0.1},
                      {'epochs': 60, 'learning rate': 0.02},
                      {'epochs': 40, 'learning rate': 0.004},
                      {'epochs': 40, 'learning rate': 0.0008}]
        super(CIAFR_WideTrainer, self).train(config)


class SVHN_WideTrainer(WideTrainer):
    # TODO : doc-string

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 80, 'learning rate': 0.01},
                      {'epochs': 40, 'learning rate': 0.001},
                      {'epochs': 40, 'learning rate': 0.0001}]
        super(SVHN_WideTrainer, self).train(config)


class SDTrainer(BaseTrainer, EpochTrainer):
    # TODO : doc-string
    weight_decay = 0.0001
    momentum_method = staticmethod(nesterov_momentum)

    def __init__(self, *args, batchsize=128, **kwargs):
        super(SDTrainer, self).__init__(*args, batchsize=batchsize, **kwargs)


class CIFAR_SDTrainer(SDTrainer):
    # TODO : doc-string

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 250, 'learning rate': 0.1},
                      {'epochs': 125, 'learning rate': 0.01},
                      {'epochs': 125, 'learning rate': 0.001}]
        super(CIFAR_SDTrainer, self).train(config)


class SVHN_SDTrainer(SDTrainer):
    # TODO : doc-string

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 30, 'learning rate': 0.1},
                      {'epochs':  5, 'learning rate': 0.01},
                      {'epochs': 15, 'learning rate': 0.001}]
        super(SVHN_SDTrainer, self).train(config)


class WeightedTrainer(ResNetTrainer):
    # TODO : doc-string

    # BUG : the weights are not learned correctly
    #       for small networks n \in [1,3] (all) the weights lay
    #       outside of [-1, 1], for larger networks this is only
    #       true for some weight ... but still ...

    def default_training(self):
        input_var = tensor.tensor4('inputs')
        target_var = tensor.ivector('targets')
        loss, _ = loss_acc(self.model, input_var, target_var,
                           deterministic=False)
        loss += regularize_layer_params(get_all_layers(self.model), l2,
                                        tags={'regularizable': True,
                                              'layer_weight': False}) * 1e-4
        # TODO : does this count as weight decay (...*1e-4) or not?
        # the learning rate is 1/100 of the normal learning rate
        # ... but we just adapt the decay
        loss += regularize_layer_params(get_all_layers(self.model), l2,
                                        tags={'layer_weight': True}) * 1e-6
        params = get_all_params(self.model, trainable=True)
        # updates = adam(loss, params, learning_rate=self.learning_rate)
        updates = self.momentum_method(loss, params, momentum=self.momentum,
                                       learning_rate=self.learning_rate)
        for weight in get_all_params(self.model, trainable=True,
                                     tags={'layer_weight': True}):
            # all residual weights are in [-1, 1]
            assert weight in updates
            updates[weight] = tensor.minimum(
                1.0, tensor.maximum(-1.0, updates[weight]))
        self.set_training(input_var, target_var, loss, updates)


class CIFAR_WeightedTrainer(WeightedTrainer, CIFAR_ResNetTrainer):
    # TODO : doc-string
    pass


class DenseNetTrainer(BaseTrainer, EpochTrainer):
    # TODO : doc-string

    momentum_method = staticmethod(nesterov_momentum)


class CIFAR_DenseNetTrainer(DenseNetTrainer):
    # TODO : doc-string

    def __init__(self, *args, batchsize=64, val_batchsize=100, **kwargs):
        super(CIFAR_DenseNetTrainer, self).__init__(
            *args, batchsize=batchsize, val_batchsize=val_batchsize, **kwargs)

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 150, 'learning rate': 0.1},
                      {'epochs':  75, 'learning rate': 0.01},
                      {'epochs':  75, 'learning rate': 0.001}]
        super(CIFAR_DenseNetTrainer, self).train(config)


class SVHN_DenseNetTrainer(DenseNetTrainer):
    # TODO : doc-string

    def __init__(self, *args, batchsize=64, val_batchsize=100, **kwargs):
        super(SVHN_DenseNetTrainer, self).__init__(
            *args, batchsize=batchsize, val_batchsize=val_batchsize, **kwargs)

    def train(self, config=None):
        if config is None:
            config = [{'epochs': 20, 'learning rate': 0.1},
                      {'epochs': 10, 'learning rate': 0.01},
                      {'epochs': 10, 'learning rate': 0.001}]
        super(SVHN_DenseNetTrainer, self).train(config)
