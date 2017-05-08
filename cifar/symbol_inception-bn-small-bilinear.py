import sys
sys.path.insert(0, "../mxnet/python/")
import mxnet as mx

eps = 1e-10 + 1e-5
fix_gamma = False


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", bn_mom=0.9):
    conv = mx.symbol.Convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(
        data=conv, eps=eps, fix_gamma=fix_gamma, momentum=bn_mom)
    act = mx.symbol.Activation(data=bn, act_type=act_type)
    return act


# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(
        2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(
        3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat


# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, act_type='relu'):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(
        1, 1), pad=(0, 0), num_filter=ch_1x1, act_type=act_type)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(
        3, 3), pad=(1, 1), num_filter=ch_3x3, act_type=act_type)
    # concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat


def get_symbol(num_classes=10, args=None):
    data = mx.symbol.Variable(name="data")
    conv1 = ConvFactory(data=data, kernel=(3, 3), pad=(
        1, 1), num_filter=96, act_type="relu")
    in3a = SimpleFactory(conv1, 32, 32)
    in3b = SimpleFactory(in3a, 32, 48)
    in3c = DownsampleFactory(in3b, 80)
    in4a = SimpleFactory(in3c, 112, 48)
    in4b = SimpleFactory(in4a, 96, 64)
    in4c = SimpleFactory(in4b, 80, 80)
    in4d = SimpleFactory(in4c, 48, 96)
    in4e = DownsampleFactory(in4d, 96)
    in5a = SimpleFactory(in4e, 176, 160)
    in5b = SimpleFactory(in5a, 176, 160)
    #pool = mx.symbol.Pooling(data=in5b, pool_type="avg",
    #                         kernel=(8, 8), name="global_pool")
    #flatten = mx.symbol.Flatten(data=pool, name="flatten")
    #fc = mx.symbol.FullyConnected(
    #    data=flatten, num_hidden=num_classes, name="fc1")

    reshape = mx.symbol.Reshape(
        data=in5b, shape=(-1, 336, 8*8),
        name='cifar_reshape')
    reshape1 = mx.symbol.SwapAxis(
        data=reshape, dim1=1, dim2=2, name='cifar_swap')

    #implement bilinear layers (refer to 'Bilinear CNN Models for Fine-grained Visual Recognition')
    bilinear = mx.symbol.batch_dot(reshape, reshape1, name='cifar_batchdot')
    bilinear = mx.symbol.Reshape(data=bilinear, shape=(-1, 336*336))

    sym_abs = mx.symbol.abs(bilinear + 1e-5)
    sym_sqrt = mx.symbol.sqrt(sym_abs)
    sym_sign = mx.symbol.sign(bilinear)
    ssqrt = sym_sign * sym_sqrt

    l2 = mx.symbol.L2Normalization(data=ssqrt, name='cifar_l2')

    fc = mx.symbol.FullyConnected(data=l2, num_hidden=num_classes, name="cifar_fc8")

    softmax = mx.symbol.SoftmaxOutput(data=fc, name="softmax")
    return softmax


if __name__ == '__main__':
    sym = get_symbol(100)
    mx.viz.print_summary(sym, {'data': (1, 3, 32, 32)})
