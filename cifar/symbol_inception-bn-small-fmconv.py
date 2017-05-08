import sys
sys.path.insert(0, "../mxnet/python/")
import mxnet as mx
import argparse

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
    in5b = SimpleFactory(in5a, 176, 160, act_type='tanh')
    bilinear = mx.symbol.FMConvolution3(data=in5b, num_filter=num_classes, num_factor=args.fmconv_factor,
                                        kernel=(1, 1), stride=(1, 1),
                                        pad=(0, 0), p=args.fmconv_drop, name='bilinear1')
    conv = mx.symbol.Convolution(data=in5b, num_filter=num_classes,
                                 kernel=(1, 1), stride=(1, 1),
                                 pad=(0, 0), name='conv_fc')
    pool = mx.symbol.Pooling(
        data=conv + bilinear, pool_type="avg", kernel=(8, 8), name="global_pool")
    flatten = mx.symbol.Flatten(data=pool, name="flatten")
    
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name="softmax")
    return softmax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="command for training cifar")
    parser.add_argument('--fmconv-factor', type=int, default=20,
                        help='the number of fmconv factor')
    parser.add_argument('--fmconv-drop', type=int, default=0.5,
                        help='fmconv drop rate')
    args = parser.parse_args()
    sym = get_symbol(100, args)
    mx.viz.print_summary(sym, {'data': (1, 3, 32, 32)})
