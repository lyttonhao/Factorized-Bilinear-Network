import sys
sys.path.insert(0, "../mxnet/python/")
import mxnet as mx
import argparse
import importlib
import logging
import numpy as np
from mxnet.optimizer import SGD
from fmconv_scheduler import FMConvScheduler


def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1, fm_scale=0.1, fm_slowstart=0):
    if fm_slowstart > 0:
        step = [fm_slowstart] + step
    step_ = [epoch_size * (x - begin_epoch)
             for x in step if x - begin_epoch > 0]
    print step, step_
    if len(step_) > 0:
        if fm_slowstart > 0:
             return FMConvScheduler(step=step_, factor=factor, fm_scale=fm_scale)
        else:
            return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor)
    return None


def get_iterator(args, kv):
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="%s/train.rec" % (args.data_dir),
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        pad=4,
        rand_crop=True,
        rand_mirror=True,
        prefetch_buffer=4,
        preprocess_threads=4,
        shuffle=True,
        data_shape=(3, args.data_shape, args.data_shape),
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    val_dataiter = mx.io.ImageRecordIter(
        path_imgrec="%s/test.rec" % (args.data_dir),
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        rand_crop=False,
        rand_mirror=False,
        prefetch_buffer=5,
        preprocess_threads=4,
        data_shape=(3, args.data_shape, args.data_shape),
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return (train_dataiter, val_dataiter)


def main(args):
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else\
        [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.begin_epoch
    batch_size = args.batch_size

    save_model_prefix = "model/{}".format(args.model_prefix)
    checkpoint = mx.callback.do_checkpoint(save_model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(
            "model/" + args.model_load_prefix, args.model_load_epoch)
    symbol = importlib.import_module(
        'symbol_' + args.network).get_symbol(args.num_classes, args)
    if args.init == 'xavier':
        init = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)
    else:
        init = MyInit(rnd_type='gaussian', factor_type="in", magnitude=2) 
    lr_scheduler = multi_factor_scheduler(
        begin_epoch, epoch_size,
        step=[int(x) for x in args.lr_step.split(',')],
        factor=0.1, fm_scale=args.fmconv_scale, fm_slowstart=args.fmconv_slowstart)
    sgd = SGD(learning_rate=args.lr, momentum=args.mom,
              wd=args.wd, clip_gradient=10,
              lr_scheduler=lr_scheduler,
              rescale_grad=1.0 / batch_size)
    arg_names = symbol.list_arguments()
    args_lrscale = {}
    if args.fmconv and args.fmconv_slowstart == 0:
        index = 0
        for name in arg_names:
            if name != 'data' and name != 'softmax_label':
                args_lrscale[index] = args.fmconv_scale if name.startswith(
                    'bilinear') else 1.0
                index += 1
        print "set lr scale"
    if args.freeze:
        index = 0
        for name in arg_names:
            if name != 'data' and name != 'softmax_label':
                if 'cifar' not in name:
                    args_lrscale[index] = 0.0
                else:
                    print name
                index += 1
    
    sgd.set_lr_mult(args_lrscale)

    logger = logging.getLogger()
    train, val = get_iterator(args, kv)
    if args.warmup:
        print "warmup"
        model = mx.model.FeedForward(symbol=symbol, ctx=devs, num_epoch=1,
                                     initializer=init,
                                     learning_rate=args.lr * args.fmconv_scale, momentum=args.mom,
                                     wd=args.wd, clip_gradient=10,
                                     arg_params=arg_params,
                                     aux_params=aux_params,
                                     )
        model.fit(
            X=train,
            eval_data=val,
            eval_metric=['acc'],
            kvstore=kv,
            logger=logger,
            batch_end_callback=mx.callback.Speedometer(
                args.batch_size, args.frequent),
            epoch_end_callback=mx.callback.do_checkpoint(save_model_prefix + '-warm'))
        _, arg_params, aux_params = mx.model.load_checkpoint(
            save_model_prefix + '-warm', 1)
        kv = mx.kvstore.create(args.kv_store)
        model = None
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=symbol,
        arg_params=arg_params,
        aux_params=aux_params,
        num_epoch=args.num_epoches,
        begin_epoch=begin_epoch,
        optimizer=sgd,
        initializer=init,
    )
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=['acc'],
        kvstore=kv,
        logger=logger,
        batch_end_callback=mx.callback.Speedometer(
            args.batch_size, args.frequent),
        epoch_end_callback=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="command for training cifar using factorized bilinear networks")
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str,
                        default='./cifar10',
                        help='the input data directory')
    parser.add_argument('--data-type', type=str,
                        default='cifar10', help='the dataset type')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9,
                        help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='the batch size')
    parser.add_argument('--data-shape', type=int,
                        default=32, help='the batch size')
    parser.add_argument('--num-epoches', type=int,
                        default=400, help='the batch size')
    parser.add_argument('--network', type=str,
                        default='inception-bn-small', help='network name')
    parser.add_argument('--res-module-num', type=int,
                        default=18, help='number of resnet module')
    parser.add_argument('--model-prefix', type=str,
                        help='save model name prefix')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the class number of your task')
    parser.add_argument('--num-examples', type=int,
                        default=50000, help='the number of training examples')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='tmp',
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50,
                        help='frequency of logging')
    parser.add_argument('--retrain', action='store_true',
                        default=False, help='true means continue training')
    parser.add_argument('--begin-epoch', type=int, default=0,
                        help='begin epoch')
    parser.add_argument('--fmconv', action='store_true',
                        default=False, help='if use fmconv way for training, such as decease lr for bilinear layer')
    parser.add_argument('--fmconv-scale', type=float,
                        default=0.1, help='scale ratio of lr in fmconv layers')
    parser.add_argument('--fmconv-slowstart', type=int,
                        default=0, help='the slowstart epoches of lr in fmconv layers')
    parser.add_argument('--fmconv-drop', type=float,
                        default=0.5, help='fmconv drop factor rate')
    parser.add_argument('--fmconv-factor', type=int,
                        default=20, help='fmconv factor')
    parser.add_argument('--lr-step', type=str, default='',
                        help='lr drop steps"')
    parser.add_argument('--warmup', action='store_true',
                        default=False, help='if warmup with small learning rate')
    parser.add_argument('--init', type=str,
                        default='xavier', help='initilization mode')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='if freeze previsou layer')
    args = parser.parse_args()
    print args.model_prefix
    logging.basicConfig(filename='log/%s.log' %
                        args.model_prefix, level=logging.DEBUG)
    logging.info(args)
    main(args)
