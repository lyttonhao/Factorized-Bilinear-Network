import argparse
import logging
import os
import sys
sys.path.insert(0, '../mxnet/python')
import mxnet as mx
import importlib
from mxnet.optimizer import SGD, NAG
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
    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
        else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, args.data_shape, args.data_shape),
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        batch_size=args.batch_size,
        pad=0,
        rand_crop=True,
        max_random_scale=1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale=1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio=0 if args.aug_level == 1 else 0.25,
        random_h=0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s=0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l=0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle=0 if args.aug_level <= 2 else 10,
        max_shear_ratio=0 if args.aug_level <= 2 else 0.1,
        rand_mirror=True,
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        batch_size=args.batch_size,
        data_shape=(3, args.data_shape, args.data_shape),
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return train, val


def main(args):
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else\
        [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.begin_epoch
    batch_size = args.batch_size
  #  if not os.path.exists(args.save_modelpath):
  #      os.mkdir(args.save_modelpath)
    save_model_prefix = "model/{}".format(args.model_prefix)
    checkpoint = mx.callback.do_checkpoint(save_model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(
            "model/" + args.model_load_prefix, args.model_load_epoch)
    symbol = importlib.import_module(
        'symbol_' + args.network).get_symbol(args.num_classes, args)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(
            symbol, data=(args.batch_size, 3, 224, 224))

    sgd = NAG(learning_rate=args.lr, momentum=args.mom,
              wd=args.wd, clip_gradient=10,
              lr_scheduler=multi_factor_scheduler(
                  begin_epoch, epoch_size,
                  step=[int(x) for x in args.lr_step.split(',')],
                  factor=0.1, fm_scale=args.fmconv_scale, fm_slowstart=args.fmconv_slowstart),
              rescale_grad=1.0 / batch_size)
    if args.fmconv and args.fmconv_slowstart == 0:
        arg_names = symbol.list_arguments()
        args_lrscale = {}
        index = 0
        for name in arg_names:
            if name != 'data' and name != 'softmax_label':
                args_lrscale[index] = 0.1 if name.startswith('bilinear') else 1.0
                index += 1
        sgd.set_lr_mult(args_lrscale)

    train, val = get_iterator(args, kv)
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=symbol,
        arg_params=arg_params,
        aux_params=aux_params,
        num_epoch=args.num_epoches,
        begin_epoch=begin_epoch,
        optimizer=sgd,
        # optimizer          = 'sgd',
        initializer=mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2),
    )
    logger = logging.getLogger()
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=['acc', mx.metric.create('top_k_accuracy', top_k=5)],
        kvstore=kv,
        logger=logger,
        batch_end_callback=mx.callback.Speedometer(
            args.batch_size, args.frequent),
        epoch_end_callback=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str,
                        default='/data/lytton/imagenet_data/', help='the input data directory')
    parser.add_argument('--data-type', type=str,
                        default='imagenet', help='the dataset type')
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
                        default=224, help='the batch size')
    parser.add_argument('--num-epoches', type=int,
                        default=80, help='the batch size')
    parser.add_argument('--network', type=str,
                        default='inception-bn', help='network name')
    parser.add_argument('--depth', type=int,
                        default=50, help='depth of resnet')
    parser.add_argument('--model-prefix', type=str,
                        help='save model name prefix')
    parser.add_argument('--workspace', type=int, default=512,
                        help='memory space size(MB) used in convolution,'
                        'if xpu memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int,
                        default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default='tmp',
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50,
                        help='frequency of logging')
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
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

    args = parser.parse_args()
    print args.model_prefix
    logging.basicConfig(filename='log/%s.log' %
                        args.model_prefix, level=logging.DEBUG)
    logging.info(args)
    main(args)
