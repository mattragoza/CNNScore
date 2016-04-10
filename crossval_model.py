import sys
import os
import argparse
import cnnscore

def parse_args(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True,
        help='.binmaps file for training and validation')

    parser.add_argument('-r', '--data_root', type=str, default='/scr/CSAR/',
        help='root directory of .binmap data')

    parser.add_argument('-m', '--model', type=str, required=True,
        help='model topology specification')

    parser.add_argument('-p', '--downsample', type=str, default='pool',
        help='downsampling method, pool or conv')

    parser.add_argument('-k', '--kfolds', type=int, default=3,
        help='number of folds for k-fold cross-validation')

    parser.add_argument('-l', '--baselr', type=float, default=0.001,
        help='initial learning rate')

    parser.add_argument('-a', '--momentum', type=float, default=0.9,
        help='learning rate momentum')

    parser.add_argument('-w', '--weightdecay', type=float, default=0.001,
        help='weight decay regularization')

    parser.add_argument('-i', '--maxiter', type=int, default=20000,
        help='number of training iterations')

    parser.add_argument('-s', '--snapshot', type=int, default=1000,
        help='period of iterations to save weights')
    
    parser.add_argument('-g', '--gpus', type=str, default=None,
        help='device IDs of GPUs for Caffe to use')

    parser.add_argument('-o', '--output', type=str, default='./',
        help='directory to write output files')

    parser.add_argument('-f', '--force', action='store_true', default=False,
        help='force output to directory if it already exists')

    return parser.parse_args(argv[1:])


def main(argv=sys.argv):

    args = parse_args(argv)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not args.force:
        sys.exit('warning: output dir already exists, use -f to output there anyway')

    n_units, n_conv_per_unit, n_filters = map(int, args.model.split('x'))
    model = cnnscore.CNNScoreModel(args.model, n_units, n_conv_per_unit, n_filters,
        downsample=args.downsample)

    if args.gpus:
        gpus = map(int, args.gpus.split(','))
    else:
        gpus = None

    all_scored_data = model.train(args.data, args.data_root,
        k=args.kfolds,
        gpus=gpus,
        base_lr=args.baselr,
        momentum=args.momentum,
        weight_decay=args.weightdecay,
        max_iter=args.maxiter,
        snapshot=args.snapshot,
        output_dir=args.output)

    tot_plot_file = cnnscore.join_filename_params(args.output, 
        [model.param, 'testontrain'], '.roc.png')
    cv_plot_file = cnnscore.join_filename_params(args.output, 
        [model.param, 'crossval'], '.roc.png')

    cnnscore.plot_roc_curves(tot_plot_file, all_scored_data[0])
    cnnscore.plot_roc_curves(cv_plot_file, all_scored_data[1])

    all_scored_data_T = zip(*all_scored_data)
    weight_iters = range(args.snapshot, args.maxiter+1, args.snapshot)
    iter_params = ['iter_'+str(i) for i in weight_iters]

    for i, iter_param in enumerate(iter_params):
        iter_plot_file = cnnscore.join_filename_params(args.output, 
            [model.param, iter_param], '.roc.png')
        cnnscore.plot_roc_curves(iter_plot_file, all_scored_data_T[i])


if __name__ == '__main__':
    sys.exit(main())
