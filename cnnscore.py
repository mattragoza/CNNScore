import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google import protobuf as pb
from operator import add
from functools import reduce
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pyplot
import matplotlib.cm as colormap

USAGE_ERROR = '''\
usage: python cnnscore.py TASK ARGUMENTS [OPTIONS]

Tasks/Arguments:

    crossval    Train a model using k-fold cross-validation
        
        DATA_FILE       Dataset for training and validation
        MODEL_FILE      Caffe model definition prototxt
        SOLVER_FILE     Caffe solver definition prototxt

    test        Test a trained model on a dataset

        DATA_FILE       Dataset to produce scores for
        MODEL_FILE      Model definition prototxt
        WEIGHT_FILE     Training iteration of weights to use

Options:

    -o OUTPUT_DIR       Directory to output generated files
    -g GPUS             Comma-delimited device ids of GPUs to use
    -b BINMAP_ROOT      Root of binmap directory tree
'''
OUTPUT_DIR_ERROR = 'error: could not make or access the output directory'


class UsageError(Exception): 
    pass


def read_data_to_target_dict(data_file):
    '''Read a .binmaps file into a dictionary where keys are
    targets and values are the list of corresponding examples,
    as [example, label].'''

    data = dict()
    with open(data_file, 'r') as f:
        for line in f:
            label, target, example = line.rstrip().split()
            try:
                label = int(label)
            except ValueError:
                label = None

            if target not in data:
                data[target] = []
            data[target].append([example, label])

    return data


def read_data_to_column_dict(data_file):
    '''Read a .binmaps file into a dictionary where keys are labels,
    targets, and examples, and values are lists of corresponding data
    in the order it appears in the file.'''

    targets, examples, labels = [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            label, target, example = line.rstrip().split()
            try:
                label = int(label)
            except ValueError:
                label = None

            labels.append(label)
            targets.append(target)
            examples.append(example)

    return dict(labels=labels, targets=targets, examples=examples)


def read_scored_data_to_column_dict(data_file):
    '''Read a .scores file into a dictionary where keys are labels,
    targets, examples, and scores, and values are lists of corresponding
    data in the order it appears in the file.'''

    targets, examples, labels, scores = [], [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            label, target, example, score = line.rstrip().split()
            try:
                label = int(label)
            except ValueError:
                label = None

            labels.append(label)
            targets.append(target)
            examples.append(example)
            scores.append(score)

    return dict(labels=labels, targets=targets, examples=examples, scores=scores)


def reduce_target_data(data, factor):
    '''Creates a reduced data set by randomly sampling from the given
    target dictioanry such that each target's number of examples has
    been reduced by the given factor.'''

    reduced = dict()
    for target in data:
        np.random.shuffle(data[target])
        reduced[target] = data[target][:len(data[target])//factor]

    return reduced


def k_fold_partitions(data, k):
    '''Returns a list of k balanced partitions of the data targets,
    where each partition is a list of the targets that should be
    withheld and used as the test set.'''

    targets = list(data.keys())
    targets.sort(key=lambda target: len(data[target]), reverse=True)

    parts = [[] for i in range(k)]
    i = 0
    forward = True
    for target in targets:
        parts[i].append(target)
        if forward:
            i += 1
            if i + 1 == k:
                forward = False
        else:
            i -= 1
            if i == 0:
                forward = True

    return parts


def write_data_to_binmaps_file(data_file, data, targets=None):

    if targets is None:
        targets = data.keys()

    lines = []
    for target in targets:
        for example in data[target]:
            lines.append(str(example[1]) + ' ' + target + ' ' + example[0])

    np.random.shuffle(lines)

    with open(data_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def plot_roc_curves(roc_plot_file, plot_data, mode):

    pyplot.clf()
    pyplot.title('Receiver Operating Characteristic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    
    colors = colormap.rainbow(np.linspace(0, 1, len(plot_data)))
    for i, series in enumerate(plot_data):

        fpr, tpr, thresholds = roc_curve(series['labels'], series['scores']) 
        pyplot.plot(fpr, tpr, '-', color=colors[i], \
            label=series.get('name', 'series %d' % i) + ' (AUC=%0.2f)' % auc(fpr, tpr))

    legend = pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyplot.savefig(roc_plot_file, bbox_extra_artists=(legend,), bbox_inches='tight')


def write_scores_to_file(score_file, results):

    buf = ''
    for i, example in enumerate(results['examples']):
        target = results['targets'][i]
        label = results['labels'][i]
        score = results['scores'][i]
        buf += ' '.join(map(str, [label, target, example, score])) + '\n'

    with open(score_file, 'w') as f:
        f.write(buf)


def score_data_with_model(model_file, weight_file, data_file):

    targets, examples, labels = read_data_to_lists(data_file)
    model = caffe.Net(model_file, weight_file, caffe.TEST)
    batch_size = model.blobs['data'].shape[0] # assumes existence of 'data' blob
    num_batches = len(examples)//batch_size + 1
    scores = []

    c = 0
    for i in range(num_batches):

        # this assumes that model scores examples in same
        # order in which they appear in the data file
        print('Scoring %s: batch %d / %d' % (data_file, i, num_batches))

        output = model.forward()
        for j in range(batch_size):
        
            if i*batch_size + j >= len(examples):
                break

            scores.append(output['pred'][j][1])

    return dict(targets=targets, examples=examples, labels=labels, scores=scores)


def write_model_prototxt(model_file, model_prototype, data_file, binmap_root, mode):
    
    model = caffe_pb2.NetParameter()
    model.CopyFrom(model_prototype)

    data_layer = model.layer[0]
    data_layer.ndim_data_param.source = data_file
    data_layer.ndim_data_param.root_folder = binmap_root

    if mode == 'train':
        data_layer.ndim_data_param.shuffle = True
        data_layer.ndim_data_param.balanced = True
        for i, layer in enumerate(model.layer):
            if layer.name == 'pred':
                model.layer.pop(i)

    elif mode == 'test':
        data_layer.ndim_data_param.shuffle = False
        data_layer.ndim_data_param.balanced = False
        for i, layer in enumerate(model.layer):
            if layer.name == 'loss':
                model.layer.pop(i)

    with open(model_file, 'w') as f:
        f.write(str(model))


def write_solver_prototxt(solver_file, solver_prototype, model_file):

    solver = caffe_pb2.SolverParameter()
    solver.CopyFrom(solver_prototype)

    solver.train_net = model_file
    weight_prefix = model_file.replace('_train.model.prototxt', '')
    solver.snapshot_prefix = weight_prefix

    with open(solver_file, 'w') as f:
        f.write(str(solver))


def read_model_prototxt(model_file):

    model = caffe_pb2.NetParameter()
    with open(model_file, 'r') as f:
        pb.text_format.Merge(str(f.read()), model)
    return model


def read_solver_prototxt(solver_file):

    solver = caffe_pb2.SolverParameter()
    with open(solver_file, 'r') as f:
        pb.text_format.Merge(str(f.read()), solver)
    return solver


def generate_crossval_files(output_dir, full_data_file, binmap_root, model_prototype_file, solver_prototype_file, k=0):

    # keep track of names of all generated files in a dictionary
    crossval_files = dict(train_data=[], test_data=[], train_models=[], test_models=[], \
        solvers=[], weights=[], roc_plots=[])

    # get parameter strings for data, model, and solver arguments
    data_param, data_ext = os.path.splitext(os.path.basename(full_data_file)) # data_param.data_ext
    model_param  = os.path.basename(model_prototype_file).split('.')[0] # model_param.model.prototxt
    solver_param = os.path.basename(solver_prototype_file).split('.')[0] # solver_param.solver.prototxt

    # split data targets into k-fold partitions
    full_data = read_data_to_dict(full_data_file)
    if k: parts = k_fold_partitions(full_data, k)

    for i in range(k+1):

        if i == 0:
            # train and/or test on full data
            part_param = 'full'
            train_data_file = full_data_file
            test_data_file  = full_data_file
        else:
            # split full data into train and test data using target partitions
            part_param = 'part' + str(i)
            train_data_file = os.path.join(output_dir, '_'.join([data_param, part_param, 'train']) + data_ext)
            test_data_file  = os.path.join(output_dir, '_'.join([data_param, part_param, 'test'])  + data_ext)
            train_targets = [t for p, part in enumerate(parts) if p != (i-1) for t in part]
            test_targets = parts[i-1]
            write_data_to_binmaps_file(train_data_file, full_data, train_targets)
            write_data_to_binmaps_file(test_data_file,  full_data, test_targets)

        crossval_files['train_data'].append(train_data_file)
        crossval_files['test_data'].append(test_data_file)


        # create prototxt for train model, test model and solver for each
        model_prototype  = read_model_prototxt(model_prototype_file)
        solver_prototype = read_solver_prototxt(solver_prototype_file)
        train_model_file = os.path.join(output_dir, '_'.join([model_param, part_param, 'train']) + '.model.prototxt')
        test_model_file  = os.path.join(output_dir, '_'.join([model_param, part_param, 'test'])  + '.model.prototxt')
        solver_file = os.path.join(output_dir, '_'.join([solver_param, part_param]) + '.solver.prototxt')
        write_model_prototxt(train_model_file, model_prototype, train_data_file, binmap_root, mode='train')
        write_model_prototxt(test_model_file,  model_prototype, test_data_file,  binmap_root, mode='test')
        write_solver_prototxt(solver_file, solver_prototype, train_model_file)
        crossval_files['train_models'].append(train_model_file)
        crossval_files['test_models'].append(test_model_file)
        crossval_files['solvers'].append(solver_file)


        # keep track of weight files and score files that will be produced
        weight_files = list()
        snap = solver_prototype.snapshot
        if snap > 0:
            weight_iters = range(snap, solver_prototype.max_iter+snap, snap)
        else:
            weight_iters = [solver_prototype.max_iter]
        for j in weight_iters:
            iter_param = 'iter_' + str(j)
            weight_file = os.path.join(output_dir, '_'.join([model_param, part_param, iter_param]) + '.caffemodel')
            weight_files.append(weight_file)

        crossval_files['weights'].append(weight_files)

    # make a results roc curve
    roc_plot_file = os.path.join(output_dir, '_'.join([model_param]) + '.roc.png')
    crossval_files['roc_plots'].append(roc_plot_file)

    return crossval_files


def generate_test_files(output_dir, test_data_file, binmap_root, model_prototype_file, weight_file):

    test_files = dict(test_data=None, test_model=None, score=None)

    # get parameter strings for data, model, and solver arguments
    data_param, data_ext = os.path.splitext(os.path.basename(test_data_file)) # data_param.data_ext
    model_param = os.path.basename(model_prototype_file).split('.')[0] # model_param.model.prototxt

    model_prototype = read_model_prototxt(model_prototype_file)
    test_model_file = os.path.join(output_dir, '_'.join([model_param, data_param])  + '.model.prototxt')
    write_model_prototxt(test_model_file, model_prototype, test_data_file, binmap_root, mode='test')

    weight_param = os.path.splitext(os.path.basename(weight_file))[0]
    score_file = os.path.join(output_dir, '_'.join([weight_param, data_param]) + '.scores')

    test_files['test_data'] = test_data_file
    test_files['test_model'] = test_model_file
    test_files['score'] = score_file

    return test_files


def crossval_model(output_dir, data_file, model_file, solver_file, opts):
    
    gpus = opts.pop('-g', '0')

    crossval_files = generate_crossval_files(output_dir, data_file, model_file, solver_file, k=3)
    
    for i in range(k+1):

        solver_file = crossval_files['solvers'][i]
        os.system('caffe train -solver ' + solver_file + ' -gpu ' + gpus)

    caffe.set_device(int(gpus[0]))
    caffe.set_mode_gpu()
    plot_data = []
    for i in range(k+1):

        test_data_file = crossval_files['test_data'][i]
        test_model_file = crossval_files['test_models'][i]
        weight_file = crossval_files['weights'][i][-1]
        results = score_data_with_model(test_model_file, weight_file, test_data_file)

        results['name'] = os.path.basename(weight_file)
        plot_data.append(results)

    roc_plot_file = crossval_files['roc_plots'][0]
    plot_roc_curves(roc_plot_file, plot_data)


def test_model(output_dir, data_file, model_file, weight_file, opts):

    gpus = opts.pop('-g', '0')
    binmap_root = opts.pop('-b', '/scr/DUDe/')
    
    test_files = generate_test_files(output_dir, data_file, binmap_root, model_file, weight_file)
    caffe.set_device(int(gpus[0]))
    caffe.set_mode_gpu()

    test_data_file = test_files['test_data']
    test_model_file = test_files['test_model']
    weight_file = weight_file
    results = score_data_with_model(test_model_file, weight_file, test_data_file)

    score_file = test_files['score']
    write_scores_to_file(score_file, results)


def parse_args(argv):

    # first get optional args
    opts = dict()
    for i, arg in enumerate(argv):
        if arg[0] == '-':
            if arg in ['-o', '-g', '-b']:
                opt = argv.pop(i)
                val = argv[i]
                opts[opt] = val
            else:
                raise UsageError()

    # then get positional args
    if len(argv) < 5:
        raise UsageError() 
    task = argv[1]
    args = argv[2:]
    if task not in ['crossval', 'test']:
        raise UsageError()

    return task, args, opts 


def main(argv=sys.argv):

    try:
        task, args, opts = parse_args(argv)
    except UsageError:
        return USAGE_ERROR

    try:
        output_dir = opts.pop('-o', '.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except IOError:
        return OUTPUT_DIR_ERROR

    if task == 'crossval':
        crossval_model(output_dir, args[0], args[1], args[2], opts)

    elif task == 'test':
        test_model(output_dir, args[0], args[1], args[2], opts)


if __name__ == '__main__':
    sys.exit(main())
