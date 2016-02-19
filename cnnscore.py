import matplotlib
matplotlib.use('Agg')
import sys
import os
import argparse
import numpy as np

import caffe
from caffe.proto import caffe_pb2
from google import protobuf as pb
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.cm as colormap


OUTPUT_DIR_ERROR = 'error: could not make or access the output directory'
GPU_BUSY_ERROR = 'error: one or more selected GPUs are unavailable'

DEFAULT_BINMAP_ROOT = '/scr/DUDe/'


class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg


# functional stuff

def read_lines_from_file(file):
    '''Read file to a list of lines split by whitespace'''
    with open(file, 'r') as f:
        return list(map(str.split, f.read().splitlines()))

def transpose(list_list):
    '''Transpose inner and outer lists'''
    return list(map(list, zip(*list_list)))

# working stuff

def read_data_to_target_dict(data_file):
    '''Read a .binmaps file into a dictionary where keys are
    targets and values are the list of corresponding examples,
    as [example, label].'''

    data = dict()
    with open(data_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            try:
                label = int(fields[0])
            except ValueError:
                label = None

            target = fields[1]
            if target not in data:
                data[target] = []

            example = fields[2]
            data[target].append([example, label])

    return data


def read_data_to_column_dict(data_file):
    '''Read a .binmaps file into a dictionary where keys are labels,
    targets, and examples, and values are lists of corresponding data
    in the order it appears in the file.'''

    targets, examples, labels = [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            try:
                label = int(fields[0])
            except ValueError:
                label = None

            labels.append(label)
            targets.append(fields[1])
            examples.append(fields[2])

    return dict(labels=labels, targets=targets, examples=examples)


def read_scored_data_to_column_dict(data_file):
    '''Read a .scores file into a dictionary where keys are labels,
    targets, examples, and scores, and values are lists of corresponding
    data in the order it appears in the file.'''

    targets, examples, labels, scores = [], [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            try:
                label = int(fields[0])
            except ValueError:
                label = None

            labels.append(label)
            targets.append(fields[1])
            examples.append(fields[2])
            scores.append(float(fields[3]))

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
    withheld from training and used as the test set.'''

    targets = list(data.keys())
    np.random.shuffle(targets)

    parts = [[] for i in range(k)]
    for i, target in enumerate(targets):
        parts[i%k].append(target)

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


def plot_roc_curves(roc_plot_file, plot_data):

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    
    colors = colormap.rainbow(np.linspace(0, 1, len(plot_data)))
    for i, series in enumerate(plot_data):

        fpr, tpr, thresholds = roc_curve(series['labels'], series['scores']) 
        plt.plot(fpr, tpr, '-', color=colors[i], \
            label=series.get('name', 'series %d' % i) + ' (AUC=%0.2f)' % auc(fpr, tpr))

    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(roc_plot_file, bbox_extra_artists=(legend,), bbox_inches='tight')


def write_scores_to_file(score_file, results):

    buf = ''
    for i, example in enumerate(results['examples']):
        target = results['targets'][i]
        label = results['labels'][i]
        score = results['scores'][i]
        buf += ' '.join(map(str, [label, target, example, score])) + '\n'

    with open(score_file, 'w') as f:
        f.write(buf)


def get_model_predictions(model_file, weight_file, data_file):

    data = read_data_to_column_dict(data_file)
    model = caffe.Net(model_file, weight_file, caffe.TEST)
    batch_size = model.blobs['data'].shape[0] # assumes existence of 'data' blob
    num_examples = len(data['examples'])
    num_batches = num_examples//batch_size + 1
    data['scores'] = []

    c = 0
    for i in range(num_batches):

        # this assumes that model scores examples in same
        # order in which they appear in the data file
        print('Scoring %s: batch %d / %d' % (data_file, i, num_batches))

        output = model.forward()
        for j in range(batch_size):
        
            data_index = i*batch_size + j
            if data_index >= num_examples:
                break

            if data['labels'][data_index] != int(output['label'][j]):
                raise IndexError('file data does not match model data')
            data['scores'].append(output['pred'][j][1])

    return data


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
    full_data = read_data_to_target_dict(full_data_file)
    if k: parts = k_fold_partitions(full_data, k)

    for i in range(k+1):

        if i == 0:
            # train and test on full data
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
            weight_iters = range(snap, solver_prototype.max_iter + snap, snap)
        else:
            weight_iters = [solver_prototype.max_iter]
        for j in weight_iters:
            iter_param = 'iter_' + str(j)
            weight_file = os.path.join(output_dir, '_'.join([model_param, part_param, iter_param]) + '.caffemodel')
            weight_files.append(weight_file)
            # TODO crossval score files

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


def crossval_model(output_dir, args):
    
    gpus = args.g or '0'
    binmap_root = args.b or DEFAULT_BINMAP_ROOT
    k = 3

    crossval_files = generate_crossval_files(output_dir, args.data_file, binmap_root, \
        args.model_file, args.solver_file, k)
    
    for i in range(k+1):

        solver_file = crossval_files['solvers'][i]
        os.system('caffe train -solver ' + solver_file + ' -gpu ' + gpus)

    caffe.set_device(int(gpus[0])) # can pycaffe do multi-gpu?
    caffe.set_mode_gpu()
    plot_data = []
    for i in range(k+1):

        test_data_file = crossval_files['test_data'][i]
        test_model_file = crossval_files['test_models'][i]
        weight_file = crossval_files['weights'][i][-1]
        results = get_model_predictions(test_model_file, weight_file, test_data_file)
        results['name'] = os.path.basename(weight_file)
        plot_data.append(results)

    roc_plot_file = crossval_files['roc_plots'][0]
    plot_roc_curves(roc_plot_file, plot_data)


def test_model(output_dir, args):

    gpus = args.g or '0'
    binmap_root = args.b or DEFAULT_BINMAP_ROOT
    
    test_files = generate_test_files(output_dir, args.data_file, binmap_root, \
        args.model_file, args.weight_file)
    
    caffe.set_device(int(gpus[0])) 
    caffe.set_mode_gpu()

    test_data_file = test_files['test_data']
    test_model_file = test_files['test_model']
    results = get_model_predictions(test_model_file, args.weight_file, test_data_file)

    score_file = test_files['score']
    write_scores_to_file(score_file, results)


def parse_args(argv):

    p_global = argparse.ArgumentParser(add_help=False)
    p_global.add_argument('-g', metavar='<gpus>', \
        help='comma-separated device ids of GPUs to use')
    p_global.add_argument('-o', metavar='<output_dir>', \
        help='directory to output generated files')
    p_global.add_argument('-b', metavar='<binmap_root>', \
        help='root of binmap directory tree')

    p_task = argparse.ArgumentParser(parents=[p_global])
    subparsers = p_task.add_subparsers(title='task', metavar='<crossval|test>', \
        help='perform cross-validation training or testing')
    subparsers.required = True

    p_crossval = subparsers.add_parser('crossval', parents=[p_global])
    p_crossval.set_defaults(task='crossval')
    p_crossval.add_argument('data_file', metavar='<data_file>', \
        help='dataset for training and validation')
    p_crossval.add_argument('model_file', metavar='<model_file>', \
        help='model definition prototxt')
    p_crossval.add_argument('solver_file', metavar='<solver_file>', \
        help='solver definition prototxt')

    p_test = subparsers.add_parser('test', parents=[p_global])
    p_test.set_defaults(task='test')
    p_test.add_argument('data_file', metavar='<data_file>', \
        help='dataset to produce scores for')
    p_test.add_argument('model_file', metavar='<model_file>', \
        help='model definition prototxt')
    p_test.add_argument('weight_file', metavar='<weight_file>', \
        help='caffemodel to use as model weights')

    return p_task.parse_args(argv[1:])


def main(argv=sys.argv):

    try:
        args = parse_args(argv)
    except SystemExit:
        return 1

    try:
        output_dir = args.o or '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except IOError:
        return OUTPUT_DIR_ERROR

    if args.task == 'crossval':
        crossval_model(output_dir, args)

    elif args.task == 'test':
        test_model(output_dir, args)


if __name__ == '__main__':
    sys.exit(main())
