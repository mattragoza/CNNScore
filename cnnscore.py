import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import numpy as np
import caffe
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pyplot
import matplotlib.cm as colormap

MAX_ITER = 10000
BATCH_SIZE = 10
GPUS = 0, 1

BINMAP_ROOT_DIRS = {
    'perigee': '/home/dkoes/DUDe/',
    'dvorak':  '/scr/DUDe/' }

USAGE_STRING = '''\
usage: python cnnscore.py [options...] <dir> <param>

    <dir>       The directory to output and look for all generated files
    <param>     A string to use as the base for filenames
    options:
        -d      Write train and test datasets
        -m      Write model definitions
        -s      Write solver definitions
        -t      Train models in workspace
        -r      Test models in workspace
'''

def write_model_prototxt(model_file, data_file, train=False):

    model = caffe.NetSpec()

    # 34*49*49*49 = 4000066
    model.data, model.label = caffe.layers.NDimData(
        ntop=2, ndim_data_param=dict(
            source=data_file, 
            root_folder=BINMAP_ROOT_DIRS['dvorak'],
            batch_size=BATCH_SIZE,
            shuffle=train,
            balanced=train and BATCH_SIZE>1,
            shape=dict(dim=[34, 49, 49, 49])))

    # 64*24^3 = 884736
    model.conv1 = caffe.layers.Convolution(model.data,
        kernel_size=3, pad=1, stride=2, num_output=64,
        weight_filler=dict(type='xavier'))
    model.conv1_relu = caffe.layers.ReLU(model.conv1,
        in_place=True)

    model.conv2 = caffe.layers.Convolution(model.conv1_relu,
        kernel_size=3, pad=1, stride=1, num_output=64,
        weight_filler=dict(type='xavier'))
    model.conv2_relu = caffe.layers.ReLU(model.conv2,
        in_place=True)

    # 128*12^3 = 221184
    model.conv3 = caffe.layers.Convolution(model.conv2_relu,
        kernel_size=3, pad=1, stride=2, num_output=128,
        weight_filler=dict(type='xavier'))
    model.conv3_relu = caffe.layers.ReLU(model.conv3,
        in_place=True)

    model.conv4 = caffe.layers.Convolution(model.conv3_relu,
        kernel_size=3, pad=1, stride=1, num_output=128,
        weight_filler=dict(type='xavier'))
    model.conv4_relu = caffe.layers.ReLU(model.conv4,
        in_place=True)

    # 256*6^3 = 55296
    model.conv5 = caffe.layers.Convolution(model.conv4_relu,
        kernel_size=3, pad=1, stride=2, num_output=256,
        weight_filler=dict(type='xavier'))
    model.conv5_relu = caffe.layers.ReLU(model.conv5,
        in_place=True)

    model.conv6 = caffe.layers.Convolution(model.conv5_relu,
        kernel_size=3, pad=1, stride=1, num_output=256,
        weight_filler=dict(type='xavier'))
    model.conv6_relu = caffe.layers.ReLU(model.conv6,
        in_place=True)

    # 512*3^3 = 13824
    model.conv7 = caffe.layers.Convolution(model.conv6_relu,
        kernel_size=3, pad=1, stride=2, num_output=512,
        weight_filler=dict(type='xavier'))
    model.conv7_relu = caffe.layers.ReLU(model.conv7,
        in_place=True)

    model.fc1 = caffe.layers.InnerProduct(model.conv7_relu,
        num_output=1024, weight_filler=dict(type='xavier'))
    model.fc1_drop = caffe.layers.Dropout(model.fc1)
    model.fc1_relu = caffe.layers.ReLU(model.fc1_drop,
        in_place=True)

    model.fc2 = caffe.layers.InnerProduct(model.fc1_relu,
        num_output=1024, weight_filler=dict(type='xavier'))
    model.fc2_drop = caffe.layers.Dropout(model.fc2)
    model.fc2_relu = caffe.layers.ReLU(model.fc2_drop,
        in_place=True)

    model.fc3 = caffe.layers.InnerProduct(model.fc2_relu,
        num_output=2, weight_filler=dict(type='xavier'))
    if train:
        model.loss = caffe.layers.SoftmaxWithLoss(model.fc3, model.label)
    else:
        model.score = caffe.layers.Softmax(model.fc3)

    with open(model_file, 'w') as f:
        f.write(str(model.to_proto()))


def write_solver_prototxt(solver_file, train_model_file):

    solver = dict(
        train_net=train_model_file,
        base_lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        lr_policy='inv',
        gamma=0.0001,
        power=0.75,
        display=10,
        max_iter=MAX_ITER,
        snapshot=1000,
        snapshot_prefix=train_model_file.replace('_train.model.prototxt', '') )

    buf = ''
    for i in solver:
        if type(solver[i]) is str:
            buf += i + ': "' + solver[i] + '"\n'
        else:
            buf += i + ': ' + str(solver[i]) + '\n'
    
    with open(solver_file, 'w') as f:
        f.write(buf)


def read_data_to_dict(data_file):

    data = dict()
    with open(data_file, 'r') as f:
        for line in f:
            label, target, example = line.rstrip().split()
            if target in data:
                data[target].append([example, int(label)])
            else:
                data[target] = [[example, int(label)]]

    return data


def read_data_to_lists(data_file):

    targets, examples, labels = [], [], []
    with open(data_file, 'r') as f:
        for line in f:
            label, target, example = line.rstrip().split()

            targets.append(target)
            examples.append(example)
            labels.append(int(label))

    return targets, examples, labels


def reduce_data(data, factor):

    reduced = dict()
    for target in data:
        np.random.shuffle(data[target])
        reduced[target] = data[target][:len(data[target])//factor]

    return reduced


def k_fold_partitions(data, k): 

    targets = list(data.keys())
    targets.sort(key=lambda target: len(data[target]), reverse=True)

    parts = [[] for i in range(k)]
    i = 0
    forward = True
    for target in targets:
        parts[i].append(target)
        if forward:
            i += 1
            if i == k - 1:
                forward = False
        else:
            i -= 1
            if i == 0:
                forward = True

    return parts


def write_data_to_binmaps_file(data_file, data, targets=None):

    if targets is None:
        targets = data.keys()

    with open(data_file, 'w') as f:

        lines = []
        for target in targets:
            for example in data[target]:
                lines.append(str(example[1]) + ' ' + target + ' ' + example[0])

        np.random.shuffle(lines)
        f.write('\n'.join(lines) + '\n')


def generate_crossval_filenames(output_dir, full_data_file, param_str, k):

    if full_data_file[-8:] != '.binmaps':
        raise TypeError('full_data_file argument must be a .binmaps file')
    if not k > 1:
        raise IndexError('k argument must be greater than 1')

    crossval_files = dict(
        train_data=[],
        test_data=[],
        train_models=[],
        test_models=[],
        solvers=[],
        weights=[[] for i in range(k)],
        roc_plot=[] )

    for i in range(k):

        train_data_file = os.path.join(output_dir, os.path.basename(full_data_file)[:-8] + '_part' + str(i) + '_train.binmaps')
        test_data_file  = os.path.join(output_dir, os.path.basename(full_data_file)[:-8] + '_part' + str(i) + '_test.binmaps')
        train_model_file = os.path.join(output_dir, param_str + '_part' + str(i) + '_train.model.prototxt')
        test_model_file  = os.path.join(output_dir, param_str + '_part' + str(i) + '_test.model.prototxt')
        solver_file = os.path.join(output_dir, param_str + '_part' + str(i) + '.solver.prototxt')

        crossval_files['train_data'].append(train_data_file)
        crossval_files['test_data'].append(test_data_file)
        crossval_files['train_models'].append(train_model_file)
        crossval_files['test_models'].append(test_model_file)
        crossval_files['solvers'].append(solver_file)

        for j in range(1000, 10001, 1000): # TODO should these iter values really be hard-coded?
            weight_file = os.path.join(output_dir, param_str + '_part' + str(i) + '_iter_' + str(j) + '.caffemodel')
            crossval_files['weights'][i].append(weight_file)

        roc_plot_file = os.path.join(output_dir, param_str + '_part' + str(i) + '.roc.png')
        crossval_files['roc_plot'].append(roc_plot_file)

    return crossval_files


def score_data_with_model(model_file, weight_file, data_file):

    targets, examples, labels = read_data_to_lists(data_file)
    model = caffe.Net(model_file, weight_file, caffe.TEST)
    scores = []

    c = 0
    for i in range(len(examples)//BATCH_SIZE + 1):

        # this assumes that model scores examples in same
        # order in which they appear in the data file

        output = model.forward()
        for j in range(BATCH_SIZE):
        
            if i*BATCH_SIZE + j >= len(examples):
                break

            print('Scoring example %s' % examples[i*BATCH_SIZE+j])
            scores.append(output['score'][j][1])

    return dict(targets=targets, examples=examples, labels=labels, scores=scores)


def plot_roc_curves(roc_plot_file, results):

    pyplot.title('Receiver Operating Characteristic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    colors = colormap.rainbow(np.linspace(0, 1, len(results)))
    legend = pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for i, data in enumerate(results):

        fpr, tpr, thresholds = roc_curve(data['labels'], data['scores']) 
        pyplot.plot(fpr, tpr, '-', color=colors[i], label='iter%d (area = %0.2f)' % (1000*i+1000, auc(fpr, tpr)))

    pyplot.savefig(roc_plot_file, bbox_extra_artists=(legend,), bbox_inches='tight')


def parse_args(argv):

    args, flags = list(), set()
    for arg in argv:
        if arg[0] == '-':
            for c in arg[1:]:
                flags.add(c)
        else:
            args.append(arg)

    return args[1], args[2], flags


def main(argv=sys.argv):

    try:
        output_dir, param_str, flags = parse_args(argv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except IndexError:
        return USAGE_STRING

    data_file = '/net/pulsar/home/koes/mtr22/CNNScore/all.binmaps'
    k = 3

    # generate filenames for everything we need for cross-validation
    crossval_files = generate_crossval_filenames(output_dir, data_file, param_str, k)

    # write k-fold data
    if 'd' in flags:
        full_data = read_data_to_dict(data_file)
        parts = k_fold_partitions(full_data, k)
        for i in range(k):

            train_data_file = crossval_files['train_data'][i]
            train_targets = [t for p, part in enumerate(parts) if p != i for t in part]
            write_data_to_binmaps_file(train_data_file, full_data, train_targets)
            
            test_data_file  = crossval_files['test_data'][i]
            test_targets = parts[i]
            write_data_to_binmaps_file(test_data_file, full_data, test_targets)

    # write model definitions
    if 'm' in flags:
        for i in range(k):
            
            train_data_file  = crossval_files['train_data'][i]
            train_model_file = crossval_files['train_models'][i]
            write_model_prototxt(train_model_file, train_data_file, train=True)

            test_data_file  = crossval_files['test_data'][i]
            test_model_file = crossval_files['test_models'][i]
            write_model_prototxt(test_model_file, test_data_file)

    # write solver definitions
    if 's' in flags:
        for i in range(k):
            
            solver_file = crossval_files['solvers'][i]
            train_model_file = crossval_files['train_models'][i]
            write_solver_prototxt(solver_file, train_model_file)

    # train models with training data and solvers
    if 't' in flags:
        for i in range(k):

            solver_file = crossval_files['solvers'][i]
            command = 'caffe train -solver ' + solver_file + ' -gpu ' + ','.join(map(str, GPUS))
            os.system(command)

    # score test data using trained models
    caffe.set_device(GPUS[0])
    caffe.set_mode_gpu()
    if 'r' in flags:
        results = [[] for i in range(k)]
        for i in range(k):

            test_data_file  = crossval_files['test_data'][i]
            test_model_file = crossval_files['test_models'][i]

            for weight_file in crossval_files['weights'][i]:
                results[i].append(score_data_with_model(test_model_file, weight_file, test_data_file))
        
            # view the results
            roc_plot_file = crossval_files['roc_plot'][i]
            plot_roc_curves(roc_plot_file, results[i])
        

if __name__ == '__main__':
    sys.exit(main())