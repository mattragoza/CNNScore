import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import argparse
from glob import glob
import numpy as np
import caffe
import lmdb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class CaffeModel(object):

    '''CaffeModel represents a neural network model in Caffe. It also
    organizes the various prototxt and data files that are used for
    training, testing, and using a model.'''

    def __init__(self):

        # files for model training
        self.train_solver_file = None
        self.train_model_file = None

        # train and test data specification
        self.train_data_file = None
        self.test_data_file = None

        # files for model usage
        self.deploy_weight_file = None
        self.deploy_model_file = None
        
        self.train_model_net = None
        self.deploy_model_net = None

    def setup_train(self):

        '''Initialize the network in Caffe for training.'''

        raise NotImplementedError()
        self.train_model_net = caffe.Net(self.train_model_file,
            None, caffe.TRAIN)

    def setup_test(self):

        '''Initialize the network in Caffe using trained weights.'''

        self.deploy_model_net = caffe.Net(self.deploy_model_file,
            self.deploy_weight_file, caffe.TEST)

    @staticmethod
    def from_solver_prototxt(solver_file):

        '''Construct a CaffeModel by parsing a solver prototxt file.
        Parses the training model from the 'net' parameter, then
        tries to parse the training model prototxt as well to get data
        files specified for training and testing.'''

        self = CaffeModel()
        self._parse_solver_prototxt(solver_file)
        return self

    def _parse_solver_prototxt(self, solver_file):

        # read the solver file into a token buffer
        with open(solver_file, 'r') as solver_fp:
            solver = [i.lstrip().rstrip().split() for i in solver_fp]

        # parse out the training model file and length of training
        for line in solver:
            if not line:
                continue
            elif line[0] == 'net:':
                train_model_file = line[1].strip('"')
            elif line[0] == 'max_iter:':
                max_iter = int(line[1])

        # use training model and length of training to get
        # the deploy model file and weights file
        deploy_model_file = re.sub(r'_part.*?.model|_full.model', \
            '_deploy.model', train_model_file)
        deploy_weight_file = re.sub('.model.prototxt', \
            '_iter_'+str(max_iter)+'.caffemodel', train_model_file)

        # initialize the CaffeModel with the parsed data
        self.train_solver_file = solver_file
        self.train_model_file = train_model_file
        self.deploy_model_file = deploy_model_file
        self.deploy_weight_file = deploy_weight_file

        # parse the training model to get data files
        self._parse_model_prototxt(train_model_file)

    def _parse_model_prototxt(self, model_file):

        # read the model file into a token buffer
        with open(model_file, 'r') as model_fp:
            model = [i.lstrip().rstrip().split() for i in model_fp]

        # getparse paths to data files from data_params in buffer
        data_files = []
        model_iter = iter(model)
        for line in model_iter:
            if not line:
                continue
            elif line[0] == 'data_param':
                while line[0] != '}':
                    if line[0] == 'source:':
                        data_files.append(line[1].strip('"'))
                    line = next(model_iter)
                    continue

        # TODO need to handle this better
        try:
            self.train_data_file = data_files[0]
            self.test_data_file = data_files[1]
        except IndexError:
            pass

    def test_lmdb(self, data_file):

        lmdb_env = lmdb.open(data_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()

        result = {'id':[], 'truth':[], 'score':[], 'num':0}
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            features = caffe.io.datum_to_array(datum)

            truth = datum.label
            score = self.deploy_model_net.forward_all(data_blob=features.reshape(1, 61, 1, 1))
            score = list(score['output_act_blob'][0])

            result['id'].append(key)
            result['truth'].append(truth)
            result['score'].append(score)
            result['num'] += 1

        return result

def main(argv=sys.argv[1:]):

    args = parse_args(argv)
    caffe.set_mode_gpu()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_plot_data = []
    plot_name = 'default'

    for s in sorted(glob(args.SOLVER_PATTERN)):

        nnet = CaffeModel.from_solver_prototxt(s)
        nnet.setup_test()
        print('Using model trained by ' + str(s))

        if args.data_pattern is None:
            if nnet.test_data_file:
                data_glob = [nnet.test_data_file]
            elif nnet.train_data_file:
                data_glob = [nnet.train_data_file]
            else:
                data_glob = []
        else:
            data_glob = glob(args.data_pattern)

        if args.output_prefix is None:
            args.output_prefix = ''

        if len(data_glob) == 0:
            print('\tNo test data')
            continue

        for test_db in data_glob:
            print('\tTesting on ' + str(test_db))
            result = nnet.test_lmdb(test_db)

            base_name = write_score_file(args.output_prefix, nnet, test_db, result)
            if not args.roc:
                continue

            fpr, tpr, thresholds = roc_curve(result['truth'], [i[-1] for i in result['score']])
            roc_auc = auc(fpr, tpr)
            data_label = os.path.basename(test_db).replace('.prototxt', '')
            if 'full' in data_label:
                plot_name.replace('full.solver.prototxt', '')
            else:
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

            plot_data = {'data_label':data_label,
                         'roc_auc':roc_auc,
                         'fpr':fpr,
                         'tpr':tpr}

            if 
            all_plot_data.append(plot_data)

    if not args.roc:
        return 0

    # mean data for crossval
    try:

        mean_tpr[0]  = 0.0
        mean_tpr /= len(all_plot_data)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plot_data = {'data_label':'mean',
                     'roc_auc':mean_auc,
                     'fpr':mean_fpr,
                     'tpr':mean_tpr}
    except TypeError:
        pass
    all_plot_data.append(plot_data)

    if args.output_prefix:
        plot_name = args.output_prefix

    write_roc_curves(all_plot_data, plot_name)

    return 0

def proto_to_dict(proto):

    '''Basic prototxt parser for files or strings.
    Builds a dict of keys with arrays of values.'''

    def parse_to_dict(iter_):
        d = dict()
        for line in iter_:
            if not line:
                continue
            elif line[0] == '}':
                return d
            elif line[0][-1] == ':':
                key = line[0][:-1]
                value = line[1]
            elif line[1] == '{':
                key = line[0]
                value = parse_to_dict(buf_iter)
            else:
                continue
            if key in d:
                d[key].append(value)
            else:
                d[key] = [value]
        return d

    ws = ' \t\n'
    toks = [line.lstrip(ws).rstrip(ws).split(' ') for line in proto]

    return parse_to_dict(iter(toks))

def write_score_file(dir_, nnet, data, result):

    model = os.path.basename(nnet.train_model_file).replace('_full.model.prototxt', '')
    train = os.path.basename(nnet.train_data_file).replace('.full', '')
    test = os.path.basename(data).replace('.full', '')
    base_file = os.path.join(dir_, model+'_'+train+'_'+test)
    with open(base_file + '.scores', 'w') as f:
        for i in range(result['num']):
            f.write(str(result['id'][i]))
            for j in result['score'][i]:
                f.write(' '+str(j))
            f.write('\n')
    return base_file

def write_roc_curves(roc_data, plot_name):

    '''plot_data should be a list of dictionaries. Each dict should have:
            data_label - the string for the legend
            auc - value as returned by auc function
            fpr - array as returned by roc_curve function
            tpr - array as returned by roc_curve function'''

    plt.figure(1)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')

    plt.figure(2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')

    # sort the plots by area under curve
    roc_data.sort(key=lambda d: d["roc_auc"], reverse=True)

    # iterate through plot data and add to graphs, full data gets its own graph
    for i in roc_data:
        if 'full' in i['data_label']:
            plt.figure(1)
            plt.plot(i['fpr'], i['tpr'], lw=1, label='%s (area = %0.2f)' % (i['data_label'], i['roc_auc']))
        elif 'mean' in i['data_label']:
            plt.figure(2)
            plt.plot(i['fpr'], i['tpr'], lw=1, label='%s (area = %0.2f)' % (i['data_label'], i['roc_auc']))
        else:
            plt.figure(2)
            plt.plot(i['fpr'], i['tpr'], 'k--', lw=2, label='%s (area = %0.2f)' % (i['data_label'], i['roc_auc']))

    # finally, add the legend and save the plots as .png files
    plt.figure(1)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_name + 'full-roc.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # figure 2, which is for the partitioned/split data, will have a mean line
    plt.figure(2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_name + 'split-roc.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def parse_args(argv):

    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Generate ROC curves for a trained model.',
        epilog=None)
    parser.add_argument('SOLVER_PATTERN')
    parser.add_argument('--data_pattern', '-d', type=str)
    parser.add_argument('--output_prefix', '-o', type=str)
    parser.add_argument('--roc', '-r', type=int)
    return parser.parse_args(argv)

if __name__ == "__main__":
    sys.exit(main())












def old_main():

    solver_glob = sorted(glob.glob(args.SOLVER_PATTERN))

    caffe.set_mode_gpu()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_plot_data = []

    for solver_file in sorted(solver_glob):

        print("Parsing the model name and iterations from " + solver_file)
        try: # parse the model name and max iterations from solver file
            solver_fp = open(solver_file, "r")
            solver = [i.lstrip().rstrip().split() for i in solver_fp.readlines()]
            for line in solver:
                if line and line[0] == "net:":
                    model_file = line[1].strip("\"")
                elif line and line[0] == "max_iter:":
                    max_iter = int(line[1])
            solver_fp.close()
        except IOError:
            print("Error: couldn't access a solver file for testing")
            sys.exit(1)
        print("\tnet: " + model_file)
        print("\tmax_iter: " + str(max_iter))


        if args.data_pattern is None:
            print("Parsing the test data source from " + model_file)
            try: # parse the test data source from model file
                model_fp = open(model_file, "r")
                model = [i.lstrip().rstrip().split() for i in model_fp.readlines()]
                for line in model:
                    if line and line[0] == "source:":
                        data_file = line[1].strip("\"") # just use the last source in model
                model_fp.close()
            except IOError:
                print("Error: couldn't access a model file for testing")
                sys.exit(1)
            print("\ttest source: " + data_file)
            data_glob = glob.glob(data_file)
        else:
            print("Gathering the test data matching the pattern " + args.data_pattern)
            data_glob = glob.glob(args.data_pattern)

        deploy_file = re.sub(r"_part.*?.model.prototxt", "_deploy.model.prototxt", model_file)
        deploy_file = re.sub("_full.model.prototxt", "_deploy.model.prototxt", deploy_file)

        weights_file = model_file.replace(".model.prototxt", "_iter_"+str(max_iter)+".caffemodel")
        print("Testing " + deploy_file + " with weights from " + weights_file)

        net = caffe.Net(deploy_file, weights_file, caffe.TEST)

        for i in sorted(data_glob):
            print(i)
            lmdb_env = lmdb.open(i)
            lmdb_txn = lmdb_env.begin()
            lmdb_cursor = lmdb_txn.cursor()

            score = []
            truth = []
            for key, value in lmdb_cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                arr = caffe.io.datum_to_array(datum)
                out = net.forward_all(data_blob=np.asarray([arr]))

                truth.append(datum.label)
                pred = out["output_act_blob"][0]

                if len(pred) == 2: # binary classes
                    # for now just use athe positive class prediction
                    # but there's probably a way of using both class
                    # predictions to get a confidence value
                    score.append(pred[1])

                elif len(pred) == 1:
                    score.append(pred[0])

                else:
                    raise IndexError('net output is not valid for binary \
                        classification')

            fpr, tpr, thresholds = roc_curve(truth, score)
            roc_auc = auc(fpr, tpr)
            data_label = os.path.basename(i).replace(".prototxt", "")
            is_full_data = False

            if "full" in i:
                is_full_data = True
            else:
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

            curr_plot_data = {"data_label":data_label, "roc_auc":roc_auc, "fpr":fpr, "tpr":tpr, "is_full_data":is_full_data}
            all_plot_data.append(curr_plot_data)

    # we now have the plot data, let's plot them

    # plot the random guess line and label the axes

