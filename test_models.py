import sys
import os
import re
import argparse
from glob import glob
import numpy as np
import caffe
import lmdb
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CaffeModel(object):

    '''CaffeModel represents a neural network model in Caffe. It also
    organizes the various prototxt and data files that are used for
    training and testing/deploying a model.'''

    def __init__(self):

        self.train_solver_file = None
        self.train_model_file = None
        self.train_model_net = None

        self.train_data_file = None
        self.test_data_file = None

        self.deploy_weight_file = None
        self.deploy_model_file = None
        self.deploy_model_net = None

    @staticmethod
    def from_solver_prototxt(solver_file):

        self = CaffeModel()
        self._parse_solver_prototxt(solver_file)
        return self

    def _parse_solver_prototxt(self, solver_file):

        with open(solver_file, 'r') as solver_fp:
            solver = [i.lstrip().rstrip().split() for i in solver_fp]

        for line in solver:
            if line is None:
                continue
            elif line[0] == 'net:':
                train_model_file = line[1].strip('"')
            elif line[0] == 'max_iter:':
                max_iter = int(line[1])

        deploy_model_file = re.sub(r'_part.*?.model|_full.model', \
            '_deploy.model', train_model_file)
        deploy_weight_file = re.sub('.model.prototxt', \
            '_iter_'+str(max_iter)+'.caffemodel', train_model_file)

        self.train_solver_file = solver_file
        self.train_model_file = train_model_file
        self._parse_model_prototxt(train_model_file)

        self.deploy_model_file = deploy_model_file
        self.deploy_weight_file = deploy_weight_file

    def _parse_model_prototxt(model_file=None):

        if model_file is None:
            model_file = self.train_model_file

        with open(model_file, 'r') as model_fp:
            model = [i.lstrip().rstrip().split() for i in model_fp]

        data_files = []
        for line in model:
            if line is None:
                continue
            elif line[0] == "source:":
                data_files.append(line[1].strip("\""))

        # TODO need to handle this better
        self.train_data_file = data_files[0]
        self.test_data_file = data_files[1]

    def test_lmdb(data_file):

        lmdb_env = lmdb.open(data_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()

        truths, scores = [], []
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            features = caffe.io.datum_to_array(datum)

            truth = datum.label
            score = self.deploy_model_net.forward_all(data_blob=features)
            score = score['output_act_blob'][0]

            truths.append(truth)
            scores.append(score[-1])

        return truths, scores

def main(argv=sys.argv[1:]):

    args = parse_args(argv)
    caffe.set_mode_gpu()


    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_plot_data = []

    for s in sorted(glob(args.SOLVER_PATTERN)):

        nnet = CaffeModel.from_solver_prototxt(s)
        nnet.setup_test()

        if args.data_pattern is None:
            data_glob = [nnet.test_data_file]
        else:
            data_glob = glob(args.data_pattern)

        for d in data_glob:

            truths, scores = nnet.test_lmdb(d)
            fpr, tpr, thresholds = roc_curve(truths, scores)
            roc_auc = auc(fpr, tpr)
            data_label = os.path.basename(d).replace('.prototxt', '')
            if 'full' in data_label:
                is_full_data = True
            else:
                is_full_data = False
                mean_tpr += numpy.interp(mean_fpr, fpr, tpr)

            plot_data = {'data_label':data_label,
                         'roc_auc':roc_auc,
                         'fpr':fpr,
                         'tpr':tpr,
                         'is_full_data':is_full_data}

            all_plot_data.append(plot_data)

    make_roc_plots(all_plot_data)

    return 0

def parse_args(argv):

    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Generate ROC curves for a trained model.',
        epilog=None)
    parser.add_argument('SOLVER_PATTERN')
    parser.add_argument('--data_pattern', "-d", type=str)
    return parser.parse_args(argv)

if __name__ == "__main__":
    sys.exit(main())












def old_main():

    solver_glob = sorted(glob.glob(args.SOLVER_PATTERN))

    caffe.set_mode_gpu()
    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0, 1, 100)
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
                out = net.forward_all(data_blob=numpy.asarray([arr]))

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
                mean_tpr += numpy.interp(mean_fpr, fpr, tpr)

            curr_plot_data = {"data_label":data_label, "roc_auc":roc_auc, "fpr":fpr, "tpr":tpr, "is_full_data":is_full_data}
            all_plot_data.append(curr_plot_data)

    # we now have the plot data, let's plot them

    # plot the random guess line and label the axes
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
    all_plot_data.sort(key=lambda curr_plot_data: curr_plot_data["roc_auc"], reverse=True)

    # iterate through plot data and add to graphs, full data gets its own graph
    for pd in all_plot_data:
        if pd["is_full_data"]:
            plt.figure(1)
            plt.plot(pd["fpr"], pd["tpr"], lw=1, label='%s (area = %0.2f)' % (pd["data_label"], pd["roc_auc"]))
        else:
            plt.figure(2)
            plt.plot(pd["fpr"], pd["tpr"], lw=1, label='%s (area = %0.2f)' % (pd["data_label"], pd["roc_auc"]))

    # finally, add format/add the legend and save the graphs as .png
    plt.figure(1)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(deploy_file.replace("deploy.model.prototxt", "full-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')

    # figure 2, which is for the partitioned/split data, will have a mean line
    plt.figure(2)
    mean_tpr[0]  = 0.0
    mean_tpr /= len(all_plot_data)-1
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='mean (area = %0.2f)' % mean_auc, lw=2) # add the mean curve last so it's on top
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(deploy_file.replace("deploy.model.prototxt", "split-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
