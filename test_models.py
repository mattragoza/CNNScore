import matplotlib
matplotlib.use('Agg')
import sys
import os
import caffe
import lmdb
import glob
import numpy
import re
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

USAGE = "python test_models.py <input_pattern>"

if __name__ == "__main__":

    usage_format = USAGE.split()[1:]
    if len(sys.argv) < len(usage_format):
        print("Usage: " + USAGE)
        sys.exit(1)

    input_arg = sys.argv[usage_format.index("<input_pattern>")]
    solver_glob = sorted(glob.glob(input_arg))

    caffe.set_mode_gpu()
    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0, 1, 100)
    for solver_file in solver_glob:

        print("Parsing " + solver_file)
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
        print("net: " + model_file)
        print("max_iter: " + str(max_iter))

        print("Parsing " + model_file)
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
        print("test source: " + data_file)

        deploy_file = re.sub(r"_part.*?.prototxt", "_deploy.prototxt", model_file)
        deploy_file = re.sub("_full.prototxt", "_deploy.prototxt", deploy_file)

        weights_file = model_file.replace(".prototxt", "_iter_"+str(max_iter)+".caffemodel")

        net = caffe.Net(deploy_file, weights_file, caffe.TEST)
        lmdb_env = lmdb.open(data_file)
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
            score.append(out["output_act_blob"][0, 0])

        fpr, tpr, thresholds = roc_curve(truth, score)
        roc_auc = auc(fpr, tpr)


        model_label = os.path.basename(model_file).replace(".prototxt", "")
        if "full.prototxt" in model_file:
            plt.figure(1)
            plt.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' % (model_label, roc_auc))
        else:
            mean_tpr += numpy.interp(mean_fpr, fpr, tpr)

            plt.figure(2)
            plt.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' % (model_label, roc_auc))

    mean_tpr[0]  = 0.0
    mean_tpr /= 10
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(deploy_file.replace("deploy.prototxt", "testontrain-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(2)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='mean (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(deploy_file.replace("deploy.prototxt", "crossval-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')