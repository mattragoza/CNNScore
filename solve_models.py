import sys
import os
import glob
import datetime
import caffe
import numpy
import re

USAGE = "python solve_models.py <input_pattern>"

if __name__ == "__main__":

    usage_format = USAGE.split()[1:]
    if len(sys.argv) < len(usage_format):
        print("Usage: " + USAGE)
        sys.exit(1)

    input_arg = sys.argv[usage_format.index("<input_pattern>")]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    solver_glob = glob.glob(input_arg)
    for solver_file in solver_glob:

        command = "caffe train --solver="+solver_file + \
            " 2>&1 | tee -a logs/"+timestamp+".log"
        os.system(command)

def test_models(): #TODO probably make this its own script, or combine everything into one

    caffe.set_mode_gpu()
    for solver_file in solver_glob:

        try:
            solver = open(solver_file, "r")
            model_file = solver.readline().split(" ")[1].strip("\"\n")
            solver.close()
        except:
            print("Error: couldn't access a solver file for testing")
            sys.exit(1)

        try:
            model = open(model_file, "r")
            model = [i.lstrip().rstrip().split() for i in model.readlines()]

            for line in model:
                if line and line[0] == "source:":
                    data_file = line[1].strip("\"") # just use the last source in model

        except:
            print("Error: couldn't access a model file for testing")
            sys.exit(1)

        weights_file = model_file.replace(".prototxt", "_iter_10000.caffemodel")
        
        print(model_file, weights_file, data_file)
        net = caffe.Net(model_file, weights_file, caffe.Test)
        lmdb_env = lmdb.open(data_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        for key, value in lmdb_cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            arr = caffe.io.datum_to_array(datum)
            out = net.forward_all(data=numpy.asarray([arr]))