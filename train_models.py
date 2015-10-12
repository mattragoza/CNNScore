import sys
import os
import glob
import datetime

USAGE = "python train_models.py <input_pattern>"

if __name__ == "__main__":

    usage_format = USAGE.split()[1:]
    if len(sys.argv) < len(usage_format):
        print("Usage: " + USAGE)
        sys.exit(1)

    input_arg = sys.argv[usage_format.index("<input_pattern>")]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    solver_glob = sorted(glob.glob(input_arg))
    for solver_file in solver_glob:

        command = "caffe train --solver="+solver_file + \
            " 2>&1 | tee -a logs/"+timestamp+".log"
        os.system(command)