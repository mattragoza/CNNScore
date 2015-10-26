import sys
import os
import glob
import datetime
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Train caffe models specified in a glob of solver files.',
        epilog=None)
    parser.add_argument('SOLVER_PATTERN')
    args = parser.parse_args()

    solver_arg = args.SOLVER_PATTERN

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    solver_glob = sorted(glob.glob(solver_arg))
    for solver_file in solver_glob:

        command = "caffe train --solver="+solver_file + \
            " 2>&1 | tee -a logs/"+timestamp+".log"
        os.system(command)