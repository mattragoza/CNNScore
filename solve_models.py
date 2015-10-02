import sys
import os
import glob
import datetime

USAGE = "python solve_models.py <input_dir>"

if __name__ == "__main__":

    usage_format = USAGE.split()[1:]
    if len(sys.argv) < len(usage_format):
        print("Usage: " + USAGE)
        sys.exit(1)

    input_arg = sys.argv[usage_format.index("<input_dir>")]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    for solver in glob.glob(input_arg):

        command = "caffe train --solver="+solver + \
            "2>&1 | tee -a logs/"+timestamp+".log"
        print(command)
        #os.system(command)