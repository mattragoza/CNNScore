import sys
import os
import re

USAGE = "python make_prototxt.py <data_source> <model_template> <solver_template> <output_dir>"


class Template:

    def __init__(self, file_):

        if file_ is not None:
            self.read_file(file_)
        else:
            self.name = None
            self.contents = None

    def read_file(self, file_):

        with open(file_) as f:
            self.name = file_
            self.contents = f.read()


def write_prototxt(data_name, model_temp, solver_temp, output_dir, id_):

    model_name = os.path.basename(model_temp.name).replace("TEMPLATE", id_)
    model_name = os.path.join(output_dir, model_name)

    solver_name = os.path.basename(solver_temp.name).replace("TEMPLATE", id_)
    solver_name = os.path.join(output_dir, solver_name)

    output_name = os.path.basename(model_temp.name).replace("TEMPLATE.prototxt", id_)
    output_prefix = os.path.join(output_dir, output_name)

    model  = model_temp.contents.replace("$model_name", model_name)
    solver = solver_temp.contents.replace("$model_name", model_name)
    solver = solver.replace("$output_prefix", output_prefix)

    if id_ == "full": # train on entire set, no test phase
        full_lmdb = data_name+".full"
        model = model.replace("$train_lmdb", full_lmdb)
        model = model.replace("$test_lmdb",  full_lmdb)
        model = re.sub(r'//.*?\n|/\*.*?\*/', "", model, re.S)
        solver = re.sub(r'//.*?\n|/\*.*?\*/', "", solver, re.S)

    else: # cross-validation training and testing
        train_lmdb = data_name+"."+id_+".train"
        test_lmdb  = data_name+"."+id_+".test"
        model = model.replace("$train_lmdb", train_lmdb)
        model = model.replace("$test_lmdb",  test_lmdb)
        model = model.replace("//", "") # uncomment lines relevant to testing
        solver = solver.replace("//", "")


    model_file = open(model_name, "w")
    model_file.write(model)
    model_file.close()

    solver_file = open(solver_name, "w")
    solver_file.write(solver)
    solver_file.close()


if __name__ == "__main__":

    usage_format = USAGE.split()[1:]
    if len(sys.argv) < len(usage_format):
        print("Usage: " + USAGE)
        sys.exit(1)

    data_arg   = sys.argv[usage_format.index("<data_source>")]
    model_arg  = sys.argv[usage_format.index("<model_template>")]
    solver_arg = sys.argv[usage_format.index("<solver_template>")]
    output_arg = sys.argv[usage_format.index("<output_dir>")]

    print("Generating prototxt")
    try: 
        model_temp  = Template(model_arg)
        solver_temp = Template(solver_arg)

        write_prototxt(data_arg, model_temp, solver_temp, output_arg, "full")
        for i in range(10):
            write_prototxt(data_arg, model_temp, solver_temp, output_arg, "part"+str(i))

    except IOError:
        print("Error: could not read or write a file")
        sys.exit(1)

    print("Done, without errors.")


