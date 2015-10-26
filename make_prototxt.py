import sys
import os
import re
import argparse
import glob



class Template:

	"""Template is a container for a TEMPLATE.prototxt file.

	Keep track of the filename, and a string buffer of its contents.
	"""

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


def write_prototxt(data_prefix, model_temp, solver_temp, output_dir, id_):

	# to get this model's name, replace TEMPLATE in model template name with the model id
	model_name = os.path.basename(model_temp.name).replace("TEMPLATE", id_)
	model_name = os.path.join(output_dir, model_name)

	# to get the model solver's name, repalce TEMPLATE in solver template name with model id
	solver_name = os.path.basename(solver_temp.name).replace("TEMPLATE", id_)
	solver_name = os.path.join(output_dir, solver_name)

	# output prefix says where to put the trained weights file, specified in the solver
	output_name = os.path.basename(model_temp.name).replace("TEMPLATE.prototxt", id_)
	output_prefix = os.path.join(output_dir, output_name)

	# replace MODEL_NAME with the model's name, in model and solver prototxt
	# also replace OUTPUT_PREFIX with the weight prefix, in solver
	model  = model_temp.contents.replace("MODEL_NAME", model_name)
	solver = solver_temp.contents.replace("MODEL_NAME", model_name)
	solver = solver.replace("OUTPUT_PREFIX", output_prefix)

	# if we're writing the deploy prototxt, keep the <deploy></deploy> section and remove the <train></train> section
	if id_ == "deploy":
		model = model.replace("<deploy>", "").replace("</deploy>", "")
		model = re.sub(r'<train>.*?</train>', "", model, flags=re.DOTALL)

	# otherwise, get rid of the <deploy></deploy> section and keep the <train></train> section
	elif id_ == "full":
		full_lmdb = data_prefix+".full"
		model = model.replace("TRAIN_LMDB", full_lmdb) # no TEST_LMDB 

		model = model.replace("<train>", "").replace("</train>", "")
		model = re.sub(r'<deploy>.*?</deploy>', "", model, flags=re.DOTALL)

		# there's no test phase when training on the full dataset, so get rid of <crossval></crossval> section
		model = re.sub(r'<crossval>.*?</crossval>', "", model, flags=re.DOTALL)
		solver = re.sub(r'<crossval>.*?</crossval>', "", solver, flags=re.DOTALL)

	# for partitioned data, we have train and test phases
	else:
		train_lmdb = data_prefix+"."+id_+".train"
		test_lmdb  = data_prefix+"."+id_+".test"
		model = model.replace("TRAIN_LMDB", train_lmdb)
		model = model.replace("TEST_LMDB",  test_lmdb)

		model = model.replace("<train>", "").replace("</train>", "")
		model = re.sub(r'<deploy>.*?</deploy>', "", model, flags=re.DOTALL)

		# keep the <crossval></crossval> sections so that we alternate train and test data while training
		model = model.replace("<crossval>", "").replace("</crossval>", "")
		solver = solver.replace("<crossval>", "").replace("</crossval>", "")

	# finally, write the model prototxt
	print("\t" + model_name)
	model_file = open(model_name, "w")
	model_file.write(model)
	model_file.close()

	# the deploy file is only used after training, so it doesn't need a solver prototxt
	if id_ != "deploy":
		print("\t" + solver_name)
		solver_file = open(solver_name, "w")
		solver_file.write(solver)
		solver_file.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog=__file__,
		description='Generate the model and solver prototxt to train in Caffe.',
		epilog=None)
	parser.add_argument('DATA_PREFIX')
	parser.add_argument('MODEL_TEMPLATE_PATTERN')
	parser.add_argument('SOLVER_TEMPLATE_FILE')
	parser.add_argument('OUTPUT_DIRECTORY')
	args = parser.parse_args()

	# Generate models and solvers for cross-validation and deployment of a glob of model templates.
	# Each model template generates 12 actual model prototxt files: one trained on the "full" dataset,
	# 10 trained on partitions of the dataset (for cross-validation), and one that has no "data layer",
	# which is used for deployment/testing after training. Any of the weight files from the other 11
	# trained models can be supplied to the "deploy" file at the command line.
	#
	# Since the solvers have a "model" specification, also need to generate a specific solver prototxt
	# for each generated model prototxt, so 12 of those too (per model template). That's a ton of prototxt
	# files, which all get outputted to the 'output directory' argument with unique filenames.

	print("Model templates:")
	model_temp_glob = glob.glob(args.MODEL_TEMPLATE_PATTERN)
	for i in model_temp_glob: print("\t" + i)

	print("Solver template:")
	solver_temp = Template(args.SOLVER_TEMPLATE_FILE)
	print("\t" + solver_temp.name)

	print("Using dataset:")
	print("\t" + args.DATA_PREFIX)

	for i in model_temp_glob:
		model_temp = Template(i)
		print("Generating prototxt for " + i)
		write_prototxt(None, model_temp, solver_temp, args.OUTPUT_DIRECTORY, "deploy") # deploy is for using the trained weights
		write_prototxt(args.DATA_PREFIX, model_temp, solver_temp, args.OUTPUT_DIRECTORY, "full") 
		for i in range(10):
			write_prototxt(args.DATA_PREFIX, model_temp, solver_temp, args.OUTPUT_DIRECTORY, "part"+str(i))

	print("Done, without errors.")


