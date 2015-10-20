import sys
import os
import re
import argparse



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

	model  = model_temp.contents.replace("MODEL_NAME", model_name)
	solver = solver_temp.contents.replace("MODEL_NAME", model_name)
	solver = solver.replace("OUTPUT_PREFIX", output_prefix)

	if id_ == "deploy": # this is for using the model once it's trained
		model = model.replace("<deploy>", "").replace("</deploy>", "")
		model = re.sub(r'<train>.*?</train>', "", model, flags=re.DOTALL)

	elif id_ == "full": # train on entire set, no test phase
		full_lmdb = data_name+".full"
		model = model.replace("TRAIN_LMDB", full_lmdb)
		model = model.replace("TEST_LMDB",  full_lmdb)

		model = model.replace("<train>", "").replace("</train>", "")
		model = re.sub(r'<deploy>.*?</deploy>', "", model, flags=re.DOTALL)
		model = re.sub(r'<crossval>.*?</crossval>', "", model, flags=re.DOTALL)
		solver = re.sub(r'<crossval>.*?</crossval>', "", solver, flags=re.DOTALL)

	else: # cross-validation training and testing
		train_lmdb = data_name+"."+id_+".train"
		test_lmdb  = data_name+"."+id_+".test"
		model = model.replace("TRAIN_LMDB", train_lmdb)
		model = model.replace("TEST_LMDB",  test_lmdb)

		model = model.replace("<train>", "").replace("</train>", "")
		model = re.sub(r'<deploy>.*?</deploy>', "", model, flags=re.DOTALL)
		model = model.replace("<crossval>", "").replace("</crossval>", "") # uncomment lines relevant to testing
		solver = solver.replace("<crossval>", "").replace("</crossval>", "")


	model_file = open(model_name, "w")
	model_file.write(model)
	model_file.close()

	if id_ != "deploy":
		solver_file = open(solver_name, "w")
		solver_file.write(solver)
		solver_file.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog=__file__,
		description='Generate the model and solver prototxt to train in Caffe.',
		epilog=None)
	parser.add_argument('INPUT_PATTERN')
	parser.add_argument('MODEL_TEMPLATE')
	parser.add_argument('SOLVER_TEMPLATE')
	parser.add_argument('OUTPUT_DIR')
	args = parser.parse_args()

	print("Generating prototxt")
	try:
		model_temp  = Template(args.MODEL_TEMPLATE)
		solver_temp = Template(args.SOLVER_TEMPLATE)

		write_prototxt(None, model_temp, solver_temp, args.OUTPUT_DIR, "deploy")
		write_prototxt(args.INPUT_PATTERN, model_temp, solver_temp, args.OUTPUT_DIR, "full") #deploy 
		for i in range(10):
			write_prototxt(args.INPUT_PATTERN, model_temp, solver_temp, args.OUTPUT_DIR, "part"+str(i))

	except IOError:
		print("Error: could not read or write a file")
		sys.exit(1)

	print("Done, without errors.")


