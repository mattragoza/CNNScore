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
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog=__file__,
		description='Generate ROC curves for a trained model.',
		epilog=None)
	parser.add_argument('SOLVER_PATTERN')
	parser.add_argument('--data_pattern', "-d", type=str)
	args = parser.parse_args()

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
		print("net: " + model_file)
		print("max_iter: " + str(max_iter))


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
			print("test source: " + data_file)
			data_glob = glob.glob(data_file)
		else:
			print("Gathering the test data matching the pattern " + args.data_pattern)
			data_glob = glob.glob(args.data_pattern)

		deploy_file = re.sub(r"_part.*?.prototxt", "_deploy.prototxt", model_file)
		deploy_file = re.sub("_full.prototxt", "_deploy.prototxt", deploy_file)

		weights_file = model_file.replace(".prototxt", "_iter_"+str(max_iter)+".caffemodel")

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
				score.append(out["output_act_blob"][0, 0])

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
	plt.savefig(deploy_file.replace("deploy.prototxt", "full-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')

	# figure 2, which is for the partitioned/split data, will have a mean line
	plt.figure(2)
	mean_tpr[0]  = 0.0
	mean_tpr /= len(all_plot_data)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='mean (area = %0.2f)' % mean_auc, lw=2) # add the mean curve last so it's on top
	lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig(deploy_file.replace("deploy.prototxt", "split-roc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
