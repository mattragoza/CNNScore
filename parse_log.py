import matplotlib
matplotlib.use('Agg')
import sys
import glob
import matplotlib.pyplot as plt



class Log:

	def __init__(self, file_):

		if file_ is not None:
			self.parse_log(file_)
		else:
			self._entries = []
			self._data = []

	def parse_log(self, file_):

		n = 0 # network
		solving = False
		self._entries = []
		self._data = []

		log_file = open(file_, "r")
		for line in log_file.readlines():

			e = line.split("]")[-1].lstrip().rstrip()
			self._entries.append(e)

			tok = e.split()

			if tok[0] == "Solving":

				solving = True
				i = 0
				name = tok[1]
				train = [[], []] # iter, error
				test  = [[], []]

			if solving:

				if tok[0] == "Iteration":
					i = int(tok[1][:-1]) # strip comma

				if tok[0] == "Train":
					loss = float(tok[6])
					train[0].append(i)
					train[1].append(loss)

				if tok[0] == "Test":
					loss = float(tok[6])
					test[0].append(i)
					test[1].append(loss)

				if tok[0] == "Optimization":
					solving = False
					self._data.append({"model":name, "train":train, "test":test})
					n += 1

	def write_plot(self, file_):

		plt.figure(1)
		plt.xlabel('Iterations')
		plt.ylabel('Euclidean Error')
		plt.title('Neural net training')
		for i in self._data:
			if i["train"]:
				plt.plot(i["train"][0], i["train"][1], lw=1, label=i["model"]+".train")
			if i["test"]:
				plt.plot(i["train"][0], i["train"][1], lw=1, label=i["model"]+".test")
		

		lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.savefig(file_, bbox_extra_artists=(lgd,), bbox_inches='tight')

	def write_csv(self, file_):

		csv_file = open(file_, "w")

		csv = "iteration"
		for n in range(len(self._data)):
			net = self._data[n]
			if len(net["test"][0]) > 0:
				csv += ",part" + str(n-1)
			else:
				csv += ",full"
		csv += "\n"

		# index in testing points
		j = 0

		# index in training points
		for i in range(len(self._data[0]["train"][0])):

			# iteration (use first partitioned model)
			tr = self._data[1]["train"][0][i]
			te = self._data[1]["test"][0][j]

			csv += str(tr)

			if te == tr: # a test iteration

				for net in self._data:

					if len(net["test"][0]) > 0:
						csv += "," + str(net["test"][1][j])
					else:
						csv += "," + str(net["train"][1][i])
						full_model = False

				j += 1

			else: # no testing done
				for net in self._data:
					csv += "," + str(net["train"][1][i]) + ","

			csv += "\n"

		csv_file.write(csv)
		csv_file.close()




if __name__ == "__main__":

	# TODO
	# read caffe training output from a pipe
	# send an early-stop signal to caffe process
	# when some error criteria is met... what criteria?

	file_arg = sys.argv[usage_format.index("<file_pattern>")]

	for log_file in glob.glob(file_arg):

		print("Reading from " + log_file)
		try: log = Log(file_=log_file)
		except IOError:
			print("Error: could not access the input file")
			sys.exit(1)

		print("Writing to " + log_file + ".csv")
		try: log.write_plot(file_=log_file+".png")
		except IOError:
			print("Error: could not access the output file")
			sys.exit(1)

	print("Done, without errors.")