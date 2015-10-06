import sys

USAGE = "python parse_log.py <log_file>"

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
				train = [[], []]
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
					self._data.append({"train":train, "test":test})
					n += 1

	def write_csv(self, file_):

		csv_file = open(file_, "w")

		csv = ""
		for n in range(len(self._data)):
			csv += "," + str(n) + "," + str(n)
		csv += "\niteration"
		for n in self._data:
			csv += ",train,test"
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
				full_model = True
				for net in self._data:
					if not full_model:
						csv += "," + str(net["train"][1][i]) + "," + str(net["test"][1][j])
					else:
						csv += "," + str(net["train"][1][i]) + ","
						full_model = False

				j += 1

			else: # no testing done
				for net in self._data:
					csv += "," + str(net["train"][1][i]) + ","

			csv += "\n"

		csv_file.write(csv)
		csv_file.close()




if __name__ == "__main__":

	usage_format = USAGE.split()[1:]
	if len(sys.argv) < len(usage_format):
		print("Usage: " + USAGE)
		sys.exit(1)

	file_arg = sys.argv[usage_format.index("<log_file>")]

	print("Reading from " + file_arg)
	try: log = Log(file_=file_arg)
	except IOError:
		print("Error: could not access the input file")
		sys.exit(1)

	print("Writing to " + file_arg + ".csv")
	try: log.write_csv(file_=file_arg+".csv")
	except IOError:
		print("Error: could not access the output file")
		sys.exit(1)

	print("Done, without errors.")