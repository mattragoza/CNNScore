import sys
import os
import numpy
import csv
import lmdb
import caffe
import random

CSV_FORMATS = {
    "DUDE_SCOREDATA": {
    	"delimiter":' ',
        "id":2,
        "group":1,
        "data":list(range(4, 65)),
        "label":[0]
    },

    "TEST_DATA": {
    	"delimiter":',',
    	"id":0,
    	"group":1,
    	"data":[2],
    	"label":[3]
    }
}

USAGE = "python make_lmdb.py <file> <format> <partitions> <dir>\n"



class Database:

	def __init__(self):

		self._source = None
		self._format = None
		self._samples = []
		self._groups = set()

		self._nparts = 0
		self._parts = []
		self._by = None

		self._nbytes = 0


	def __repr__(self):

		s = "source: " + self._source + "\n" + \
		    "samples: " + str(len(self._samples)) + "\n" + \
		    "groups: " + str(len(self._groups)) + "\n" + \
		    "partitions: " + str(self._nparts) + "\n" + \
		    "bytes: " + str(self._nbytes) + "\n"
		return s

	def read_csv(self, file_, format_):

		# parse the csv_file in a known format
		csv_file   = open(file_, "r")
		csv_format = CSV_FORMATS[format_]
		csv_reader = csv.reader(csv_file, delimiter=csv_format["delimiter"])

		self.empty()
		self._source = file_
		self._format = csv_format
		# create a Database.Sample() object for each row in the csv_file
		# also keep track of all groups present in the data in a set()
		for row in csv_reader:
			s = Database.Sample(self, row[csv_format["id"]], row[csv_format["group"]],
				               [row[i] for i in csv_format["data"]],
				               [row[i] for i in csv_format["label"]])
			self._samples.append(s)
			self._groups.add(s._group)
		
		# get a *rough* estimate of the number of bytes of data by summing the numpy.array() bytes
		self._nbytes = 8 * (len(self._format["data"]) + len(self._format["label"])) * len(self._samples)
		csv_file.close()
		return self

	def write_lmdb(self, dir_):

		# for each partition in the database
		for i in range(self._nparts):

			partition  = self._parts[i]
			source_file = os.path.basename(self._source)

			# we need to create 4 lmdbs- training data, training label, test data, and test label
			train_data_path  = os.path.join(dir_, source_file+"."+str(i)+".train.data")
			train_label_path = os.path.join(dir_, source_file+"."+str(i)+".train.label")
			test_data_path   = os.path.join(dir_, source_file+"."+str(i)+".test.data")
			test_label_path  = os.path.join(dir_, source_file+"."+str(i)+".test.label")
			
			# open the memory mapped environment associated with each lmdb
			train_data_lmdb  = lmdb.open(train_data_path,  map_size=2*self._nbytes)
			train_label_lmdb = lmdb.open(train_label_path, map_size=2*self._nbytes)
			test_data_lmdb   = lmdb.open(test_data_path,   map_size=2*self._nbytes)
			test_label_lmdb  = lmdb.open(test_label_path,  map_size=2*self._nbytes)

			# writing training set data and labels
			with train_data_lmdb.begin(write=True) as data_txn:
				with train_label_lmdb.begin(write=True) as label_txn:
					for s in partition.train_set():
						data_datum  = caffe.io.array_to_datum(s._data)
						label_datum = caffe.io.array_to_datum(s._label)
						data_txn.put(key=s._id.encode("ascii"), value=data_datum.SerializeToString())
						label_txn.put(key=s._id.encode("ascii"), value=label_datum.SerializeToString())

			# write test set data and labels
			with test_data_lmdb.begin(write=True) as data_txn:
				with test_label_lmdb.begin(write=True) as label_txn:
					for s in partition.test_set():
						data_datum  = caffe.io.array_to_datum(s._data)
						label_datum = caffe.io.array_to_datum(s._label)
						data_txn.put(key=s._id.encode("ascii"), value=data_datum.SerializeToString())
						label_txn.put(key=s._id.encode("ascii"), value=label_datum.SerializeToString())

			# close the lmdb environments
			train_data_lmdb.close()
			train_label_lmdb.close()
			test_data_lmdb.close()
			test_label_lmdb.close()
	
	def make_partitions(self, num, by_group):

		# for by_group, we divide the database's groups into num partitons, where in each one
		# we use 1/num of the groups' samples as the test set and the rest as the training set
		if by_group:

			# randomly shuffle the order of the groups in a list
			groups = list(self._groups)
			random.shuffle(groups)

			# iterate through the groups in steps of size #groups/#partitions
			self._parts = []
			p = int(len(groups)/num)
			for i in range(0, len(groups), p):

				# use the current step as the test set and the rest as the training set in a partition
				train  = groups[0:i] + groups[i+p:]
				test = groups[i:i+p]
				self._parts.append(Database.Partition(self, train, test))

			self._nparts = num
			self._by = "_group"

		else: # TODO should have this even though we're probably not using it

			self.unpartition()
			print("TODO")

	def unpartition(self):

		self._parts = []
		self._nparts = 0
		self._by = None

	def empty(self):

		self.unpartition()
		self._groups = set()
		self._samples = []
		self._nbytes = 0
		self._source = None



	class Partition:

		def __init__(self, db, train, test):

			self._db = db
			self._train_set = set(train)
			self._test_set  = set(test)

		def train_set(self):

			for s in self._db._samples:
				if getattr(s, self._db._by) in self._train_set:
					yield s

		def test_set(self):

			for s in self._db._samples:
				if getattr(s, self._db._by) in self._test_set:
					yield s



	class Sample:

		def __init__(self, db, id_, group, data, label):

			self._id    = id_
			self._group = group
			self._data  = numpy.empty((len(db._format["data"]), 1, 1),  dtype=numpy.float64)
			self._label = numpy.empty((len(db._format["label"]), 1, 1), dtype=numpy.float64)
			for i in range(len(db._format["data"])):  self._data[i, 0, 0]  = data[i]
			for i in range(len(db._format["label"])): self._label[i, 0, 0] = label[i]

		def __repr__(self):

			s = "id: " + str(self._id) + "\n" + \
				"group: " + str(self._group) + "\n" + \
				"data: " + str(self._data.shape) + "\n" + \
				"label: " + str(self._label.shape) + "\n"
			return s




if __name__ == "__main__":

	usage_format = USAGE[7:-1].split()
	if len(sys.argv) < len(usage_format):
		print("Usage: " + USAGE)
		sys.exit(1)

	file_arg   = sys.argv[usage_format.index("<file>")]
	format_arg = sys.argv[usage_format.index("<format>")]
	parts_arg  = sys.argv[usage_format.index("<partitions>")]
	output_arg = sys.argv[usage_format.index("<dir>")]

	db = Database()
	print("Gathering data from " + file_arg)
	try: db.read_csv(file_arg, format_arg)
	except IOError:
		print("Error: could not access the input file")
		sys.exit(1)
	except KeyError:
		print("Error: unknown input file format")
		sys.exit(1)

	print("Generating " + parts_arg + " partitions")
	db.make_partitions(num=int(parts_arg), by_group=True)

	print("Converting to lmdb format in " + output_arg)
	try: db.write_lmdb(output_arg)
	except IOError:
		print("Error: could not access the output location")
		sys.exit(1)

	print("Done, without errors.")
	sys.exit(0)