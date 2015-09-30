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
        "label":0
    },
}

USAGE = "python make_lmdb.py <data>\n"



class Database:

	def __init__(self, file_=None, format_=None):

		if file_ is not None and format_ is not None:
			self.read_csv(file_, format_)

		else:
			self._source = None
			self._format = None

			self._samples = []
			self._groups  = {}
			self._parts = []

			self.nbytes = 0


	def read_csv(self, file_, format_):

		# parse the csv_file in a known format
		csv_file   = open(file_, "r")
		csv_format = CSV_FORMATS[format_]
		csv_reader = csv.reader(csv_file, delimiter=csv_format["delimiter"])

		self._source = file_
		self._format = csv_format
		self._samples = []
		self._groups = {}
		self._parts = []

		# create a Database.Sample() object for each row in the csv_file
		# also keep track of all groups present in the data in a set()
		for row in csv_reader:
			s = Database.Sample(self, row[csv_format["id"]], row[csv_format["group"]],
				               [float(row[i]) for i in csv_format["data"]], int(row[csv_format["label"]]))
			self._samples.append(s)

			if s._group in self._groups:
				self._groups[s._group].append(s)
			else:
				self._groups[s._group] = [s]
		
		# get a *rough* estimate of the number of bytes of data, assuming float64
		self.nbytes = 8 * len(self._format["data"]) * len(self._samples)
		csv_file.close()


	def write_lmdb(self, dir_):

		# for each partition in the database
		for i in range(len(self._parts)):

			partition  = self._parts[i]
			source_file = os.path.basename(self._source)

			# we need to create train and test lmdbs
			train_path = os.path.join(dir_, source_file+"."+str(i)+".train")
			test_path  = os.path.join(dir_, source_file+"."+str(i)+".test")
			
			# open the memory mapped environment associated with each lmdb
			train_lmdb = lmdb.open(train_path,  map_size=4*self.nbytes)
			test_lmdb  = lmdb.open(test_path,   map_size=4*self.nbytes)

			# writing training set data and labels
			with train_lmdb.begin(write=True) as txn:
				for s in partition.train_set():
					datum = caffe.proto.caffe_pb2.Datum()
					datum.channels = len(self._format["data"])
					datum.height = 1
					datum.width  = 1
					datum.float_data.extend(s._data);
					datum.label = s._label
					txn.put(key=s._id.encode("ascii"), value=datum.SerializeToString())

			# write test set data and labels
			with test_lmdb.begin(write=True) as txn:
				for s in partition.test_set():
					datum = caffe.proto.caffe_pb2.Datum()
					datum.channels = len(self._format["data"])
					datum.height = 1
					datum.width  = 1
					datum.float_data.extend(s._data);
					datum.label = s._label
					txn.put(key=s._id.encode("ascii"), value=datum.SerializeToString())

			# close the lmdb environments
			train_lmdb.close()
			test_lmdb.close()
	
	def balanced_partition(self, n):

		# sort the groups in a list based on number of samples
		sorted_groups = [(g, len(self._groups[g])) for g in self._groups]
		sorted_groups.sort(key=lambda tup: tup[1], reverse=True)

		index = 0
		forward = True
		folds = [[] for i in range(n)]
		for g in sorted_groups:
			folds[index].append(g[0])
			if forward:
				if index < n-1:
					index += 1
				else:
					forward = False
			else:
				if index > 0:
					index -= 1
				else:
					forward = True

		for i in range(n):
			test_set = folds[i]
			train_set = [j for f in folds[:i] for j in f] + [j for f in folds[i+1:] for j in f]
			self._parts.append(Database.Partition(self, train_set, test_set))


	class Partition:

		def __init__(self, db, train, test):

			self._db = db
			self._train_set = set(train)
			self._test_set  = set(test)

		def train_set(self):

			for s in self._db._samples:
				if s._group in self._train_set:
					yield s

		def test_set(self):

			for s in self._db._samples:
				if s._group in self._test_set:
					yield s


	class Sample:

		def __init__(self, db, id_, group, data, label):

			self._id    = id_
			self._group = group
			self._data  = data
			self._label = label






if __name__ == "__main__":

	usage_format = USAGE.strip("\n").split()[1:]
	if len(sys.argv) < len(usage_format):
		print("Usage: " + USAGE)
		sys.exit(1)

	file_arg   = sys.argv[usage_format.index("<data>")]

	print("Gathering data from " + file_arg)
	try: db = Database(file_arg, "DUDE_SCOREDATA")
	except IOError:
		print("Error: could not access the input file")
		sys.exit(1)
	except KeyError:
		print("Error: unknown input file format")
		sys.exit(1)
	print(str(db.nbytes) + " bytes read")

	print("Generating balanced partitions")
	db.balanced_partition(n=10)

	print("Converting to lmdb format")
	try: db.write_lmdb("lmdb/")
	except IOError:
		print("Error: could not access the output location")
		sys.exit(1)

	print("Done, without errors.")
	sys.exit(0)