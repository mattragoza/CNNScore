import sys
import os
import argparse
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

		# create a dict for each row in the csv_file
		# also keep track of all groups present in the data in a set()
		for row in csv_reader:
			s= {"id": row[csv_format["id"]], "group": row[csv_format["group"]],
				"x": [float(row[i]) for i in csv_format["data"]],
				"y": int(row[csv_format["label"]])}

			# append to master data list
			self._samples.append(s)

			# also append to the dictionary of samples grouped by target
			if s["group"] in self._groups:
				self._groups[s["group"]].append(s)
			else:
				self._groups[s["group"]] = [s]
		
		csv_file.close()
		
		# get a *rough* estimate of the number of bytes of data, assuming float64
		self.nbytes = 8 * len(self._format["data"]) * len(self._samples)

	def write_lmdb(self, dir_):

		# write the entire database
		source_file = os.path.basename(self._source)
		full_path = os.path.join(dir_, source_file+".full")
		full_lmdb = lmdb.open(full_path, map_size=4*self.nbytes)
		with full_lmdb.begin(write=True) as txn:
			for s in self._samples:
				datum = caffe.proto.caffe_pb2.Datum()
				datum.channels = len(self._format["data"])
				datum.height = 1
				datum.width  = 1
				datum.float_data.extend(s["x"]);
				datum.label = s["y"]
				txn.put(key=s["id"].encode("ascii"), value=datum.SerializeToString())
		full_lmdb.close()

		# for each partition in the database
		for i in range(len(self._parts)):

			partition  = self._parts[i]
			n = partition._name

			# we need to create train and test lmdbs
			# open the memory mapped environment associated with each lmdb
			# writing training set data and labels

			if partition._train_set is not None:
				train_path = os.path.join(dir_, source_file+"."+n+".train")
				train_lmdb = lmdb.open(train_path, map_size=4*self.nbytes)
				print("\t" + train_path)
				with train_lmdb.begin(write=True) as txn:
					for s in partition.train_set():
						datum = caffe.proto.caffe_pb2.Datum()
						datum.channels = len(self._format["data"])
						datum.height = 1
						datum.width  = 1
						datum.float_data.extend(s["x"]);
						datum.label = s["y"]
						txn.put(key=s["id"].encode("ascii"), value=datum.SerializeToString())
				train_lmdb.close()

			if partition._test_set is not None:
				test_path  = os.path.join(dir_, source_file+"."+n+".test")
				test_lmdb  = lmdb.open(test_path, map_size=4*self.nbytes)
				print("\t" + test_path)
				with test_lmdb.begin(write=True) as txn:
					for s in partition.test_set():
						datum = caffe.proto.caffe_pb2.Datum()
						datum.channels = len(self._format["data"])
						datum.height = 1
						datum.width  = 1
						datum.float_data.extend(s["x"]);
						datum.label = s["y"]
						txn.put(key=s["id"].encode("ascii"), value=datum.SerializeToString())
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
			self._parts.append(Database.Partition(self, "part"+str(i), train_set, test_set))

	def split_by_target(self):

		for g in self._groups:
			self._parts.append(Database.Partition(self, g, None, [g]))


	def normalize(self):

		n = len(self._samples)
		d = len(self._format["data"])

		# compute mean for each feature
		mean = [0 for i in range(d)]
		for i in self._samples:
			for j in range(d):
				mean[j] += i["x"][j]
		mean = [i/n for i in mean]

		# compute standard deviation for each feature
		sd = [0 for i in range(d)]
		for i in self._samples:
			for j in range(d):
				sd[j] += (i["x"][j] - mean[j])**2
		sd = [(i/n)**(0.5) for i in sd]

		# normalize each feature- subtract mean, divide by std dev		
		for i in self._samples:
			for j in range(d):
				if sd[j]:
					i["x"][j] = (i["x"][j] - mean[j])/float(sd[j])
				else:
					i["x"][j] = 0.0

	def randomize(self):
		
		# shuffle the order of samples in the master list,
		# and also in the dictionary of by-target groupings
		random.shuffle(self._samples)
		for group in self._groups:
			random.shuffle(self._groups[group])

	class Partition:

		def __init__(self, db, name, train, test):

			self._db = db
			self._name = name
			if train: self._train_set = set(train)
			else: self._train_set = None
			if test: self._test_set  = set(test)
			else: self._test_set = None

		def train_set(self):

			for s in self._db._samples:
				if s["group"] in self._train_set:
					yield s

		def test_set(self):

			for s in self._db._samples:
				if s["group"] in self._test_set:
					yield s






if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog=__file__,
		description='Convert a csv database into partitioned lmdb files.',
		epilog=None)
	parser.add_argument('INPUT_FILE')
	parser.add_argument('OUTPUT_DIR')
	parser.add_argument('--mode', '-m', type=str)
	args = parser.parse_args()

	if args.mode == "balanced_partitions":
		mode = 0
	elif args.mode == "split_by_target":
		mode = 1
	else:
		print("Error: mode argument not recognized, try 'balanced_partitions' or 'split_by_target'")
		exit(1)

	print("Gathering data from " + args.INPUT_FILE)
	try: db = Database(args.INPUT_FILE, "DUDE_SCOREDATA")
	except IOError:
		print("Error: could not access the input file")
		exit(1)
	except KeyError:
		print("Error: unknown input file format")
		exit(1)
	print(str(db.nbytes) + " bytes read")

	print("Normalizing and shuffling data")
	db.normalize()
	db.randomize()

	if mode == 0:
		print("Generating balanced partitions")
		db.balanced_partition(n=10)
	elif mode == 1:
		print("Splitting data by individual targets")
		db.split_by_target()

	print("Converting to lmdb format")
	try: db.write_lmdb(args.OUTPUT_DIR)
	except IOError:
		print("Error: could not access the output location")
		exit(1)

	print("Done, without errors.")